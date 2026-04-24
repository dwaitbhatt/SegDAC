"""
Unified SegDAC inference: detection, segmentation, token encoding, profiling, and viz hooks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from segdac.networks.image_segmentation_models.grounded_efficientvit_sam import (
    GroundedEfficientVitSam,
)
from segdac.networks.segments_encoders.sam_encoder_segments_encoder import (
    SamEncoderEmbeddingsSegmentsEncoder,
)
from segdac.networks.segments_encoders.segment_token_utils import (
    pool_spatial_map_to_per_segment_embeddings,
    segments_token_tensordict,
)
from segdac.processor_viz import (
    TokenPcaVizState,
    build_token_encoder_viz,
    render_yolo_world_detections,
    segdac_masks_to_instance_rgb,
    to_uint8_image,
    viz_mask_union_tensor,
)

TokenEncoderName = Literal["sam", "dinov2", "random", "sam_image"]


def _sync_cuda(device: torch.device | str) -> None:
    d = torch.device(device) if isinstance(device, str) else device
    if d.type == "cuda":
        torch.cuda.synchronize()


def pixels_to_segmenter_res(
    pixels_01: torch.Tensor, device: torch.device | str, seg_model: GroundedEfficientVitSam
) -> torch.Tensor:
    s = int(seg_model.segmenter_image_size)
    x = pixels_01.to(device)
    if int(x.shape[2]) == s and int(x.shape[3]) == s:
        return x
    return F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)


@dataclass
class SegDACTimings:
    """Per-frame wall times (seconds) for the last completed stage(s)."""

    rgb_to_segmenter_s: float | None = None
    object_detection_s: float | None = None
    segmentation_s: float | None = None
    token_encoder_s: float | None = None


@dataclass
class SegDACDetectionBoxes:
    """Object-detection boxes for one batch image, in **native input** pixel coordinates."""

    xyxy: torch.Tensor
    class_ids: torch.Tensor


class SegDACProcessResult:
    """
    One ``process(image)`` call. Heavy work is memoized; properties share one forward.

    Access order does not change correctness: e.g. ``object_tokens`` first fills the cache
    so ``bboxes`` / ``seg_masks`` / ``img_features`` return without extra model work.
    """

    def __init__(self, processor: SegDACProcessor, image_01: torch.Tensor) -> None:
        self._processor = processor
        self._image_01 = image_01
        self._timings = SegDACTimings()

        self._pixels_seg: torch.Tensor | None = None
        self._segments_data: TensorDict | None = None
        self._sam_raw: torch.Tensor | None = None
        self._encoder_out: TensorDict | None = None
        self._feat_map: torch.Tensor | None = None  # spatial enc map when applicable
        self._stage_segment = False
        self._stage_tokens = False

    @property
    def native_hw(self) -> tuple[int, int]:
        """Input spatial size ``(H, W)`` for batch layout ``(B, 3, H, W)``."""
        _ = self._processor
        _, _, h, w = self._image_01.shape
        return int(h), int(w)

    def _run_segmentation_pipeline(self) -> None:
        """Resize to segmenter resolution, then detection + SAM masks (optionally record phase timings)."""
        p = self._processor
        dev = p.device
        img = self._image_01.to(dev)
        if p.profiling:
            _sync_cuda(dev)
            t_prep0 = time.perf_counter()
        self._pixels_seg = pixels_to_segmenter_res(img, dev, p.seg_model)
        if p.profiling:
            _sync_cuda(dev)
            t_prep1 = time.perf_counter()
            self._timings.rgb_to_segmenter_s = t_prep1 - t_prep0
        tok = cast(TokenEncoderName, p.token_encoder)
        r = p.seg_model.segment(
            self._pixels_seg,
            return_sam_encoder_embeddings=(tok == "sam"),
            return_phase_timings=p.profiling,
        )
        if tok == "sam":
            self._segments_data, self._sam_raw = r
        else:
            self._segments_data = r
            self._sam_raw = None
        if p.profiling:
            _sync_cuda(dev)
            ph = getattr(p.seg_model, "last_segment_phase_timings", None) or {}
            self._timings.object_detection_s = float(ph.get("object_detection_s", 0.0))
            self._timings.segmentation_s = float(ph.get("segmentation_s", 0.0))
            # token_encoder_s set in _run_object_token_encoder
        self._stage_segment = True

    def _run_object_token_encoder(self) -> None:
        """Pool / encode per-segment embeddings from segmentation output (optionally record timings)."""
        if not self._stage_segment:
            self._run_segmentation_pipeline()
        assert self._pixels_seg is not None and self._segments_data is not None
        p = self._processor
        dev = p.device
        tok = cast(TokenEncoderName, p.token_encoder)
        sd = self._segments_data
        enc = p.segments_encoder
        mp = int(getattr(enc, "min_pixels", 4))
        S = int(p.seg_model.segmenter_image_size)

        if p.profiling:
            _sync_cuda(dev)
            t_tok0 = time.perf_counter()

        if tok == "sam":
            assert self._sam_raw is not None
            sam_5d = self._sam_raw.unsqueeze(1)
            self._encoder_out = enc(sd, sam_5d)
            self._feat_map = None
        elif getattr(enc, "needs_full_image", False):
            if self._feat_map is None:
                feat = enc.image_encoder(self._pixels_seg)
                if not isinstance(feat, torch.Tensor) or feat.dim() != 4:
                    raise TypeError("spatial token encoder must return a 4D tensor")
                self._feat_map = feat
            feat = self._feat_map
            emb = pool_spatial_map_to_per_segment_embeddings(
                sd,
                feat,
                segmenter_image_size=S,
                min_pixels=mp,
            )
            self._encoder_out = segments_token_tensordict(sd, emb)
        else:
            self._feat_map = None
            self._encoder_out = enc(sd)

        if p.profiling:
            _sync_cuda(dev)
            t_tok1 = time.perf_counter()
            self._timings.token_encoder_s = t_tok1 - t_tok0

        self._stage_tokens = True

    def _ensure_segment(self) -> None:
        if not self._stage_segment:
            self._run_segmentation_pipeline()

    def _ensure_tokens(self) -> None:
        if not self._stage_tokens:
            self._run_object_token_encoder()

    @property
    def bboxes(self) -> SegDACDetectionBoxes:
        """YOLO-World boxes mapped to native input resolution (batch index 0)."""
        self._ensure_segment()
        p = self._processor
        h, w = self.native_hw
        xyxy_list = getattr(p.seg_model, "last_yolo_xyxy", None) or []
        cls_list = getattr(p.seg_model, "last_yolo_classes", None) or []
        if not xyxy_list or not cls_list:
            device = p.device
            return SegDACDetectionBoxes(
                xyxy=torch.zeros(0, 4, device=device),
                class_ids=torch.zeros(0, dtype=torch.long, device=device),
            )
        xyxy = xyxy_list[0]
        cls_t = cls_list[0]
        if xyxy is None or xyxy.numel() == 0:
            device = p.device
            return SegDACDetectionBoxes(
                xyxy=torch.zeros(0, 4, device=device),
                class_ids=torch.zeros(0, dtype=torch.long, device=device),
            )
        scale = torch.tensor([w, h, w, h], dtype=xyxy.dtype, device=xyxy.device) / 640.0
        boxes = (xyxy * scale).round()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, float(w - 1))
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, float(h - 1))
        return SegDACDetectionBoxes(xyxy=boxes, class_ids=cls_t.long().flatten())

    @property
    def seg_masks(self) -> torch.Tensor:
        """Per-instance binary masks ``(N, 1, S, S)`` uint8 at segmenter resolution."""
        self._ensure_segment()
        assert self._segments_data is not None
        return self._segments_data["binary_masks"]

    @property
    def seg_mask_class_ids(self) -> torch.Tensor:
        """
        Per-mask YOLO-World class indices ``(N,)`` ``torch.long``, **aligned 1:1** with
        ``seg_masks`` rows (same ordering as ``segments_data['binary_masks']``).
        """
        self._ensure_segment()
        assert self._segments_data is not None
        return self._segments_data["classes"].long().flatten()

    @property
    def seg_mask_classes(self) -> list[str]:
        """
        Per-mask text labels from :attr:`SegDACProcessor.grounding_text_tags`, **same
        length and order** as ``seg_masks`` (index ``i`` corresponds to mask row ``i``).

        Class IDs come from ``segments_data['classes']``; each ID is mapped to
        ``grounding_text_tags[id]`` when in range, otherwise ``\"class_{id}\"`` (same
        convention as bbox visualization in :func:`segdac.processor_viz.render_yolo_world_detections`).
        """
        self._ensure_segment()
        assert self._segments_data is not None
        tags = self._processor.grounding_text_tags
        ids = self._segments_data["classes"].long().flatten()
        n_tag = len(tags)
        out: list[str] = []
        for i in range(int(ids.numel())):
            ci = int(ids[i].item())
            if 0 <= ci < n_tag:
                out.append(tags[ci])
            else:
                out.append(f"class_{ci}")
        return out

    @property
    def img_features(self) -> torch.Tensor:
        """
        Full-image token-encoder representation (no per-mask pooling).

        * ``sam``: SAM ViT map ``(B, C, h, w)`` from the segmenter forward.
        * ``dinov2`` / ``sam_image``: dense map ``(B, C, h, w)`` (single encoder call).
        * ``random``: global vector per batch row ``(B, D)`` from the random encoder on
          segmenter-res RGB.
        """
        self._ensure_segment()
        tok = cast(TokenEncoderName, self._processor.token_encoder)
        if tok == "sam":
            assert self._sam_raw is not None
            return self._sam_raw
        enc = self._processor.segments_encoder
        assert self._pixels_seg is not None
        if getattr(enc, "needs_full_image", False):
            if self._feat_map is not None:
                return self._feat_map
            out = enc.image_encoder(self._pixels_seg)
            if not isinstance(out, torch.Tensor):
                raise TypeError("image_encoder must return a Tensor for img_features")
            if out.dim() == 4:
                self._feat_map = out
                return out
            return out
        out = enc.image_encoder(self._pixels_seg)
        if isinstance(out, dict):
            raise TypeError("img_features: unexpected dict encoder output")
        return out

    @property
    def object_tokens(self) -> torch.Tensor:
        """Per-segment embeddings ``(N, D)`` (same as ``segments_encoder_output['embeddings']``)."""
        self._ensure_tokens()
        assert self._encoder_out is not None
        return self._encoder_out["embeddings"]

    @property
    def segments_encoder_output(self) -> TensorDict:
        """Full encoder TensorDict (requires token stage)."""
        self._ensure_tokens()
        assert self._encoder_out is not None
        return self._encoder_out

    @property
    def segments_data(self) -> TensorDict:
        """Raw segment tensor dict from ``GroundedEfficientVitSam.segment``."""
        self._ensure_segment()
        assert self._segments_data is not None
        return self._segments_data

    @property
    def pixels_segmenter_resolution(self) -> torch.Tensor:
        """Input RGB resized to ``(B, 3, S, S)`` in ``[0, 1]``."""
        self._ensure_segment()
        assert self._pixels_seg is not None
        return self._pixels_seg

    def last_processing_times(self) -> SegDACTimings | None:
        """Timings for work done on this result (when processor profiling is on)."""
        if not self._processor.profiling:
            return None
        return self._timings


class SegDACProcessor:
    """
    High-level SegDAC runner: build models once, then ``process(image)`` per frame.

    Parameters mirror ``test.py`` defaults (weights under ``./weights/`` relative to CWD).
    """

    def __init__(
        self,
        *,
        token_encoder: TokenEncoderName = "sam",
        device: str = "cuda",
        grounding_text_tags: list[str] | None = None,
        object_detector_weights_path: str | Path = "weights/yolov8s-worldv2.pt",
        segmenter_weights_path: str | Path = "weights/efficientvit_sam_l0.pt",
        segmenter_model_name: str = "efficientvit-sam-l0",
        random_encoder_dim: int = 128,
        profiling: bool = False,
    ) -> None:
        if grounding_text_tags is None:
            grounding_text_tags = ["robot arm", "red cube"]
        self.token_encoder: TokenEncoderName = token_encoder
        self.profiling = profiling
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.seg_model = GroundedEfficientVitSam(
            device=str(self.device),
            grounding_text_tags=list(grounding_text_tags),
            object_detector_weights_path=str(object_detector_weights_path),
            segmenter_weights_path=str(segmenter_weights_path),
            segmenter_model_name=segmenter_model_name,
        )
        S = int(self.seg_model.segmenter_image_size)
        self._S = S

        if token_encoder == "sam":
            self.segments_encoder = SamEncoderEmbeddingsSegmentsEncoder(
                S, min_pixels=4
            )
        elif token_encoder == "dinov2":
            from segdac.networks.image_encoders.dinov2 import DinoV2DenseMapEncoder
            from segdac.networks.segments_encoders.image_encoder_segment_adapter import (
                ImageEncoderSegmentTokensAdapter,
            )

            self.segments_encoder = ImageEncoderSegmentTokensAdapter(
                DinoV2DenseMapEncoder(),
                mode="spatial_from_full_image",
                segmenter_image_size=S,
                min_pixels=4,
            )
        elif token_encoder == "random":
            from segdac.networks.image_encoders.random import RandomImageEncoder
            from segdac.networks.segments_encoders.image_encoder_segment_adapter import (
                ImageEncoderSegmentTokensAdapter,
            )

            self.segments_encoder = ImageEncoderSegmentTokensAdapter(
                RandomImageEncoder(int(random_encoder_dim)),
                mode="global",
                segmenter_image_size=S,
                min_pixels=4,
            )
        elif token_encoder == "sam_image":
            from segdac.networks.image_encoders.sam import SamImageEncoder
            from segdac.networks.segments_encoders.image_encoder_segment_adapter import (
                ImageEncoderSegmentTokensAdapter,
            )

            s_img = SamImageEncoder(
                segmenter_model_name=segmenter_model_name,
                segmenter_weights_path=str(segmenter_weights_path),
            )
            self.segments_encoder = ImageEncoderSegmentTokensAdapter(
                s_img,
                mode="spatial_from_full_image",
                segmenter_image_size=S,
                min_pixels=4,
            )
        else:
            raise ValueError(f"Unknown token_encoder: {token_encoder!r}")

        if self.device.type == "cuda":
            self.segments_encoder = self.segments_encoder.to(self.device)

        self.grounding_text_tags = list(grounding_text_tags)
        self._pca_viz = TokenPcaVizState()
        self._last_result: SegDACProcessResult | None = None

    @torch.inference_mode()
    def process(self, image: torch.Tensor) -> SegDACProcessResult:
        """
        Args:
            image: ``(B, 3, H, W)`` float32 in ``[0, 1]``.

        Returns:
            :class:`SegDACProcessResult` with lazy properties.
        """
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError("image must be (B, 3, H, W)")
        self._last_result = SegDACProcessResult(self, image)
        return self._last_result

    def last_frame_processing_times(self) -> SegDACTimings | None:
        """Processing times for the **last** ``process()`` result (if profiling enabled)."""
        if not self.profiling or self._last_result is None:
            return None
        return self._last_result.last_processing_times()

    def _native_rgb_u8(self, result: SegDACProcessResult, batch_index: int = 0) -> np.ndarray:
        img = result._image_01[batch_index].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return to_uint8_image(img)

    @torch.inference_mode()
    def viz_bboxes(
        self,
        result: SegDACProcessResult,
        grounding_tags: list[str] | None = None,
        *,
        batch_index: int = 0,
    ) -> np.ndarray:
        """RGB uint8 with YOLO-World boxes (native resolution)."""
        result._ensure_segment()
        tags = grounding_tags if grounding_tags is not None else self.grounding_text_tags
        rgb_u8 = self._native_rgb_u8(result, batch_index)
        xyxy_list = getattr(self.seg_model, "last_yolo_xyxy", None) or []
        cls_list = getattr(self.seg_model, "last_yolo_classes", None) or []
        conf_list = getattr(self.seg_model, "last_yolo_confidences", None) or []
        return render_yolo_world_detections(
            rgb_u8, xyxy_list, cls_list, conf_list, batch_index, list(tags)
        )

    @torch.inference_mode()
    def viz_seg_masks(
        self,
        result: SegDACProcessResult,
        *,
        target_hw: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Instance-colored mask visualization (uint8 RGB), optionally resized."""
        result._ensure_segment()
        h, w = result.native_hw if target_hw is None else target_hw
        sd = result.segments_data
        bm_t = sd["binary_masks"].float()
        if int(bm_t.shape[-2]) != h or int(bm_t.shape[-1]) != w:
            bm_t = F.interpolate(bm_t, size=(h, w), mode="nearest")
        bm = bm_t.cpu().numpy()
        return to_uint8_image(segdac_masks_to_instance_rgb(bm, h, w))

    @torch.inference_mode()
    def viz_mask_union(
        self,
        result: SegDACProcessResult,
        *,
        target_hw: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Union of predicted masks times RGB (uint8), optionally resized to ``target_hw``."""
        result._ensure_segment()
        u = viz_mask_union_tensor(result.pixels_segmenter_resolution, result.segments_data)
        h, w = result.native_hw if target_hw is None else target_hw
        if int(u.shape[2]) != h or int(u.shape[3]) != w:
            u = F.interpolate(u, size=(h, w), mode="bilinear", align_corners=False)
        return to_uint8_image(u[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())

    @torch.inference_mode()
    def viz_token_encoder(
        self,
        result: SegDACProcessResult,
        *,
        target_hw: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Token-encoder PCA panel (uint8), optionally resized to native ``target_hw``."""
        result._ensure_tokens()
        h, w = result.native_hw if target_hw is None else target_hw
        mp = int(getattr(self.segments_encoder, "min_pixels", 4))
        sam_for_viz = result._sam_raw if self.token_encoder == "sam" else None
        override = (
            result._feat_map
            if self.token_encoder in ("dinov2", "sam_image")
            and result._feat_map is not None
            else None
        )
        tok_hw = build_token_encoder_viz(
            result.segments_data,
            result.pixels_segmenter_resolution,
            self.segments_encoder,
            self.seg_model,
            self.token_encoder,
            sam_for_viz,
            mp,
            self._pca_viz,
            dense_feature_map_override=override,
        )
        if int(tok_hw.shape[0]) != h or int(tok_hw.shape[1]) != w:
            tok_t = tok_hw.permute(2, 0, 1).unsqueeze(0).contiguous()
            tok_t = F.interpolate(
                tok_t, size=(h, w), mode="bilinear", align_corners=False
            )
            tok_hw = tok_t[0].permute(1, 2, 0).contiguous()
        return to_uint8_image(tok_hw.cpu().numpy())
