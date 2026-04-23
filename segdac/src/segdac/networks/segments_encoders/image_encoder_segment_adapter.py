"""
Wrap any :mod:`segdac.networks.image_encoders` module that implements
``forward(image)`` on ``(B,3,H,W)`` float in ``[0,1]`` into the standard
per-segment token ``TensorDict`` layout.

* ``mode="global"`` — encoder returns one vector per image row (e.g. DINO, Random);
  we call it on ``rgb_segments`` stacked as a batch.
* ``mode="spatial_from_full_image"`` — encoder returns a feature map ``(B,C,h,w)``;
  we call it on the full segmenter-res frame once and pool with
  :func:`pool_spatial_map_to_per_segment_embeddings` (second EfficientViT-SAM forward
  if used next to :class:`GroundedEfficientVitSam`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Literal

from segdac.networks.segments_encoders.segment_token_utils import (
    pool_spatial_map_to_per_segment_embeddings,
    segments_token_tensordict,
)

ModeT = Literal["global", "spatial_from_full_image"]


def _normalize_backbone_out(out: object) -> torch.Tensor:
    if isinstance(out, dict):
        if "x_norm_clstoken" in out:
            return out["x_norm_clstoken"]
        return next(iter(out.values()))  # type: ignore[return-value]
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            f"image_encoder must return a Tensor or dict, got {type(out)}"
        )
    return out


def _infer_global_out_dim(image_encoder: nn.Module) -> int:
    od = getattr(image_encoder, "output_dim", None)
    if isinstance(od, int):
        return int(od)
    m = getattr(image_encoder, "model", image_encoder)
    ed = getattr(m, "embed_dim", None)
    if ed is not None:
        return int(ed)
    p = next(image_encoder.parameters(), None)
    if p is None:
        raise ValueError("Cannot infer embedding dim: pass out_dim= to the adapter")
    t = p.device
    dt = p.dtype
    with torch.inference_mode():
        d = _normalize_backbone_out(
            image_encoder(
                torch.zeros(1, 3, 224, 224, device=t, dtype=dt)
            )
        )
    return int(d.shape[-1])


class ImageEncoderSegmentTokensAdapter(nn.Module):
    """
    ``forward(segments_data, image_01=None)`` — when ``mode`` is
    ``spatial_from_full_image``, pass segmenter-res ``(B,3,S,S)`` in ``image_01``.
    Set :attr:`needs_full_image` is True in that mode (see :func:`test.extract_object_tokens`).
    """

    needs_full_image: bool

    def __init__(
        self,
        image_encoder: nn.Module,
        *,
        mode: ModeT = "global",
        segmenter_image_size: int = 512,
        min_pixels: int = 4,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self._mode: ModeT = mode
        self.segmenter_image_size = segmenter_image_size
        self.min_pixels = min_pixels
        self._out_dim_override: int | None = out_dim
        self.needs_full_image = mode == "spatial_from_full_image"

    def out_dim(self) -> int:
        if self._out_dim_override is not None:
            return int(self._out_dim_override)
        return _infer_global_out_dim(self.image_encoder)

    @torch.inference_mode()
    def forward(
        self,
        segments_data: TensorDict,
        image_01: torch.Tensor | None = None,
    ) -> TensorDict:
        if self._mode == "global":
            n = int(segments_data.batch_size[0])
            if n == 0:
                d = self.out_dim()
                p = next(self.image_encoder.parameters(), None)
                dt = p.dtype if p is not None else torch.float32
                return segments_token_tensordict(
                    segments_data,
                    torch.empty(0, d, device=segments_data.device, dtype=dt),
                )
            out = self.image_encoder(segments_data["rgb_segments"])
            out = _normalize_backbone_out(out)
            if out.dim() != 2:
                raise ValueError(
                    f"global mode expects (N, D) encoder output, got {tuple(out.shape)}"
                )
            return segments_token_tensordict(segments_data, out)

        if self._mode == "spatial_from_full_image":
            if image_01 is None:
                raise ValueError(
                    "spatial_from_full_image requires image_01 (B,3,S,S) in [0,1]"
                )
            raw = self.image_encoder(image_01)
            if isinstance(raw, dict):
                raise ValueError(
                    "spatial mode: image_encoder must return a 4D tensor, not a dict"
                )
            feat = raw
            if feat.dim() != 4:
                raise ValueError(
                    "spatial mode expects (B, C, h, w) feature map from image_encoder, "
                    f"got {tuple(feat.shape)}"
                )
            if int(segments_data.batch_size[0]) == 0:
                c = int(feat.shape[1])
                return segments_token_tensordict(
                    segments_data,
                    torch.empty(0, c, device=feat.device, dtype=feat.dtype),
                )
            segs = pool_spatial_map_to_per_segment_embeddings(
                segments_data,
                feat,
                segmenter_image_size=self.segmenter_image_size,
                min_pixels=self.min_pixels,
            )
            return segments_token_tensordict(segments_data, segs)

        raise NotImplementedError(self._mode)
