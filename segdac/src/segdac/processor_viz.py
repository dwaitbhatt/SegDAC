"""Visualization helpers for :class:`~segdac.processor.SegDACProcessor` (ported from ``test.py``)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes

from segdac.networks.image_segmentation_models.grounded_efficientvit_sam import (
    GroundedEfficientVitSam,
    get_image_covered_by_predicted_masks,
)
from segdac.networks.segments_encoders.segment_token_utils import (
    spatial_pool_selection_mask,
)

# Same palette as ManiSkill demo_vis_segmentation.py / test.py
COLOR_PALETTE = np.array(
    [
        [164, 74, 82],
        [85, 200, 95],
        [149, 88, 210],
        [111, 185, 57],
        [89, 112, 223],
        [194, 181, 43],
        [219, 116, 216],
        [71, 146, 48],
        [214, 70, 164],
        [157, 183, 57],
        [154, 68, 158],
        [82, 196, 133],
        [225, 64, 121],
        [50, 141, 77],
        [224, 59, 84],
        [74, 201, 189],
        [237, 93, 68],
        [77, 188, 225],
        [182, 58, 29],
        [77, 137, 200],
        [230, 155, 53],
        [93, 90, 162],
        [213, 106, 38],
        [150, 153, 224],
        [120, 134, 37],
        [186, 135, 220],
        [78, 110, 27],
        [182, 61, 117],
        [106, 184, 145],
        [184, 62, 65],
        [44, 144, 124],
        [229, 140, 186],
        [48, 106, 60],
        [167, 102, 155],
        [160, 187, 114],
        [150, 74, 107],
        [204, 177, 86],
        [34, 106, 77],
        [226, 129, 94],
        [72, 106, 45],
        [222, 125, 129],
        [101, 146, 86],
        [150, 89, 44],
        [147, 138, 73],
        [210, 156, 106],
        [102, 96, 32],
        [168, 124, 34],
    ],
    dtype=np.uint8,
)


def maniskill_seg_to_color_rgb(
    obs_seg_1env: np.ndarray, selected_id: int | None
) -> np.ndarray:
    """Colorize simulator segmentation ids (H, W, 1) int, optional single-id highlight."""
    seg = obs_seg_1env.astype(np.int64)
    if selected_id is not None:
        seg = seg == int(selected_id)
    seg = np.remainder(seg, len(COLOR_PALETTE))
    seg_rgb = np.zeros(
        (obs_seg_1env.shape[0], obs_seg_1env.shape[1], 3), dtype=np.uint8
    )
    for seg_id, color in enumerate(COLOR_PALETTE):
        seg_rgb[seg[..., 0] == seg_id] = color
    return seg_rgb


def to_uint8_image(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def segdac_masks_to_instance_rgb(
    binary_masks: np.ndarray, height: int, width: int
) -> np.ndarray:
    n = binary_masks.shape[0]
    out = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(n):
        c = COLOR_PALETTE[i % len(COLOR_PALETTE)].astype(np.float32) / 255.0
        m = binary_masks[i, 0] > 0.5
        out[m] = c
    return out


def _fg_boundary_hw(mask_hw: torch.Tensor) -> torch.Tensor:
    u = (mask_hw > 0.5).float().view(1, 1, mask_hw.shape[-2], mask_hw.shape[-1])
    eroded = 1.0 - F.max_pool2d(1.0 - u, kernel_size=3, stride=1, padding=1)
    inside = u > 0.5
    core = eroded > 0.99
    return (inside & ~core).squeeze(0).squeeze(0)


def _thicken_bool_hw(b_hw: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return b_hw
    h, w = int(b_hw.shape[-2]), int(b_hw.shape[-1])
    x = b_hw.float().view(1, 1, h, w)
    k = 2 * int(radius) + 1
    pad = int(radius)
    return (F.max_pool2d(x, kernel_size=k, stride=1, padding=pad) > 0.5).squeeze(
        0
    ).squeeze(0)


_TOKEN_OUTLINE_PASTELS: tuple[tuple[float, float, float], ...] = (
    (0.78, 0.86, 0.96),
    (0.96, 0.82, 0.90),
    (0.82, 0.92, 0.84),
    (0.94, 0.88, 0.78),
    (0.88, 0.82, 0.95),
    (0.80, 0.90, 0.92),
    (0.92, 0.86, 0.82),
    (0.84, 0.88, 0.92),
    (0.90, 0.84, 0.88),
    (0.78, 0.90, 0.88),
    (0.92, 0.90, 0.78),
    (0.86, 0.88, 0.90),
)


class TokenPcaVizState:
    """Holds PCA basis from the previous frame for Procrustes alignment (per-processor)."""

    __slots__ = ("_pca_prev_vh", "_pca_prev_c")

    def __init__(self) -> None:
        self._pca_prev_vh: np.ndarray | None = None
        self._pca_prev_c: int | None = None

    def reset(self) -> None:
        self._pca_prev_vh = None
        self._pca_prev_c = None

    def pca_features_to_rgb_01(self, feats: np.ndarray) -> np.ndarray:
        if feats.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        c = int(feats.shape[1])
        if feats.shape[0] == 1:
            return np.full((1, 3), 0.5, dtype=np.float64)
        x = feats.astype(np.float64)
        mean = x.mean(axis=0, keepdims=True)
        xc = x - mean
        _, _, vh = np.linalg.svd(xc, full_matrices=False)
        n_comp = min(3, int(vh.shape[0]))
        if n_comp < 3:
            self._pca_prev_vh = None
            self._pca_prev_c = None
            proj = xc @ vh[:n_comp].T
            if n_comp < 3:
                proj = np.concatenate(
                    [proj, np.zeros((proj.shape[0], 3 - n_comp), dtype=np.float64)],
                    axis=1,
                )
        else:
            b = vh[:3].copy()
            if (
                self._pca_prev_vh is not None
                and self._pca_prev_c == c
                and self._pca_prev_vh.shape == (3, c)
            ):
                m_align = b @ self._pca_prev_vh.T
                u_a, _, vt_a = np.linalg.svd(m_align, full_matrices=False)
                r_a = u_a @ vt_a
                b = r_a @ b
            self._pca_prev_vh = b.copy()
            self._pca_prev_c = c
            proj = xc @ b.T
        lo = proj.min(axis=0)
        hi = proj.max(axis=0)
        denom = np.maximum(hi - lo, 1e-6)
        return np.clip((proj - lo) / denom, 0.0, 1.0)


@torch.inference_mode()
def build_token_encoder_viz(
    sd,
    pixels_seg: torch.Tensor,
    seg_enc: torch.nn.Module,
    seg_model: GroundedEfficientVitSam,
    token_encoder: str,
    sam_feature_map: torch.Tensor | None,
    min_pixels: int,
    pca_state: TokenPcaVizState,
    *,
    dense_feature_map_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Segmenter-res float RGB (S, S, 3) in [0, 1], matching ``test.py`` token panel logic.
    """
    s = int(seg_model.segmenter_image_size)
    device = pixels_seg.device
    bg = pixels_seg[0].clamp(0, 1).permute(1, 2, 0).contiguous()
    n_seg = int(sd.batch_size[0])
    if n_seg == 0:
        return bg

    alpha_feat = 0.56
    bg_dim_mul = 0.64
    pca_cell_dim = 0.38
    pca_cell_bright = 1.06

    if token_encoder == "random":
        out = bg.clone()
        out[:, :, 1] = out[:, :, 1] * 0.55 + 0.22
        return out.clamp(0, 1)

    panel_b = 0

    if token_encoder in ("sam", "sam_image", "dinov2"):
        if token_encoder == "sam":
            if sam_feature_map is None:
                return bg
            feat = sam_feature_map
        else:
            if dense_feature_map_override is not None:
                feat = dense_feature_map_override
            else:
                enc = getattr(seg_enc, "image_encoder", None)
                if enc is None:
                    return bg
                feat = enc(pixels_seg)
        feat = feat.float()
        if int(feat.shape[0]) <= panel_b:
            return bg
        hf, wf = int(feat.shape[-2]), int(feat.shape[-1])
        c = int(feat.shape[1])
        vecs = feat[panel_b].permute(1, 2, 0).reshape(-1, c).contiguous().cpu().numpy()
        rgb_k = pca_state.pca_features_to_rgb_01(vecs)
        rgb_low = torch.from_numpy(rgb_k.reshape(hf, wf, 3)).to(
            device=device, dtype=torch.float32
        ).permute(2, 0, 1)
        pca_hi = (
            F.interpolate(
                rgb_low.unsqueeze(0),
                size=(s, s),
                mode="nearest",
            )[0]
            .permute(1, 2, 0)
            .contiguous()
        )
        mp = int(min_pixels)
        sel = spatial_pool_selection_mask(
            sd,
            feat,
            segmenter_image_size=s,
            min_pixels=mp,
        )
        pooled_union = torch.zeros(hf, wf, dtype=torch.bool, device=device)
        ids = sd["image_ids"]
        for si in range(n_seg):
            if int(ids[si].item()) != panel_b:
                continue
            pooled_union = pooled_union | sel[si]
        w_low = pooled_union.float().view(1, 1, hf, wf)
        w_up = F.interpolate(w_low, size=(s, s), mode="nearest")[0, 0].unsqueeze(-1)
        gain = pca_cell_dim + (pca_cell_bright - pca_cell_dim) * w_up
        pca_hi = (pca_hi * gain).clamp(0.0, 1.0)
        bg_dim = bg * bg_dim_mul
        out = (1.0 - alpha_feat) * bg_dim + alpha_feat * pca_hi
        rel = sd["relative_segment_ids"]
        line_w = 0.66
        n_pal = len(_TOKEN_OUTLINE_PASTELS)
        for si in range(n_seg):
            if int(ids[si].item()) != panel_b:
                continue
            one = sel[si]
            if not bool(one.any().item()):
                continue
            w_i = F.interpolate(
                one.float().view(1, 1, hf, wf), size=(s, s), mode="nearest"
            )[0, 0]
            bd = _fg_boundary_hw(w_i)
            if not bool(bd.any().item()):
                continue
            bd = _thicken_bool_hw(bd, radius=2)
            pi = int(rel[si].item()) % n_pal
            col = _TOKEN_OUTLINE_PASTELS[pi]
            b3 = bd.unsqueeze(-1).to(dtype=out.dtype)
            c_t = out.new_tensor(col).view(1, 1, 3)
            out = out * (1.0 - line_w * b3) + c_t * (line_w * b3)
        return out.clamp(0, 1)

    return bg


def render_yolo_world_detections(
    rgb_u8: np.ndarray,
    xyxy_list: list,
    classes_list: list,
    batch_index: int,
    grounding_tags: list[str],
) -> np.ndarray:
    """YOLO-World boxes are in 640x640 space; map to native (H, W) of rgb_u8."""
    h, w = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
    if (
        not xyxy_list
        or batch_index >= len(xyxy_list)
        or batch_index >= len(classes_list)
    ):
        return rgb_u8.copy()
    xyxy = xyxy_list[batch_index]
    cls_t = classes_list[batch_index]
    if xyxy is None or xyxy.numel() == 0:
        return rgb_u8.copy()
    scale = torch.tensor([w, h, w, h], dtype=xyxy.dtype, device=xyxy.device) / 640.0
    boxes = (xyxy * scale).round()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, float(w - 1))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, float(h - 1))
    labels: list[str] = []
    cls_flat = cls_t.long().flatten()
    n_box = int(boxes.shape[0])
    pal_n = int(len(COLOR_PALETTE))
    box_colors: list[tuple[int, int, int]] = []
    for bi in range(n_box):
        ci = int(cls_flat[bi].item()) if bi < cls_flat.numel() else 0
        rgb = COLOR_PALETTE[ci % pal_n]
        box_colors.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
        if 0 <= ci < len(grounding_tags):
            labels.append(grounding_tags[ci])
        else:
            labels.append(f"class_{ci}")
    img = torch.from_numpy(rgb_u8).permute(2, 0, 1).contiguous()
    out = draw_bounding_boxes(
        img,
        boxes.cpu(),
        labels=labels,
        colors=box_colors,
        width=max(1, min(3, w // 128)),
    )
    return out.permute(1, 2, 0).contiguous().numpy().astype(np.uint8)


def viz_mask_union_tensor(
    pixels_seg: torch.Tensor, segments_data
) -> torch.Tensor:
    """BCHW float [0,1] union of predicted masks times RGB (segmenter resolution)."""
    return get_image_covered_by_predicted_masks(pixels_seg, segments_data)
