"""
Shared building blocks for per-segment token TensorDicts (SAM map pooling, metadata).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict


def spatial_pool_cell_counts(
    segments_data: TensorDict,
    feature_map: torch.Tensor,
    *,
    segmenter_image_size: int,
) -> torch.Tensor:
    """
    For each segment mask, count how many binary mask pixels fall in each
    ``feature_map`` cell (strided ``segmenter_image_size // h_f`` window).

    Returns ``(N, 1, h_f, w_f)`` float tensor (same device/dtype as masks after float()).
    """
    mask_to_embedding_ratio = segmenter_image_size // int(feature_map.shape[-1])
    return F.conv2d(
        input=segments_data["binary_masks"].float(),
        weight=torch.ones(
            size=(1, 1, mask_to_embedding_ratio, mask_to_embedding_ratio),
            dtype=torch.float32,
            device=segments_data.device,
        ),
        bias=None,
        stride=(mask_to_embedding_ratio, mask_to_embedding_ratio),
        padding=0,
    )


def spatial_pool_selection_mask(
    segments_data: TensorDict,
    feature_map: torch.Tensor,
    *,
    segmenter_image_size: int,
    min_pixels: int,
) -> torch.Tensor:
    """
    Per-segment boolean mask over encoder cells: ``True`` iff that cell is
    included in mean-pooling (``count >= min_pixels``).

    Shape: ``(N, h_f, w_f)``. If ``N == 0``, returns shape ``(0, h_f, w_f)``.
    """
    h_f = int(feature_map.shape[-2])
    w_f = int(feature_map.shape[-1])
    if int(segments_data.batch_size[0]) == 0:
        return torch.empty(
            (0, h_f, w_f),
            dtype=torch.bool,
            device=feature_map.device,
        )
    counts = spatial_pool_cell_counts(
        segments_data,
        feature_map,
        segmenter_image_size=segmenter_image_size,
    )
    return counts.squeeze(1) >= min_pixels


def pool_spatial_map_to_per_segment_embeddings(
    segments_data: TensorDict,
    feature_map: torch.Tensor,
    *,
    segmenter_image_size: int,
    min_pixels: int,
) -> torch.Tensor:
    """
    Pool a dense spatial feature map to one vector per segment, using the same
    strided window count over ``binary_masks`` as ``SamEncoderEmbeddingsSegmentsEncoder``.

    feature_map: (B, C, h, w) for B input images, aligned with ``image_ids`` per mask.
    Returns: (N, C) for N = number of segments.
    """
    if int(segments_data.batch_size[0]) == 0:
        c = int(feature_map.shape[1])
        return torch.empty(0, c, device=feature_map.device, dtype=feature_map.dtype)

    embeddings_masks_pixels_count = spatial_pool_cell_counts(
        segments_data,
        feature_map,
        segmenter_image_size=segmenter_image_size,
    )

    nb_segments = int(embeddings_masks_pixels_count.shape[0])

    sam_embeddings_selection_mask = (
        embeddings_masks_pixels_count.squeeze(1) >= min_pixels
    )

    segments_embeddings: list[torch.Tensor] = []
    for segment_index in range(nb_segments):
        image_id = int(segments_data["image_ids"][segment_index].item())
        image_feats = feature_map[image_id]
        segment_embeddings = image_feats[
            :, sam_embeddings_selection_mask[segment_index]
        ]
        segment_embeddings = segment_embeddings.mean(dim=1)
        segments_embeddings.append(segment_embeddings)
    return torch.stack(segments_embeddings)


def segments_token_tensordict(
    segments_data: TensorDict,
    embeddings: torch.Tensor,
) -> TensorDict:
    """
    Assemble the standard ``segments_encoder_output`` TensorDict from
    per-segment embeddings and ``segments_data`` (ids, coords, ...).
    For ``N==0`` segments, pass an empty ``embeddings`` tensor of shape (0, D)
    (``D`` used for the zero-row layout).
    """
    nb_segments = int(segments_data.batch_size[0])
    device = segments_data.device
    image_ids = segments_data["image_ids"]
    is_latent_tokens = torch.zeros(
        (nb_segments,), device=device, dtype=torch.bool
    )
    if nb_segments == 0:
        return TensorDict(
            source={
                "image_ids": image_ids,
                "relative_segment_ids": segments_data["relative_segment_ids"],
                "is_latent_tokens": is_latent_tokens,
                "embeddings": embeddings,
                "coords": segments_data["coords"],
            },
            batch_size=(0,),
            device=device,
        )
    return TensorDict(
        source={
            "image_ids": image_ids,
            "relative_segment_ids": segments_data["relative_segment_ids"],
            "is_latent_tokens": is_latent_tokens,
            "embeddings": embeddings,
            "coords": segments_data["coords"],
        },
        batch_size=torch.Size([nb_segments]),
        device=device,
    )
