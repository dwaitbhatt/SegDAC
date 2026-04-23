import torch
import torch.nn as nn
from tensordict import TensorDict

from segdac.networks.segments_encoders.segment_token_utils import (
    pool_spatial_map_to_per_segment_embeddings,
    segments_token_tensordict,
)


class SamEncoderEmbeddingsSegmentsEncoder(nn.Module):
    def __init__(
        self,
        segmenter_image_size: int,
        min_pixels: int,
    ):
        super().__init__()
        self.segmenter_image_size = segmenter_image_size
        self.min_pixels = min_pixels

    def forward(
        self, segments_data: TensorDict, sam_encoder_embeddings: torch.Tensor
    ) -> TensorDict:
        """
        segments_data: batch_size (num_segments,)
        sam_encoder_embeddings: batch_size (num_envs, 1, C, h, w) after one squeeze
            becomes (num_envs, C, h, w) inside the pool helper.
        """
        feature_map = sam_encoder_embeddings.squeeze(1)
        segments_embeddings = pool_spatial_map_to_per_segment_embeddings(
            segments_data,
            feature_map,
            segmenter_image_size=self.segmenter_image_size,
            min_pixels=self.min_pixels,
        )
        return segments_token_tensordict(segments_data, segments_embeddings)
