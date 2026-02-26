import torch.nn.functional as F
import torch
import torch.nn as nn
from tensordict import TensorDict


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
        sam_encoder_embeddings: batch_size (num_envs, 1)
        """
        sam_encoder_embeddings = sam_encoder_embeddings.squeeze(1)
        mask_to_embedding_ratio = (
            self.segmenter_image_size // sam_encoder_embeddings.shape[-1]
        )
        embeddings_masks_pixels_count = F.conv2d(
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

        segments_embeddings = []
        nb_segments = embeddings_masks_pixels_count.shape[0]
        sam_embeddings_selection_mask = (
            embeddings_masks_pixels_count.squeeze(1) >= self.min_pixels
        )

        for segment_index in range(nb_segments):
            image_id = segments_data["image_ids"][segment_index]
            image_sam_encoder_embeddings = sam_encoder_embeddings[image_id]
            segment_embeddings = image_sam_encoder_embeddings[
                :, sam_embeddings_selection_mask[segment_index]
            ]
            segment_embeddings = segment_embeddings.mean(axis=1)
            segments_embeddings.append(segment_embeddings)

        segments_embeddings = torch.stack(segments_embeddings)

        nb_segments = segments_data.batch_size[0]

        image_ids = segments_data["image_ids"]
        is_latent_tokens = torch.zeros(
            (nb_segments,), device=segments_data.device, dtype=torch.bool
        )
        return TensorDict(
            source={
                "image_ids": image_ids,
                "relative_segment_ids": segments_data["relative_segment_ids"],
                "is_latent_tokens": is_latent_tokens,
                "embeddings": segments_embeddings,
                "coords": segments_data["coords"],
            },
            batch_size=torch.Size([nb_segments]),
            device=segments_data.device,
        )
