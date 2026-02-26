import torch.nn as nn
import torch
from tensordict import TensorDict
from segdac.networks.transformers.decoder import TransformerDecoder
from einops import repeat
from einops import rearrange
from segdac.data.mdp import MdpData


class SegDacActorNetwork(nn.Module):
    def __init__(
        self,
        decoder: TransformerDecoder,
        embedding_dim: int,
        nb_query_tokens: int,
        action_projection_head: nn.Module,
        segments_embeddings_dim: int,
        proprioception_dim: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.proprioception_projection = nn.Linear(
            in_features=proprioception_dim, out_features=embedding_dim
        )
        self.segments_embeddings_projection = nn.Linear(
            in_features=segments_embeddings_dim, out_features=embedding_dim
        )
        self.segment_position_encoder = nn.Linear(
            in_features=4, out_features=embedding_dim
        )
        self.decoder = decoder
        self.nb_query_tokens = nb_query_tokens
        scale = embedding_dim**-0.5
        self.learned_query_tokens = nn.Parameter(
            scale * torch.randn(1, nb_query_tokens, embedding_dim)
        )
        self.action_projection_head = action_projection_head
        """
        0: learned query token
        1: segment
        2: proprioception
        """
        self.input_type_embedding = nn.Embedding(
            num_embeddings=3, embedding_dim=embedding_dim
        )

    def forward(
        self,
        mdp_data: MdpData,
    ) -> TensorDict:
        """
        Inputs:
            data:
                image_ids: (b,)
                proprioception: (b, dim_p)
            segmentation_data:
                segments_encoder_output:
                    image_ids: (src_seq_len,)
                    embeddings: (src_seq_len, dim_s)
                    coords:
                        masks_normalized_bboxes: (src_seq_len, 4)

            Where b = number of images in batch
                  h = history stack (number of frames)
                  src_seq_len = total number of tokens outputted by encoder for all images (sequence packing used here)
                  z = number of latent action tokens per image
                  dim_s = whatever embeddings dim the segments embeddings have
                  dim_p = proprioception dim
        Outputs:
            action: (b, ?), depends on the algorithm used, for instance SAC can output mean and std of a Gaussian distribution.
        """
        device = mdp_data.data.device

        segments_encoder_output = mdp_data.segmentation_data["segments_encoder_output"]
        segments_embeddings = segments_encoder_output[
            "embeddings"
        ]  # (src_seq_len, dim_s)
        segments_embeddings = self.segments_embeddings_projection(
            segments_embeddings
        )  # (src_seq_len, d)

        proprioception = mdp_data.data["proprioception"]  # (b, dim_p)
        b, dim_p = proprioception.shape
        proprioception_embeddings = self.proprioception_projection(
            proprioception
        )  # (b, d)

        total_nb_query_tokens = self.nb_query_tokens * b
        query_tokens_type_encoding = self.input_type_embedding(
            torch.full(
                (total_nb_query_tokens,), fill_value=0, dtype=torch.long, device=device
            )
        )
        total_nb_segments = segments_embeddings.shape[0]
        segment_type_encoding = self.input_type_embedding(
            torch.full(
                (total_nb_segments,), fill_value=1, dtype=torch.long, device=device
            )
        )
        total_nb_proprio = b
        proprioception_type_encoding = self.input_type_embedding(
            torch.full(
                (total_nb_proprio,), fill_value=2, dtype=torch.long, device=device
            )
        )

        segments_coords = segments_encoder_output["coords"][
            "masks_normalized_bboxes"
        ]  # (src_seq_len, 4)
        segments_pos_encoding = self.segment_position_encoder(segments_coords)

        segments_embeddings = (
            segments_embeddings + segment_type_encoding + segments_pos_encoding
        )
        segments_embeddings = segments_embeddings.unsqueeze(0)  # (1, src_seq_len, d)

        proprioception_embeddings = (
            proprioception_embeddings + proprioception_type_encoding
        )
        proprioception_embeddings = proprioception_embeddings.unsqueeze(0)

        cross_attention_kv = torch.cat(
            [
                segments_embeddings,
                proprioception_embeddings,
            ],
            dim=1,
        )  # (1, (src_seq_len)+(b), d)

        image_ids = mdp_data.data["image_ids"]  # (b,)

        segments_embeddings_image_ids = segments_encoder_output[
            "image_ids"
        ]  # (src_seq_len,)

        proprioception_embeddings_image_ids = image_ids

        cross_attention_image_ids = torch.cat(
            [
                segments_embeddings_image_ids,
                proprioception_embeddings_image_ids,
            ]
        )  # (src_seq_len+b)

        learned_query_tokens = repeat(
            self.learned_query_tokens, "1 z d -> 1 (b z) d", b=b
        )
        learned_query_tokens = learned_query_tokens + query_tokens_type_encoding

        learned_query_image_ids = image_ids.repeat_interleave(
            repeats=self.nb_query_tokens
        )  # (b*z)

        mhsa_attn_mask = learned_query_image_ids.unsqueeze(
            1
        ) == learned_query_image_ids.unsqueeze(
            0
        )  # (b*z, b*z)

        mhca_attn_mask = learned_query_image_ids.unsqueeze(
            1
        ) == cross_attention_image_ids.unsqueeze(
            0
        )  # (b*z, src_seq_len+b)

        output_tokens = self.decoder(
            decoder_output=learned_query_tokens,
            cross_attention_kv=cross_attention_kv,
            mhsa_attn_mask=mhsa_attn_mask,
            mhca_attn_mask=mhca_attn_mask,
        )  # (1, b*z, d)

        flat_output_tokens = rearrange(
            output_tokens,
            "1 (b z) d -> b (z d)",
            b=b,
            z=self.nb_query_tokens,
            d=self.embedding_dim,
        )

        projected_action_output = self.action_projection_head(
            flat_output_tokens
        )  # (b, *) depends on algorithm

        return TensorDict(
            {
                "env_action": projected_action_output,
            },
            batch_size=torch.Size([b]),
            device=device,
        )
