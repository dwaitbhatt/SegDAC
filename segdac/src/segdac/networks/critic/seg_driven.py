import torch.nn as nn
import torch
from tensordict import TensorDict
from segdac.networks.transformers.decoder import TransformerDecoder
from segdac.data.mdp import MdpData
from einops import repeat


class SegDacCriticNetwork(nn.Module):
    def __init__(
        self,
        decoder: TransformerDecoder,
        embedding_dim: int,
        nb_query_tokens: int,
        q_value_projection_head: nn.Module,
        segments_embeddings_dim: int,
        proprioception_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.proprioception_dim = proprioception_dim
        if proprioception_dim is not None:
            self.proprioception_projection = nn.Linear(
                in_features=proprioception_dim, out_features=embedding_dim
            )
        self.segments_embeddings_projection = nn.Linear(
            in_features=segments_embeddings_dim, out_features=embedding_dim
        )
        self.action_projection = nn.Linear(
            in_features=action_dim, out_features=embedding_dim
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
        self.q_value_projection_head = q_value_projection_head
        self.query_proj = nn.Linear(
            in_features=embedding_dim + nb_query_tokens * embedding_dim,
            out_features=embedding_dim,
        )
        """
        0: learned query token
        1: segment
        2: proprioception
        3: action
        """
        self.input_type_embedding = nn.Embedding(
            num_embeddings=4, embedding_dim=embedding_dim
        )

    def forward(
        self,
        mdp_data: MdpData,
    ) -> TensorDict:
        if self.proprioception_dim is not None:
            return self._forward_with_proprioception(mdp_data)

        return self._forward_without_proprioception(mdp_data)

    def _forward_with_proprioception(
        self,
        mdp_data: MdpData,
    ) -> TensorDict:
        """
        Inputs:
            data:
                action: (b, dim_a)
                image_ids: (b,)
                proprioception : (b, dim_p)
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
                  dim_a = env action space dim
                  dim_g = whatever embeddings dim the global image embeddings have
                  dim_s = whatever embeddings dim the segments embeddings have
                  dim_p = proprioception dim
                  d = embedding dimension
        Outputs:
            q_value: (b, 1)
        """
        device = mdp_data.data.device

        segments_encoder_output = mdp_data.segmentation_data["segments_encoder_output"]
        segments_embeddings = segments_encoder_output["embeddings"]
        segments_embeddings = self.segments_embeddings_projection(
            segments_embeddings
        )  # (src_seq_len, d)
        src_seq_len, _ = segments_embeddings.shape

        proprioception = mdp_data.data["proprioception"]  # (b, dim_p)
        b, dim_p = proprioception.shape
        proprioception_embeddings = self.proprioception_projection(
            proprioception
        )  # (b, d)

        action = mdp_data.data["action"]  # (b, ?)
        action_embeddings = self.action_projection(action)
        b, d = action_embeddings.shape

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
        total_nb_action = b
        action_type_encoding = self.input_type_embedding(
            torch.full(
                (total_nb_action,), fill_value=3, dtype=torch.long, device=device
            )
        )

        segments_coords = mdp_data.segmentation_data["segments_encoder_output"][
            "coords"
        ]["masks_normalized_bboxes"]
        segments_pos_encoding = self.segment_position_encoder(
            segments_coords
        )  # (src_seq_len, d)

        segments_embeddings = (
            segments_embeddings + segment_type_encoding + segments_pos_encoding
        )
        segments_embeddings = segments_embeddings.unsqueeze(0)  # (1, src_seq_len, d)

        proprioception_embeddings = (
            proprioception_embeddings + proprioception_type_encoding
        )
        proprioception_embeddings = proprioception_embeddings.unsqueeze(0)

        cross_attention_kv = torch.cat(
            [segments_embeddings, proprioception_embeddings],
            dim=1,
        )  # (1, (src_seq_len)+(b), d)

        image_ids = mdp_data.data["image_ids"]  # (b,)

        segments_embeddings_image_ids = segments_encoder_output[
            "image_ids"
        ]  # (src_seq_len,)

        proprioception_embeddings_image_ids = image_ids

        cross_attention_image_ids = torch.cat(
            [segments_embeddings_image_ids, proprioception_embeddings_image_ids]
        )  # (src_seq_len+b)

        learned_query_tokens = repeat(
            self.learned_query_tokens, "1 q d -> 1 b (d q)", b=b
        )
        learned_query_tokens = learned_query_tokens + query_tokens_type_encoding

        action_embeddings = action_embeddings + action_type_encoding
        action_embeddings = action_embeddings.unsqueeze(0)

        cond_query = self.query_proj(
            torch.cat([learned_query_tokens, action_embeddings], dim=-1)
        )  # (1 b d)

        query_token_image_ids = image_ids  # (b,)

        mhsa_attn_mask = query_token_image_ids.unsqueeze(
            1
        ) == query_token_image_ids.unsqueeze(
            0
        )  # (b, b)

        mhca_attn_mask = query_token_image_ids.unsqueeze(
            1
        ) == cross_attention_image_ids.unsqueeze(
            0
        )  # (b, src_seq_len+b)

        output_tokens, cross_attn_weights = self.decoder(
            decoder_output=cond_query,
            cross_attention_kv=cross_attention_kv,
            mhsa_attn_mask=mhsa_attn_mask,
            mhca_attn_mask=mhca_attn_mask,
            return_attn_weights=True,
        )  # (1, b, d), (1, b, src_seq_len+b)

        q_value = self.q_value_projection_head(output_tokens).squeeze(0)  # (b, 1)

        output_data = {}

        output_data["q_value"] = q_value

        cross_attn_weights = cross_attn_weights.squeeze(0)  # (b, src_seq_len+b)

        # q_value_segments_attn ---
        match_matrix = mdp_data.segmentation_data["image_ids"].unsqueeze(
            1
        ) == image_ids.unsqueeze(0)
        batch_indices_for_segment = torch.argmax(match_matrix.long(), dim=1)
        segment_indices_in_kv = torch.arange(src_seq_len, device=device)

        q_value_segments_attn = cross_attn_weights[
            batch_indices_for_segment, segment_indices_in_kv
        ]
        output_data["q_value_segments_attn"] = q_value_segments_attn

        # q_value_proprio_attn ---
        batch_indices_for_proprio = torch.arange(b, device=device)
        proprio_indices_in_kv = src_seq_len + torch.arange(b, device=device)

        q_value_proprio_attn = cross_attn_weights[
            batch_indices_for_proprio, proprio_indices_in_kv
        ]
        output_data["q_value_proprio_attn"] = q_value_proprio_attn

        return TensorDict(output_data, batch_size=torch.Size([]), device=device)

    def _forward_without_proprioception(
        self,
            mdp_data: MdpData,
    ) -> TensorDict:
        """
        Inputs:
            data:
                action: (b, dim_a)
                image_ids: (b,)
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
                  dim_a = env action space dim
                  dim_g = whatever embeddings dim the global image embeddings have
                  dim_s = whatever embeddings dim the segments embeddings have
                  d = embedding dimension
        Outputs:
            q_value: (b, 1)
        """
        device = mdp_data.data.device

        segments_encoder_output = mdp_data.segmentation_data["segments_encoder_output"]
        segments_embeddings = segments_encoder_output["embeddings"]
        segments_embeddings = self.segments_embeddings_projection(
            segments_embeddings
        )  # (src_seq_len, d)
        src_seq_len, _ = segments_embeddings.shape

        b = mdp_data.data["image_ids"].shape[0]

        action = mdp_data.data["action"]  # (b, ?)
        action_embeddings = self.action_projection(action)
        b, d = action_embeddings.shape

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
        
        total_nb_action = b
        action_type_encoding = self.input_type_embedding(
            torch.full(
                (total_nb_action,), fill_value=3, dtype=torch.long, device=device
            )
        )

        segments_coords = mdp_data.segmentation_data["segments_encoder_output"][
            "coords"
        ]["masks_normalized_bboxes"]
        segments_pos_encoding = self.segment_position_encoder(
            segments_coords
        )  # (src_seq_len, d)

        segments_embeddings = (
            segments_embeddings + segment_type_encoding + segments_pos_encoding
        )
        segments_embeddings = segments_embeddings.unsqueeze(0)  # (1, src_seq_len, d)

        cross_attention_kv = torch.cat(
            [segments_embeddings],
            dim=1,
        )  # (1, src_seq_len, d)

        image_ids = mdp_data.data["image_ids"]  # (b,)

        segments_embeddings_image_ids = segments_encoder_output[
            "image_ids"
        ]  # (src_seq_len,)

        cross_attention_image_ids = torch.cat(
            [segments_embeddings_image_ids]
        )  # (src_seq_len+b)

        learned_query_tokens = repeat(
            self.learned_query_tokens, "1 q d -> 1 b (d q)", b=b
        )
        learned_query_tokens = learned_query_tokens + query_tokens_type_encoding

        action_embeddings = action_embeddings + action_type_encoding
        action_embeddings = action_embeddings.unsqueeze(0)

        cond_query = self.query_proj(
            torch.cat([learned_query_tokens, action_embeddings], dim=-1)
        )  # (1 b d)

        query_token_image_ids = image_ids  # (b,)

        mhsa_attn_mask = query_token_image_ids.unsqueeze(
            1
        ) == query_token_image_ids.unsqueeze(
            0
        )  # (b, b)

        mhca_attn_mask = query_token_image_ids.unsqueeze(
            1
        ) == cross_attention_image_ids.unsqueeze(
            0
        )  # (b, src_seq_len)

        output_tokens, cross_attn_weights = self.decoder(
            decoder_output=cond_query,
            cross_attention_kv=cross_attention_kv,
            mhsa_attn_mask=mhsa_attn_mask,
            mhca_attn_mask=mhca_attn_mask,
            return_attn_weights=True,
        )  # (1, b, d), (1, b, src_seq_len)

        q_value = self.q_value_projection_head(output_tokens).squeeze(0)  # (b, 1)

        output_data = {}

        output_data["q_value"] = q_value

        cross_attn_weights = cross_attn_weights.squeeze(0)  # (b, src_seq_len)

        # q_value_segments_attn
        match_matrix = mdp_data.segmentation_data["image_ids"].unsqueeze(
            1
        ) == image_ids.unsqueeze(0)
        batch_indices_for_segment = torch.argmax(match_matrix.long(), dim=1)
        segment_indices_in_kv = torch.arange(src_seq_len, device=device)

        q_value_segments_attn = cross_attn_weights[
            batch_indices_for_segment, segment_indices_in_kv
        ]
        output_data["q_value_segments_attn"] = q_value_segments_attn

        return TensorDict(output_data, batch_size=torch.Size([]), device=device)


class SegDacCriticNetworkDiscreteAction(nn.Module):
    def __init__(
        self,
        decoder: TransformerDecoder,
        embedding_dim: int,
        nb_query_tokens: int,
        q_value_projection_head: nn.Module,
        segments_embeddings_dim: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
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
        self.q_value_projection_head = q_value_projection_head
        self.query_proj = nn.Linear(
            in_features=nb_query_tokens * embedding_dim,
            out_features=embedding_dim,
        )
        """
        0: learned query token
        1: segment
        """
        self.input_type_embedding = nn.Embedding(
            num_embeddings=2, embedding_dim=embedding_dim
        )

    def forward(
        self,
        mdp_data: MdpData,
    ) -> TensorDict:
        return self._forward_without_proprioception(mdp_data)

    def _forward_without_proprioception(
        self,
            mdp_data: MdpData,
    ) -> TensorDict:
        """
        Inputs:
            data:
                image_ids: (b,)
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
                  dim_a = env action space dim
                  dim_g = whatever embeddings dim the global image embeddings have
                  dim_s = whatever embeddings dim the segments embeddings have
                  d = embedding dimension
        Outputs:
            q_value: (b, 1)
        """
        device = mdp_data.data.device

        segments_encoder_output = mdp_data.segmentation_data["segments_encoder_output"]
        segments_embeddings = segments_encoder_output["embeddings"]
        segments_embeddings = self.segments_embeddings_projection(
            segments_embeddings
        )  # (src_seq_len, d)
        src_seq_len, _ = segments_embeddings.shape

        b = mdp_data.data["image_ids"].shape[0]

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

        segments_coords = mdp_data.segmentation_data["segments_encoder_output"][
            "coords"
        ]["masks_normalized_bboxes"]
        segments_pos_encoding = self.segment_position_encoder(
            segments_coords
        )  # (src_seq_len, d)

        segments_embeddings = (
            segments_embeddings + segment_type_encoding + segments_pos_encoding
        )
        segments_embeddings = segments_embeddings.unsqueeze(0)  # (1, src_seq_len, d)

        cross_attention_kv = torch.cat(
            [segments_embeddings],
            dim=1,
        )  # (1, src_seq_len, d)

        image_ids = mdp_data.data["image_ids"]  # (b,)

        segments_embeddings_image_ids = segments_encoder_output[
            "image_ids"
        ]  # (src_seq_len,)

        cross_attention_image_ids = torch.cat(
            [segments_embeddings_image_ids]
        )  # (src_seq_len)

        learned_query_tokens = repeat(
            self.learned_query_tokens, "1 q d -> 1 b (d q)", b=b
        )
        learned_query_tokens = learned_query_tokens + query_tokens_type_encoding

        cond_query = self.query_proj(
            learned_query_tokens
        )  # (1 b d)

        query_token_image_ids = image_ids  # (b,)

        mhsa_attn_mask = query_token_image_ids.unsqueeze(
            1
        ) == query_token_image_ids.unsqueeze(
            0
        )  # (b, b)

        mhca_attn_mask = query_token_image_ids.unsqueeze(
            1
        ) == cross_attention_image_ids.unsqueeze(
            0
        )  # (b, src_seq_len)

        output_tokens, cross_attn_weights = self.decoder(
            decoder_output=cond_query,
            cross_attention_kv=cross_attention_kv,
            mhsa_attn_mask=mhsa_attn_mask,
            mhca_attn_mask=mhca_attn_mask,
            return_attn_weights=True,
        )  # (1, b, d), (1, b, src_seq_len)

        q_value = self.q_value_projection_head(output_tokens).squeeze(0)  # (b, nb_actions)

        output_data = {}

        output_data["q_value"] = q_value

        cross_attn_weights = cross_attn_weights.squeeze(0)  # (b, src_seq_len)

        # q_value_segments_attn
        match_matrix = mdp_data.segmentation_data["image_ids"].unsqueeze(
            1
        ) == image_ids.unsqueeze(0) # (src_seq_len, b)
        batch_indices_for_segment = torch.argmax(match_matrix.long(), dim=1) # (src_seq_len,)
        segment_indices_in_kv = torch.arange(src_seq_len, device=device)

        q_value_segments_attn = cross_attn_weights[
            batch_indices_for_segment, segment_indices_in_kv
        ]
        output_data["q_value_segments_attn"] = q_value_segments_attn

        return TensorDict(output_data, batch_size=torch.Size([]), device=device)
