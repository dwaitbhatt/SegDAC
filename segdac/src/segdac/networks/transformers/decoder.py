import torch
import torch.nn as nn
from tensordict import TensorDict
from segdac.networks.transformers.encoder import Mlp
from segdac.networks.transformers.multi_head_attention import MultiHeadAttention


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mhsa = MultiHeadAttention(
            embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mhca = MultiHeadAttention(
            embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout
        )
        self.ln3 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.ln4 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = Mlp(embedding_dim=embedding_dim, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        decoder_output: torch.Tensor,
        cross_attention_kv: torch.Tensor,
        mhsa_attn_mask: torch.Tensor,
        mhca_attn_mask: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple:
        mhsa_input = self.ln1(decoder_output)
        x = decoder_output + self.mhsa(
            query=mhsa_input,
            key=mhsa_input,
            value=mhsa_input,
            attn_mask=mhsa_attn_mask,
            return_attn_weights=False,
        )

        q = self.ln2(x)
        kv = self.ln3(cross_attention_kv)
        cross_attn_output = self.mhca(
            query=q,
            key=kv,
            value=kv,
            attn_mask=mhca_attn_mask,
            return_attn_weights=return_attn_weights,
        )
        if return_attn_weights:
            cross_attn_output, cross_attn_weights = cross_attn_output

        x = x + cross_attn_output

        mlp_input = self.ln4(x)
        x = x + self.mlp(mlp_input)

        if return_attn_weights:
            return x, cross_attn_weights

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        device,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.device = device

    def forward(
        self,
        decoder_output: torch.Tensor,
        cross_attention_kv: TensorDict,
        mhsa_attn_mask: torch.Tensor,
        mhca_attn_mask: torch.Tensor,
        return_attn_weights: bool = False,
    ):
        for i, layer in enumerate(self.layers):
            return_layer_attn_weights = (
                return_attn_weights and i == len(self.layers) - 1
            )
            decoder_output = layer(
                decoder_output=decoder_output,
                cross_attention_kv=cross_attention_kv,
                mhsa_attn_mask=mhsa_attn_mask,
                mhca_attn_mask=mhca_attn_mask,
                return_attn_weights=return_layer_attn_weights,
            )

        return decoder_output
