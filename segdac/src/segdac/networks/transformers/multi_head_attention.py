import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.q_projection = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim, bias=False
        )
        self.k_projection = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.v_projection = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.output_projection = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim, bias=False
        )
        self.output_dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple:
        q = self.q_projection(query)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)
        k = self.k_projection(key)
        v = self.v_projection(value)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)
        if return_attn_weights:
            # (b, h, s_q, d) @ (b, h, d, s_kv) -> (b, h, s_q, s_kv)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(attn_mask == False, -float("inf"))

            attn_weights = F.softmax(attn_scores, dim=-1)  # (b, h, s_q, s_kv)
            avg_attn_weights = attn_weights.mean(dim=1)  # (b, s_q, s_kv)

            attn_weights_dropped = F.dropout(
                attn_weights, p=self.dropout_p if self.training else 0.0
            )

            # (b, h, s_q, s_kv) @ (b, h, s_kv, d) -> (b, h, s_q, d)
            heads = torch.matmul(attn_weights_dropped, v)

        else:
            """
            query=q.unsqueeze(0),
            key=k.unsqueeze(0),
            value=v.unsqueeze(0),
            Is needed to make it compatible with torch.vmap which adds an extra batch dim to the weights.
            It still works with non-torch.vmap code.
            """
            heads = F.scaled_dot_product_attention(
                query=q.unsqueeze(0),
                key=k.unsqueeze(0),
                value=v.unsqueeze(0),
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            ).squeeze(0)
            avg_attn_weights = None

        concatenated_heads = rearrange(heads, "b h s d_head -> b s (h d_head)")
        output = self.output_projection(concatenated_heads)
        output = self.output_dropout(output)

        if avg_attn_weights is None:
            return output

        return output, avg_attn_weights
