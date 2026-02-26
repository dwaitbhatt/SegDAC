import torch
import torch.nn as nn
from segdac.networks.transformers.multi_head_attention import MultiHeadAttention
from typing import Optional


class Mlp(nn.Module):
    def __init__(self, embedding_dim: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_ff, out_features=embedding_dim, bias=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mhsa = MultiHeadAttention(
            embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = Mlp(embedding_dim=embedding_dim, d_ff=d_ff, dropout=dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        mha_input = self.ln1(x)
        x = x + self.mhsa(
            query=mha_input, key=mha_input, value=mha_input, attn_mask=attn_mask
        )
        mlp_input = self.ln2(x)
        x = x + self.mlp(mlp_input)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(normalized_shape=embedding_dim)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.ln(x)
        return x
