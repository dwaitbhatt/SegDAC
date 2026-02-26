import torch
import torch.nn as nn
from segdac.networks.mlp import Mlp


class TrunkMlp(nn.Module):
    def __init__(self, in_features: int, features_dim: int, mlp: Mlp):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh(),
        )
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk(x)
        return self.mlp(x)
