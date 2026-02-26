import torch
import torch.nn as nn


class RandomImageEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.randn(image.shape[0], self.output_dim, device=image.device)
