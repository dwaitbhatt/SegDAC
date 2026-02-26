import torch
import torch.nn as nn


class DqnC51Network(nn.Module):
    def __init__(self, nb_actions: int, in_channels=3, feature_dim: int = 1024):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.q_head = nn.Linear(in_features=feature_dim, out_features=nb_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5: # (b, stack, c, h, w) -> (b, c*stack, h, w)
            b = x.shape[0]
            h = x.shape[-2]
            w = x.shape[-1]
            x = x.reshape(b, -1, h, w)
        features = self.feature_extractor(x)
        q_values = self.q_head(features)
        return q_values
