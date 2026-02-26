import torch
import torch.nn as nn


class CnnPixelEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim: int = 50):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=32 * 35 * 35, out_features=feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CnnPixelDecoder(nn.Module):
    def __init__(self, feature_dim: int = 50, out_channels=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=32 * 35 * 35),
            nn.ReLU(inplace=True),
        )
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                output_padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.reshape(-1, 32, 35, 35)
        x = self.up_conv(x)
        return x
