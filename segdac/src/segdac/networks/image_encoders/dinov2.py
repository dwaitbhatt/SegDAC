import os

import torch
import torch.nn as nn
from torchvision.transforms import v2


class DinoV2ImageEncoder(nn.Module):
    """DINOv2 image encoder. Builds the backbone via torch.hub and loads a local
    pretrain file at ``{weights_dir}/dinov2/{arch}.pth`` (e.g. ``weights/dinov2/dinov2_vits14.pth``).
    A plain checkpoint directory (weights only) is *not* a valid ``torch.hub`` local
    ``repo``; the hub *code* comes from ``facebookresearch/dinov2`` (cached after first
    use). Only the .pth is expected in ``weights/``.
    """

    def __init__(
        self,
        hub_repo: str = "facebookresearch/dinov2",
        model: str = "dinov2_vits14",
        source: str = "github",
        pretrained: bool = False,
        weights_dir: str = "weights",
    ) -> None:
        super().__init__()
        arch = model
        weights_path = os.path.join(weights_dir, "dinov2", f"{arch}.pth")
        backbone = torch.hub.load(
            hub_repo, arch, source=source, pretrained=pretrained
        ).eval()
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(sd, strict=True)
        for param in backbone.parameters():
            param.requires_grad = False

        self.model = backbone

        # Image is assumed to be float32 in range [0,1] already
        self.transforms = v2.Compose(
            [
                v2.Resize(224),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        preprocessed_image = self.transforms(image)
        image_embeddings = self.model(preprocessed_image)
        return image_embeddings
