import torch
import torch.nn as nn
from torchvision.transforms import v2


class DinoV2ImageEncoder(nn.Module):
    def __init__(
        self,
        repo_or_dir: str = "weights/dinov2",
        model: str = "dinov2_vits14",
        source: str = "local",
        pretrained: bool = True,
    ):
        super().__init__()
        model = torch.hub.load(
            repo_or_dir=repo_or_dir, model=model, source=source, pretrained=pretrained
        ).eval()
        if source == "local":
            model.load_state_dict(torch.load(f"weights/{model}.pth"))
        for param in model.parameters():
            param.requires_grad = False

        self.model = model

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
