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

    @torch.no_grad()
    def forward_patch_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """
        Last-layer **patch** tokens (no CLS / registers), normalized.

        Returns ``(B, Gh, Gw, D)`` on the same grid as DINO patches for the
        transformed image (``Resize`` then ``Normalize`` as in :meth:`forward`).
        """
        preprocessed = self.transforms(image)
        ret = self.model.forward_features(preprocessed)
        pt = ret["x_norm_patchtokens"]
        b, n, d = pt.shape
        ps = int(getattr(self.model, "patch_size", 14))
        h, w = int(preprocessed.shape[-2]), int(preprocessed.shape[-1])
        gh, gw = h // ps, w // ps
        if gh * gw != n:
            s = int(round(n**0.5))
            if s * s != n:
                raise RuntimeError(
                    f"Cannot reshape {n} patch tokens to a grid "
                    f"(preprocessed {h}x{w}, patch_size={ps})"
                )
            gh = gw = s
        return pt.reshape(b, gh, gw, d)

    @torch.no_grad()
    def forward_feature_map(self, image: torch.Tensor) -> torch.Tensor:
        """
        Dense patch grid for SAM-style spatial pooling: ``(B, D, Gh, Gw)``.

        Same preprocessing as :meth:`forward_patch_tokens`; channels are patch
        embedding dim ``D``.
        """
        pt = self.forward_patch_tokens(image)
        return pt.permute(0, 3, 1, 2).contiguous()


class DinoV2DenseMapEncoder(nn.Module):
    """
    Wraps :class:`DinoV2ImageEncoder` so ``forward`` returns a dense map
    ``(B, D, Gh, Gw)`` for :class:`~segdac.networks.segments_encoders.image_encoder_segment_adapter.ImageEncoderSegmentTokensAdapter`
    ``mode="spatial_from_full_image"``. Delegates :meth:`forward_patch_tokens` for viz.
    """

    def __init__(self, inner: DinoV2ImageEncoder | None = None, **kwargs) -> None:
        super().__init__()
        self.inner = inner if inner is not None else DinoV2ImageEncoder(**kwargs)

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.inner.forward_feature_map(image)

    @torch.no_grad()
    def forward_patch_tokens(self, image: torch.Tensor) -> torch.Tensor:
        return self.inner.forward_patch_tokens(image)
