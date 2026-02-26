import torch.nn as nn
from segdac.data.mdp import MdpData
from segdac_dev.envs.transforms.transform import Transform


class GlobalImageEncoderMeanPooledTransform(Transform):
    def __init__(
        self, device: str, image_encoder: nn.Module, pool_dims: tuple = (2, 3)
    ):
        super().__init__(device)
        self.image_encoder = image_encoder.to(device)
        self.pool_dims = list(pool_dims)

    def reset(self, mdp_data: MdpData) -> MdpData:
        return self.step(mdp_data)

    def step(self, mdp_data: MdpData) -> MdpData:
        pixels_transformed = mdp_data.data["pixels_transformed"].squeeze(1)
        image_embeddings = self.image_encoder(pixels_transformed).mean(
            dim=self.pool_dims
        )
        mdp_data.data["global_image_embeddings"] = image_embeddings.unsqueeze(1)

        return mdp_data
