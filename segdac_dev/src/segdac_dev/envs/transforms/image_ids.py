import torch
from segdac.data.mdp import MdpData
from segdac_dev.envs.transforms.transform import Transform


class ImageIdsTransform(Transform):
    def __init__(
        self,
        device: str,
        num_envs: int,
    ):
        super().__init__(device)
        self.num_envs = num_envs
        self.image_id = 0

    def reset(self, mdp_data: MdpData) -> MdpData:
        return self.step(mdp_data)

    def step(self, mdp_data: MdpData) -> MdpData:
        mdp_data.data["image_ids"] = torch.arange(
            self.image_id,
            self.image_id + self.num_envs,
            device=self.device,
        ).unsqueeze(1)

        self.image_id += self.num_envs

        return mdp_data
