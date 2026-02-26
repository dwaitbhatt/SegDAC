import torch
from segdac.data.mdp import MdpData
from segdac_dev.envs.transforms.transform import Transform


class InitTrackerTransform(Transform):
    def __init__(self, device: str, num_envs: int):
        super().__init__(device)
        self.num_envs = num_envs
        self.is_init = torch.ones((num_envs, 1), dtype=torch.bool, device=device)

    def reset(self, mdp_data: MdpData) -> MdpData:
        return self.step(mdp_data)

    def step(self, mdp_data: MdpData) -> MdpData:
        mdp_data.data["is_init"] = self.is_init.clone()
        self.is_init = mdp_data.data["done"]
        return mdp_data
