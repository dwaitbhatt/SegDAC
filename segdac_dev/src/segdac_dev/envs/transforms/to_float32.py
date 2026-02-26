import torch
import torchvision.transforms.v2 as v2
from segdac.data.mdp import MdpData
from segdac_dev.envs.transforms.transform import Transform


class ToFloat32Transform(Transform):
    def __init__(self, device: str, in_key: str, out_key: str):
        super().__init__(device)
        self.in_key = in_key
        self.out_key = out_key
        self.transform = v2.ToDtype(dtype=torch.float32, scale=True).to(device)

    def reset(self, mdp_data: MdpData) -> MdpData:
        return self.step(mdp_data)

    def step(self, mdp_data: MdpData) -> MdpData:
        in_data = mdp_data.data[self.in_key].to(self.device)
        out_data = self.transform(in_data)
        mdp_data.data[self.out_key] = out_data
        return mdp_data
