import torch
from segdac.data.mdp import MdpData
from tensordict import TensorDict
from segdac_dev.envs.transforms.transform import Transform


class RandomLatentActionTransform(Transform):
    def __init__(
        self,
        device: str,
        num_envs: int,
        latent_action_shape: list,
        out_key: str,
        seed: int,
        num_random_steps: int,
    ):
        super().__init__(device)
        self.num_envs = num_envs
        self.latent_action_shape = tuple([num_envs, 1] + latent_action_shape)
        self.out_key = out_key
        self.device = device
        self.generator = torch.Generator(device).manual_seed(seed)
        self.num_random_steps = num_random_steps
        self.current_step = 1

    def reset(self, mdp_data: MdpData) -> MdpData:
        return self.step(mdp_data)

    def step(self, mdp_data: MdpData) -> MdpData:
        if self.current_step <= self.num_random_steps:
            random_latent_action = (
                2
                * torch.rand(
                    *self.latent_action_shape,
                    generator=self.generator,
                    dtype=torch.float32,
                    device=self.device
                )
                - 1
            )  # Scale to [-1,1]
            if mdp_data.data.get("extras", None) is None:
                mdp_data.data["extras"] = TensorDict(
                    {self.out_key: random_latent_action},
                    batch_size=torch.Size([self.num_envs, 1]),
                    device=self.device,
                )
            else:
                mdp_data.data["extras"][self.out_key] = random_latent_action

        self.current_step += 1

        return mdp_data
