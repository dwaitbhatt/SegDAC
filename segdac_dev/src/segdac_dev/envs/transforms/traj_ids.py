import torch
from tensordict import TensorDict
from segdac.data.mdp import MdpData
from segdac_dev.envs.transforms.transform import Transform


class TrajIdsTransform(Transform):
    def __init__(self, device: str, num_envs: int):
        super().__init__(device)
        self.num_envs = num_envs
        self.traj_ids = torch.arange(start=0, end=num_envs, device=device).unsqueeze(1)
        self.previous_step_done = torch.zeros(
            (num_envs,), dtype=torch.bool, device=device
        ).unsqueeze(1)

    def reset(self, mdp_data: MdpData) -> MdpData:
        return self.step(mdp_data)

    def step(self, mdp_data: MdpData) -> MdpData:
        if self.previous_step_done.any():
            nb_envs_done = self.previous_step_done.sum()
            new_traj_id_start_inclusive = self.traj_ids.max() + 1
            new_traj_id_end_exclusive = new_traj_id_start_inclusive + nb_envs_done
            new_traj_ids = torch.arange(
                new_traj_id_start_inclusive,
                new_traj_id_end_exclusive,
                dtype=self.traj_ids.dtype,
                device=self.device,
            )
            updated_traj_ids = self.traj_ids.clone()
            updated_traj_ids[self.previous_step_done] = new_traj_ids
            self.traj_ids = updated_traj_ids

        mdp_data.data["collector"] = TensorDict(
            {"traj_ids": self.traj_ids},
            batch_size=torch.Size([self.num_envs, 1]),
            device=self.device,
        )
        self.previous_step_done = mdp_data.data["done"]

        return mdp_data
