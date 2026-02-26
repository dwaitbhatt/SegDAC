from segdac_dev.replay_buffers.segments.replay_buffer import TensorDict
import torch
from collections import deque
from segdac_dev.envs.transforms.transform import Transform
from segdac.data.mdp import MdpData


class FrameStackTransform(Transform):
    def __init__(
        self, device: str, num_envs: int, history_size: int, in_key: str, out_key: str
    ):
        super().__init__(device)
        self.num_envs = num_envs
        self.envs_framestack = [deque(maxlen=history_size) for _ in range(num_envs)]
        self.history_size = history_size
        self.first_apply_done = False
        self.in_key = in_key
        self.out_key = out_key

    def reset(self, mdp_data: MdpData) -> MdpData:
        return self.step(mdp_data)

    def step(self, mdp_data: MdpData) -> MdpData:
        data = mdp_data.data[self.in_key]

        env_done = mdp_data.data["done"]
        for env_id in range(self.num_envs):
            # If the env is done or if it's our first time saving a frame, we fill the history with the latest data
            # For the first case, this prevents the history from containing data from the previous traj
            if env_done[env_id] or len(self.envs_framestack[env_id]) == 0:
                self.fill_env_frames_history(env_id=env_id, data=data)
            else:
                # Otherwise we append the data to the existing history
                self.add_env_data(env_id, data)

        envs_stacked_frames = torch.stack(
            [
                torch.stack(list(env_framestack))
                for env_framestack in self.envs_framestack
            ]
        ).unsqueeze(
            1
        )  # (num_envs, 1, history_size, *)
        mdp_data.data[self.out_key] = envs_stacked_frames

        return mdp_data

    def fill_env_frames_history(self, env_id: int, data: TensorDict):
        for _ in range(self.history_size):
            self.add_env_data(env_id, data)

    def add_env_data(self, env_id: int, data: TensorDict):
        env_data = data.squeeze(1)[env_id]
        self.envs_framestack[env_id].append(env_data.to(self.device))
