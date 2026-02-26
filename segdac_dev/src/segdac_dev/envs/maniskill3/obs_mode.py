import torch
from tensordict import TensorDict
from typing import Union


class ObsMode:
    def get_name(self) -> str:
        raise NotImplementedError

    def get_obs_data(self, raw_obs: Union[dict, torch.Tensor]) -> TensorDict:
        raise NotImplementedError

    def get_obs_keys(self) -> list[str]:
        raise NotImplementedError


class ProprioRgbObsMode:
    def __init__(self, camera_name: str):
        self.camera_name = camera_name

    def get_name(self) -> str:
        return "state_dict+rgb"

    def get_obs_data(self, raw_obs: dict) -> TensorDict:
        pixels = raw_obs["sensor_data"][self.camera_name]["rgb"].permute(
            0, 3, 1, 2
        )  # (B, H, W, C) -> (B, C, H, W)

        qpos = raw_obs["agent"]["qpos"]

        return TensorDict(
            {"proprioception": qpos, "pixels": pixels},
            batch_size=torch.Size([qpos.shape[0]]),
        )

    def get_obs_keys(self) -> list[str]:
        return ["proprioception", "pixels"]


class StateObsMode:
    def __init__(self, camera_name: str):
        self.camera_name = camera_name

    def get_name(self) -> str:
        return "state+rgb"

    def get_obs_data(self, raw_obs: dict) -> TensorDict:
        state = raw_obs["state"]

        pixels = raw_obs["sensor_data"][self.camera_name]["rgb"].permute(
            0, 3, 1, 2
        )  # (B, H, W, C) -> (B, C, H, W)

        return TensorDict(
            {"state": state, "pixels": pixels},
            batch_size=torch.Size([state.shape[0]]),
        )

    def get_obs_keys(self) -> list[str]:
        return ["state", "pixels"]
