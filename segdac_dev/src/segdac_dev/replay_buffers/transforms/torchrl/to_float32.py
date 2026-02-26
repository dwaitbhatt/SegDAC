import torch
from torchrl.envs.transforms import Transform


class ToFloat32Transform(Transform):
    def __init__(self, in_keys: list[str], out_keys: list[str], scale: bool = True):
        in_keys_formatted = []
        for k in in_keys:
            in_keys_formatted.append(k)
            in_keys_formatted.append(("next", k))
        out_keys_formatted = []
        for k in out_keys:
            out_keys_formatted.append(k)
            out_keys_formatted.append(("next", k))
        super().__init__(in_keys=in_keys_formatted, out_keys=out_keys_formatted)
        self.scale = scale

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(torch.float32)
        if self.scale:
            obs = obs / 255.0

        return obs
