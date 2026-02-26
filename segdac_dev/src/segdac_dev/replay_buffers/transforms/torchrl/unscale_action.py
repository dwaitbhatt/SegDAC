import torch
from torchrl.envs.transforms import Transform
from segdac.action_scaling.env_action_scaler import TanhEnvActionScaler


class UnscaleActionTransform(Transform):
    def __init__(self, env_action_scaler: TanhEnvActionScaler):
        super().__init__(in_keys_inv=["action"], out_keys_inv=["action"])
        self.env_action_scaler = env_action_scaler
        self.inverse = True

    def _apply_transform(self, scaled_action: torch.Tensor) -> torch.Tensor:
        return scaled_action

    def _inv_apply_transform(self, scaled_action: torch.Tensor) -> torch.Tensor:
        return self.env_action_scaler.unscale(scaled_action)
