from segdac.action_scaling.env_action_scaler import TanhEnvActionScaler
from segdac_dev.replay_buffers.transforms.transform import Transform
from segdac.data.mdp import MdpData


class UnscaleActionTransform(Transform):
    def __init__(self, device: str, env_action_scaler: TanhEnvActionScaler):
        super().__init__(device=device)
        self.env_action_scaler = env_action_scaler

    def apply(self, mdp_data: MdpData) -> MdpData:
        scaled_action = mdp_data.data["action"]
        unscaled_action = self.env_action_scaler.unscale(scaled_action)
        mdp_data.data["action"] = unscaled_action
        return mdp_data
