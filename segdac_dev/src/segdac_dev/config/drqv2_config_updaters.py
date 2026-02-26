from omegaconf import DictConfig
from omegaconf import OmegaConf


class Drqv2ConfigUpdater:
    def update_config(self, env, cfg: DictConfig) -> DictConfig:
        OmegaConf.set_readonly(cfg, False)

        action_dim = env.action_space.shape[-1]

        cfg["algo"]["action_dim"] = action_dim
        num_rgb_channels = 3
        cfg['algo']['agent']['num_channels'] = num_rgb_channels * int(cfg['algo']['frame_stack'])
        return cfg
