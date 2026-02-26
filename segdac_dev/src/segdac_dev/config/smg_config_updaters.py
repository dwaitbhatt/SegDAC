from omegaconf import DictConfig
from omegaconf import OmegaConf


class SmgConfigUpdater:
    def update_config(self, env, cfg: DictConfig) -> DictConfig:
        OmegaConf.set_readonly(cfg, False)

        action_dim = env.action_space.shape[-1]

        cfg["algo"]["agent"]['action_shape'] = [action_dim]
        num_rgb_channels = 3
        frame_stack = cfg['algo']['frame_stack']
        height = cfg['algo']['agent']['obs_shape'][1]
        width = cfg['algo']['agent']['obs_shape'][2]
        cfg['algo']['agent']['obs_shape'] = [frame_stack * num_rgb_channels, height, width]
        return cfg
