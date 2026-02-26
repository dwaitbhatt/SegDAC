from omegaconf import DictConfig
from omegaconf import OmegaConf


class FtdConfigUpdater:
    def update_config(self, env, cfg: DictConfig) -> DictConfig:
        OmegaConf.set_readonly(cfg, False)

        action_dim = env.action_space.shape[-1]

        cfg["algo"]["agent"]["action_shape"] = [action_dim]

        return cfg
