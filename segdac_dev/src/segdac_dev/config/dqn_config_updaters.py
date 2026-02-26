from omegaconf import DictConfig
from omegaconf import OmegaConf


class DqnAllPossibilitiesConfigUpdater:
    def update_config(self, env, cfg: DictConfig) -> DictConfig:
        OmegaConf.set_readonly(cfg, False)

        nb_actions = 2**int(env.action_space.n)

        cfg["algo"]["q_function"]["network"]["nb_actions"] = nb_actions

        cfg["algo"]["agent"]["nb_actions"] = nb_actions

        return cfg


class DqnDiscreteConfigUpdater:
    def update_config(self, env, cfg: DictConfig) -> DictConfig:
        OmegaConf.set_readonly(cfg, False)

        nb_actions = int(env.action_space.n)

        cfg["algo"]["q_function"]["network"]["nb_actions"] = nb_actions
        cfg["algo"]["q_function"]["network"]["in_channels"] = 3 * int(cfg['algo']['frame_stack'])

        cfg["algo"]["agent"]["nb_actions"] = nb_actions

        return cfg
