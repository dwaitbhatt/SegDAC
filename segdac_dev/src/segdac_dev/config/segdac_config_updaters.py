from omegaconf import DictConfig
from omegaconf import OmegaConf


class SegdacTd3ConfigUpdater:
    def update_config(self, env, cfg: DictConfig) -> DictConfig:
        OmegaConf.set_readonly(cfg, False)

        action_dim = env.action_space.shape[-1]

        cfg["algo"]["agent"]["action_sampling_strategy"]["actor"][
            "network"
        ]["action_projection_head"]["out_features"] = (
            action_dim
        ) 

        cfg["algo"]["q_function"]["action_dim"] = action_dim

        cfg["algo"]["agent"]["critic"]["q_function_1"][
            "action_dim"
        ] = action_dim

        cfg["algo"]["agent"]["critic"]["q_function_2"][
            "action_dim"
        ] = action_dim

        return cfg


class SegdacSacConfigUpdater:
    def update_config(self, env, cfg: DictConfig) -> DictConfig:
        OmegaConf.set_readonly(cfg, False)

        action_dim = env.action_space.shape[-1]

        cfg["algo"]["agent"]["action_sampling_strategy"]["actor"][
            "network"
        ]["action_projection_head"]["out_features"] = (
            2 * action_dim
        )  # We multiply by 2 because SAC predicts both mean + std

        cfg["algo"]["agent"]["action_sampling_strategy"]["actor"][
            "action_dim"
        ] = action_dim

        cfg["algo"]["q_function"]["action_dim"] = action_dim

        cfg["algo"]["agent"]["critic"]["q_function_1"][
            "action_dim"
        ] = action_dim

        cfg["algo"]["agent"]["critic"]["q_function_2"][
            "action_dim"
        ] = action_dim

        return cfg


class SegdacDqnConfigUpdater:
    def update_config(self, env, cfg: DictConfig) -> DictConfig:
        OmegaConf.set_readonly(cfg, False)

        action_dim = int(env.action_space.n)

        cfg["algo"]["agent"]["critic"]["q_function"][
            "q_value_projection_head"
        ]["out_features"] = action_dim

        cfg["algo"]["q_function"][
            "q_value_projection_head"
        ]["out_features"] = action_dim

        cfg["algo"]["agent"]["nb_actions"] = action_dim

        return cfg
