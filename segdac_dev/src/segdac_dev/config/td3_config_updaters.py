from omegaconf import DictConfig
from omegaconf import OmegaConf


class Td3StateConfigUpdater:
    def update_config(self, env, cfg: DictConfig) -> DictConfig:
        OmegaConf.set_readonly(cfg, False)

        states_dim = env.observation_space["state"].shape[-1]

        action_dim = env.action_space.shape[-1]

        cfg["algo"]["agent"]["action_sampling_strategy"]["actor"]["network"]["network"][
            "in_features"
        ] = states_dim
        cfg["algo"]["agent"]["action_sampling_strategy"]["actor"]["network"]["network"][
            "out_features"
        ] = action_dim

        cfg["algo"]["agent"]["critic"]["q_function_1"]["network"]["in_features"] = (
            states_dim + action_dim
        )
        cfg["algo"]["agent"]["critic"]["q_function_2"]["network"]["in_features"] = (
            states_dim + action_dim
        )
        cfg["algo"]["q_function"]["network"]["in_features"] = states_dim + action_dim

        return cfg
