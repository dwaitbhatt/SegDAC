from omegaconf import DictConfig
from omegaconf import OmegaConf


class SamSamEncoderConfigUpdater:
    def update_config(self, env, cfg: DictConfig) -> DictConfig:
        OmegaConf.set_readonly(cfg, False)

        sam_enc_embed_dim = 256
        proprioception_dim = cfg["algo"]["proprioception_dim"]

        action_dim = env.action_space.shape[-1]

        cfg["algo"]["agent"]["action_sampling_strategy"]["actor"]["network"]["network"][
            "in_features"
        ] = sam_enc_embed_dim + proprioception_dim
        cfg["algo"]["agent"]["action_sampling_strategy"]["actor"]["network"]["network"][
            "out_features"
        ] = (
            2 * action_dim
        )  # mean and std for each action dim
        cfg["algo"]["agent"]["action_sampling_strategy"]["actor"][
            "action_dim"
        ] = action_dim

        cfg["algo"]["agent"]["critic"]["q_function_1"]["network"]["in_features"] = (
            sam_enc_embed_dim + proprioception_dim + action_dim
        )
        cfg["algo"]["agent"]["critic"]["q_function_2"]["network"]["in_features"] = (
            sam_enc_embed_dim + proprioception_dim + action_dim
        )
        cfg["algo"]["q_function"]["network"]["in_features"] = sam_enc_embed_dim + proprioception_dim + action_dim

        return cfg
