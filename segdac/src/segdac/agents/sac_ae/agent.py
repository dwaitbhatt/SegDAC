import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict import merge_tensordicts
from copy import deepcopy
from segdac.agents.target_networks_params_updaters.polyak_average_updater import (
    PolyakAverageParametersUpdater,
)
from segdac.agents.sac.agent import SacAgent
from segdac.agents.sac.critic import SacCritic
from segdac.action_scaling.env_action_scaler import TanhEnvActionScaler
from einops import rearrange
from segdac.data.mdp import MdpData
from segdac.agents.action_sampling_strategy import StochasticActionSamplingStrategy


class SacAeAgent(SacAgent):
    """
    Custom implementation of https://arxiv.org/abs/1910.01741 in a way that is agnostic of the model-free algorithm.
    One can change the base agent to any model-free algorithm (eg: SAC, DDPG, TD3, etc).
    """
    def __init__(
        self,
        env_action_scaler: TanhEnvActionScaler,
        action_sampling_strategy: StochasticActionSamplingStrategy,
        critic: SacCritic,
        critic_update_frequency: int,
        actor_update_frequency: int,
        target_networks_update_frequency: int,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_optimizer,
        decoder_optimizer,
        decoder_latent_lambda: float,
        decoder_update_frequency: int,
        encoder_target_params_updater: PolyakAverageParametersUpdater,
        encoder_input_observation_key: str,
        encoder_output_observation_key: str,
    ):
        super().__init__(
            env_action_scaler=env_action_scaler,
            action_sampling_strategy=action_sampling_strategy,
            critic=critic,
            critic_update_frequency=critic_update_frequency,
            actor_update_frequency=actor_update_frequency,
            target_networks_update_frequency=target_networks_update_frequency,
        )
        self.encoder = encoder
        self.decoder = decoder
        self.target_encoder = deepcopy(self.encoder).eval()
        self.encoder_optimizer = encoder_optimizer(params=self.encoder.parameters())
        self.decoder_optimizer = decoder_optimizer(params=self.decoder.parameters())
        self.action_sampling_strategy.actor = self.action_sampling_strategy.actor(
            encoder=self.encoder,
            encoder_input_observation_key=encoder_input_observation_key,
            encoder_output_observation_key=encoder_output_observation_key,
        )
        self.critic = self.critic(
            encoder=self.encoder,
            target_encoder=self.target_encoder,
            encoder_input_observation_key=encoder_input_observation_key,
            encoder_output_observation_key=encoder_output_observation_key,
            encoder_optimizer=self.encoder_optimizer,
        )
        self.decoder_update_frequency = decoder_update_frequency
        self.encoder_target_params_updater = encoder_target_params_updater
        self.decoder_latent_lambda = decoder_latent_lambda
        self.encoder_input_observation_key = encoder_input_observation_key
        self.encoder_output_observation_key = encoder_output_observation_key
        self.apply(self.weight_init)
        
    def weight_init(self, m):
        """
        Custom weight init for Conv2D and Linear layers.
        Taken from https://github.com/denisyarats/pytorch_sac_ae/blob/master/sac_ae.py
        """
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
            assert m.weight.size(2) == m.weight.size(3)
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)
            mid = m.weight.size(2) // 2
            gain = nn.init.calculate_gain("relu")
            nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

    def predict_unscaled_action(self, mdp_data: MdpData) -> TensorDict:
        img = mdp_data.data[self.encoder_input_observation_key]
        obs = rearrange(img, "B N C H W -> B (N C) H W")
        mdp_data.data[self.encoder_input_observation_key] = obs
        obs_latent = self.encoder(mdp_data)[self.encoder_output_observation_key]
        mdp_data.data[self.encoder_output_observation_key] = obs_latent
        return super().predict_unscaled_action(mdp_data)

    def update(
        self, train_mdp_data: MdpData, env_step: int, is_time_to_evaluate: bool
    ) -> TensorDict:
        train_mdp_data = self.preprocess_data_for_training(
            train_mdp_data=train_mdp_data
        )

        critic_logs = self.update_critic(
            train_mdp_data=train_mdp_data,
            env_step=env_step,
            is_time_to_evaluate=is_time_to_evaluate,
        )

        actor_logs = self.update_actor(
            train_mdp_data=train_mdp_data,
            env_step=env_step,
            is_time_to_evaluate=is_time_to_evaluate,
        )

        other_logs = self.udpate_decoder(
            train_mdp_data=train_mdp_data,
            env_step=env_step,
            is_time_to_evaluate=is_time_to_evaluate,
        )

        output_logs = merge_tensordicts(critic_logs, actor_logs, other_logs)

        # Fixes merge_tensordicts removing actor_grad_layers key
        if actor_logs.get("actor_grad_stats", None) is not None:
            output_logs["actor_grad_stats"]["actor_grad_layers"] = actor_logs[
                "actor_grad_stats"
            ]["actor_grad_layers"]
            output_logs["critic_grad_stats"]["critic_grad_layers"] = critic_logs[
                "critic_grad_stats"
            ]["critic_grad_layers"]

        return output_logs


    def preprocess_data_for_training(self, train_mdp_data: MdpData) -> MdpData:
        raw_obs = train_mdp_data.data[self.encoder_input_observation_key]
        raw_obs = rearrange(raw_obs, "B N C H W -> B (N C) H W")
        train_mdp_data.data[self.encoder_input_observation_key] = raw_obs

        next_raw_obs = train_mdp_data.next.data[self.encoder_input_observation_key]
        next_raw_obs = rearrange(next_raw_obs, "B N C H W -> B (N C) H W")
        train_mdp_data.next.data[self.encoder_input_observation_key] = next_raw_obs

        return train_mdp_data

    def udpate_decoder(
        self, train_mdp_data: MdpData, env_step: int, is_time_to_evaluate: bool
    ) -> TensorDict:
        """
        Encoder and decoder are updated using a reconstruction loss.
        """
        if env_step % self.decoder_update_frequency != 0:
            return TensorDict(
                {},
                batch_size=torch.Size([]),
            )
        
        self.encoder.train()
        self.decoder.train()
        obs_latent = self.encoder(train_mdp_data)[self.encoder_output_observation_key]
        train_mdp_data.data[self.encoder_output_observation_key] = obs_latent
        reconstructed_obs = self.decoder(train_mdp_data)[
            self.encoder_input_observation_key
        ]

        target_observation = train_mdp_data.data[self.encoder_input_observation_key]
        target_observation = self.preprocess_obs(target_observation)
        reconstruction_loss = F.mse_loss(reconstructed_obs, target_observation)

        # L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * obs_latent.pow(2).sum(1)).mean()

        loss = reconstruction_loss + self.decoder_latent_lambda * latent_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        logs_data = {}
        if is_time_to_evaluate:
            logs_data["decoder_loss"] = loss.detach()
            logs_data["reconstruction_loss"] = reconstruction_loss.detach()
            logs_data["latent_loss"] = latent_loss.detach()

        return TensorDict(
            logs_data,
            batch_size=torch.Size([]),
        )

    def preprocess_obs(self, obs, bits=5):
        """
        See B.6. Pixels Preprocessing from https://arxiv.org/abs/1910.01741
        Preprocessing reduces bit depth to 5 bits as in https://arxiv.org/abs/1807.03039.
        Reuces complexity, the network doesn't have to represent extremely fine-grained pixel-level details.
        """
        bins = 2**bits
        assert obs.dtype == torch.float32
        if bits < 8:
            obs = torch.floor(
                obs / 2 ** (8 - bits)
            )  # Quantization to reduce 8 bits values per channel to lower dim (eg: 5 bits)
        obs = obs / bins  # Renormalize to (0,1) range
        obs = (
            obs + torch.rand_like(obs) / bins
        )  # Prevents network from overfitting to discrete hard quantization, smoothes out input distribution slightly.
        obs = obs - 0.5  # Center to 0 can help with training stability
        return obs

    def update_target_networks(self, env_step: int):
        if env_step % self.target_networks_update_frequency != 0:
            return

        super().update_target_networks(env_step=env_step)

        self.encoder_target_params_updater.update_target_network_params(
            params=self.encoder.parameters(),
            target_network_params=self.target_encoder.parameters(),
        )
