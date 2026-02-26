import torch
import torch.nn as nn
from segdac.agents.target_networks_params_updaters.polyak_average_updater import (
    PolyakAverageParametersUpdater,
)
from segdac.agents.sac.critic import SacCritic
from segdac.agents.distribution_factory import DistributionFactory
from segdac.data.mdp import MdpData
from tensordict import TensorDict
from tensordict import NonTensorData
from segdac.stats.grad import compute_grad_stats


class SacAeCritic(SacCritic):
    def __init__(
        self,
        target_params_updater: PolyakAverageParametersUpdater,
        gamma: float,
        q_function_1: nn.Module,
        q_function_2: nn.Module,
        q_function_loss: nn.Module,
        q_function_optimizer,
        distribution_factory: DistributionFactory,
        device: str,
        max_grad_norm: float,
        encoder: nn.Module,
        target_encoder: nn.Module,
        encoder_input_observation_key: str,
        encoder_output_observation_key: str,
        encoder_optimizer
    ):
        super().__init__(
            target_params_updater=target_params_updater,
            gamma=gamma,
            q_function_1=q_function_1,
            q_function_2=q_function_2,
            q_function_loss=q_function_loss,
            q_function_optimizer=q_function_optimizer,
            distribution_factory=distribution_factory,
            device=device,
            max_grad_norm=max_grad_norm,
        )
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.encoder_input_observation_key = encoder_input_observation_key
        self.encoder_output_observation_key = encoder_output_observation_key
        self.encoder_optimizer = encoder_optimizer
    
    def update(
        self, actor, train_mdp_data: MdpData, env_step: int, is_time_to_evaluate: bool
    ) -> TensorDict:
        """
        In the paper's implementation, the encoder is nested inside the critic, so it is updated via the critic optimizer.
        In our case, we have a separate encoder optimizer, so we need to call it explicitly.
        """
        logs_data = {}

        self.encoder.train()
        self.encoder_optimizer.zero_grad()

        target_next_q_value = self.compute_target_next_q_value(
            actor=actor,
            train_mdp_data=train_mdp_data,
            alpha=actor.alpha,
        )  # (train_batch_size, 1)

        critic_mdp_data = self.preprocess_data_for_critic(train_mdp_data)

        critic_outputs, q_value_losses = torch.vmap(
            self.batched_q_function, (0, None, None)
        )(
            self.q_function_params, critic_mdp_data, target_next_q_value
        )  # (2, train_batch_size, 1), (2,)

        q_value = critic_outputs["q_value"]
        loss = q_value_losses.sum(dim=0)  # (,)

        self.q_function_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.q_function.parameters(),
                max_norm=self.max_grad_norm,
                norm_type=2.0,
            )
        self.q_function_optimizer.step()
        self.encoder_optimizer.step()

        if is_time_to_evaluate:
            q_value_detach = q_value.detach()
            logs_data["critic_target_q"] = target_next_q_value.detach().mean()
            logs_data["critic_q1"] = q_value_detach[0].mean()
            logs_data["critic_q2"] = q_value_detach[1].mean()
            logs_data["critic_loss"] = loss.detach()
            layers, avg_grads, max_grads, l2_norms = compute_grad_stats(
                self.q_function.named_parameters()
            )
            logs_data["critic_grad_stats"] = TensorDict(
                {
                    "critic_grad_layers": NonTensorData(layers),
                    "critic_grad_avg": avg_grads,
                    "critic_grad_max": max_grads,
                    "critic_grad_l2_norms": l2_norms,
                },
                batch_size=torch.Size([len(layers)]),
            )
        
        return TensorDict(
            logs_data,
            batch_size=torch.Size([]),
        )

    @torch.no_grad()
    def compute_target_next_q_value(
        self,
        actor,
        train_mdp_data: MdpData,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        actor_next_mdp_data = (
            self.preprocess_next_data_for_actor_during_target_q_computation(
                train_mdp_data
            )
        )
        next_actor_outputs = actor(actor_next_mdp_data)
        mu, log_std = next_actor_outputs["env_action"].chunk(2, dim=1)
        next_action_distribution = self.distribution_factory.create(mu, log_std)
        next_action, next_action_log_prob = next_action_distribution.rsample()

        critic_next_mdp_data = (
            self.preprocess_next_data_for_critic_during_target_q_computation(
                train_mdp_data
            )
        )
        # SAC Critic does not take action from replay buffer
        critic_data = critic_next_mdp_data.data.exclude("action")
        critic_data["action"] = next_action
        critic_mdp_data = MdpData(
            data=critic_data,
            segmentation_data=critic_next_mdp_data.segmentation_data,
            next=critic_next_mdp_data.next,
        )

        next_q_value = self.target_network_forward(critic_mdp_data)[
            "q_value"
        ]  # (2, train_batch_size, 1)

        next_q_value = next_q_value.min(dim=0).values  # (train_batch_size, 1)

        next_v_value = next_q_value - alpha * next_action_log_prob

        replay_buffer_reward = train_mdp_data.next.data["reward"].reshape(-1, 1)
        replay_buffer_done = train_mdp_data.next.data["done"].reshape(-1, 1)
        gamma = train_mdp_data.next.data.get(
            "gamma", torch.tensor([self.gamma], device=replay_buffer_done.device)
        ).expand_as(replay_buffer_done)

        assert (
            replay_buffer_reward.shape == replay_buffer_done.shape
            and replay_buffer_done.shape == next_q_value.shape
            and gamma.shape == replay_buffer_done.shape
        )

        return (
            replay_buffer_reward
            + (1 - replay_buffer_done.float()) * gamma * next_v_value
        )

    def preprocess_data_for_critic(
        self,
        train_mdp_data: MdpData
    ) -> MdpData:
        obs_latent = self.encoder(train_mdp_data)[self.encoder_output_observation_key]
        train_mdp_data.data[self.encoder_output_observation_key] = obs_latent
        return train_mdp_data
        
    def preprocess_next_data_for_actor_during_target_q_computation(
        self,
        train_mdp_data: MdpData,
    ) -> MdpData:
        """
        SAC AE uses the main encoder to compute the latent observation for the next data used by the actor when computing the target Q value.
        See https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L320
            and
            https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L410-L413
            (Only the critic target uses the target encoder)
        """
        next_mdp_data = train_mdp_data.next
        next_obs_latent = self.encoder(next_mdp_data)[self.encoder_output_observation_key]
        next_mdp_data.data[self.encoder_output_observation_key] = next_obs_latent
        return next_mdp_data
    
    def preprocess_next_data_for_critic_during_target_q_computation(
        self,
        train_mdp_data: MdpData
    ) -> MdpData:
        """
        SAC AE uses the **target encoder** to compute the latent observation for the next data used by the critic when computing the target Q value.
        See https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L321C13-L321C79
            and
            https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L410-L413
            (Only the critic target uses the target encoder)
        """
        next_mdp_data = train_mdp_data.next
        next_obs_latent = self.target_encoder(next_mdp_data)[self.encoder_output_observation_key]
        next_mdp_data.data[self.encoder_output_observation_key] = next_obs_latent
        return next_mdp_data
