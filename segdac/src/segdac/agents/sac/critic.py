import torch
import torch.nn as nn
from segdac.agents.target_networks_params_updaters.polyak_average_updater import (
    PolyakAverageParametersUpdater,
)
from copy import deepcopy
from segdac.agents.critic import Critic
from tensordict import TensorDict
from tensordict import NonTensorData
from segdac.agents.distribution_factory import DistributionFactory
from segdac.data.mdp import MdpData
from tensordict import from_modules
from segdac.stats.grad import compute_grad_stats


class SacCritic(Critic):
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
    ):
        super().__init__()
        self.target_params_updater = target_params_updater
        self.gamma = gamma
        self.q_function_params = from_modules(
            q_function_1, q_function_2, as_module=True
        )
        self.target_q_function_params = self.q_function_params.data.clone().to(device)  # Since we clone the params, we need to move them to the device manually
        self.q_function = deepcopy(q_function_1).train()
        self.q_function_params.to_module(self.q_function)
        self.q_function_optimizer_factory = q_function_optimizer
        self.q_function_loss = q_function_loss
        self.distribution_factory = distribution_factory
        self.create_optimizers()
        self.max_grad_norm = max_grad_norm

    def create_optimizers(self):
        # If the instance has no grad (eg: eval only instance), then we skip optimizers creation
        if len(list(self.q_function.parameters())) == 0:
            return
        self.q_function_optimizer = self.q_function_optimizer_factory(
            params=self.q_function.parameters(),
        )

    def compile(self, compile_config: dict):
        if compile_config.get("critic", None) is None:
            return
        torch_compile = compile_config["critic"]["compile"]
        if torch_compile:
            self.q_function = torch.compile(
                self.q_function, **compile_config["critic"]["compile_kwargs"]
            )

    def forward(self, mdp_data: MdpData) -> TensorDict:
        critic_outputs = torch.vmap(self.batched_q_function, (0, None))(
            self.q_function_params, mdp_data
        )  # (2, train_batch_size, 1)
        # The 2 from the shape comes from the number of critic networks used in SAC, we batch their computation using torch.vmap
        min_q_value = (
            critic_outputs["q_value"].min(dim=0).values
        )  # (train_batch_size, 1)

        output_data = {**critic_outputs, "q_value": min_q_value}

        return TensorDict(output_data, batch_size=torch.Size([]))

    def batched_q_function(self, params, mdp_data: MdpData, next_q_value=None):
        """
        Inputs:
            params: (,)
            mdp_data: (train_batch_size,)
            next_q_value: (train_batch_size,)

        params is parameters from 1 of the 2 Q functions, torch.vmap will pass a slice from the 2 Q functions params of shape (2,).
        So this function performs a forward pass on the critic network and self.q_function is 1 network inside this function.
        Externally this function is repeated by torch.vmap to call it for each Q function networks in a vectorized fashion.
        So even if we have 2 Q function networks, torch.vmap will run both in a vectorized fashion.
        Note that params is the Q function params or the target Q function params depending on when it is being called.

        If next_q_value is passed then we compute the loss for the q function network as well and return that, this allows
            computing both Q1 and Q2 losses in parallel via torch.vmap

        Outputs:
            q_value : (train_batch_size, 1)
            or
            q_value, loss_q : (train_batch_size, 1), (,)
        """
        with params.to_module(self.q_function):
            critic_outputs = self.q_function(mdp_data)
            if next_q_value is not None:
                q_value = critic_outputs["q_value"]
                assert q_value.shape == next_q_value.shape
                loss_q = self.q_function_loss(q_value, next_q_value)
                return critic_outputs, loss_q
        return critic_outputs

    def update(
        self, actor, train_mdp_data: MdpData, env_step: int, is_time_to_evaluate: bool
    ) -> TensorDict:
        logs_data = {}

        target_next_q_value = self.compute_target_next_q_value(
            actor=actor,
            train_mdp_data=train_mdp_data,
            alpha=actor.alpha,
        )  # (train_batch_size, 1)

        critic_mdp_data = train_mdp_data

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
        actor_next_mdp_data = train_mdp_data.next
        next_actor_outputs = actor(actor_next_mdp_data)
        mu, log_std = next_actor_outputs["env_action"].chunk(2, dim=1)
        next_action_distribution = self.distribution_factory.create(mu, log_std)
        next_action, next_action_log_prob = next_action_distribution.rsample()

        critic_next_mdp_data = train_mdp_data.next
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

    def target_network_forward(self, mdp_data: MdpData) -> TensorDict:
        return torch.vmap(self.batched_q_function, (0, None))(
            self.target_q_function_params, mdp_data
        )

    def update_target_networks(self, env_step: int):
        self.target_params_updater.update_target_network_batched_params(
            params=self.q_function_params,
            target_network_params=self.target_q_function_params,
        )
