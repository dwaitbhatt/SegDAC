import torch
import torch.nn as nn
from segdac.agents.target_networks_params_updaters.polyak_average_updater import (
    PolyakAverageParametersUpdater,
)
from copy import deepcopy
from segdac.agents.critic import Critic
from segdac.data.mdp import MdpData
from segdac.stats.grad import compute_grad_stats
from tensordict import TensorDict
from tensordict import NonTensorData
from segdac.networks.network_wrapper import TensorDictNetworkWrapper


class DdpgCritic(Critic):
    def __init__(
        self,
        target_params_updater: PolyakAverageParametersUpdater,
        gamma: float,
        q_function: TensorDictNetworkWrapper,
        q_function_loss: nn.Module,
        q_function_optimizer,
        max_grad_norm: float,
    ):
        super().__init__()
        self.target_params_updater = target_params_updater
        self.gamma = gamma
        self.q_function = q_function
        self.q_function_loss = q_function_loss
        self.q_function_optimizer_factory = q_function_optimizer
        self.target_q_function = deepcopy(q_function).eval()
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

    def forward(self, data: TensorDict) -> TensorDict:
        return self.q_function(data)

    def update(
        self,
        actor,
        train_mdp_data: MdpData,
        env_step: int,
        is_time_to_evaluate: bool,
    ) -> TensorDict:
        target_next_q_value = self.compute_target_next_q_value(
            actor=actor, next_train_mdp_data=train_mdp_data.next
        )  # (train_batch_size, 1)

        q_value = self.q_function(train_mdp_data)["q_value"]
        assert q_value.shape == target_next_q_value.shape
        loss = self.q_function_loss(q_value, target_next_q_value)

        self.q_function_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.q_function.parameters(),
                max_norm=self.max_grad_norm,
                norm_type=2.0,
            )
        self.q_function_optimizer.step()

        logs_data = {}
        if is_time_to_evaluate:
            logs_data["critic_target_q"] = target_next_q_value.detach().mean()
            logs_data["critic_q1"] = q_value.detach().mean()
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
        self, actor, next_train_mdp_data: MdpData
    ) -> torch.Tensor:
        next_actor_outputs = actor.get_target_network()(next_train_mdp_data)
        next_action = next_actor_outputs["env_action"]

        critic_data = next_train_mdp_data.data.exclude("action")
        critic_data["action"] = next_action
        critic_mdp_data = MdpData(
            data=critic_data,
            segmentation_data=next_train_mdp_data.segmentation_data,
            next=next_train_mdp_data.next,
        )

        next_q_value = self.target_q_function(critic_mdp_data)["q_value"]

        replay_buffer_reward = next_train_mdp_data.data["reward"].reshape(-1, 1)
        replay_buffer_done = next_train_mdp_data.data["done"].reshape(-1, 1)
        gamma = next_train_mdp_data.data.get(
            "gamma", torch.tensor([self.gamma], device=replay_buffer_done.device)
        ).expand_as(replay_buffer_done)

        assert (
            replay_buffer_reward.shape == replay_buffer_done.shape
            and replay_buffer_done.shape == next_q_value.shape
            and gamma.shape == replay_buffer_done.shape
        )

        return (
            replay_buffer_reward
            + (1 - replay_buffer_done.float()) * gamma * next_q_value
        )

    def update_target_networks(self, env_step: int):
        self.target_params_updater.update_target_network_params(
            params=self.q_function.parameters(),
            target_network_params=self.target_q_function.parameters(),
        )
