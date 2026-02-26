import torch
import torch.nn as nn
from copy import deepcopy
from tensordict import TensorDict
from tensordict import NonTensorData
from segdac.stats.grad import compute_grad_stats
from segdac.data.mdp import MdpData
from segdac.agents.critic import Critic
from segdac.agents.target_networks_params_updaters.polyak_average_updater import (
    PolyakAverageParametersUpdater,
)


class DqnCritic(Critic):
    def __init__(
        self,
        target_params_updater: PolyakAverageParametersUpdater,
        gamma: float,
        q_function: nn.Module,
        q_function_loss: nn.Module,
        q_function_optimizer,
        max_grad_norm: float,
    ):
        super().__init__()
        self.target_params_updater = target_params_updater
        self.gamma = gamma
        self.q_function = q_function
        self.target_q_function = deepcopy(q_function).eval()
        self.q_function_loss = q_function_loss
        self.q_function_optimizer = q_function_optimizer(params=self.q_function.parameters())
        self.max_grad_norm = max_grad_norm

    def forward(self, mdp_data: MdpData) -> TensorDict:
        return self.q_function(mdp_data)

    def update(
        self,
        actor,
        train_mdp_data: MdpData,
        env_step: int,
        is_time_to_evaluate: bool
    ) -> TensorDict:
        target = self.compute_target(train_mdp_data.next)

        q_values = self.q_function(train_mdp_data)["q_value"] # (batch_size, num_actions)
        action_from_rb = train_mdp_data.data["action"].unsqueeze(1) # (batch_size, 1)
        q_value = q_values.gather(dim=1, index=action_from_rb.long()).squeeze(1) # (batch_size, 1)

        assert q_value.shape == target.shape

        loss = self.q_function_loss(
            q_value,
            target
        )

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
            logs_data["critic_target_q"] = target.detach().mean()
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
    def compute_target(
        self,
        next_train_mdp_data: MdpData
    ) -> torch.Tensor:
        next_q_values = self.target_q_function(next_train_mdp_data)["q_value"]
        max_next_q_value = next_q_values.max(dim=1)[0]
        target = next_train_mdp_data.data["reward"] + (
            1 - next_train_mdp_data.data["done"].float()
        ) * self.gamma * max_next_q_value
        return target

    def update_target_networks(self, env_step: int):
        self.target_params_updater.update_target_network_params(
            params=self.q_function.parameters(),
            target_network_params=self.target_q_function.parameters(),
        )
