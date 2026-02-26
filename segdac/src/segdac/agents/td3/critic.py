import torch
import torch.nn as nn
from segdac.agents.target_networks_params_updaters.polyak_average_updater import (
    PolyakAverageParametersUpdater,
)

from copy import deepcopy
from segdac.agents.critic import Critic
from segdac.data.mdp import MdpData
from tensordict import TensorDict
from tensordict import from_modules
from segdac.stats.grad import compute_grad_stats
from tensordict import NonTensorData


class Td3Critic(Critic):
    def __init__(
        self,
        target_params_updater: PolyakAverageParametersUpdater,
        gamma: float,
        q_function_1: nn.Module,
        q_function_2: nn.Module,
        q_function_loss: nn.Module,
        q_function_optimizer,
        noise_std: float,
        noise_clip_value: float,
        device: str,
        max_grad_norm: float,
    ):
        super().__init__()
        self.target_params_updater = target_params_updater
        self.gamma = gamma
        self.q_function_params = from_modules(
            q_function_1, q_function_2, as_module=True
        ).to(device)
        self.target_q_function_params = self.q_function_params.data.clone()
        self.q_function = deepcopy(q_function_1).to(device).train()
        self.q_function_params.to_module(self.q_function)
        self.q_function_optimizer_factory = q_function_optimizer
        self.q_function_loss = q_function_loss
        self.noise_std = noise_std
        self.noise_clip_value = noise_clip_value
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
        # Use weights of the first Q function as done in the paper
        with self.q_function_params[0].to_module(self.q_function):
            critic_outputs = self.q_function(mdp_data)
        return critic_outputs

    def update(
        self, actor, train_mdp_data: MdpData, env_step: int, is_time_to_evaluate: bool
    ) -> TensorDict:
        target_next_q_value = self.compute_target_next_q_value(
            actor=actor, train_mdp_data=train_mdp_data
        )  # (train_batch_size, 1)

        critic_outputs, q_value_losses = torch.vmap(
            self.batched_q_function, (0, None, None)
        )(
            self.q_function_params, train_mdp_data, target_next_q_value
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

        logs_data = {}
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
        self, actor, train_mdp_data: MdpData
    ) -> torch.Tensor:
        action_noise = torch.normal(
            mean=0.0,
            std=self.noise_std,
            size=train_mdp_data.data["action"].shape,
            device=train_mdp_data.data["action"].device,
            dtype=train_mdp_data.data["action"].dtype,
        ).clamp(-self.noise_clip_value, self.noise_clip_value)

        next_mdp_data = train_mdp_data.next
        next_actor_outputs = actor.get_target_network()(next_mdp_data)

        next_action = next_actor_outputs["env_action"] + action_noise
        tanh_min = -1.0
        tanh_max = 1.0
        next_action = next_action.clamp(tanh_min, tanh_max)

        critic_data = next_mdp_data.data.exclude("action")
        critic_data["action"] = next_action
        critic_mdp_data = MdpData(
            data=critic_data,
            segmentation_data=next_mdp_data.segmentation_data,
            next=next_mdp_data.next,
        )

        next_q_value = torch.vmap(self.batched_q_function, (0, None))(
            self.target_q_function_params, critic_mdp_data
        )[
            "q_value"
        ]  # (2, train_batch_size, 1)

        next_q_value = next_q_value.min(dim=0).values  # (train_batch_size, 1)

        replay_buffer_reward = next_mdp_data.data["reward"].reshape(-1, 1)
        replay_buffer_done = next_mdp_data.data["done"].reshape(-1, 1)
        gamma = next_mdp_data.data.get(
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

    def batched_q_function(self, params, mdp_data: MdpData, next_q_value=None):
        """
        Inputs:
            params: (,)
            mdp_data: (train_batch_size,)
            next_q_value: (train_batch_size,)

        Outputs:
            critic_outputs:
                q_value : (train_batch_size, 1)
            or
            critic_outputs:
                q_value (train_batch_size, 1)
            loss_q : (,)
        """
        with params.to_module(self.q_function):
            critic_outputs = self.q_function(mdp_data)
            if next_q_value is not None:
                q_value = critic_outputs["q_value"]
                assert q_value.shape == next_q_value.shape
                loss_q = self.q_function_loss(q_value, next_q_value)
                return critic_outputs, loss_q
        return critic_outputs

    def update_target_networks(self, env_step: int):
        self.target_params_updater.update_target_network_batched_params(
            params=self.q_function_params,
            target_network_params=self.target_q_function_params,
        )
