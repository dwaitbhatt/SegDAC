import torch
import torch.nn as nn
import numpy as np
from segdac.agents.distribution_factory import DistributionFactory
from segdac.agents.actor import Actor
from segdac.agents.critic import Critic
from tensordict import TensorDict
from tensordict import merge_tensordicts
from segdac.data.mdp import MdpData
from segdac.stats.grad import compute_grad_stats
from tensordict import NonTensorData


class SacActor(Actor):
    def __init__(
        self,
        network: nn.Module,
        policy_optimizer,
        entropy_optimizer,
        device: str,
        action_dim: int,
        distribution_factory: DistributionFactory,
        initial_entropy: float,
        max_grad_norm: float,
    ):
        super().__init__()
        self.target_entropy = -action_dim
        self.network = network
        self.policy_optimizer_factory = policy_optimizer
        self.entropy_optimizer_factory = entropy_optimizer
        self.log_alpha = torch.tensor(
            np.log(initial_entropy),
            device=device,
            requires_grad=True,
            dtype=torch.float32,
        )
        self.distribution_factory = distribution_factory
        self.create_optimizers()
        self.max_grad_norm = max_grad_norm

    def create_optimizers(self):
        # If the instance has no grad (eg: eval only instance), then we skip optimizers creation
        if len(list(self.network.parameters())) == 0:
            return
        self.policy_optimizer = self.policy_optimizer_factory(
            params=self.network.parameters(),
        )
        self.entropy_optimizer = self.entropy_optimizer_factory(
            params=[self.log_alpha],
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def forward(self, mdp_data: MdpData, **kwargs) -> TensorDict:
        return self.network(mdp_data, **kwargs)

    def compile(self, compile_config: dict):
        if compile_config.get("actor", None) is None:
            return
        torch_compile = compile_config["actor"]["compile"]
        if torch_compile:
            self.network = torch.compile(
                self.network,
                **compile_config["actor"]["compile_kwargs"],
            )

    def update(
        self,
        train_mdp_data: MdpData,
        critic: Critic,
        env_step: int,
        is_time_to_evaluate: bool,
    ) -> TensorDict:
        network_outputs = self(train_mdp_data)
        mu, log_std = network_outputs["env_action"].chunk(2, dim=1)
        action_distribution = self.distribution_factory.create(mu, log_std)
        action, action_log_prob = action_distribution.rsample()

        actor_logs = self.update_actor(
            train_mdp_data,
            action,
            network_outputs,
            critic,
            action_log_prob,
            is_time_to_evaluate,
        )
        entropy_logs = self.update_entropy(action_log_prob, is_time_to_evaluate)

        return merge_tensordicts(actor_logs, entropy_logs)

    def update_actor(
        self,
        train_mdp_data: MdpData,
        action: torch.Tensor,
        actor_outputs: TensorDict,
        critic: nn.Module,
        action_log_prob: torch.Tensor,
        is_time_to_evaluate: bool,
    ) -> TensorDict:
        logs_data = {}

        critic_mdp_data = train_mdp_data
        # SAC Critic does not take action from replay buffer
        critic_data = critic_mdp_data.data.exclude("action")
        critic_data["action"] = action
        critic_mdp_data = MdpData(
            data=critic_data,
            segmentation_data=critic_mdp_data.segmentation_data,
            next=critic_mdp_data.next,
        )

        q_value = critic(critic_mdp_data)["q_value"]

        assert action_log_prob.shape == q_value.shape

        policy_loss = (self.alpha.detach() * action_log_prob - q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.network.parameters(),
                max_norm=self.max_grad_norm,
                norm_type=2.0,
            )
        self.policy_optimizer.step()

        if is_time_to_evaluate:
            logs_data["actor_loss"] = policy_loss.detach()
            layers, avg_grads, max_grads, l2_norms = compute_grad_stats(
                self.network.named_parameters()
            )
            logs_data["actor_grad_stats"] = TensorDict(
                {
                    "actor_grad_layers": NonTensorData(layers),
                    "actor_grad_avg": avg_grads,
                    "actor_grad_max": max_grads,
                    "actor_grad_l2_norms": l2_norms,
                },
                batch_size=torch.Size([len(layers)]),
            )

        return TensorDict(
            logs_data,
            batch_size=torch.Size([]),
        )

    def update_entropy(
        self, action_log_prob: torch.Tensor, is_time_to_evaluate: bool
    ) -> TensorDict:
        entropy_loss = (
            -self.log_alpha * action_log_prob.detach()
            - self.log_alpha * self.target_entropy
        ).mean()

        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()

        logs_data = {}
        if is_time_to_evaluate:
            logs_data["entropy_loss"] = entropy_loss.detach()
            logs_data["log_alpha"] = self.log_alpha.detach()
            logs_data["alpha"] = self.alpha.detach()

        return TensorDict(
            logs_data,
            batch_size=torch.Size([]),
        )

    def update_target_networks(self, env_step: int):
        pass
