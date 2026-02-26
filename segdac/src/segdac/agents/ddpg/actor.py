import torch
import torch.nn as nn
from segdac.agents.target_networks_params_updaters.polyak_average_updater import (
    PolyakAverageParametersUpdater,
)
from copy import deepcopy
from segdac.agents.actor import Actor
from segdac.agents.critic import Critic
from segdac.data.mdp import MdpData
from segdac.stats.grad import compute_grad_stats
from tensordict import TensorDict
from tensordict import NonTensorData


class DdpgActor(Actor):
    def __init__(
        self,
        target_params_updater: PolyakAverageParametersUpdater,
        network: nn.Module,
        optimizer,
        max_grad_norm: float,
    ):
        super().__init__()
        self.target_params_updater = target_params_updater
        self.network = network
        self.optimizer_factory = optimizer
        self.target_network = deepcopy(network).eval()
        self.create_optimizers()
        self.max_grad_norm = max_grad_norm

    def create_optimizers(self):
        # If the instance has no grad (eg: eval only instance), then we skip optimizers creation
        if len(list(self.network.parameters())) == 0:
            return
        self.optimizer = self.optimizer_factory(params=self.network.parameters())

    def forward(self, mdp_data: MdpData) -> TensorDict:
        return self.network(mdp_data)

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
        actor_outputs = self(train_mdp_data)
        action = actor_outputs["env_action"]

        # DDPG Actor update does not take action from replay buffer
        critic_data = train_mdp_data.data.exclude("action")
        critic_data["action"] = action

        critic_mdp_data = MdpData(
            data=critic_data,
            segmentation_data=train_mdp_data.segmentation_data,
            next=train_mdp_data.next,
        )

        loss = -critic(critic_mdp_data)["q_value"].mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.network.parameters(),
                max_norm=self.max_grad_norm,
                norm_type=2.0,
            )
        self.optimizer.step()

        logs_data = {}
        if is_time_to_evaluate:
            logs_data["actor_loss"] = loss.detach()
            layers, ave_grads, max_grads, l2_norms = compute_grad_stats(
                self.network.named_parameters()
            )
            logs_data["actor_grad_stats"] = TensorDict(
                {
                    "actor_grad_layers": NonTensorData(layers),
                    "actor_grad_avg": ave_grads,
                    "actor_grad_max": max_grads,
                    "actor_grad_l2_norms": l2_norms,
                },
                batch_size=torch.Size([len(layers)]),
            )

        return TensorDict(
            logs_data,
            batch_size=torch.Size([]),
        )

    def update_target_networks(self, env_step: int):
        self.target_params_updater.update_target_network_params(
            params=self.network.parameters(),
            target_network_params=self.target_network.parameters(),
        )

    def get_target_network(self) -> nn.Module:
        return self.target_network
