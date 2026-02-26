import torch
from segdac.agents.actor import Actor
from segdac.agents.critic import Critic
from tensordict import TensorDict
from segdac.data.mdp import MdpData


class DqnActor(Actor):
    def __init__(
        self,
        critic: Critic,
    ):
        super().__init__()
        self.critic = critic

    def forward(self, mdp_data: MdpData) -> TensorDict:
        q_values = self.critic(mdp_data)["q_value"]
        actions = q_values.argmax(dim=1, keepdim=True)
        return TensorDict(
            {
                "env_action": actions,
            },
            batch_size=torch.Size([mdp_data.data.batch_size[0], 1]),
        )

    def update(
        self,
        train_mdp_data: MdpData,
        critic: Critic,
        env_step: int,
        is_time_to_evaluate: bool,
    ) -> TensorDict:
        return TensorDict(
            {},
            batch_size=torch.Size([]),
        )

    def update_target_networks(self, env_step: int):
        pass
