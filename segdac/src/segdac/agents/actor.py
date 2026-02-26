import torch.nn as nn
from abc import ABC
from abc import abstractmethod
from segdac.agents.critic import Critic
from tensordict import TensorDict
from segdac.data.mdp import MdpData


class Actor(ABC, nn.Module):
    def compile(self, compile_config: dict):
        pass

    @abstractmethod
    def update(
        self,
        train_mdp_data: MdpData,
        critic: Critic,
        env_step: int,
        is_time_to_evaluate: bool,
    ) -> TensorDict:
        pass

    @abstractmethod
    def update_target_networks(self, env_step: int):
        pass
