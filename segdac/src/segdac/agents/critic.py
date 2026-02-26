import torch.nn as nn
from abc import ABC
from abc import abstractmethod
from tensordict import TensorDict
from segdac.data.mdp import MdpData


class Critic(ABC, nn.Module):
    def compile(self, compile_config: dict):
        pass

    @abstractmethod
    def update(
        self, actor, train_mdp_data: MdpData, env_step: int, is_time_to_evaluate: bool
    ) -> TensorDict:
        pass

    @abstractmethod
    def update_target_networks(self, env_step: int):
        pass
