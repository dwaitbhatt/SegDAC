import torch
import torch.nn as nn
from tensordict import TensorDict
from segdac.data.mdp import MdpData


class TensorDictNetworkWrapper(nn.Module):
    """
    Small wrapper to make it easier to use pure nn.Module networks with TensorDicts.
    This allows support for networks with multiple inputs and outputs without having to change the base algo code.
    The inputs are taken from the TensorDict using the provided in_keys, and the outputs
    are stored in the TensorDict under the provided out_key.
    """

    def __init__(self, network: nn.Module, in_keys: list, out_key: str):
        super().__init__()
        self.network = network
        self.in_keys = in_keys
        self.out_key = out_key

    def forward(self, mdp_data: MdpData) -> TensorDict:
        network_inputs = self.get_network_inputs(mdp_data)
        network_outputs = self.network(network_inputs)
        return TensorDict(
            {
                self.out_key: network_outputs,
            },
            batch_size=mdp_data.data.batch_size,
        )

    def get_network_inputs(self, mdp_data: MdpData) -> torch.Tensor:
        if len(self.in_keys) == 1:
            network_inputs = mdp_data.data[self.in_keys[0]]
        else:
            network_inputs = torch.cat(
                [mdp_data.data[key] for key in self.in_keys], dim=-1
            )

        return network_inputs


class ActorTensorDictNetworkWrapper(TensorDictNetworkWrapper):
    def __init__(
        self,
        network: nn.Module,
        in_keys: list,
    ):
        super().__init__(network=network, in_keys=in_keys, out_key="env_action")


class CriticTensorDictNetworkWrapper(TensorDictNetworkWrapper):
    def __init__(
        self,
        network: nn.Module,
        in_keys: list,
    ):
        super().__init__(network=network, in_keys=in_keys, out_key="q_value")
