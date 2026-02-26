import torch
from collections.abc import Iterator
from torch.nn.parameter import Parameter


class HardParametersUpdater:

    @torch.no_grad()
    def update_target_network_params(
        self, params: Iterator[Parameter], target_network_params: Iterator[Parameter]
    ):
        for param, target_param in zip(params, target_network_params):
            # Copy weights directly (hard update)
            target_param.data.copy_(param.data)
