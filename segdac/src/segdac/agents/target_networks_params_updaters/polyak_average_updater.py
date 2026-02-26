import torch
from collections.abc import Iterator
from torch.nn.parameter import Parameter


class PolyakAverageParametersUpdater:
    def __init__(self, tau: float):
        self.tau = tau

    @torch.no_grad()
    def update_target_network_params(
        self, params: Iterator[Parameter], target_network_params: Iterator[Parameter]
    ):
        for param, target_param in zip(params, target_network_params):
            # In-place update: target_param = tau * param + (1-tau) * target_param.detach()
            target_param.mul_(1 - self.tau).add_(self.tau * param.detach())

    @torch.no_grad
    def update_target_network_batched_params(self, params, target_network_params):
        """
        Taken from https://github.com/pytorch-labs/LeanRL/blob/a416e61058ffa2dfe571bfe1d69a7f62d622b503/leanrl/sac_continuous_action_torchcompile.py#L352C17-L353C17
        """
        # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
        target_network_params.lerp_(params.data, self.tau)
