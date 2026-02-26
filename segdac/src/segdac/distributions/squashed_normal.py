import torch
from torch.distributions import Normal


class SquashedNormal:
    def __init__(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        min_log_std: float,
        max_log_std: float,
    ):
        self.compute_squashed_std(
            raw_log_std=log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )
        self._mean = mean
        self.normal = Normal(mean, self.std)

    def compute_squashed_std(
        self,
        raw_log_std: torch.Tensor,
        min_log_std: float,
        max_log_std: float,
    ):
        log_std_tanh = torch.tanh(raw_log_std)  # Scale to [-1, 1]
        self.log_std = min_log_std + 0.5 * (max_log_std - min_log_std) * (
            log_std_tanh + 1
        )  # Scale to [min_log_std, max_log_std]
        self.std = self.log_std.exp()

    def apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def sample(self):
        return self.apply_transform(self.normal.sample())

    def rsample(self) -> tuple:
        x = self.normal.rsample()
        action = self.apply_transform(x)

        log_prob = self.normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        # clamp into (−1+ε,1−ε)
        y = value.clamp(-1 + eps, 1 - eps)
        # invert tanh via log:
        inv_tanh = 0.5 * (torch.log(1 + y) - torch.log(1 - y))  # same as atanh(y)
        # base Gaussian log‐prob
        log_base = self.normal.log_prob(inv_tanh)
        # tanh‐Jacobian correction
        log_jac = -torch.log(1 - y.pow(2) + eps)
        # sum over action dims
        return (log_base + log_jac).sum(dim=1, keepdim=True)

    @property
    def mean(self):
        return self.apply_transform(self._mean)
