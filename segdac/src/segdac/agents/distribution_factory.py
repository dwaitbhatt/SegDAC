import torch
from segdac.distributions.squashed_normal import SquashedNormal


class DistributionFactory:
    def create(self, mu: torch.Tensor, log_std: torch.Tensor):
        pass


class SquashedNormalFactory(DistributionFactory):
    def __init__(self, min_logstd: float, max_logstd: float):
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

    def create(self, mu: torch.Tensor, log_std: torch.Tensor) -> SquashedNormal:
        return SquashedNormal(
            mean=mu,
            log_std=log_std,
            min_log_std=self.min_logstd,
            max_log_std=self.max_logstd,
        )
