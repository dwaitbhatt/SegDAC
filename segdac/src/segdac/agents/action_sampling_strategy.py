import torch
import torch.nn as nn
from tensordict import TensorDict
from segdac.agents.actor import Actor
from segdac.agents.distribution_factory import DistributionFactory
from segdac.data.mdp import MdpData


class ActionSamplingStrategy(nn.Module):
    def __init__(self, actor: Actor):
        super().__init__()
        self.actor = actor
        self.disable_exploration()
        self.disable_stochasticity()

    def enable_exploration(self):
        self.is_exploration_enabled = True

    def disable_exploration(self):
        self.is_exploration_enabled = False

    def enable_stochasticity(self):
        self.is_stochasticity_enabled = True

    def disable_stochasticity(self):
        self.is_stochasticity_enabled = False

    def compile(self, compile_config: dict):
        self.actor.compile(compile_config=compile_config)

    @torch.no_grad()
    def forward(self, mdp_data: MdpData) -> TensorDict:
        actor_outputs = self.actor(mdp_data)

        unscaled_action = self.extract_env_action(actor_outputs, mdp_data)

        output_data = {
            "unscaled_action": unscaled_action,
        }

        out = TensorDict(output_data, batch_size=torch.Size([unscaled_action.shape[0]]))

        return out

    def extract_env_action(
        self, actor_outputs: TensorDict, mdp_data: MdpData
    ) -> torch.Tensor:
        raise NotImplementedError

    def step(self, frames: int = 1):
        """
        Function that is called on env collection step, used to update exploration noise schedulers.
        """
        pass


class StochasticActionSamplingStrategy(ActionSamplingStrategy):
    def __init__(
        self,
        actor: Actor,
        distribution_factory: DistributionFactory,
    ):
        super().__init__(actor=actor)
        self.distribution_factory = distribution_factory

    def extract_env_action(
        self, actor_outputs: TensorDict, mdp_data: MdpData
    ) -> torch.Tensor:
        mu, log_std = actor_outputs["env_action"].chunk(2, dim=1)
        action_distribution = self.distribution_factory.create(mu, log_std)

        if self.is_stochasticity_enabled:
            action = action_distribution.sample()
        else:
            action = action_distribution.mean

        return action


class DeterministicActionSamplingStrategy(ActionSamplingStrategy):
    def extract_env_action(
        self, actor_outputs: TensorDict, mdp_data: MdpData
    ) -> torch.Tensor:
        return actor_outputs["env_action"]


class EpsilonGreedyActionSamplingStrategy(ActionSamplingStrategy):
    def __init__(self, actor: Actor, device: str, seed: int, nb_actions: int, epsilon_end: float = 0.1, epsilon_decay_frames: int = 1_000_000):
        super().__init__(actor)
        self.register_buffer("epsilon", torch.tensor(1.0, dtype=torch.float32, device=device))
        self.register_buffer("epsilon_end", torch.tensor(epsilon_end, dtype=torch.float32, device=device))
        self.epsilon_decay_frames = epsilon_decay_frames
        self.generator = torch.Generator().manual_seed(seed)
        self.nb_actions = nb_actions
        self.device = device

    def step(self, frames: int = 1) -> None:
        if not self.is_exploration_enabled:
            return
        for _ in range(frames):
            self.epsilon.data.copy_(
                torch.maximum(
                    self.epsilon_end,
                    self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay_frames,
                )
            )

    @torch.no_grad()
    def forward(self, mdp_data: MdpData) -> TensorDict:
        num_envs = mdp_data.data.shape[0]
        if self.is_exploration_enabled and torch.rand(1, generator=self.generator).item() < self.epsilon:
            unscaled_action = torch.randint(low=0, high=self.nb_actions, size=(num_envs,), generator=self.generator).to(self.device)
        else:
            unscaled_action = self.actor(mdp_data)["env_action"].squeeze(1)

        output_data = {
            "unscaled_action": unscaled_action,
        }

        out = TensorDict(output_data, batch_size=torch.Size([unscaled_action.shape[0]]))

        return out


class GaussianNoiseActionSamplingStrategy(ActionSamplingStrategy):
    def __init__(
        self,
        actor: Actor,
        device: str,
        seed: int,
        sigma_init: float = 1.0,
        sigma_end: float = 0.1,
        annealing_num_steps: int = 1000,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        """
        sigma_init : initial epsilon value. default: 1.0
        sigma_end : final epsilon value. default: 0.1
        annealing_num_steps : number of steps it will take for sigma to reach the sigma_end value. default: 1000
        mean : mean of each output element's normal distribution. default: 0.0
        std : standard deviation of each output element's normal distribution. default: 1.0
        seed : Seed to reproduce results
        """
        super().__init__(actor=actor)
        self.register_buffer("sigma_init", torch.tensor(sigma_init, device=device))
        self.register_buffer("sigma_end", torch.tensor(sigma_end, device=device))
        self.annealing_num_steps = annealing_num_steps
        self.register_buffer("mean", torch.tensor(mean, device=device))
        self.register_buffer("std", torch.tensor(std, device=device))
        self.register_buffer(
            "sigma", torch.tensor(sigma_init, dtype=torch.float32, device=device)
        )
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def step(self, frames: int = 1):
        if not self.is_exploration_enabled:
            return
        for _ in range(frames):
            self.sigma.data.copy_(
                torch.maximum(
                    self.sigma_end,
                    (
                        self.sigma
                        - (self.sigma_init - self.sigma_end) / self.annealing_num_steps
                    ),
                )
            )

    def extract_env_action(
        self, actor_outputs: TensorDict, mdp_data: MdpData
    ) -> torch.Tensor:
        raw_action = actor_outputs["env_action"]

        if self.is_exploration_enabled:
            exploration_noise = torch.normal(
                mean=self.mean,
                std=self.std,
                size=raw_action.shape,
                device=raw_action.device,
                dtype=raw_action.dtype,
                generator=self.generator,
            )
            sigma = self.sigma.expand(raw_action.shape)
            action = torch.clamp(raw_action + exploration_noise * sigma, -1.0, 1.0)
        else:
            action = raw_action

        return action


class OrnsteinUhlenbeckNoiseActionSamplingStrategy(ActionSamplingStrategy):
    def __init__(
        self,
        actor: Actor,
        device: str,
        seed: int,
        eps_init: float = 1.0,
        eps_end: float = 0.1,
        annealing_num_steps: int = 1000,
        theta: float = 0.15,
        mu: float = 0.0,
        sigma: float = 0.2,
    ):
        """
        eps_init : initial scaling factor for the OU noise (default 1.0).
        eps_end : final scaling factor for the OU noise (default 0.1).
        annealing_num_steps : number of steps to anneal eps from eps_init to eps_end.
        theta : speed of mean reversion in the OU process (default 0.15).
        mu : long-term mean of the OU process (default 0.0).
        sigma : volatility parameter for the OU process (default 0.2).
        seed : Seed for reproducibility.
        """
        super().__init__(actor=actor)
        self.register_buffer("eps_init", torch.tensor(eps_init, device=device))
        self.register_buffer("eps_end", torch.tensor(eps_end, device=device))
        self.annealing_num_steps = annealing_num_steps
        self.register_buffer(
            "eps", torch.tensor(eps_init, dtype=torch.float32, device=device)
        )
        self.register_buffer("theta", torch.tensor(theta, device=device))
        self.register_buffer("mu", torch.tensor(mu, device=device))
        self.register_buffer("sigma", torch.tensor(sigma, device=device))
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.state = None

    def step(self, frames: int = 1) -> None:
        """Anneal eps from eps_init to eps_end over annealing_num_steps."""
        if not self.is_exploration_enabled:
            return
        for _ in range(frames):
            self.eps.data.copy_(
                torch.maximum(
                    self.eps_end,
                    self.eps
                    - (self.eps_init - self.eps_end) / self.annealing_num_steps,
                )
            )

    def extract_env_action(
        self, actor_outputs: TensorDict, mdp_data: MdpData
    ) -> torch.Tensor:
        raw_action = actor_outputs["env_action"]

        if self.is_exploration_enabled:
            is_init = mdp_data.data["is_init"]

            if self.state is None:
                self.state = torch.full_like(raw_action, self.mu.item())
            else:
                reset_mask = is_init.unsqueeze(1).expand(self.state.shape)
                self.state = torch.where(
                    reset_mask,
                    torch.full_like(self.state, self.mu.item()),
                    self.state,
                )

            normal_sample = torch.normal(
                mean=0.0,
                std=1.0,
                size=raw_action.shape,
                device=raw_action.device,
                dtype=raw_action.dtype,
                generator=self.generator,
            )
            dx = self.theta * (self.mu - self.state) + self.sigma * normal_sample
            self.state = self.state + dx
            ou_noise = self.state
            action = torch.clamp(raw_action + ou_noise * self.eps, -1.0, 1.0)
        else:
            action = raw_action

        return action
