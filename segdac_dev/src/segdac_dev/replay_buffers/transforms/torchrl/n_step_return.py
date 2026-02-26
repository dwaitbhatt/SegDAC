import torch
from tensordict import TensorDict
from torchrl.envs.transforms.transforms import Transform


class NStepReturnTransform(Transform):
    def __init__(
        self,
        num_envs: int,
        device: str,
        n_steps: int,
        gamma: float,
    ):
        super().__init__()
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer.")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be between 0.0 and 1.0.")

        self.n_steps = n_steps
        self.gamma = gamma
        self._buffer = None
        self.nb_transitions_per_envs = torch.zeros(
            (num_envs, 1), dtype=torch.int, device=device
        )

    @torch.no_grad()
    def _inv_call(self, tensordict: TensorDict) -> TensorDict | None:
        if self.n_steps == 1:
            reward_val = tensordict.get(self.reward_key)
            gamma_val = torch.full_like(reward_val, self.gamma)
            tensordict.set("gamma", gamma_val)
            return tensordict

        if tensordict.batch_dims == 0:
            raise RuntimeError(
                "CustomNStepReturn requires batched data (batch_dims > 0)"
            )

        if self._buffer is None:
            self._buffer = tensordict.clone()
        else:
            self._buffer = torch.cat([self._buffer, tensordict], dim=-1)

            if self._buffer.shape[-1] > self.n_steps:
                self._buffer = self._buffer[..., -self.n_steps :].clone()

        self.nb_transitions_per_envs += 1

        envs_with_enough_transitions = (
            self.nb_transitions_per_envs == self.n_steps or tensordict["next"]["done"]
        )

        if not envs_with_enough_transitions.any():
            return None

        nb_transitions = self.nb_transitions_per_envs[envs_with_enough_transitions]

        """
        current : buffer[-3]
        next : buffer[-1] but with reward being discounted sum using buffer[-3]["next"]["reward"]
                            and with done being buffer[-1]["next"]["done"]
        """
        current = self._buffer[envs_with_enough_transitions, -nb_transitions]

        prev_states = self._buffer[envs_with_enough_transitions, -nb_transitions:][
            "next"
        ]
        next_state = self._buffer[envs_with_enough_transitions, -1]["next"].clone()

        gammas_pow = self.gamma ** torch.arange(self.n_steps, device=device)

        discounted_rewards = prev_states["reward"] * gammas_pow

        sum_discounted_rewards = torch.sum(discounted_rewards, dim=1)

        next_state["reward"] = sum_discounted_rewards
        next_state["gamma"] = gammas_pow

        current["next"] = next_state

        return current
