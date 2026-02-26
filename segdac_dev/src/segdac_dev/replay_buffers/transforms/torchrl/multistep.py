import torch
from typing import Optional, Tuple

from tensordict import TensorDictBase

from tensordict.utils import _unravel_key_to_tuple
from torchrl.envs.transforms.transforms import Transform


class CustomNStepReturn(Transform):
    """
    Custom N-Step return transform for Replay Buffers, applied during extend.

    Handles parallel environments and partial resets correctly when data is added
    step-by-step (shape [B] or [B, 1]).

    Calculates the N-step transition (reward, next_state, done, effective_gamma)
    for the state added N-1 steps ago and returns it for storage.

    Attributes:
        n_steps (int): The number of steps N.
        gamma (float): The base discount factor.
        reward_key_in (Tuple[str, ...]): Nested key for the input 1-step reward.
        done_key_in (Tuple[str, ...]): Nested key for the input 1-step done flag.
        out_reward_key (Tuple[str, ...]): Nested key where the N-step reward is stored.
        out_done_key (Tuple[str, ...]): Nested key where the N-step done flag is stored.
        out_gamma_key (Tuple[str, ...]): Nested key where the effective gamma^k is stored.

    Input to _inv_call:
        tensordict (TensorDictBase): Contains data for a single time step across
            parallel environments. Expected shape [B] or [B, 1]. Contains keys like
            "action", "observation", and "next" which holds the results of the
            environment step (reward, done, next_observation).

    Output from _inv_call:
        TensorDictBase | None: A TensorDict of shape [B, 1] containing the
            completed N-step transition (original s_t, a_t, calculated N-step
            reward, N-step next state, N-step done, effective gamma^k) ready
            to be stored, or None if not enough steps have been buffered yet.
    """

    # Default keys used by the calculation (relative to the input TD to extend)
    _reward_key: Tuple[str, ...] = ("next", "reward")
    _done_key: Tuple[str, ...] = ("next", "done")
    # Default output keys (relative to the TD being returned for storage)
    _out_reward_key: Tuple[str, ...] = (
        "next",
        "reward",
    )  # Overwrites original reward in 'next'
    _out_done_key: Tuple[str, ...] = (
        "next",
        "done",
    )  # Overwrites original done in 'next'
    _out_gamma_key: Tuple[str, ...] = (
        "gamma",
    )  # Adds effective gamma to the *root* level

    def __init__(
        self,
        n_steps: int,
        gamma: float,
        *,
        reward_key=None,
        done_key=None,
        out_reward_key=None,
        out_done_key=None,
        out_gamma_key=None,
    ):
        super().__init__()
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer.")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be between 0.0 and 1.0.")

        self.n_steps = n_steps
        self.gamma = gamma
        self._buffer: Optional[TensorDictBase] = None

        self.reward_key_in = _unravel_key_to_tuple(
            reward_key if reward_key is not None else self._reward_key
        )
        self.done_key_in = _unravel_key_to_tuple(
            done_key if done_key is not None else self._done_key
        )
        self.out_reward_key = _unravel_key_to_tuple(
            out_reward_key if out_reward_key is not None else self._out_reward_key
        )
        self.out_done_key = _unravel_key_to_tuple(
            out_done_key if out_done_key is not None else self._out_done_key
        )
        self.out_gamma_key = _unravel_key_to_tuple(
            out_gamma_key if out_gamma_key is not None else self._out_gamma_key
        )

    @torch.no_grad()
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase | None:
        """
        Processes a single step TD, updates buffer, calculates N-step if possible.
        Called during replay_buffer.extend().
        """
        if self.n_steps == 1:
            try:
                reward_val = tensordict.get(self.reward_key_in)
                gamma_val = torch.full_like(reward_val, self.gamma)
            except KeyError:
                gamma_val = torch.full(
                    (*tensordict.batch_size, 1), self.gamma, device=tensordict.device
                )  # Assume float32?

            tensordict.set(self.out_gamma_key, gamma_val)
            tensordict.set(self.out_reward_key, tensordict.get(self.reward_key_in))
            tensordict.set(self.out_done_key, tensordict.get(self.done_key_in))
            return tensordict  # Return the original TD with gamma added

        if tensordict.batch_dims == 0:
            raise RuntimeError(
                "CustomNStepReturn requires batched data (batch_dims > 0)"
            )
        if tensordict.batch_dims > 1:
            if tensordict.shape[-1] == 1:
                td_t = tensordict.squeeze(-1)  # Work with shape [B]
            else:
                if tensordict.ndim == len(tensordict.batch_size):
                    td_t = tensordict
                else:
                    raise ValueError(
                        f"Expected input shape [B] or [B, 1], got {tensordict.shape}"
                    )
        else:
            td_t = tensordict

        td_t = td_t.unsqueeze(-1)

        if self._buffer is None:
            self._buffer = td_t.clone()  # Start buffer
        else:
            if self._buffer.device != td_t.device:
                self._buffer = self._buffer.to(td_t.device)
            self._buffer = torch.cat([self._buffer, td_t], dim=-1)

            if self._buffer.shape[-1] > self.n_steps:
                self._buffer = self._buffer[..., -self.n_steps :].clone()

        if self._buffer.shape[-1] < self.n_steps:
            return (
                None  # Not enough steps buffered yet to calculate for the oldest step
            )

        # Buffer now has shape [B, N] where N = self.n_steps
        buffer = self._buffer

        try:
            # Get rewards and dones for the N steps in the buffer
            # These are the 1-step rewards/dones from R_{t+1} to R_{t+N}
            # relative to the state s_t at buffer index 0.
            rewards = buffer.get(self.reward_key_in)  # Shape [B, N] or [B, N, 1]
            dones = buffer.get(self.done_key_in)  # Shape [B, N] or [B, N, 1]
        except KeyError as e:
            raise KeyError(
                f"Missing required key for N-step calculation: {e}. "
                f"Buffer keys: {buffer.keys(True, True)}"
            ) from e

        rewards_squeezed = (
            rewards.squeeze(-1)
            if rewards.shape[-1] == 1 and rewards.ndim > 1
            else rewards
        )
        dones_squeezed = (
            dones.squeeze(-1) if dones.shape[-1] == 1 and dones.ndim > 1 else dones
        )

        if (
            rewards_squeezed.shape != dones_squeezed.shape
            or rewards_squeezed.ndim != 2
            or rewards_squeezed.shape[-1] != self.n_steps
        ):
            raise ValueError(
                f"Rewards shape {rewards.shape} / {rewards_squeezed.shape} and/or "
                f"Dones shape {dones.shape} / {dones_squeezed.shape} are incompatible "
                f"or time dim != N={self.n_steps}"
            )

        B, N = rewards_squeezed.shape
        device = rewards_squeezed.device

        # Precompute powers of gamma: [gamma^0, gamma^1, ..., gamma^(N-1)]
        gammas_pow = self.gamma ** torch.arange(N, device=device)  # Shape [N]

        is_done = dones_squeezed.bool()  # Shape [B, N]

        # Mask to identify steps before the first 'done' occurs in the N-step window
        cumulative_dones = torch.cumsum(is_done, dim=-1)
        not_done_yet_mask = (
            cumulative_dones == 0
        )  # True for steps before *and including* the first done

        # Calculate the actual number of steps 'k' taken before termination (or N)
        # Summing the mask gives the count of steps where cumulative_dones was 0
        actual_n_steps = (
            torch.sum(not_done_yet_mask, dim=-1) + 1
        )  # Shape [B], values from 1 to N
        actual_n_steps = torch.clamp_max(
            actual_n_steps, N
        )  # Ensure it doesn't exceed N

        # Create a mask for summing rewards: [B, N]
        # True for steps k < actual_n_steps (i.e., steps 0 to k-1)
        valid_step_mask = torch.arange(N, device=device).unsqueeze(
            0
        ) < actual_n_steps.unsqueeze(-1)

        # Calculate N-step reward G_t^(k) = sum_{i=0}^{k-1} gamma^i * R_{t+i+1}
        n_step_rewards = torch.sum(
            rewards_squeezed * gammas_pow * valid_step_mask, dim=-1
        )  # Shape [B]

        # Calculate effective discount gamma^k
        effective_gammas = self.gamma**actual_n_steps  # Shape [B]
        # If terminated early (k < N), the bootstrap term is zeroed out by gamma^k=0
        terminated_early_mask = actual_n_steps < N  # Shape [B]
        effective_gammas[terminated_early_mask] = 0.0

        # Index (relative to buffer start) of the N-step next state s_{t+k}
        # Indices are 0 to N-1
        next_state_rel_idx = actual_n_steps - 1  # Shape [B]

        # --- Construct the Output TensorDict ---
        # This output corresponds to the transition starting at buffer index 0 (t_start)

        # Get the initial state, action etc. from buffer index 0
        batch_idx = torch.arange(B, device=device)
        output_td = buffer[
            batch_idx, 0
        ].clone()  # Shape [B], contains s_t, a_t, original next_t

        # Get the N-step 'next' state information by gathering from the buffer
        next_td_in_buffer = buffer.get("next")  # Shape [B, N]
        # Select the TensorDict at the computed relative index for each batch element
        next_state_final_td = next_td_in_buffer[
            batch_idx, next_state_rel_idx
        ].clone()  # Shape [B]

        # Update the 'next' field of the output TD with the N-step next state info
        output_td["next"] = (
            next_state_final_td  # Contains s_{t+k}, original r_{t+k}, d_{t+k}, etc.
        )

        # Overwrite the reward in 'next' with the calculated N-step reward
        # Ensure reward has a trailing dimension [B, 1] like typical buffer storage
        output_td.set(self.out_reward_key, n_step_rewards.unsqueeze(-1))

        # Overwrite the done flag in 'next' with the N-step done flag (True if terminated early)
        # Ensure done has a trailing dimension [B, 1]
        output_td.set(self.out_done_key, terminated_early_mask.unsqueeze(-1))

        # Add the effective gamma (gamma^k) to the root level of the output TD
        # Ensure gamma has a trailing dimension [B, 1]
        output_td.set(self.out_gamma_key, effective_gammas.unsqueeze(-1))

        # Return the single completed N-step transition, adding back the time dimension
        return output_td.unsqueeze(-1)  # Shape [B, 1]
