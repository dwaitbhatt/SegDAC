import torch
from collections import deque
from segdac_dev.replay_buffers.transforms.transform import Transform
from segdac.data.mdp import MdpData


class NStepReturnTransform(Transform):
    def __init__(
        self,
        device: str,
        num_envs: int,
        n_step: int,
        gamma: float,
        compute_all_truncated_returns: bool,
    ):
        """
        Args:
            device: The device (e.g., 'cpu', 'cuda') for tensor computations.
            num_envs: The number of parallel environments.
            n_step: The number of steps (N) to look ahead for returns (N >= 1).
            gamma: The discount factor for future rewards.
            compute_all_truncated_returns: Controls N-step calculation at episode
            termination (`done=True`).
                - If `False`: Only the single N-step return starting
                from the oldest transition in the buffer window is calculated.
                Example (N=3, buffer holds transitions starting at s1, s2, s3
                when done occurs at s4): Only the transition for s1 (using
                rewards r2, r3, r4) is generated.
                - If `True`: Calculates all possible truncated returns ending at
                the terminal state (one for each starting transition in the
                buffer). Maximizes data usage but generates more transitions.
                Example (N=3, buffer holds transitions starting at s1, s2, s3
                when done occurs at s4): Transitions for s1 (using r2, r3, r4),
                s2 (using r3, r4), and s3 (using r4) are all generated.
        """
        super().__init__(device=device)
        if n_step < 1:
            raise ValueError("n_step must be >= 1")
        self.num_envs = num_envs
        self.n_step = n_step
        self.gamma = gamma
        self.buffers = [deque(maxlen=n_step) for _ in range(num_envs)]
        self.compute_all_truncated_returns = compute_all_truncated_returns

    def apply(self, mdp_data: MdpData) -> MdpData:
        complete_n_step_transitions = []

        for e in range(self.num_envs):
            buffer = self.buffers[e]
            new_env_data = mdp_data.data[e]
            new_env_next_data = mdp_data.next.data[e]
            if mdp_data.segmentation_data is None:
                new_env_seg_data = None
                new_env_next_seg_data = None
            else:
                image_id = new_env_data["image_ids"]
                next_image_id = new_env_next_data["image_ids"]
                new_env_seg_data = mdp_data.segmentation_data[
                    mdp_data.segmentation_data["image_ids"] == image_id
                ]
                new_env_next_seg_data = mdp_data.next.segmentation_data[
                    mdp_data.next.segmentation_data["image_ids"] == next_image_id
                ]
            new_env_mdp_data = MdpData(
                data=new_env_data.unsqueeze(1),
                segmentation_data=new_env_seg_data,
                next=MdpData(
                    data=new_env_next_data.unsqueeze(1),
                    segmentation_data=new_env_next_seg_data,
                ),
            )
            buffer.append(new_env_mdp_data)

            effective_nb_of_steps = len(buffer)
            is_env_done = new_env_mdp_data.next.data["done"].item()

            if effective_nb_of_steps == self.n_step or is_env_done:

                transition = self.compute_complete_n_step_transition(
                    buffer, is_env_done
                )
                complete_n_step_transitions.append(transition)

                if is_env_done and self.compute_all_truncated_returns:
                    nb_truncated_transitions_we_can_also_generate = max(
                        0, effective_nb_of_steps - 1
                    )
                    for _ in range(nb_truncated_transitions_we_can_also_generate):
                        buffer.popleft()
                        transition = self.compute_complete_n_step_transition(
                            buffer, is_env_done
                        )
                        complete_n_step_transitions.append(transition)

            if is_env_done:
                buffer.clear()

        if len(complete_n_step_transitions) == 0:
            return None

        return MdpData.cat(complete_n_step_transitions)

    def compute_complete_n_step_transition(
        self, buffer: deque[MdpData], is_env_done: bool
    ) -> MdpData:
        effective_nb_of_steps = len(buffer)
        state = buffer[0]
        rewards = torch.cat([s.next.data["reward"] for s in buffer])
        exponents = torch.arange(start=0, end=effective_nb_of_steps, device=self.device)
        bases = torch.full((effective_nb_of_steps,), self.gamma, device=self.device)
        gammas = torch.pow(bases, exponents)
        n_step_return = (gammas * rewards).sum().unsqueeze(0).unsqueeze(0)
        gamma_next = (
            torch.zeros((1, 1), device=self.device)
            if is_env_done
            else (gammas[-1] * self.gamma).unsqueeze(0).unsqueeze(0)
        )
        next_state = buffer[-1].next.clone()
        next_state.data["reward"] = n_step_return
        next_state.data["gamma"] = gamma_next
        return MdpData(
            data=state.data,
            segmentation_data=state.segmentation_data,
            next=next_state,
        )
