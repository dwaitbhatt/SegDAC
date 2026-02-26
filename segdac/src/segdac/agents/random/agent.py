import torch
from segdac.agents.agent import Agent
from segdac.action_scaling.env_action_scaler import TanhEnvActionScaler
from segdac.agents.actor import Actor
from segdac.data.mdp import MdpData
from tensordict import TensorDict


from segdac.agents.action_sampling_strategy import ActionSamplingStrategy


class RandomActionSamplingStrategy(ActionSamplingStrategy):
    def __init__(self, action_dim: int, seed: int, device: str):
        super().__init__(actor=None)
        self.action_dim = action_dim
        self.device = device
        self.generator = torch.Generator(device).manual_seed(seed)

    @torch.no_grad()
    def forward(self, mdp_data: MdpData) -> TensorDict:
        batch_size = mdp_data.data.shape[0]
        high = 1
        low = -1

        output_data = {
            "unscaled_action": torch.rand((batch_size, self.action_dim), generator=self.generator, device=self.device)* (high - low) + low,
        }

        out = TensorDict(output_data, batch_size=torch.Size([batch_size]))

        return out

class RandomAgent(Agent):
    def __init__(
        self,
        env_action_scaler: TanhEnvActionScaler,
        action_dim: int,
        seed: int,
        device: str
    ):
        super().__init__(
            env_action_scaler=env_action_scaler,
            action_sampling_strategy=RandomActionSamplingStrategy(
                action_dim=action_dim,
                seed=seed,
                device=device
            )
        )
        self.action_dim = action_dim
        self.seed = seed
        self.device = device
        self.counter = 1
    
    def update(
        self, train_mdp_data: MdpData, env_step: int, is_time_to_evaluate: bool
    ) -> TensorDict:
        return TensorDict(
            {},
            batch_size=torch.Size([]),
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.action_sampling_strategy =RandomActionSamplingStrategy(
                action_dim=self.action_dim,
                seed=self.seed+self.counter,
                device=self.device
            )
        self.counter += 1
