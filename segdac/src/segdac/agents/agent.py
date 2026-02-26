import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tensordict import TensorDict
from segdac.action_scaling.env_action_scaler import TanhEnvActionScaler
from segdac.agents.action_sampling_strategy import ActionSamplingStrategy
from segdac.data.mdp import MdpData


class Agent(ABC, nn.Module):
    def __init__(
        self,
        env_action_scaler: TanhEnvActionScaler,
        action_sampling_strategy: ActionSamplingStrategy,
    ):
        super().__init__()
        self.env_action_scaler = env_action_scaler
        self.action_sampling_strategy = action_sampling_strategy

    def compile(self, compile_config: dict):
        if compile_config == {}:
            return
        self.action_sampling_strategy.compile(compile_config=compile_config)

    def enable_exploration(self):
        self.action_sampling_strategy.enable_exploration()

    def disable_exploration(self):
        self.action_sampling_strategy.disable_exploration()

    def enable_stochasticity(self):
        self.action_sampling_strategy.enable_stochasticity()

    def disable_stochasticity(self):
        self.action_sampling_strategy.disable_stochasticity()

    def step(self, frames: int = 1):
        self.action_sampling_strategy.step(frames=frames)

    @torch.no_grad()
    def forward(self, mdp_data: MdpData) -> MdpData:
        """
        Performs a forward pass through the agent's policy.
        Returns a MdpData that now has at least the key "action" in .data which represents the action to be performed in the environment.
        Some methods/algo might also return other keys that can then be saved into the replay buffer for instance.

        Args:
            mdp_data (MdpData)

        Returns:
            MdpData: Contains the predicted action.
        """
        inference_inputs_data = MdpData(
            data=mdp_data.data.squeeze(1),
            segmentation_data=mdp_data.segmentation_data,
        )

        predicted_data = self.predict_unscaled_action(inference_inputs_data)
        unscaled_action = predicted_data["unscaled_action"]

        scaled_action = self.env_action_scaler.scale(unscaled_action)
        mdp_data.data["action"] = scaled_action.unsqueeze(1)

        return mdp_data

    def predict_unscaled_action(self, mdp_data: MdpData) -> TensorDict:
        """
        Returns TensorDict with keys:
            unscaled_action
        """
        return self.action_sampling_strategy(mdp_data)

    @abstractmethod
    def update(
        self, train_mdp_data: MdpData, env_step: int, is_time_to_evaluate: bool
    ) -> TensorDict:
        """
        Update the agent networks' parameters using the provided training data.

        Args:
            train_mdp_data (MdpData): Training MDP data for the update step, it contains the .next data as well.
            env_step (int)
            is_time_to_evaluate (bool): Whether or not this env_step is also gonna generate evaluation data.
                                        This is useful so the agent can know if it has to compute train metrics or not that is then returned.

        Returns:
            TensorDict: logging information (e.g., loss values, etc.) or an empty TensorDict if is_time_to_evaluate is False.
        """
        pass

    def update_target_networks(self, env_step: int):
        pass
