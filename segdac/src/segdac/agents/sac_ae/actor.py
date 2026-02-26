import torch.nn as nn
from segdac.agents.distribution_factory import DistributionFactory
from segdac.agents.sac.actor import SacActor
from segdac.data.mdp import MdpData
from tensordict import TensorDict


class SacAeActor(SacActor):
    def __init__(
        self,
        network: nn.Module,
        policy_optimizer,
        entropy_optimizer,
        device: str,
        action_dim: int,
        distribution_factory: DistributionFactory,
        initial_entropy: float,
        max_grad_norm: float,
        encoder: nn.Module,
        encoder_input_observation_key: str,
        encoder_output_observation_key: str,
    ):
        super().__init__(
            network=network,
            policy_optimizer=policy_optimizer,
            entropy_optimizer=entropy_optimizer,
            device=device,
            action_dim=action_dim,
            distribution_factory=distribution_factory,
            initial_entropy=initial_entropy,
            max_grad_norm=max_grad_norm,
        )
        self.encoder = encoder
        self.encoder_input_observation_key = encoder_input_observation_key
        self.encoder_output_observation_key = encoder_output_observation_key

    def update(self, train_mdp_data: MdpData, critic, env_step: int, is_time_to_evaluate: bool) -> TensorDict:
        self.encoder.eval()
        processed_train_mdp_data = self.preprocess_data(train_mdp_data)
        return super().update(processed_train_mdp_data, critic, env_step, is_time_to_evaluate)
    
    def preprocess_data(self, train_mdp_data: MdpData) -> MdpData:
        """
        SAC AE detaches the encoder output from the graph to avoid backpropagating through the encoder.
        See https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L342-L343
        """
        obs_latent = self.encoder(train_mdp_data)[self.encoder_output_observation_key]
        train_mdp_data.data[self.encoder_output_observation_key] = obs_latent.detach()
        return train_mdp_data