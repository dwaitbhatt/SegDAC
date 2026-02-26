from segdac.agents.actor_critic_agent import ActorCriticAgent
from segdac.action_scaling.env_action_scaler import IdentityEnvActionScaler
from segdac.agents.dqn.critic import DqnCritic
from segdac.agents.action_sampling_strategy import EpsilonGreedyActionSamplingStrategy
from segdac.agents.dqn.actor import DqnActor


class DqnAgent(ActorCriticAgent):
    def __init__(
        self,
        env_action_scaler: IdentityEnvActionScaler,
        device: str,
        seed: int,
        epsilon_end: float,
        epsilon_decay_frames: int,
        critic: DqnCritic,
        critic_update_frequency: int,
        target_networks_update_frequency: int,
        nb_actions: int,
    ):
        super().__init__(
            env_action_scaler=env_action_scaler,
            action_sampling_strategy=None,
            critic=critic,
            critic_update_frequency=critic_update_frequency,
            actor_update_frequency=1,
            target_networks_update_frequency=target_networks_update_frequency,
        )
        self.action_sampling_strategy = EpsilonGreedyActionSamplingStrategy(
            actor=DqnActor(
                critic=self.critic
            ),
            device=device,
            seed=seed,
            nb_actions=nb_actions,
            epsilon_end=epsilon_end,
            epsilon_decay_frames=epsilon_decay_frames
        )
