from segdac.agents.actor_critic_agent import ActorCriticAgent
from segdac.action_scaling.env_action_scaler import TanhEnvActionScaler
from segdac.agents.ddpg.critic import DdpgCritic
from segdac.agents.action_sampling_strategy import ActionSamplingStrategy


class DdpgAgent(ActorCriticAgent):
    def __init__(
        self,
        env_action_scaler: TanhEnvActionScaler,
        action_sampling_strategy: ActionSamplingStrategy,
        critic: DdpgCritic,
        critic_update_frequency: int,
        actor_update_frequency: int,
        target_networks_update_frequency: int,
    ):
        super().__init__(
            env_action_scaler=env_action_scaler,
            action_sampling_strategy=action_sampling_strategy,
            critic=critic,
            critic_update_frequency=critic_update_frequency,
            actor_update_frequency=actor_update_frequency,
            target_networks_update_frequency=target_networks_update_frequency,
        )
