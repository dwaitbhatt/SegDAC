import torch
from segdac.agents.agent import Agent
from segdac.action_scaling.env_action_scaler import TanhEnvActionScaler
from segdac.agents.critic import Critic
from segdac.agents.action_sampling_strategy import ActionSamplingStrategy
from tensordict import TensorDict
from tensordict import merge_tensordicts
from segdac.data.mdp import MdpData


class ActorCriticAgent(Agent):
    def __init__(
        self,
        env_action_scaler: TanhEnvActionScaler,
        action_sampling_strategy: ActionSamplingStrategy,
        critic: Critic,
        critic_update_frequency: int,
        actor_update_frequency: int,
        target_networks_update_frequency: int,
    ):
        super().__init__(
            env_action_scaler=env_action_scaler,
            action_sampling_strategy=action_sampling_strategy,
        )
        self.critic = critic
        self.critic_update_frequency = critic_update_frequency
        self.actor_update_frequency = actor_update_frequency
        self.target_networks_update_frequency = target_networks_update_frequency

    @property
    def actor(self):
        return self.action_sampling_strategy.actor

    def compile(self, compile_config: dict):
        """
        Optional to call this, but can improve speed for some methods (torch.compile is used). 
        """
        self.action_sampling_strategy.compile(compile_config=compile_config)
        self.critic.compile(compile_config=compile_config)

    def update(
        self, train_mdp_data: MdpData, env_step: int, is_time_to_evaluate: bool
    ) -> TensorDict:
        critic_logs = self.update_critic(
            train_mdp_data=train_mdp_data,
            env_step=env_step,
            is_time_to_evaluate=is_time_to_evaluate,
        )

        actor_logs = self.update_actor(
            train_mdp_data=train_mdp_data,
            env_step=env_step,
            is_time_to_evaluate=is_time_to_evaluate,
        )

        output_logs = merge_tensordicts(critic_logs, actor_logs)

        # Fixes merge_tensordicts removing actor_grad_layers key
        if actor_logs.get("actor_grad_stats", None) is not None:
            output_logs["actor_grad_stats"]["actor_grad_layers"] = actor_logs[
                "actor_grad_stats"
            ]["actor_grad_layers"]
            output_logs["critic_grad_stats"]["critic_grad_layers"] = critic_logs[
                "critic_grad_stats"
            ]["critic_grad_layers"]

        return output_logs

    def update_critic(
        self, train_mdp_data: MdpData, env_step: int, is_time_to_evaluate: bool
    ) -> TensorDict:
        if env_step % self.critic_update_frequency != 0:
            return TensorDict(
                {},
                batch_size=torch.Size([]),
            )

        self.critic.train()
        self.actor.eval()
        critic_logs = self.critic.update(
            actor=self.actor,
            train_mdp_data=train_mdp_data,
            env_step=env_step,
            is_time_to_evaluate=is_time_to_evaluate,
        )

        return critic_logs

    def update_actor(
        self, train_mdp_data: MdpData, env_step: int, is_time_to_evaluate: bool
    ) -> TensorDict:
        if env_step % self.actor_update_frequency != 0:
            return TensorDict(
                {},
                batch_size=torch.Size([]),
            )

        self.critic.eval()
        self.actor.train()
        actor_logs = self.actor.update(
            train_mdp_data=train_mdp_data,
            critic=self.critic,
            env_step=env_step,
            is_time_to_evaluate=is_time_to_evaluate,
        )

        return actor_logs

    def update_target_networks(self, env_step: int):
        if env_step % self.target_networks_update_frequency != 0:
            return
        self.actor.update_target_networks(env_step=env_step)
        self.critic.update_target_networks(env_step=env_step)
