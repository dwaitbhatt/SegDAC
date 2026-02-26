import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from segdac.agents.agent import Agent
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict import NonTensorData
from segdac_dev.evaluation.rl_evaluator import RlEvaluator
from segdac_dev.envs.env import TensorDictEnvWrapper
from segdac_dev.metrics.metrics import Metrics
from segdac_dev.replay_buffers.facade import ReplayBufferFacade
from segdac.data.mdp import MdpData


class RlOnlineTrainer:
    def __init__(
        self,
        agent_train: Agent,
        agent_env_collect: Agent,
        train_env: TensorDictEnvWrapper,
        evaluator: RlEvaluator,
        replay_buffer: ReplayBufferFacade,
        cfg: DictConfig,
        job_id: str,
    ):
        self.agent_train = agent_train
        self.agent_env_collect = agent_env_collect
        self.train_env = train_env
        self.evaluator = evaluator
        self.replay_buffer = replay_buffer
        self.train_data_collection_total_nb_frames = cfg["training"]["data_collector"][
            "total_frames"
        ]
        self.train_num_envs = cfg["training"]["env_config"]["num_envs"]
        self.train_batch_size = cfg["training"]["batch_size"]
        self.init_random_frames = cfg["training"]["data_collector"][
            "init_random_frames"
        ]
        self.policy_device = torch.device(cfg["policy_device"])
        self.num_updates_per_env_step = cfg["training"]["num_updates_per_env_step"]
        self.assets_folder_path = Path(cfg["final_job_data_dir"]) / Path(job_id) / Path(cfg["logging"]["assets_dir"])
        self.assets_folder_path.mkdir(parents=True, exist_ok=True)
        self.agent_weights_output_dir = (
            self.assets_folder_path
            / Path("weights/")
            / Path("rl_agent/")
        )
        self.agent_weights_output_dir.mkdir(parents=True, exist_ok=True)
        self.optimization_metric = cfg["logging"]["agent_optimization_metric"]["name"]
        self.maximize_optimization_metric = cfg["logging"]["agent_optimization_metric"][
            "maximize"
        ]
        self.best_optimization_metric_value = (
            float("-inf") if self.maximize_optimization_metric else float("inf")
        )
        self.best_agent_file_path = None
        self.final_agent_file_path = None
        self.task_id = cfg["env"]["id"]
        self.algo_name = cfg["algo"]["name"]
        self.seed = cfg["training"]["seed"]
        self.env_auto_resets = cfg["training"]["env_auto_resets"]
        self.action_repeat = cfg["env"]["action_repeat"]

    def train(self) -> tuple[Path, Path]:
        try:
            self.env_step = 0
            num_env_iters_to_achive_nb_frames = (
                self.compute_nb_env_iters_to_achieve_nb_frames(
                    total_nb_frames=self.train_data_collection_total_nb_frames,
                    num_envs=self.train_num_envs,
                )
            )

            train_env_mdp_data = self.train_env.reset()

            for i in tqdm(
                range(num_env_iters_to_achive_nb_frames), "Env Data Collection"
            ):
                is_before_first_env_step = i == 0

                if is_before_first_env_step:
                    eval_metrics = self.evaluator.evaluate(
                        env_step=self.env_step,
                        is_before_first_env_step=True,
                        train_metrics={},
                        grad_metrics={},
                    )

                train_env_mdp_data = self.do_one_collect_step(train_env_mdp_data, i)

                if not self.env_auto_resets:
                    if train_env_mdp_data.data["done"].any():
                        train_env_mdp_data = self.train_env.reset()

                if self.can_train():
                    train_metrics = Metrics()
                    last_grad_metrics = Metrics()
                    for _ in range(self.num_updates_per_env_step):
                        train_mdp_data = self.replay_buffer.sample()
                        train_mdp_data = train_mdp_data.to(self.policy_device)
                        train_update_metrics = self.agent_train.update(
                            train_mdp_data,
                            self.env_step,
                            self.evaluator.is_time_to_evaluate(self.env_step, is_before_first_env_step=False),
                        )
                        actor_grad_stats = train_update_metrics.get(
                            "actor_grad_stats", None
                        )
                        critic_grad_stats = train_update_metrics.get(
                            "critic_grad_stats", None
                        )
                        last_grad_metrics.data["actor_grad_stats"] = actor_grad_stats
                        last_grad_metrics.data["critic_grad_stats"] = critic_grad_stats
                        self.accumulate_train_metrics(train_metrics, train_update_metrics)

                    train_metrics = train_metrics.compute_aggregated_metrics()
                    grad_metrics = last_grad_metrics.data

                    self.agent_train.update_target_networks(env_step=self.env_step)
                else:
                    train_metrics = {}
                    grad_metrics = {}

                eval_metrics = self.evaluator.evaluate(
                    env_step=self.env_step,
                    is_before_first_env_step=False,
                    train_metrics=train_metrics,
                    grad_metrics=grad_metrics,
                )
                self.save_agent_best_checkpoint(eval_metrics=eval_metrics)
        finally:
            self.save_agent_final_checkpoint()

        return self.best_agent_file_path, self.final_agent_file_path

    def compute_nb_env_iters_to_achieve_nb_frames(
        self, total_nb_frames: int, num_envs: int
    ) -> int:
        per_iter_frames = num_envs * self.action_repeat
        assert total_nb_frames % per_iter_frames == 0, (
            f"total_frames must be divisible by num_envs*action_repeat. "
            f"Got {total_nb_frames} % {per_iter_frames} = {total_nb_frames % per_iter_frames}"
        )
        return total_nb_frames // per_iter_frames

    def do_one_collect_step(self, train_env_mdp_data: MdpData, i) -> MdpData:
        train_env_mdp_data_updated = self.train_env.consume_auto_reset_data(train_env_mdp_data)

        if self.env_step * self.action_repeat < self.init_random_frames:
            env_random_action = self.train_env.get_random_action()
            train_env_mdp_data_updated.data["action"] = env_random_action
            train_env_step_next_mdp_data = self.train_env.step(
                train_env_mdp_data_updated
            )
        else:
            policy_input = train_env_mdp_data_updated.to(self.policy_device)
            train_env_mdp_data_updated = self.agent_env_collect(policy_input)

            train_env_step_next_mdp_data = self.train_env.step(
                train_env_mdp_data_updated
            )

        train_env_mdp_data_updated.next = train_env_step_next_mdp_data

        self.evaluator.log_train_frames(
            train_env_mdp_data_updated, env_step=self.env_step
        )
        self.replay_buffer.extend(train_env_mdp_data_updated)

        self.env_step += self.train_num_envs

        if self.env_step * self.action_repeat > self.init_random_frames:
            self.agent_env_collect.step(frames=self.train_num_envs * self.action_repeat)

        return train_env_mdp_data_updated.step()

    def save_agent_final_checkpoint(self):
        checkpoint_type = "final"
        self.final_agent_file_path = self.agent_weights_output_dir / Path(
            f"{self.task_id.lower()}_{self.algo_name.lower()}_{checkpoint_type.lower()}_seed_{self.seed}_step_{self.env_step}.pt"
        )
        torch.save(
            self.agent_train.state_dict(),
            self.final_agent_file_path,
        )

    def save_agent_best_checkpoint(self, eval_metrics: dict):
        if eval_metrics == {}:
            return

        current_optimization_metric_value = eval_metrics[self.optimization_metric]

        if self.maximize_optimization_metric:
            is_better = (
                current_optimization_metric_value >= self.best_optimization_metric_value
            )
        else:
            is_better = (
                current_optimization_metric_value <= self.best_optimization_metric_value
            )

        if is_better:
            if self.best_agent_file_path is not None:
                self.best_agent_file_path.unlink(missing_ok=True)

            self.best_optimization_metric_value = current_optimization_metric_value
            checkpoint_type = "best"
            self.best_agent_file_path = self.agent_weights_output_dir / Path(
                f"{self.task_id.lower()}_{self.algo_name.lower()}_{checkpoint_type.lower()}_{self.optimization_metric}_seed_{self.seed}_step_{self.env_step}.pt"
            )
            torch.save(
                self.agent_train.state_dict(),
                self.best_agent_file_path,
            )

    def can_train(self) -> bool:
        return (
            len(self.replay_buffer) >= self.train_batch_size
            and self.env_step * self.action_repeat >= self.init_random_frames
        )

    def accumulate_train_metrics(
        self, output_train_metrics: Metrics, train_update_metrics: TensorDict
    ):
        for k, v in train_update_metrics.items():
            if isinstance(v, TensorDict):
                self.accumulate_train_metrics(output_train_metrics, v)
            elif isinstance(v, NonTensorData):
                pass
            else:
                if v.numel() > 1:
                    v = v.mean()
                output_train_metrics.accumulate_metric(
                    metric_name=k, metric_value=v.item(), agg_fn=np.mean
                )
