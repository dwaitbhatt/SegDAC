import torch
import numpy as np
from segdac.agents.agent import Agent
from segdac_dev.envs.env import TensorDictEnvWrapper
from segdac_dev.logging.loggers.logger import Logger
from omegaconf import DictConfig
from segdac_dev.metrics.metrics import Metrics
from segdac.data.mdp import MdpData
from segdac_dev.logging.rgb_pixels_writer import RgbPixelsWritter


class AgentTester:
    def __init__(self, logger: Logger, cfg: DictConfig):
        self.logger = logger
        self.agent_device = cfg['policy_device']
        self.nb_envs = cfg['evaluation']['env_config']['num_envs']
        reward_scaling_config = cfg.get("algo", {}).get(
            "reward_scaling", {"loc": 0.0, "scale": 1.0}
        )
        self.reward_scale = reward_scaling_config["scale"]
        self.reward_loc = reward_scaling_config["loc"]
        self.video_height = cfg["logging"]["video_height"]
        self.video_width = cfg["logging"]["video_width"]
        self.video_nb_rollouts_logged = 0
        self.video_nb_rollouts_to_log = cfg['logging']['num_rollouts']
        self.cfg = cfg

    @torch.no_grad()
    def test(
        self,
        agent: Agent,
        test_env: TensorDictEnvWrapper,
        num_rollouts: int,
        logging_prefix: str,
        seed_number: int,
        env_max_frames_per_traj: int,
        action_repeat: int
    ) -> dict:
        assert num_rollouts % self.nb_envs == 0, f"Expected num_rollouts % self.nb_envs == 0 but num_rollouts % self.nb_envs = {num_rollouts % self.nb_envs}"

        self.rollouts_video_writter = RgbPixelsWritter(
            cfg=self.cfg,
            num_envs=self.nb_envs,
            max_steps=self.video_nb_rollouts_to_log * env_max_frames_per_traj,
            height=self.video_height,
            width=self.video_width,
            video_tag=f"{logging_prefix}pixels_rollouts",
            max_steps_per_traj=None,
            grid=False,
        )
        self.video_nb_rollouts_logged = 0

        self.metrics = {
            "rollouts_metrics" : Metrics(),
            "current_rollout_metrics": [Metrics() for _ in range(self.nb_envs)],
        }

        for env_ids, step_number, step_mdp_data in test_env.rollouts(
            num_rollouts=num_rollouts,
            agent=agent,
            agent_device=self.agent_device
        ):
            self.accumulate_rollout_metrics(step_mdp_data, logging_prefix, action_repeat)

        metrics = self.log_metrics(seed_number)
        self.log_videos(seed_number)

        return metrics

    def accumulate_rollout_metrics(self, step_mdp_data: MdpData, logging_prefix: str, action_repeat: int) -> dict:
        for env_i in range(self.nb_envs):
            data = step_mdp_data.data[env_i]
            next_data = step_mdp_data.next.data[env_i]
            segmentation_data = None
            next_segmentation_data = None
            if step_mdp_data.segmentation_data is not None:
                image_id = data["image_ids"]
                segmentation_data = step_mdp_data.segmentation_data[
                    torch.isin(
                        step_mdp_data.segmentation_data["image_ids"], image_id
                    )
                ]
                next_image_id = next_data["image_ids"]
                next_segmentation_data = step_mdp_data.next.segmentation_data[
                    torch.isin(
                        step_mdp_data.next.segmentation_data["image_ids"],
                        next_image_id,
                    )
                ]
            env_step_mdp_data = MdpData(
                data=data,
                segmentation_data=segmentation_data,
                next=MdpData(
                    data=next_data, segmentation_data=next_segmentation_data
                ),
            ).to(device="cpu", non_blocking=True)

            current_rollout_metrics = self.metrics["current_rollout_metrics"][env_i]

            reward_unscaled = (
                env_step_mdp_data.next.data["reward"] - self.reward_loc
            ) / self.reward_scale

            current_rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}return",
                metric_value=reward_unscaled.item(),
                agg_fn=np.sum,
            )

            current_rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}return_std",
                metric_value=reward_unscaled.item(),
                agg_fn=np.std,
            )

            current_rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}reward_mean",
                metric_value=reward_unscaled.item() / action_repeat,
                agg_fn=np.mean,
            )

            current_rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}episode_length",
                metric_value=action_repeat,
                agg_fn=np.sum,
            )

            current_rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}action_mean",
                metric_value=env_step_mdp_data.data["action"].numpy(),
                agg_fn=np.mean,
            )

            current_rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}action_std",
                metric_value=env_step_mdp_data.data["action"].numpy(),
                agg_fn=np.std,
            )

            if env_step_mdp_data.next.data.get("info", {}).get("success", None) is not None:
                current_rollout_metrics.accumulate_metric(
                    metric_name=f"{logging_prefix}success_once_mean",
                    metric_value=env_step_mdp_data.next.data["info"]["success"].item(),
                    agg_fn=lambda x: np.any(x).astype(np.float32)
                )
                current_rollout_metrics.accumulate_metric(
                    metric_name=f"{logging_prefix}success_at_end_mean",
                    metric_value=env_step_mdp_data.next.data["info"]["success"].float().item(),
                    agg_fn=lambda x: x[-1]
                )

        pixels = step_mdp_data.data["pixels"]

        if self.video_nb_rollouts_logged < self.video_nb_rollouts_to_log:
            self.rollouts_video_writter.add_step_frame(
                pixels
            )

        if step_mdp_data.next.data["done"].any():
            self.video_nb_rollouts_logged += step_mdp_data.next.data["done"].sum().item()
            rollouts_metrics = self.metrics["rollouts_metrics"]
            for env_i in range(self.nb_envs):
                if step_mdp_data.next.data["done"][env_i].item():
                    current_rollout_metrics = self.metrics["current_rollout_metrics"][env_i]
                    aggregated_rollout_metrics = current_rollout_metrics.compute_aggregated_metrics()
                    for k, v in aggregated_rollout_metrics.items():
                        rollouts_metrics.accumulate_metric(
                            metric_name=k,
                            metric_value=v,
                            agg_fn=np.mean,
                        )

    def log_metrics(self, seed_number: int) -> dict:
        aggregated_test_metrics = self.metrics["rollouts_metrics"].compute_aggregated_metrics()
        self.logger.log_metrics(aggregated_test_metrics, step=seed_number)
        return aggregated_test_metrics

    def log_videos(self, seed_number: int):
        video_file = self.rollouts_video_writter.write_video_to_disk(
            env_step=seed_number
        )
        if video_file is not None:
            self.logger.log_video(str(video_file.resolve()), step=seed_number)
