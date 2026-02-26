import torch
import numpy as np
import hydra
import torchvision.transforms.v2 as v2
from segdac_dev.logging.loggers.logger import Logger
from segdac.agents.agent import Agent
from segdac_dev.metrics.metrics import Metrics
from omegaconf import DictConfig
from segdac_dev.envs.env import TensorDictEnvWrapper
from segdac.data.mdp import MdpData
from segdac_dev.logging.rgb_pixels_writer import RgbPixelsWritter
from segdac.networks.image_segmentation_models.grounded_efficientvit_sam import (
    get_image_covered_by_predicted_masks,
)
from torchvision.utils import draw_bounding_boxes
from segdac_dev.visualization.segments import draw_rgb_segments_grid
from segdac_dev.visualization.segments import (
    draw_q_value_segments_attn_heatmap,
)
from typing import Optional
from segdac_dev.visualization.grad import generate_grad_flow_interactive_html
from pathlib import Path


class RlEvaluator:
    def __init__(
        self,
        logger: Logger,
        eval_env: TensorDictEnvWrapper,
        cfg: DictConfig,
        agent: Agent,
        job_id: str
    ):
        self.logger = logger
        self.agent = agent
        self.eval_env = eval_env
        self.eval_num_envs = cfg["evaluation"]["env_config"]["num_envs"]
        self.env_max_frames_per_traj = cfg["env"]["max_frames_per_traj"]
        self.eval_num_rollouts = cfg["evaluation"]["num_rollouts"]
        self.video_max_steps_per_traj = cfg["logging"]["video_max_steps_per_traj"]
        self.video_max_steps = cfg["logging"]["video_max_steps"]
        self.video_height = cfg["logging"]["video_height"]
        self.video_width = cfg["logging"]["video_width"]
        self.eval_pixels_video_writter = RgbPixelsWritter(
            cfg=cfg,
            num_envs=self.eval_num_envs,
            max_steps=self.video_max_steps,
            height=self.video_height,
            width=self.video_width,
            video_tag="eval_pixels_step",
            grid=False,
            max_steps_per_traj=self.video_max_steps_per_traj
        )
        self.eval_masked_pixels_video_writter = RgbPixelsWritter(
            cfg=cfg,
            num_envs=self.eval_num_envs,
            max_steps=self.video_max_steps,
            height=self.video_height,
            width=self.video_width,
            video_tag="eval_masked_pixels_step",
            grid=False,
            max_steps_per_traj=self.video_max_steps_per_traj
        )
        if "segdac" in cfg["algo"]["name"]:
            self.eval_q_value_attn_video_writter = RgbPixelsWritter(
                cfg=cfg,
                num_envs=self.eval_num_envs,
                max_steps=self.video_max_steps,
                height=600,
                width=800,
                video_tag="eval_q_value_attn_step",
                grid=False,
                max_steps_per_traj=self.video_max_steps_per_traj,
                interpolation=v2.InterpolationMode.NEAREST_EXACT
            )
        else:
            self.eval_q_value_attn_video_writter = None
        self.train_num_envs = cfg["training"]["env_config"]["num_envs"]
        self.train_pixels_video_writter = RgbPixelsWritter(
            cfg=cfg,
            num_envs=self.train_num_envs,
            max_steps=self.video_max_steps,
            height=self.video_height,
            width=self.video_width,
            video_tag="train_pixels_step",
            grid=True,
            max_steps_per_traj=None
        )
        self.logging_frequency = cfg["logging"]["frequency"]
        reward_scaling_config = cfg.get("algo", {}).get(
            "reward_scaling", {"loc": 0.0, "scale": 1.0}
        )
        self.reward_scale = reward_scaling_config["scale"]
        self.reward_loc = reward_scaling_config["loc"]
        self.log_success = cfg["env"].get("log_success", False)
        self.action_repeat = cfg["env"]["action_repeat"]
        self.agent_device = cfg["policy_device"]
        self.generator = torch.Generator().manual_seed(cfg["training"]["seed"])
        self.nb_segments_image_to_log = cfg["evaluation"]["nb_segments_image_to_log"]
        self.eval_segments_bboxes = []
        self.eval_segments_grid = []
        if cfg["algo"].get("grounding_text_tags", None) is not None:
            self.class_index_to_name = {
                c: n for c, n in enumerate(list(cfg["algo"]["grounding_text_tags"]))
            }
        self.cfg = cfg
        self.assets_folder_path = Path(cfg["final_job_data_dir"]) / Path(job_id) / Path(cfg["logging"]["assets_dir"])
        self.assets_folder_path.mkdir(parents=True, exist_ok=True)
        self.html_folder_path = self.assets_folder_path / Path("html")
        self.html_folder_path.mkdir(parents=True, exist_ok=True)
        self.heavy_logging_frequency = cfg["logging"]["heavy_logging_frequency"]
        self.ignore_terminations = cfg['evaluation']['env_config']['ignore_terminations']

    def log_train_frames(self, mdp_data: MdpData, env_step: int):
        env_step_video_logging_start_inclusive = (
            env_step // self.logging_frequency + 1
        ) * self.logging_frequency - self.video_max_steps * self.train_num_envs
        env_step_video_logging_end_exclusive = (
            env_step_video_logging_start_inclusive
            + self.video_max_steps * self.train_num_envs
        )
        if (
            env_step >= env_step_video_logging_start_inclusive
            and env_step < env_step_video_logging_end_exclusive
        ):
            self.train_pixels_video_writter.add_step_frame(mdp_data.data["pixels"].to(
                device="cpu", non_blocking=True
            ))

    @torch.no_grad()
    def evaluate(self, env_step: int, is_before_first_env_step: bool, train_metrics: dict, grad_metrics: dict) -> dict:
        if not self.is_time_to_evaluate(env_step, is_before_first_env_step):
            return {}

        self.logger.log_metric("env_step", env_step, env_step)

        eval_metrics = self.log_eval_metrics(env_step)
        self.log_train_metrics(train_metrics, grad_metrics, env_step)

        return eval_metrics

    def is_time_to_evaluate(self, env_step: int, is_before_first_env_step: bool) -> bool:
        return env_step % self.logging_frequency == 0 or is_before_first_env_step

    def log_eval_metrics(self, env_step: int) -> dict:
        eval_metrics = self.compute_eval_rollouts_metrics(env_step)
        self.logger.log_metrics(metrics=eval_metrics, step=env_step)
        self.log_eval_videos(env_step)
        self.log_eval_images(env_step)
        return eval_metrics

    def compute_eval_rollouts_metrics(self, env_step: int) -> dict:
        self.metrics = {
            "rollouts_metrics": Metrics(),
            "current_rollout_metrics": [Metrics() for _ in range(self.eval_num_envs)],
        }

        self.eval_segments_bboxes = []
        self.eval_segments_grid = []

        self.random_step_indexes_to_log = torch.multinomial(
            torch.full((self.env_max_frames_per_traj,), fill_value=1 / self.env_max_frames_per_traj),
            num_samples=self.nb_segments_image_to_log,
            replacement=False,
            generator=self.generator,
        )
        self.random_step_indexes_to_log = torch.sort(self.random_step_indexes_to_log)[0]

        for env_ids, step_number, step_mdp_data in self.eval_env.rollouts(
            num_rollouts=self.eval_num_rollouts,
            agent=self.agent,
            agent_device=self.agent_device,
        ):
            self.accumulate_eval_rollout_metrics(step_mdp_data, step_number, env_step, env_ids)

        aggregated_eval_metrics = self.metrics["rollouts_metrics"].compute_aggregated_metrics()

        return aggregated_eval_metrics

    def _accum_eval_rollout_metrics_ignore_terminations(self, step_mdp_data: MdpData, step_number: int, env_step: int, logging_prefix: str, step_mdp_data_cpu: MdpData):
        for env_i in range(self.eval_num_envs):
            data = step_mdp_data_cpu.data[env_i]
            next_data = step_mdp_data_cpu.next.data[env_i]
            segmentation_data = None
            next_segmentation_data = None
            if step_mdp_data_cpu.segmentation_data is not None:
                image_id = data["image_ids"]
                segmentation_data = step_mdp_data_cpu.segmentation_data[
                    torch.isin(
                        step_mdp_data_cpu.segmentation_data["image_ids"], image_id
                    )
                ]
                next_image_id = next_data["image_ids"]
                next_segmentation_data = step_mdp_data_cpu.next.segmentation_data[
                    torch.isin(
                        step_mdp_data_cpu.next.segmentation_data["image_ids"],
                        next_image_id,
                    )
                ]
            env_step_mdp_data = MdpData(
                data=data,
                segmentation_data=segmentation_data,
                next=MdpData(
                    data=next_data, segmentation_data=next_segmentation_data
                ),
            )
            env_step_mdp_data_cpu = env_step_mdp_data

            current_rollout_metrics = self.metrics["current_rollout_metrics"][env_i]

            reward_unscaled = (
                env_step_mdp_data_cpu.next.data["reward"] - self.reward_loc
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
                metric_value=reward_unscaled.item() / self.action_repeat,
                agg_fn=np.mean,
            )

            current_rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}episode_length",
                metric_value=self.action_repeat,
                agg_fn=np.sum,
            )

            current_rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}action_mean",
                metric_value=env_step_mdp_data_cpu.data["action"].numpy(),
                agg_fn=np.mean,
            )

            current_rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}action_std",
                metric_value=env_step_mdp_data_cpu.data["action"].numpy(),
                agg_fn=np.std,
            )

            if env_step_mdp_data_cpu.next.data.get("info", {}).get("success", None) is not None:
                current_rollout_metrics.accumulate_metric(
                    metric_name=f"{logging_prefix}success_once_mean",
                    metric_value=env_step_mdp_data_cpu.next.data["info"]["success"].item(),
                    agg_fn=lambda x: np.any(x).astype(np.float32)
                )
                current_rollout_metrics.accumulate_metric(
                    metric_name=f"{logging_prefix}success_at_end_mean",
                    metric_value=env_step_mdp_data_cpu.next.data["info"]["success"].float().item(),
                    agg_fn=lambda x: x[-1]
                )

        if step_mdp_data.segmentation_data is not None:
            segmentation_data_cpu = step_mdp_data_cpu.segmentation_data

        if self.eval_q_value_attn_video_writter is not None and (
            len(self.eval_q_value_attn_video_writter) < self.video_max_steps
        ) and env_step % self.heavy_logging_frequency == 0:
            agent_input = MdpData(
                data=step_mdp_data.data.squeeze(1),
                segmentation_data=step_mdp_data.segmentation_data,
                next=MdpData(
                    data=step_mdp_data.next.data.squeeze(1),
                    segmentation_data=step_mdp_data.next.segmentation_data,
                )
            ).to(device=self.agent_device)
            critic_outputs = self.agent.critic(
                agent_input
            ).to(device="cpu", non_blocking=True)
            step_q_value = critic_outputs["q_value"]
            if step_q_value.shape[-1] > 1 and step_q_value.ndim == 2:
                step_q_value = step_q_value.max(dim=-1, keepdims=True)[0]
            if critic_outputs["q_value_segments_attn"].ndim == 2:
                step_q_value_segments_attn = critic_outputs["q_value_segments_attn"][0]  # Take predictions from first critic network
                if critic_outputs.get("q_value_proprio_attn", None) is not None:
                    step_q_value_proprio_attn = critic_outputs["q_value_proprio_attn"][0]  # Take predictions from first critic network
                else:
                    step_q_value_proprio_attn = None
            else:
                step_q_value_segments_attn = critic_outputs["q_value_segments_attn"]
                if critic_outputs.get("q_value_proprio_attn", None) is not None:
                    step_q_value_proprio_attn = critic_outputs["q_value_proprio_attn"]
                else:
                    step_q_value_proprio_attn = None

            segments_binary_masks = segmentation_data_cpu['segments_data']['binary_masks'].squeeze(1)

            q_value_attn_heatmap_imgs = []
            img_ids_cpu = step_mdp_data_cpu.data["image_ids"]
            seg_img_ids_cpu = segmentation_data_cpu["segments_data"]["image_ids"]
            for i in range(self.eval_num_envs):
                env_image_id = img_ids_cpu[i]
                seg_img_id = seg_img_ids_cpu == env_image_id
                q_value = step_q_value[i].unsqueeze(0)
                q_value_segments_attn = step_q_value_segments_attn[seg_img_id]
                if step_q_value_proprio_attn is not None:
                    q_value_proprio_attn = step_q_value_proprio_attn[i].unsqueeze(0)
                else:
                    q_value_proprio_attn = None
                env_segments_binary_masks = segments_binary_masks[seg_img_id]
                q_value_attn_heatmap_img = (
                    draw_q_value_segments_attn_heatmap(
                        q_value=q_value,
                        q_value_segments_attn=q_value_segments_attn,
                        q_value_proprio_attn=q_value_proprio_attn,
                        segments_binary_masks=env_segments_binary_masks,
                        figsize=(8, 6)
                    )
                )
                q_value_attn_heatmap_imgs.append(q_value_attn_heatmap_img)

            q_value_attn_heatmap_imgs = np.stack(q_value_attn_heatmap_imgs)

            self.eval_q_value_attn_video_writter.add_step_frame(
                torch.as_tensor(q_value_attn_heatmap_imgs)
                .permute(0, 3, 1, 2)  # (num_envs, 3, H, W)
                .unsqueeze(1)
            )

        pixels = step_mdp_data_cpu.data["pixels"]

        if torch.isin(torch.tensor([step_number-1]), self.random_step_indexes_to_log) and step_mdp_data.segmentation_data is not None and len(self.eval_segments_grid) < self.nb_segments_image_to_log:
            image_id = step_mdp_data_cpu.data["image_ids"][0]
            segments_data = segmentation_data_cpu["segments_data"]
            image_segments_data = segments_data[
                segments_data["image_ids"] == image_id
            ]
            bboxes = image_segments_data["coords"]["masks_absolute_bboxes"]
            labels = [
                self.class_index_to_name[i]
                for i in image_segments_data["classes"].tolist()
            ]
            image_with_bboxes = draw_bounding_boxes(
                pixels[0].squeeze(0),
                bboxes,
                labels,
                width=1,
            )
            self.eval_segments_bboxes.append(image_with_bboxes)

            segments_grid = torch.as_tensor(
                draw_rgb_segments_grid(
                    segments_data, image_id, show=False, unscale=True
                )
            ).permute(2, 0, 1)
            self.eval_segments_grid.append(segments_grid)

        if len(self.eval_pixels_video_writter) < self.video_max_steps:
            self.eval_pixels_video_writter.add_step_frame(pixels)

        if len(self.eval_masked_pixels_video_writter) < self.video_max_steps and step_mdp_data.segmentation_data is not None:
            segments_data = segmentation_data_cpu["segments_data"]
            masked_pixels_images = get_image_covered_by_predicted_masks(
                original_images=pixels.squeeze(1), segments_data=segments_data
            )
            self.eval_masked_pixels_video_writter.add_step_frame(
                masked_pixels_images.unsqueeze(1)
            )

        if step_mdp_data.next.data["done"].any():
            rollouts_metrics = self.metrics["rollouts_metrics"]
            for env_i in range(self.eval_num_envs):
                if step_mdp_data.next.data["done"][env_i].item():
                    current_rollout_metrics = self.metrics["current_rollout_metrics"][env_i]
                    aggregated_rollout_metrics = current_rollout_metrics.compute_aggregated_metrics()
                    for k, v in aggregated_rollout_metrics.items():
                        rollouts_metrics.accumulate_metric(
                            metric_name=k,
                            metric_value=v,
                            agg_fn=np.mean,
                        )

    def _accum_eval_rollout_metrics_respect_terminations(self, step_mdp_data: MdpData, step_number: int, env_step: int, logging_prefix: str, step_mdp_data_cpu: MdpData, env_id: int):

        current_rollout_metrics = self.metrics["current_rollout_metrics"][env_id]

        reward_unscaled = (
            step_mdp_data_cpu.next.data["reward"] - self.reward_loc
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
            metric_value=reward_unscaled.item() / self.action_repeat,
            agg_fn=np.mean,
        )

        current_rollout_metrics.accumulate_metric(
            metric_name=f"{logging_prefix}episode_length",
            metric_value=self.action_repeat,
            agg_fn=np.sum,
        )

        current_rollout_metrics.accumulate_metric(
            metric_name=f"{logging_prefix}action_mean",
            metric_value=step_mdp_data_cpu.data["action"].numpy(),
            agg_fn=np.mean,
        )

        current_rollout_metrics.accumulate_metric(
            metric_name=f"{logging_prefix}action_std",
            metric_value=step_mdp_data_cpu.data["action"].numpy(),
            agg_fn=np.std,
        )

        if step_mdp_data_cpu.next.data.get("info", {}).get("success", None) is not None:
            current_rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}success_once_mean",
                metric_value=step_mdp_data_cpu.next.data["info"]["success"].item(),
                agg_fn=lambda x: np.any(x).astype(np.float32)
            )
            current_rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}success_at_end_mean",
                metric_value=step_mdp_data_cpu.next.data["info"]["success"].float().item(),
                agg_fn=lambda x: x[-1]
            )

        if step_mdp_data.segmentation_data is not None:
            segmentation_data_cpu = step_mdp_data_cpu.segmentation_data

        if self.eval_q_value_attn_video_writter is not None and (
            len(self.eval_q_value_attn_video_writter) < self.video_max_steps
        ) and env_step % self.heavy_logging_frequency == 0:
            if step_mdp_data.data.ndim == 1:
                agent_input = MdpData(
                    data=step_mdp_data.data,
                    segmentation_data=step_mdp_data.segmentation_data,
                    next=MdpData(
                        data=step_mdp_data.next.data,
                        segmentation_data=step_mdp_data.next.segmentation_data,
                    )
                ).to(device=self.agent_device)
            else:
                agent_input = MdpData(
                    data=step_mdp_data.data.squeeze(1),
                    segmentation_data=step_mdp_data.segmentation_data,
                    next=MdpData(
                        data=step_mdp_data.next.data.squeeze(1),
                        segmentation_data=step_mdp_data.next.segmentation_data,
                    )
                ).to(device=self.agent_device)
            critic_outputs = self.agent.critic(
                agent_input
            ).to(device="cpu", non_blocking=True)
            step_q_value = critic_outputs["q_value"]
            if step_q_value.shape[-1] > 1 and step_q_value.ndim == 2:
                step_q_value = step_q_value.max(dim=-1, keepdims=True)[0]
            if critic_outputs["q_value_segments_attn"].ndim == 2:
                step_q_value_segments_attn = critic_outputs["q_value_segments_attn"][0]  # Take predictions from first critic network
                if critic_outputs.get("q_value_proprio_attn", None) is not None:
                    step_q_value_proprio_attn = critic_outputs["q_value_proprio_attn"][0]  # Take predictions from first critic network
                else:
                    step_q_value_proprio_attn = None
            else:
                step_q_value_segments_attn = critic_outputs["q_value_segments_attn"]
                if critic_outputs.get("q_value_proprio_attn", None) is not None:
                    step_q_value_proprio_attn = critic_outputs["q_value_proprio_attn"]
                else:
                    step_q_value_proprio_attn = None

            segments_binary_masks = segmentation_data_cpu['segments_data']['binary_masks'].squeeze(1)

            q_value_attn_heatmap_imgs = []
            img_ids_cpu = step_mdp_data_cpu.data["image_ids"]
            seg_img_ids_cpu = segmentation_data_cpu["segments_data"]["image_ids"]
 
            env_image_id = img_ids_cpu[0]
            seg_img_id = seg_img_ids_cpu == env_image_id
            q_value = step_q_value[0].unsqueeze(0)
            q_value_segments_attn = step_q_value_segments_attn[seg_img_id]
            if step_q_value_proprio_attn is not None:
                q_value_proprio_attn = step_q_value_proprio_attn[0].unsqueeze(0)
            else:
                q_value_proprio_attn = None
            env_segments_binary_masks = segments_binary_masks[seg_img_id]
            q_value_attn_heatmap_img = (
                draw_q_value_segments_attn_heatmap(
                    q_value=q_value,
                    q_value_segments_attn=q_value_segments_attn,
                    q_value_proprio_attn=q_value_proprio_attn,
                    segments_binary_masks=env_segments_binary_masks,
                    figsize=(8, 6)
                )
            )
            q_value_attn_heatmap_imgs.append(q_value_attn_heatmap_img)

            q_value_attn_heatmap_imgs = np.stack(q_value_attn_heatmap_imgs)

            self.eval_q_value_attn_video_writter.add_step_frame(
                torch.as_tensor(q_value_attn_heatmap_imgs)
                .permute(0, 3, 1, 2)  # (1, 3, H, W)
                .unsqueeze(1)
            )

        pixels = step_mdp_data_cpu.data["pixels"]

        if torch.isin(torch.tensor([step_number-1]), self.random_step_indexes_to_log) and step_mdp_data.segmentation_data is not None and len(self.eval_segments_grid) < self.nb_segments_image_to_log:
            image_id = step_mdp_data_cpu.data["image_ids"][0]
            segments_data = segmentation_data_cpu["segments_data"]
            image_segments_data = segments_data[
                segments_data["image_ids"] == image_id
            ]
            bboxes = image_segments_data["coords"]["masks_absolute_bboxes"]
            labels = [
                self.class_index_to_name[i]
                for i in image_segments_data["classes"].tolist()
            ]
            image_with_bboxes = draw_bounding_boxes(
                pixels[0].squeeze(0),
                bboxes,
                labels,
                width=1,
            )
            self.eval_segments_bboxes.append(image_with_bboxes)

            segments_grid = torch.as_tensor(
                draw_rgb_segments_grid(
                    segments_data, image_id, show=False, unscale=True
                )
            ).permute(2, 0, 1)
            self.eval_segments_grid.append(segments_grid)

        if len(self.eval_pixels_video_writter) < self.video_max_steps:
            self.eval_pixels_video_writter.add_step_frame(pixels)

        if len(self.eval_masked_pixels_video_writter) < self.video_max_steps and step_mdp_data.segmentation_data is not None:
            segments_data = segmentation_data_cpu["segments_data"]
            masked_pixels_images = get_image_covered_by_predicted_masks(
                original_images=pixels.squeeze(1), segments_data=segments_data
            )
            self.eval_masked_pixels_video_writter.add_step_frame(
                masked_pixels_images.unsqueeze(1)
            )

        if step_mdp_data.next.data["done"].any():
            rollouts_metrics = self.metrics["rollouts_metrics"]
            if step_mdp_data.next.data["done"].squeeze().item():
                current_rollout_metrics = self.metrics["current_rollout_metrics"][env_id]
                aggregated_rollout_metrics = current_rollout_metrics.compute_aggregated_metrics()
                for k, v in aggregated_rollout_metrics.items():
                    rollouts_metrics.accumulate_metric(
                        metric_name=k,
                        metric_value=v,
                        agg_fn=np.mean,
                    )

    def accumulate_eval_rollout_metrics(self, step_mdp_data: MdpData, step_number: int, env_step: int, env_id: Optional[int]):
        logging_prefix = "eval_"

        step_mdp_data_cpu = step_mdp_data.to(device="cpu", non_blocking=True)
        # TODO : Cleanup this function, we could probably avoid some code duplication
        if self.ignore_terminations:
            self._accum_eval_rollout_metrics_ignore_terminations(
                step_mdp_data,
                step_number,
                env_step,
                logging_prefix,
                step_mdp_data_cpu,
            )
        else:
            self._accum_eval_rollout_metrics_respect_terminations(
                step_mdp_data,
                step_number,
                env_step,
                logging_prefix,
                step_mdp_data_cpu,
                env_id
            )

    def log_eval_videos(self, env_step: int):
        video_file = self.eval_pixels_video_writter.write_video_to_disk(
            env_step=env_step
        )
        if video_file is not None:
            self.logger.log_video(str(video_file.resolve()), step=env_step)

        video_file = self.eval_masked_pixels_video_writter.write_video_to_disk(
            env_step=env_step
        )
        if video_file is not None:
            self.logger.log_video(str(video_file.resolve()), step=env_step)

        if self.eval_q_value_attn_video_writter is not None:
            video_file = self.eval_q_value_attn_video_writter.write_video_to_disk(
                env_step=env_step
            )
            if video_file is not None:
                self.logger.log_video(str(video_file.resolve()), step=env_step)

    def log_eval_images(self, env_step: int):
        for i, image_eval_segments_bboxes in enumerate(self.eval_segments_bboxes):
            self.logger.log_image(
                image_eval_segments_bboxes.permute(1, 2, 0).numpy(),
                f"eval_segments_bboxes_step_{env_step}_{i}",
                env_step,
            )
        self.eval_segments_bboxes = []

        for i, image_eval_segments_grid in enumerate(self.eval_segments_grid):
            self.logger.log_image(
                image_eval_segments_grid.permute(1, 2, 0).numpy(),
                f"eval_segments_grid_step_{env_step}_{i}",
                env_step,
            )

        self.eval_segments_grid = []

    def log_train_metrics(self, train_metrics: dict, grad_metrics: dict, env_step: int):
        self.log_grads_stats(grad_metrics=grad_metrics, env_step=env_step)
        self.logger.log_metrics(metrics=train_metrics, step=env_step)
        self.log_train_video(env_step)

    def log_grads_stats(self, grad_metrics: dict, env_step: int):
        actor_grad_stats = grad_metrics.get("actor_grad_stats", None)
        if actor_grad_stats is not None:
            actor_grad_stats_html = generate_grad_flow_interactive_html(
                layers=actor_grad_stats["actor_grad_layers"],
                avg_grads=actor_grad_stats["actor_grad_avg"],
                max_grads=actor_grad_stats["actor_grad_max"],
                l2_norms=actor_grad_stats["actor_grad_l2_norms"],
            )
            self.logger.log_html(html=actor_grad_stats_html, clear=True)
            actor_grad_stats_file_path = (
                self.html_folder_path / f"actor_grad_stats_step_{env_step}.html"
            )
            with open(actor_grad_stats_file_path, "w") as f:
                f.write(actor_grad_stats_html)
            self.logger.log_asset(asset_path=actor_grad_stats_file_path, step=env_step)

        critic_grad_stats = grad_metrics.get("critic_grad_stats", None)
        if critic_grad_stats is not None:
            critic_grad_stats_html = generate_grad_flow_interactive_html(
                layers=critic_grad_stats["critic_grad_layers"],
                avg_grads=critic_grad_stats["critic_grad_avg"],
                max_grads=critic_grad_stats["critic_grad_max"],
                l2_norms=critic_grad_stats["critic_grad_l2_norms"],
            )
            self.logger.log_html(html=critic_grad_stats_html, clear=True)
            critic_grad_stats_file_path = (
                self.html_folder_path / f"critic_grad_stats_step_{env_step}.html"
            )
            with open(critic_grad_stats_file_path, "w") as f:
                f.write(critic_grad_stats_html)
            self.logger.log_asset(asset_path=critic_grad_stats_file_path, step=env_step)

    def log_train_video(self, env_step: int):
        video_file = self.train_pixels_video_writter.write_video_to_disk(
            env_step=env_step
        )
        if video_file is not None:
            self.logger.log_video(str(video_file.resolve()), step=env_step)
