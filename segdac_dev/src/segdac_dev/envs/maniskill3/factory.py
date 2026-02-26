import sapien
import mani_skill.envs  # Needed to register the gym environments
import segdac_dev.envs.maniskill3.tasks  # Needed to register the gym environments
import segdac_dev.envs.maniskill3.visual_generalization.tasks  # Needed to register the gym environments
import gymnasium as gym
from transforms3d.euler import euler2quat
from pathlib import Path
from hydra.utils import instantiate
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from omegaconf import DictConfig
from segdac_dev.envs.maniskill3.obs_mode import ObsMode
from segdac_dev.envs.maniskill3.env import (
    ManiSkillEnvWrapper,
)
from mani_skill.utils.wrappers.action_repeat import ActionRepeatWrapper
from mani_skill.utils import sapien_utils


class Maniskill3EnvFactory:
    def create(
        self,
        cfg: DictConfig,
        env_config: DictConfig,
        job_id: str,
        env_transforms_configs: list,
        **kwargs
    ) -> ManiSkillEnvWrapper:

        sim_backend = "cpu" if env_config["device"] == "cpu" else "gpu"
        render_backend = "cpu" if env_config["device"] == "cpu" else "gpu"

        obs_height = env_config["pixels"]["height"]
        obs_width = env_config["pixels"]["width"]

        obs_mode: ObsMode = instantiate(env_config["maniskill3"]["obs_mode"])

        camera_config = dict(
            width=obs_width,
            height=obs_height,
        )

        custom_cam_look_at_pose = False
        if env_config.get("camera", {}).get("look_at", None) is not None:
            look_at_config = env_config["camera"]["look_at"]
            custom_cam_look_at_pose = True
        if env_config.get("camera", {}).get("pose", None) is not None:
            p = env_config["camera"]["pose"]["p"]
            q_euler = env_config["camera"]["pose"]["q_euler"]
            camera_config["pose"] = sapien.Pose(p=p, q=euler2quat(q_euler[0], q_euler[1], q_euler[2]))
        if kwargs.get("test_config", {}).get("camera", {}).get("look_at", None) is not None \
            and kwargs["test_config"]["camera"]["look_at"]["from"] is not None \
                and kwargs["test_config"]["camera"]["look_at"]["to"] is not None:
            look_at_config = kwargs["test_config"]["camera"]["look_at"]
            custom_cam_look_at_pose = True
        if kwargs.get("test_config", {}).get("camera", {}).get("pose", None) is not None \
            and kwargs["test_config"]["camera"]["pose"]["p"] is not None \
                and kwargs["test_config"]["camera"]["pose"]["q_euler"] is not None:
            p = kwargs["test_config"]["camera"]["pose"]["p"]
            q_euler = kwargs["test_config"]["camera"]["pose"]["q_euler"]
            camera_config["pose"] = sapien.Pose(p=p, q=euler2quat(q_euler[0], q_euler[1], q_euler[2]))

        if custom_cam_look_at_pose:
            from_xyz = look_at_config["from"]
            to_xyz = look_at_config["to"]
            pose = sapien_utils.look_at(from_xyz, to_xyz)
            camera_config["pose"] = pose

        custom_cam_fov = False
        if env_config.get("camera", {}).get("fov", None) is not None:
            fov = env_config["camera"]["fov"]
            custom_cam_fov = True
        if kwargs.get("test_config", {}).get("camera", {}).get("fov", None) is not None:
            fov = kwargs["test_config"]["camera"]["fov"]
            custom_cam_fov = True

        if custom_cam_fov:
            camera_config["fov"] = fov
        ms3_env = gym.make(
            id=env_config["id"],
            sensor_configs=camera_config,
            human_render_camera_configs=camera_config,
            max_episode_steps=env_config["max_frames_per_traj"],
            num_envs=env_config["num_envs"],
            obs_mode=obs_mode.get_name(),
            control_mode=env_config["control_mode"],
            render_mode="rgb_array",
            reconfiguration_freq=env_config["reconfiguration_freq"],
            sim_backend=sim_backend,
            render_backend=render_backend,
            **kwargs
        )

        action_repeat = env_config["action_repeat"]

        ms3_env = ActionRepeatWrapper(env=ms3_env, repeat=action_repeat)

        if env_config["record_trajectories"]:
            episode_output_dir = (
                Path(cfg["final_job_data_dir"]) / Path(job_id) / Path("trajectories")
            )
            episode_output_dir.mkdir(parents=True, exist_ok=True)
            ms3_env = RecordEpisode(
                ms3_env,
                output_dir=str(episode_output_dir.resolve()),
                save_trajectory=True,
                trajectory_name="trajectory",
                save_video=env_config["save_trajectories_video"],
                video_fps=30,
                max_steps_per_video=env_config["max_frames_per_traj"],
            )

        base_env_auto_reset = not env_config["ignore_terminations"]

        ms3_env = ManiSkillVectorEnv(
            ms3_env,
            auto_reset=base_env_auto_reset,
            ignore_terminations=env_config["ignore_terminations"],
        )

        num_envs = env_config["num_envs"]

        env_transforms = []

        for env_transform_config in env_transforms_configs:
            env_transform = instantiate(env_transform_config)
            env_transforms.append(env_transform)

        return ManiSkillEnvWrapper(
            num_envs=num_envs,
            does_base_env_auto_reset=base_env_auto_reset,
            transforms=env_transforms,
            tmp_job_data_dir=cfg["tmp_job_data_dir"],
            base_env=ms3_env,
            obs_mode=obs_mode,
            agent_device=cfg["policy_device"],
        )
