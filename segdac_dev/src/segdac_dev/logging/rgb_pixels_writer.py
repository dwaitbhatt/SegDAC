import torch
import torchvision.transforms.v2 as v2
import math
from tensordict import MemoryMappedTensor
from segdac_dev.logging.video_recorder_factory import create_video_recorder
from omegaconf import DictConfig
from torchvision.utils import make_grid
from pathlib import Path
from typing import Optional


class RgbPixelsWritter:
    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        max_steps: int,
        height: int,
        width: int,
        video_tag: str,
        max_steps_per_traj: int,
        grid: bool = True,
        interpolation: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
    ):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.height = height
        self.width = width
        self.channels = 3
        self.pixels = MemoryMappedTensor.empty(
            (
                self.num_envs,
                self.max_steps,
                self.channels,
                self.height,
                self.width,
            )
        )
        self.grid = grid
        self.max_steps_per_traj = max_steps_per_traj
        self.video_tag = video_tag
        self.video_recorder = create_video_recorder(cfg, tag=video_tag)
        self.video_frame_resize_transform = v2.Resize(
            (self.height, self.width), interpolation=interpolation
        )
        self.frame_index = 0
        self.steps_count = 0
        self.video_grid_nrows = int(math.ceil(math.sqrt(self.num_envs)))

    def __len__(self) -> int:
        return self.steps_count

    def add_step_frame(self, new_pixels: torch.Tensor):
        """
        new_pixels: (num_envs, 1, 3, H, W) or (num_envs, 1, frame_stack, 3, H, W)
        """
        is_frame_stack = new_pixels.ndim == 6
        if is_frame_stack:
            new_pixels = new_pixels[:, :, -1, :, :, :]  # (num_envs, 1, 3, H, W)
        if self.frame_index > self.max_steps - 1:
            self.frame_index = 0
            self.steps_count = 0

        for env_id, env_new_pixels in enumerate(new_pixels.squeeze(1)):
            self._add_frame(env_new_pixels, env_id)

        self.steps_count += 1
        self.frame_index += 1

    def _add_frame(self, new_pixels: torch.Tensor, env_id: int):
        pixels_to_log = new_pixels  # (3, H, W)
        h = pixels_to_log.shape[-2]
        w = pixels_to_log.shape[-1]
        if h != self.height or w != self.width:
            pixels_to_log = self.video_frame_resize_transform(pixels_to_log)

        if pixels_to_log.is_floating_point():
            pixels_to_log = (pixels_to_log * 255).to(torch.uint8)

        self.pixels[env_id, self.frame_index, :, :, :] = pixels_to_log.to(
            device="cpu", non_blocking=False
        )

    def write_video_to_disk(self, env_step: int) -> Path:
        frames = []

        if self.grid or self.num_envs == 1:
            for frame_index in range(self.steps_count):
                if self.max_steps_per_traj is not None and frame_index + 1 > self.max_steps_per_traj:
                    break
                envs_frame = self.pixels[:, frame_index, :, :, :]
                if self.num_envs > 1:
                    pixels_obs = make_grid(envs_frame, nrow=self.video_grid_nrows)
                else:
                    pixels_obs = envs_frame.squeeze(0)
                frames.append(pixels_obs)
        else:
            for env_id in range(self.num_envs):
                for frame_index in range(self.steps_count):
                    if self.max_steps_per_traj is not None and frame_index + 1 > self.max_steps_per_traj:
                        break
                    pixels_obs = self.pixels[env_id, frame_index, :, :, :]

                    frames.append(pixels_obs)

                    if len(frames) == self.max_steps:
                        break
                if len(frames) == self.max_steps:
                    break

        if len(frames) == 0:
            return None

        output_path = self.video_recorder.dump(
            frames=torch.stack(frames), name_suffix=str(env_step)
        )

        self.frame_index = 0
        self.steps_count = 0

        return output_path
