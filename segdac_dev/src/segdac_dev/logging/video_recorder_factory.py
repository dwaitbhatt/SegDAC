import hydra
import torch
from pathlib import Path
from omegaconf import DictConfig
from torchvision.io import write_video


class VideoRecorder:
    def __init__(self, tag: str, output_dir: str, fps: int):
        self.tag = tag
        self.output_dir = output_dir
        self.extension = ".mp4"
        self.fps = fps
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def dump(self, frames: torch.Tensor, name_suffix: str) -> Path:
        name = f"{self.tag}_{name_suffix}{self.extension}"
        output_path = Path(self.output_dir) / Path(name)
        write_video(
            filename=str(output_path.resolve()),
            video_array=frames.permute(0, 2, 3, 1),
            fps=self.fps,
        )
        return output_path


def create_video_recorder(
    cfg: DictConfig, tag: str = "step"
) -> tuple[VideoRecorder, Path]:
    video_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ) / Path(cfg["logging"]["videos_dir"])

    video_recorder = VideoRecorder(
        tag=tag, output_dir=video_dir, fps=cfg["logging"]["video_fps"]
    )

    return video_recorder
