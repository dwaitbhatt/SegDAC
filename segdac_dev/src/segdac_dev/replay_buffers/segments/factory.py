import hydra
from omegaconf import DictConfig
from typing import Optional
from hydra.utils import instantiate
from segdac_dev.replay_buffers.segments.replay_buffer import (
    SegmentationDataReplayBuffer,
)


def create_segmentation_data_replay_buffer(
    cfg: DictConfig,
) -> Optional[SegmentationDataReplayBuffer]:
    segments_data_keys_to_save = cfg["algo"]["replay_buffer"].get(
        "segments_data_keys_to_save", []
    )
    if len(segments_data_keys_to_save) == 0:
        return None

    segments_sample_transforms = []

    for segments_sample_transform_config in cfg["algo"]["replay_buffer"][
        "segments_sample_transforms"
    ]:
        segments_sample_transforms.append(instantiate(segments_sample_transform_config))

    return SegmentationDataReplayBuffer(
        data_storage_folder_path=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        max_nb_images=cfg["algo"]["replay_buffer"]["capacity"],
        max_nb_segments_per_image=cfg["algo"]["replay_buffer"][
            "max_nb_segments_per_image"
        ],
        segments_data_keys_to_save=segments_data_keys_to_save,
        segments_sample_transforms=segments_sample_transforms,
    )
