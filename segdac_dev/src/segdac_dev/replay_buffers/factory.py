from segdac_dev.replay_buffers.transforms.torchrl.multistep import CustomNStepReturn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchrl.data import ReplayBuffer
from torchrl.data import LazyTensorStorage
from torchrl.envs.transforms import Compose
from torchrl.envs.transforms import ExcludeTransform
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from segdac_dev.replay_buffers.transforms.torchrl.unscale_action import (
    UnscaleActionTransform,
)


def create_replay_buffer(cfg: DictConfig, env_action_scaler):
    capacity = cfg["algo"]["replay_buffer"]["capacity"]
    training_batch_size = int(cfg["training"]["batch_size"])

    keys_to_exclude = []

    for k in list(cfg["algo"]["replay_buffer"]["keys_to_exclude"]):
        keys_to_exclude.append(k)
        keys_to_exclude.append(("next", k))

    rb_transforms = []

    for save_transform_config in cfg["algo"]["replay_buffer"]["save_transforms"]:
        save_transform = instantiate(save_transform_config)
        rb_transforms.append(save_transform)

    rb_transforms.append(ExcludeTransform(*keys_to_exclude, inverse=True))
    rb_transforms.append(UnscaleActionTransform(env_action_scaler=env_action_scaler))

    for transform in rb_transforms:
        if isinstance(transform, CustomNStepReturn):
            continue
        assert transform.inverse is True

    for sample_transform_config in cfg["algo"]["replay_buffer"]["sample_transforms"]:
        sample_transform = instantiate(sample_transform_config)
        rb_transforms.append(sample_transform)

    storage = LazyTensorStorage(
        max_size=capacity,
        device="cpu",
        ndim=1
    )
    
    sampler = SamplerWithoutReplacement()

    replay_buffer = ReplayBuffer(
        storage=storage,
        sampler=sampler,
        transform=Compose(*rb_transforms),
        batch_size=training_batch_size,
    )

    return replay_buffer
