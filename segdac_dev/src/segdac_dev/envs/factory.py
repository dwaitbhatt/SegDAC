from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import OmegaConf,open_dict
from segdac_dev.envs.env import TensorDictEnvWrapper


class EnvFactory:
    def __init__(self, base_env_factory):
        self.base_env_factory = base_env_factory

    def create(
        self,
        cfg: DictConfig,
        instance_specific_env_config: DictConfig,
        job_id: str,
        env_transforms_configs: list,
        **kwargs
    ) -> TensorDictEnvWrapper:
        env_config = OmegaConf.to_container(cfg["env"])
        instance_specific_config = OmegaConf.to_container(instance_specific_env_config)

        env_config.update(instance_specific_config)

        env_config = OmegaConf.create(env_config)

        env = self.base_env_factory.create(
            cfg=cfg,
            env_config=env_config,
            job_id=job_id,
            env_transforms_configs=env_transforms_configs,
            **kwargs
        )

        return env


def create_env_from_config(
    cfg: DictConfig,
    instance_specific_env_config: DictConfig,
    job_id: str,
    env_transforms_configs: list = [],
    **kwargs
):
    env_factory = EnvFactory(instantiate(cfg["env"]["factory"]))

    env = env_factory.create(
        cfg=cfg,
        instance_specific_env_config=instance_specific_env_config,
        job_id=job_id,
        env_transforms_configs=env_transforms_configs,
        **kwargs
    )

    return env


def create_train_env(cfg: DictConfig, job_id: str):
    with open_dict(cfg):
        cfg.training.env_config.seed = cfg['training']['seed']
    return create_env_from_config(
        cfg=cfg,
        instance_specific_env_config=cfg["training"]["env_config"],
        job_id=job_id,
        env_transforms_configs=cfg["algo"]["env"]["train_transforms"],
    )


def create_eval_env(cfg: DictConfig, job_id: str):
    with open_dict(cfg):
        cfg.evaluation.env_config.seed = cfg['evaluation']['seed']
    return create_env_from_config(
        cfg=cfg,
        instance_specific_env_config=cfg["evaluation"]["env_config"],
        job_id=job_id,
        env_transforms_configs=cfg["algo"]["env"]["eval_transforms"],
    )


def create_test_env(cfg: DictConfig, job_id: str, test_config: dict):
    return create_env_from_config(
        cfg=cfg,
        instance_specific_env_config=cfg["evaluation"]["env_config"],
        job_id=job_id,
        env_transforms_configs=cfg.get("algo", {}).get("env", {}).get("eval_transforms", []),
        test_config=test_config,
        enable_shadow=True
    )
