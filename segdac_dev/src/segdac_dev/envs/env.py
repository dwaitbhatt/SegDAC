import torch
from abc import abstractmethod
from segdac.agents.agent import Agent
from segdac_dev.envs.transforms.transform import Transform
from segdac.data.mdp import MdpData
from tqdm import tqdm
from typing import Generator


class TensorDictEnvWrapper:
    def __init__(self, num_envs: int, transforms: list[Transform], tmp_job_data_dir: str):
        self.num_envs = num_envs
        self.transforms = transforms
        self.auto_reset_mdp_data = None
        self.tmp_job_data_dir = tmp_job_data_dir

    @abstractmethod
    def set_seed(self, seed: int):
        pass

    def reset(self) -> MdpData:
        reset_mdp_data = self._reset()

        for transform in self.transforms:
            reset_mdp_data = transform.reset(reset_mdp_data)

        return reset_mdp_data

    @abstractmethod
    def _reset(self) -> MdpData:
        pass

    def consume_auto_reset_data(self, mdp_data: MdpData) -> MdpData:
        """
        If an auto reset was performed, we need to use the reset obs from the auto reset.
        Must be called before predicting the action.
        if self.auto_reset_mdp_data is not None: It is expected to contain the full step data with the obs being the reset obs for envs that were done.
        """
        if self.auto_reset_mdp_data is not None:
            mdp_data.data.batch_size = torch.Size([])
            for k, v in self.auto_reset_mdp_data.data.items():
                mdp_data.data[k] = v
            mdp_data.data.batch_size = self.auto_reset_mdp_data.data.batch_size

            if self.auto_reset_mdp_data.segmentation_data is not None:
                mdp_data.segmentation_data.batch_size = torch.Size([])
                for k, v in self.auto_reset_mdp_data.segmentation_data.items():
                    mdp_data.segmentation_data[k] = v
                mdp_data.segmentation_data.batch_size = (
                    self.auto_reset_mdp_data.segmentation_data.batch_size
                )
            
            self.auto_reset_mdp_data = None

        return mdp_data

    def step(self, mdp_data: MdpData) -> MdpData:
        step_mdp_data = self._step(mdp_data)

        for transform in self.transforms:
            step_mdp_data = transform.step(step_mdp_data)

        if self.auto_reset_mdp_data is not None:
            for transform in self.transforms:
                self.auto_reset_mdp_data = transform.reset(
                    self.auto_reset_mdp_data,
                )

        return step_mdp_data

    @abstractmethod
    def _step(self, mdp_data: MdpData) -> MdpData:
        pass

    @abstractmethod
    def get_random_action(self) -> torch.Tensor:
        pass

    @abstractmethod
    def close(self):
        pass

    @torch.no_grad()
    def rollouts(
        self, num_rollouts: int, agent: Agent, agent_device: str
    ) -> Generator[tuple[int, MdpData], None, None]:
        """
        Follows the recommended evaluation protocol from https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/setup.html
        This method stores the steps data for num_envs parallel envs and once an episode is complete, it yields an MdpData object with the steps data for 1 env,
            then yields the MdpData for the next env etc.
        This means that this method will consume memory for at most nb_steps * num_envs data regardless of num_rollouts.
        This was done to save memory since storing nb_steps * num_rollouts is >= than nb_steps * num_envs
        """
        assert (
            num_rollouts % self.num_envs == 0
        ), "num_rollouts must be a multiple of num_envs"
        nb_parallel_rollouts = num_rollouts // self.num_envs

        for parallel_rollout_i in tqdm(range(nb_parallel_rollouts), desc="Env Rollouts"):
            parallel_rollout_done = False
            env_mdp_data = self.reset()

            step_count = 0
            while not parallel_rollout_done:
                env_mdp_data = agent(env_mdp_data.to(agent_device, non_blocking=False)).to(self.device, non_blocking=False)
                next_mdp_data = self.step(env_mdp_data)
                env_mdp_data.next = next_mdp_data

                step_count += 1

                yield None, step_count, env_mdp_data

                if next_mdp_data.data["done"].any():
                    assert next_mdp_data.data[
                        "done"
                    ].all(), "Env::rollouts does not support partial resets, so all envs must be done at the same time"
                    parallel_rollout_done = True
                else:
                    env_mdp_data = env_mdp_data.step()

            torch.cuda.empty_cache()