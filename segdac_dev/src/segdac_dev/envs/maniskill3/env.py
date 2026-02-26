import torch
import copy
from tensordict import TensorDict
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from segdac_dev.envs.maniskill3.obs_mode import ObsMode
from segdac_dev.envs.env import TensorDictEnvWrapper
from segdac_dev.envs.transforms.transform import Transform
from segdac.data.mdp import MdpData
from tensordict import merge_tensordicts


class ManiSkillEnvWrapper(TensorDictEnvWrapper):
    def __init__(
        self,
        num_envs: int,
        does_base_env_auto_reset: bool,
        transforms: list[Transform],
        tmp_job_data_dir: str,
        base_env: ManiSkillVectorEnv,
        obs_mode: ObsMode,
        agent_device: str,
    ):
        super().__init__(
            num_envs=num_envs, transforms=transforms, tmp_job_data_dir=tmp_job_data_dir
        )
        self.does_base_env_auto_reset = does_base_env_auto_reset
        self.base_env = base_env
        self.obs_mode = obs_mode
        self.generator = torch.Generator()
        self.envs_seeds = torch.randint(
            torch.iinfo(torch.uint32).min,
            torch.iinfo(torch.uint32).max,
            (num_envs,),
            generator=self.generator,
            dtype=torch.uint32,
        ).tolist()
        self.agent_device = agent_device
        self.info_keys_of_interest = [
            "elapsed_steps",
            "success",
            "final_observation",
            "final_info",
        ]

    @property
    def observation_space(self):
        return self.base_env.unwrapped.observation_space

    @property
    def action_space(self):
        return self.base_env.action_space

    @property
    def device(self):
        return self.base_env.unwrapped.device

    def set_seed(self, seed: int):
        self.generator = torch.Generator().manual_seed(seed)
        self.envs_seeds = torch.randint(
            torch.iinfo(torch.uint32).min,
            torch.iinfo(torch.uint32).max,
            (self.num_envs,),
            generator=self.generator,
            dtype=torch.uint32,
        ).tolist()
        self.base_env.reset(seed=self.envs_seeds)

    def _reset(self) -> MdpData:
        obs, info = self.base_env.reset()

        # Maniskill obs uses shared tensor so previous obs pixels == new obs pixels
        # We clone here so that the next step will not overwrite the previous step tensors
        obs = self.get_obs_data_of_interest(obs).clone()

        done = torch.zeros((self.num_envs), dtype=torch.bool, device=self.device)

        info = self.filter_info(info)

        done_info_data = TensorDict(
            {"done": done, "info": info},
            batch_size=torch.Size([self.num_envs]),
        )

        reset_data = merge_tensordicts(done_info_data, obs).unsqueeze(1)

        return MdpData(
            data=reset_data,
        )

    def filter_info(self, info: dict) -> dict:
        filtered_info = copy.deepcopy(
            {k: v for k, v in info.items() if k in self.info_keys_of_interest}
        )
        if (
            info.get("final_info", None) is not None
            and "final_info" in self.info_keys_of_interest
        ):
            filtered_final_info = copy.deepcopy(
                {
                    k: v
                    for k, v in info["final_info"].items()
                    if k in self.info_keys_of_interest
                }
            )
            filtered_info["final_info"] = filtered_final_info
        return filtered_info

    def _step(self, mdp_data: MdpData) -> MdpData:
        action = mdp_data.data["action"].to(self.base_env.unwrapped.device).squeeze(1)

        obs, reward, done, truncated, info = self.base_env.step(action)

        done = done | truncated

        # Maniskill obs uses shared tensor so previous obs pixels == new obs pixels
        # We clone here so that the next step will not overwrite the previous step tensors
        obs = self.get_obs_data_of_interest(obs).clone()

        info = self.filter_info(info)

        reward_done_info_data = TensorDict(
            {"reward": reward, "done": done, "info": info},
            batch_size=torch.Size([self.num_envs]),
        )

        step_data = merge_tensordicts(reward_done_info_data, obs).unsqueeze(1)

        step_data = self._get_true_step_data(step_data)

        out = MdpData(data=step_data)

        return out

    def _get_true_step_data(self, step_data: TensorDict) -> TensorDict:
        """
        If at least one sub-env was done, then it means that at least one sub-env has its obs returned as the reset obs (next traj obs).
            and the final obs is in a special info dict instead.
        We need to swap the reset obs with the true final obs so that when we save the save in the replay buffer it has the correct next obs.
        Then we can save the reset obs in the dict so that we can use it at the beggining of the next step.
        """
        if self.does_base_env_auto_reset and step_data["done"].any():
            auto_reset_data = step_data.exclude(
                "done", "reward", ("info", "final_info"), ("info", "final_observation")
            )  # done and reward are for the true step data not the reset data
            auto_reset_data["done"] = torch.zeros_like(
                step_data["done"], dtype=torch.bool, device=step_data["done"].device
            )  # We assume that done = false after a reset
            self.auto_reset_mdp_data = MdpData(data=auto_reset_data)

            final_info = step_data["info"]["final_info"]
            final_info = self.filter_info(final_info)
            raw_final_obs = step_data["info"]["final_observation"].squeeze(1)
            final_obs = self.get_obs_data_of_interest(raw_final_obs).unsqueeze(1)
            keys_affected_by_auto_reset = ["info"] + self.obs_mode.get_obs_keys()
            true_step_data = step_data.exclude(*keys_affected_by_auto_reset)
            true_step_data["info"] = final_info
            for obs_key in self.obs_mode.get_obs_keys():
                true_step_data[obs_key] = final_obs[obs_key]
        else:
            true_step_data = step_data
            self.auto_reset_mdp_data = None

        return true_step_data

    def get_obs_data_of_interest(self, raw_obs: dict) -> TensorDict:
        return self.obs_mode.get_obs_data(raw_obs)

    def get_random_action(self) -> torch.tensor:
        return torch.as_tensor(
            self.action_space.sample(),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

    def close(self):
        self.base_env.close()
