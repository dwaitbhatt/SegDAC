import torch
from tensordict import TensorDict
from segdac.data.mdp import MdpData
from segdac_dev.envs.transforms.transform import Transform


class SegmentationDataTrajIdsTransform(Transform):
    def __init__(self, device: str):
        super().__init__(device)

    def reset(self, mdp_data: MdpData) -> MdpData:
        return self.step(mdp_data)

    def step(self, mdp_data: MdpData) -> MdpData:
        absolute_image_ids = mdp_data.data["image_ids"].flatten()  # shape: (num_envs,)
        traj_ids = mdp_data.data["collector"]["traj_ids"].flatten()  # (num_envs,)

        seg_image_ids = mdp_data.segmentation_data["image_ids"].flatten()  # (N,)

        # Since absolute_image_ids is sorted (generated via torch.arange) and since absolute_image_ids
        # values are the same as seg_image_ids (it's just that seg_image_ids repeats values for segments
        # within the same image), we can use searchsorted.
        # For each segmentation image id, this returns the index into the absolute_image_ids.
        seg_image_ids_indexes = torch.searchsorted(absolute_image_ids, seg_image_ids)

        seg_traj_ids = traj_ids[seg_image_ids_indexes].view_as(
            mdp_data.segmentation_data["image_ids"]
        )

        mdp_data.segmentation_data["collector"] = TensorDict(
            {"traj_ids": seg_traj_ids},
            batch_size=mdp_data.segmentation_data.batch_size,
            device=self.device,
        )

        return mdp_data
