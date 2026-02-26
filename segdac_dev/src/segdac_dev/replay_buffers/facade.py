from segdac_dev.replay_buffers.segments.replay_buffer import (
    SegmentationDataReplayBuffer,
)
from torchrl.data import ReplayBuffer
from segdac.data.mdp import MdpData
from typing import Optional


class ReplayBufferFacade:
    def __init__(
        self,
        pre_save_transform: list,
        data_replay_buffer: ReplayBuffer,
        segmentation_data_replay_buffer: Optional[SegmentationDataReplayBuffer] = None,
    ):
        self.pre_save_transform = pre_save_transform
        self.data_replay_buffer = data_replay_buffer
        self.segmentation_data_replay_buffer = segmentation_data_replay_buffer
        self.nb_transitions_stored = 0

    def __len__(self) -> int:
        return self.nb_transitions_stored

    def extend(self, mdp_data: MdpData):
        for transform in self.pre_save_transform:
            mdp_data = transform.apply(mdp_data)
            if mdp_data is None:
                return
        current_step_data = mdp_data.data.copy()
        next_step_data = mdp_data.next.data.copy()
        current_step_data["next"] = next_step_data
        self.data_replay_buffer.extend(current_step_data)

        if self.segmentation_data_replay_buffer is not None:
            self.segmentation_data_replay_buffer.save(mdp_data)

        self.nb_transitions_stored += mdp_data.data.batch_size[0]

    def sample(self) -> MdpData:
        data_with_next_data = self.data_replay_buffer.sample().copy().squeeze(1)

        next_data = data_with_next_data["next"]
        data = data_with_next_data.exclude("next")

        if self.segmentation_data_replay_buffer is None:
            return MdpData(data=data, next=MdpData(data=next_data))

        segmentation_data, next_segmentation_data = (
            self.segmentation_data_replay_buffer.get_all(data["image_ids"])
        )

        return MdpData(
            data=data,
            segmentation_data=segmentation_data,
            next=MdpData(data=next_data, segmentation_data=next_segmentation_data),
        )
