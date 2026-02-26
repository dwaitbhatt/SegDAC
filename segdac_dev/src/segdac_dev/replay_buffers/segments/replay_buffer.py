import torch
from collections import deque
from segdac.data.mdp import MdpData
from tensordict import TensorDict
from pathlib import Path
from tensordict import MemoryMappedTensor


class SegmentationDataReplayBuffer:
    def __init__(
        self,
        data_storage_folder_path: str,
        max_nb_images: int,
        max_nb_segments_per_image: int,
        segments_data_keys_to_save: list,
        segments_sample_transforms: list = [],
    ):
        self.data_storage_folder_path = Path(data_storage_folder_path) / Path(
            "segmentation_data_rb"
        )
        self.data_storage_folder_path.mkdir(parents=True, exist_ok=True)
        self.segments_data_save_dir = Path(self.data_storage_folder_path) / Path(
            "segments_data/"
        )
        self.segments_data_save_dir.mkdir(parents=True, exist_ok=True)
        self.next_segments_data_save_dir = Path(self.data_storage_folder_path) / Path(
            "next_segments_data/"
        )
        self.next_segments_data_save_dir.mkdir(parents=True, exist_ok=True)
        self.max_nb_images = max_nb_images
        self.max_nb_segments_per_image = max_nb_segments_per_image
        self.max_nb_segments = self.max_nb_images * max_nb_segments_per_image
        self.image_ids = deque(maxlen=max_nb_images)
        self.image_id_to_seg_metadata = {}
        self.segments_data_keys_to_save = [
            "image_ids",
            ("collector", "traj_ids"),
        ] + list(segments_data_keys_to_save)
        self.segments_sample_transforms = segments_sample_transforms
        self.segments_data = None
        self.next_segments_data = None
        self.insert_index = 0

    def save(self, mdp_data: MdpData):
        segmentation_data_to_save = mdp_data.segmentation_data.select(
            *self.segments_data_keys_to_save
        )
        next_segmentation_data_to_save = mdp_data.next.segmentation_data.select(
            *self.segments_data_keys_to_save
        )

        self.create_segments_data_storage(
            segmentation_data_to_save, next_segmentation_data_to_save
        )

        for image_id, next_image_id in zip(
            mdp_data.data["image_ids"], mdp_data.next.data["image_ids"]
        ):
            image_id_int = image_id.item()
            if len(self.image_ids) == self.max_nb_images:
                oldest_image_id = self.image_ids.popleft()
                self.image_id_to_seg_metadata.pop(oldest_image_id)
            self.image_ids.append(image_id_int)
            image_segmentation_data_to_save = segmentation_data_to_save[
                segmentation_data_to_save["image_ids"] == image_id
            ]
            segment_data_insert_index_start_inclusive = (
                self.insert_index * self.max_nb_segments_per_image
            )
            nb_segments = image_segmentation_data_to_save.batch_size[0]
            segment_data_insert_index_end_exclusive = (
                segment_data_insert_index_start_inclusive + nb_segments
            )
            self.segments_data[
                segment_data_insert_index_start_inclusive:segment_data_insert_index_end_exclusive
            ] = image_segmentation_data_to_save

            image_next_segmentation_data_to_save = next_segmentation_data_to_save[
                next_segmentation_data_to_save["image_ids"] == next_image_id
            ]
            next_nb_segments = image_next_segmentation_data_to_save.batch_size[0]
            next_segment_data_insert_index_end_exclusive = (
                segment_data_insert_index_start_inclusive + next_nb_segments
            )
            self.next_segments_data[
                segment_data_insert_index_start_inclusive:next_segment_data_insert_index_end_exclusive
            ] = image_next_segmentation_data_to_save

            self.image_id_to_seg_metadata[image_id_int] = {
                "segmentation_data_index_start": segment_data_insert_index_start_inclusive,
                "segmentation_data_index_end": segment_data_insert_index_end_exclusive,
                "next_segmentation_data_index_start": segment_data_insert_index_start_inclusive,
                "next_segmentation_data_index_end": next_segment_data_insert_index_end_exclusive,
            }

            self.insert_index = (self.insert_index + 1) % self.max_nb_images

    def create_segments_data_storage(
        self,
        segmentation_data_to_save: TensorDict,
        next_segmentation_data_to_save: TensorDict,
    ):
        if self.segments_data is None:
            fake_segments_data_dict = {}
            self.populate_fake_segments_data_dict(
                segmentation_data_to_save, fake_segments_data_dict
            )
            self.segments_data = TensorDict(
                fake_segments_data_dict, batch_size=torch.Size([self.max_nb_segments])
            ).memmap_like(prefix=self.segments_data_save_dir)
        if self.next_segments_data is None:
            fake_next_segments_data_dict = {}
            self.populate_fake_segments_data_dict(
                next_segmentation_data_to_save, fake_next_segments_data_dict
            )
            self.next_segments_data = TensorDict(
                fake_next_segments_data_dict,
                batch_size=torch.Size([self.max_nb_segments]),
            ).memmap_like(prefix=self.next_segments_data_save_dir)

    def populate_fake_segments_data_dict(
        self, segmentation_data_to_save: TensorDict, fake_segments_data_dict: dict
    ):
        for k, v in segmentation_data_to_save.items():
            if isinstance(v, TensorDict):
                fake_segments_data_dict[k] = {}
                self.populate_fake_segments_data_dict(
                    segmentation_data_to_save[k], fake_segments_data_dict[k]
                )
            else:
                shape = list([self.max_nb_segments])
                if len(v.shape) > 1:
                    shape += list(v.shape[1:])

                fake_segments_data_dict[k] = MemoryMappedTensor.empty(
                    *shape, dtype=v.dtype
                )

    def get_all(self, absolute_image_ids: list) -> tuple[TensorDict, TensorDict]:
        current_indices = []
        next_indices = []
        for image_id in absolute_image_ids:
            seg_metadata = self.image_id_to_seg_metadata[image_id.item()]

            current_start = seg_metadata["segmentation_data_index_start"]
            current_end = seg_metadata["segmentation_data_index_end"]
            current_indices.extend(range(current_start, current_end))

            next_start = seg_metadata["next_segmentation_data_index_start"]
            next_end = seg_metadata["next_segmentation_data_index_end"]
            next_indices.extend(range(next_start, next_end))

        current_indices_tensor = torch.tensor(current_indices, dtype=torch.int32)
        next_indices_tensor = torch.tensor(next_indices, dtype=torch.int32)

        all_step_seg_data = self.segments_data[current_indices_tensor]
        all_next_step_data = self.next_segments_data[next_indices_tensor]

        for transform in self.segments_sample_transforms:
            all_step_seg_data = transform.apply(all_step_seg_data)

        for transform in self.segments_sample_transforms:
            all_next_step_data = transform.apply(all_next_step_data)

        return all_step_seg_data, all_next_step_data
