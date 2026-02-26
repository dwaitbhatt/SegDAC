from segdac_dev.envs.transforms.transform import Transform
from segdac.data.mdp import MdpData
from tensordict import TensorDict


class SegmentationDataImageIdsTransform(Transform):
    """
    Uses the image_ids field in the MdpData to convert the segmentation_data image_ids to absolute image_ids.
    This transform assumes that ImageIdsTransform was applied before this transform.
    It also assumes that mdp_data.segmentation_data exists (which we expect to be created from grounded_segmentation transform)
    This transform adds the key "image_ids" to the segmentation_data
    """

    def __init__(self, device: str):
        super().__init__(device)

    def reset(self, mdp_data: MdpData) -> MdpData:
        return self.step(mdp_data)

    def step(self, mdp_data: MdpData) -> MdpData:
        self.absolute_image_ids = mdp_data.data["image_ids"]
        self.recursive_update_image_ids(mdp_data.segmentation_data)
        mdp_data.segmentation_data["image_ids"] = self.segmentation_data_image_ids.to(
            self.device
        )

        return mdp_data

    def recursive_update_image_ids(self, segmentation_data: TensorDict):
        for key, value in segmentation_data.items():
            if key == "image_ids":
                image_ids_shape = segmentation_data[key].shape
                self.segmentation_data_image_ids = self.absolute_image_ids[
                    segmentation_data[key]
                ].reshape(image_ids_shape)
                segmentation_data[key] = self.segmentation_data_image_ids.to(
                    self.device
                )
            elif isinstance(value, TensorDict):
                self.recursive_update_image_ids(value)
