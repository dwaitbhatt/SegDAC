from segdac.data.mdp import MdpData
from segdac_dev.envs.transforms.transform import Transform
from segdac.networks.image_segmentation_models.grounded_efficientvit_sam import (
    GroundedEfficientVitSam,
)
from segdac_dev.replay_buffers.segments.replay_buffer import TensorDict


class GroundedSegmentationWithSamEncoderEmbeddingsTransform(Transform):
    def __init__(self, device: str, segmentation_model: GroundedEfficientVitSam):
        super().__init__(device)
        self.segmentation_model = segmentation_model

    def reset(self, mdp_data: MdpData) -> MdpData:
        return self.step(mdp_data)

    def step(self, mdp_data: MdpData) -> MdpData:
        image = mdp_data.data["pixels_transformed"].squeeze(1)

        segments_data, sam_encoder_embeddings = self.segmentation_model.segment(
            image, return_sam_encoder_embeddings=True
        )

        mdp_data.segmentation_data = TensorDict(
            {
                "segments_data": segments_data.to(self.device),
            },
            batch_size=segments_data.batch_size,
        )

        mdp_data.data["sam_encoder_embeddings"] = sam_encoder_embeddings.to(
            self.device
        ).unsqueeze(1)

        return mdp_data
