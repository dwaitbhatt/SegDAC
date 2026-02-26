from segdac.data.mdp import MdpData
from segdac_dev.envs.transforms.transform import Transform
from segdac.networks.segments_encoders.sam_encoder_segments_encoder import (
    SamEncoderEmbeddingsSegmentsEncoder,
)


class SamEncoderEmbeddingsSegmentsEncoderTransform(Transform):
    def __init__(
        self, device: str, segments_encoder: SamEncoderEmbeddingsSegmentsEncoder
    ):
        super().__init__(device)
        self.segments_encoder = segments_encoder

    def reset(self, mdp_data: MdpData) -> MdpData:
        return self.step(mdp_data)

    def step(self, mdp_data: MdpData) -> MdpData:
        segments_data = mdp_data.segmentation_data["segments_data"]
        sam_encoder_embeddings = mdp_data.data["sam_encoder_embeddings"]

        segments_encoder_output = self.segments_encoder(
            segments_data, sam_encoder_embeddings
        )

        mdp_data.segmentation_data["segments_encoder_output"] = (
            segments_encoder_output.to(self.device)
        )

        return mdp_data
