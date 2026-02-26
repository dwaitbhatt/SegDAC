import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor


class SamImageEncoder(nn.Module):
    def __init__(
        self,
        segmenter_model_name: str = "efficientvit-sam-l0",
        segmenter_weights_path: str = "weights/efficientvit_sam_l0.pt",
    ):
        super(SamImageEncoder, self).__init__()
        self.segmenter_model = create_efficientvit_sam_model(
            name=segmenter_model_name,
            pretrained=True,
            weight_url=segmenter_weights_path,
        )
        self.segmenter_model = self.segmenter_model.eval()
        self.segments_predictor = EfficientViTSamPredictor(self.segmenter_model)
        self.segmenter_image_size = self.get_segmenter_image_size(segmenter_model_name)
        self.sam_transform = v2.Compose(
            [
                v2.Resize(size=(self.segmenter_image_size, self.segmenter_image_size)),
                v2.Normalize(
                    mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                    std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
                ),
            ]
        )

    def get_segmenter_image_size(self, segmenter_model_name: str) -> int:
        if segmenter_model_name in [
            "efficientvit-sam-l0",
            "efficientvit-sam-l1",
            "efficientvit-sam-l2",
        ]:
            segmenter_image_size = 512
        else:
            segmenter_image_size = 1024

        return segmenter_image_size

    @torch.no_grad()
    def forward(self, image):
        """
        Forward pass through the SAM model to get image embeddings.

        Args:
            image (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Image embeddings of shape (B, D, 64, 64), where D is the embedding dimension.
        """
        sam_image = self.preprocess_image_for_sam(image)
        self.segments_predictor.set_image_batch(sam_image)
        sam_encoder_embeddings = self.segments_predictor.features
        return sam_encoder_embeddings

    def preprocess_image_for_sam(self, image: torch.Tensor) -> torch.Tensor:
        """
        Source: https://github.com/mit-han-lab/efficientvit/blob/95842539d7e9bd70def2fcdffc96b727722d801b/efficientvit/models/efficientvit/sam.py#L212
        """
        return self.sam_transform(image).contiguous()
