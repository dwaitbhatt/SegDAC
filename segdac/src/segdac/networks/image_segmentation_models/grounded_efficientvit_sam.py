import time

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from ultralytics import YOLOWorld
try:
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model
except:
    pass
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from tensordict import TensorDict
from segdac.masks.coords_extractor import MaskCoordsExtractor


class RgbToBgr(torch.nn.Module):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return img[:, [2, 1, 0], :, :]


@torch.no_grad()
def get_image_covered_by_predicted_masks(
    original_images: torch.Tensor, segments_data: TensorDict
) -> torch.Tensor:
    B, C, H, W = original_images.shape

    binary_masks = segments_data["binary_masks"]

    union_mask = torch.zeros(
        (B, 1, H, W), dtype=binary_masks.dtype, device=binary_masks.device
    )

    image_ids = segments_data["image_ids"]

    unique_vals, relative_image_ids = torch.unique(
        image_ids, sorted=True, return_inverse=True
    )

    union_mask = torch.index_add(
        input=union_mask, dim=0, index=relative_image_ids, source=binary_masks, alpha=1
    ).clamp_max(1)

    return original_images * union_mask


class GroundedEfficientVitSam:
    def __init__(
        self,
        device: str,
        grounding_text_tags: list,
        object_detector_weights_path: str = "weights/yolov8s-worldv2.pt",
        object_detection_confidence_threshold: float = 0.0001,
        object_detection_iou_threshold: float = 0.01,
        segmenter_model_name: str = "efficientvit-sam-l0",
        segmenter_weights_path: str = "weights/efficientvit_sam_l0.pt",
        masks_post_process_kernel_size: int = 9,
        max_nb_segments: int = 30  # For memory limitation only
    ):
        self.device = device
        self.object_detector = YOLOWorld(
            object_detector_weights_path).to(device).eval()
        self.object_detector.set_classes(grounding_text_tags)
        self.object_detection_confidence_threshold = (
            object_detection_confidence_threshold
        )
        self.object_detection_iou_threshold = object_detection_iou_threshold
        self.object_detection_img_size = 640
        self.yolo_transform = v2.Compose(
            [
                v2.Resize(
                    size=(
                        self.object_detection_img_size,
                        self.object_detection_img_size,
                    )
                ),
                RgbToBgr(),
            ]
        )
        self.segmenter_model = create_efficientvit_sam_model(
            name=segmenter_model_name,
            pretrained=True,
            weight_url=segmenter_weights_path,
        )
        self.segmenter_model = self.segmenter_model.to(device).eval()
        self.segments_predictor = EfficientViTSamPredictor(
            self.segmenter_model)
        self.segmenter_image_size = self.get_segmenter_image_size(
            segmenter_model_name)
        self.sam_transform = v2.Compose(
            [
                v2.Resize(size=(self.segmenter_image_size,
                          self.segmenter_image_size)),
                v2.Normalize(
                    mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                    std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
                ),
            ]
        )
        self.masks_post_process_kernel_size = masks_post_process_kernel_size
        self.fallback_bounding_box = torch.tensor(
            [[0.0, 0.0, self.segmenter_image_size, self.segmenter_image_size]],
            dtype=torch.float32,
            device=device,
        )
        self.fallback_mask = torch.ones(
            size=(1, 1, self.segmenter_image_size, self.segmenter_image_size),
            dtype=torch.uint8,
            device=device,
        )
        self.mask_coords_extractor = MaskCoordsExtractor()
        self.max_nb_segments = max_nb_segments

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

    def _is_cuda(self) -> bool:
        """`device` is stored as str (e.g. 'cuda'); accept torch.device too."""
        d = self.device
        if isinstance(d, torch.device):
            return d.type == "cuda"
        s = str(d)
        return s == "cuda" or s.startswith("cuda:")

    def filter_duplicate_bounding_boxes(self, xyxy_coords: list, classes: list, confidences: list) -> tuple[list, list, list]:
        """
        If multiple bounding boxes are predicted for the same class, we only retain the one with the highest confidence.
        """
        filtered_xyxy_coords = []
        filtered_classes = []
        filtered_confidences = []
        # Iterate over results for different images
        for image_xyxy_coords, image_class_ids, image_confidences in zip(xyxy_coords, classes, confidences):
            valid_class_ids = image_class_ids.unique()
            valid_box_inds = []
            for class_id in valid_class_ids:
                class_inds = torch.where(image_class_ids == class_id)[0]
                class_max_conf_box_ind = class_inds[image_confidences[class_inds].argmax()]
                valid_box_inds.append(class_max_conf_box_ind.item())
                
            filtered_xyxy_coords.append(image_xyxy_coords[valid_box_inds])
            filtered_classes.append(image_class_ids[valid_box_inds])
            filtered_confidences.append(image_confidences[valid_box_inds])
        return filtered_xyxy_coords, filtered_classes, filtered_confidences

    @torch.no_grad()
    def segment(
        self,
        image: torch.Tensor,
        save_bounding_boxes: bool = False,
        verbose: bool = False,
        return_sam_encoder_embeddings: bool = False,
        return_phase_timings: bool = False,
        filter_duplicate_bounding_boxes: bool = True,
    ) -> TensorDict | tuple[TensorDict, torch.Tensor]:
        """
        Inputs :
            image: (B,C,H,W) torch.float32 in range (0,1)
            (Optional) return_sam_encoder_embeddings: Whether or not to return the embeddings from SAM's encoder.
        Output :
            image_ids: (N,) torch.int64 where N is the number of segments
            absolute_segment_ids: (N,) torch.int64 where N is the number of segments
            relative_segment_ids: (N,) torch.int64 where N is the number of segments
            binary_masks: (N,1,H,W) torch.uint8
            rgb_segments: (N,3,H,W) torch.float32 in range (0,1)
            coords:
                masks_absolute_bboxes: (N, 4) absolute bbox coords between (0,H-1) and (0,W-1) in (xmin, ymin, xmax, ymax) format.
                masks_normalized_bboxes: (N, 4) The normalized bbox coords in (0,1) range, still in (xmin, ymin, xmax, ymax) format.
        """
        def _sync_for_timings() -> None:
            if return_phase_timings and self._is_cuda():
                torch.cuda.synchronize()

        _sync_for_timings()
        t_segment_start = time.perf_counter()
        images_bboxes_xyxy, images_bboxes_classes, images_bboxes_confidences = self.predict_bounding_boxes(
            image, save=save_bounding_boxes, verbose=verbose
        )
        # Same YOLO outputs as used for SAM; test harness can draw boxes without a second forward. Below used for plotting (unfiltered) bounding boxes.
        self.last_yolo_xyxy = images_bboxes_xyxy
        self.last_yolo_classes = images_bboxes_classes
        self.last_yolo_confidences = images_bboxes_confidences

        # If multiple bounding boxes are predicted for the same class, we only retain the one with the highest confidence.
        if filter_duplicate_bounding_boxes:
            images_bboxes_xyxy, images_bboxes_classes, images_bboxes_confidences = self.filter_duplicate_bounding_boxes(
                images_bboxes_xyxy, images_bboxes_classes, images_bboxes_confidences
            )

        _sync_for_timings()
        t_after_object_detection = time.perf_counter()

        image_ids = []
        images_binary_masks = []
        images_rgb_segments = []
        mask_classes = []

        sam_image = self.preprocess_image_for_sam(image)

        self.segments_predictor.set_image_batch(sam_image)

        sam_encoder_embeddings = self.segments_predictor.features

        for image_id, image_bboxes_xyxy in enumerate(images_bboxes_xyxy):
            preprocessed_image_boxes = self.preprocess_boxes_for_sam(
                image_bboxes_xyxy)

            image_binary_masks = self.predict_binary_masks(
                preprocessed_image_boxes, image_id
            )[:self.max_nb_segments]

            num_masks_generated = image_binary_masks.shape[0]

            image_ids.extend([image_id] * image_binary_masks.shape[0])
            images_binary_masks.extend(image_binary_masks)

            image_bboxes_classes = images_bboxes_classes[image_id]
            if image_bboxes_classes.numel() > 0:
                classes_to_add = image_bboxes_classes[:num_masks_generated]
                mask_classes.extend(classes_to_add.long())
            else:
                # No original classes - use placeholders for all generated masks
                # Use first class as placeholder (eg: if background is first item in grounding text tags list then background will be used)
                placeholder_class_value = 0
                dtype_to_use = torch.long
                placeholder_classes = torch.full(
                    (num_masks_generated,),
                    placeholder_class_value,
                    dtype=dtype_to_use,
                    device=self.device
                )
                mask_classes.extend(placeholder_classes)

        image_ids = torch.tensor(
            image_ids, dtype=torch.int64, device=self.device)
        images_binary_masks = torch.stack(images_binary_masks)

        images_rgb_segments = image[image_ids] * images_binary_masks

        coords = self.mask_coords_extractor.extract(images_binary_masks)

        nb_segments = len(image_ids)

        absolute_segment_ids = torch.arange(nb_segments, device=self.device)

        _, counts = torch.unique_consecutive(image_ids, return_counts=True)
        relative_segment_ids = torch.cat(
            [
                torch.arange(
                    count,
                )
                for count in counts
            ]
        )

        mask_classes = torch.stack(mask_classes)

        segments_data = TensorDict(
            source={
                "image_ids": image_ids,
                "absolute_segment_ids": absolute_segment_ids,
                "relative_segment_ids": relative_segment_ids,
                "binary_masks": images_binary_masks,
                "rgb_segments": images_rgb_segments,
                "coords": coords,
                "classes": mask_classes,
            },
            batch_size=torch.Size([nb_segments]),
            device=self.device,
        )

        if return_phase_timings:
            _sync_for_timings()
            t_segment_end = time.perf_counter()
            self.last_segment_phase_timings = {
                "object_detection_s": t_after_object_detection
                - t_segment_start,
                "segmentation_s": t_segment_end - t_after_object_detection,
            }

        if return_sam_encoder_embeddings:
            return segments_data, sam_encoder_embeddings
        else:
            return segments_data

    def predict_binary_masks(
        self, image_bounding_boxes: torch.Tensor, image_index: int
    ) -> torch.Tensor:
        """
        Returns binary masks (N,1,H,W) torch.uint8
        """
        image_binary_masks, _, _ = self.segments_predictor.predict_torch(
            image_index=image_index,
            point_coords=None,
            point_labels=None,
            boxes=image_bounding_boxes,
            multimask_output=False,
        )

        if not self.is_there_at_least_1_mask(image_binary_masks):
            return self.fallback_mask

        image_processed_binary_masks = self.post_process_masks(
            image_binary_masks,
            kernel_size=self.masks_post_process_kernel_size,
        )

        if not self.is_there_at_least_1_mask(image_processed_binary_masks):
            return self.fallback_mask

        image_processed_binary_masks = self.get_non_empty_masks(
            image_processed_binary_masks
        )

        return image_processed_binary_masks

    def is_there_at_least_1_mask(self, predicted_masks: torch.Tensor) -> bool:
        return predicted_masks.numel() > 0

    def get_non_empty_masks(self, masks: torch.Tensor) -> torch.Tensor:
        masks_bool = masks.bool()
        if masks_bool.any():
            non_empty_masks = masks_bool.any(dim=(-3, -2, -1))
            return masks[non_empty_masks]
        else:
            return self.fallback_mask

    def predict_bounding_boxes(
        self, image: torch.Tensor, save: bool, verbose: bool
    ) -> tuple[list, list]:
        xyxy_coords = []
        classes = []
        confidences = []
        preprocessed_image = self.preprocess_image_for_yolo_world(image)
        detections = self.object_detector.predict(
            preprocessed_image,
            conf=self.object_detection_confidence_threshold,
            iou=self.object_detection_iou_threshold,
            save=save,
            verbose=verbose,
        )
        for detection in detections:
            xyxy_coords.append(detection.boxes.xyxy)
            classes.append(detection.boxes.cls)
            confidences.append(detection.boxes.conf)
        
        return xyxy_coords, classes, confidences

    def preprocess_image_for_yolo_world(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B,C,H,W) torch.Tensor with dtype torch.float32 in (0,1) range.
        Source: https://github.com/ultralytics/ultralytics/blob/a672bf79dd8e22091a9fc637199552c7282968d8/ultralytics/engine/predictor.py#L116
        """
        return self.yolo_transform(image).contiguous()

    def preprocess_image_for_sam(self, image: torch.Tensor) -> torch.Tensor:
        """
        Source: https://github.com/mit-han-lab/efficientvit/blob/95842539d7e9bd70def2fcdffc96b727722d801b/efficientvit/models/efficientvit/sam.py#L212
        """
        return self.sam_transform(image).contiguous()

    def preprocess_boxes_for_sam(
        self, image_bounding_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Yolo outputs bounding boxes that are in (0, self.object_detection_img_size) range.
        SAM expects bounding boxes in (0, self.segmenter_image_size) range.
        We bring the yolo bounding boxes to (0,1) range then we scale the result to be in (0, self.segmenter_image_size) range.
        We assume H = W.
        In case no bounding boxes were predicted, we return 1 bounding box covering the entire image as a fallback.

        Yolo returns bounding boxes in range (0, self.object_detection_img_size) if we manually resize the input to (self.object_detection_img_size, self.object_detection_img_size)
        beforehand, otherwise when yolo does the preprocessing (eg: If we pass a numpy array) then it returns bounding boxes in range(0, original_img_size).
        """
        if image_bounding_boxes.numel() > 0:
            preprocessed_bboxes = (
                image_bounding_boxes
                / self.object_detection_img_size
                * self.segmenter_image_size
            ).clip(0.0, self.segmenter_image_size)
        else:
            preprocessed_bboxes = self.fallback_bounding_box

        return self.segments_predictor.apply_boxes_torch(preprocessed_bboxes)

    def post_process_masks(self, masks: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        masks: (N,1,H,W)
        """
        opened = self.apply_morphological_opening(
            masks.to(torch.float32), kernel_size=kernel_size
        )
        return self.apply_morphological_closing(opened, kernel_size=kernel_size).to(
            torch.uint8
        )

    def apply_morphological_opening(
        self, masks: torch.Tensor, kernel_size: int
    ) -> torch.Tensor:
        eroded = self.apply_erosion(masks, kernel_size)
        return self.apply_dilation(eroded, kernel_size)

    def apply_erosion(self, masks: torch.Tensor, kernel_size: int) -> torch.Tensor:
        return -self.max_pool2d_same_dim(-masks, kernel_size=kernel_size)

    def apply_dilation(self, masks: torch.Tensor, kernel_size: int) -> torch.Tensor:
        return self.max_pool2d_same_dim(masks, kernel_size=kernel_size)

    def max_pool2d_same_dim(self, masks: torch.Tensor, kernel_size: int):
        stride = 1
        dilation = 1
        pad_h_top, pad_h_bottom = self.compute_padding(
            kernel_size, stride, dilation)
        pad_w_left, pad_w_right = self.compute_padding(
            kernel_size, stride, dilation)

        padded_input = F.pad(
            masks, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom))

        return F.max_pool2d(
            padded_input, kernel_size=kernel_size, stride=stride, dilation=dilation
        )

    def compute_padding(self, kernel_size, stride=1, dilation=1):
        padding_total = max(0, (kernel_size - 1) * dilation - stride + 1)
        pad_before = padding_total // 2
        pad_after = padding_total - pad_before
        return pad_before, pad_after

    def apply_morphological_closing(
        self, masks: torch.Tensor, kernel_size: int
    ) -> torch.Tensor:
        dilated = self.apply_dilation(masks, kernel_size)
        return self.apply_erosion(dilated, kernel_size)
