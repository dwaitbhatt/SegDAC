import torch
from tensordict import TensorDict


class MaskCoordsExtractor:
    def extract(self, binary_masks: torch.Tensor) -> TensorDict:
        """
        Inputs:
            binary_masks: (N,1,H,W) torch.uint8
        Outputs:
            masks_absolute_bboxes: (N, 4) absolute bbox coords between (0,H-1) and (0,W-1) in (xmin, ymin, xmax, ymax) format.
            masks_normalized_bboxes: (N, 4) The normalized bbox coords in (0,1) range, in (xmin, ymin, xmax, ymax) format.
        """
        device = binary_masks.device
        binary_masks = binary_masks.squeeze(1)

        batch_indices, non_zero_i_indices, non_zero_j_indices = (
            self.get_masks_non_zero_indices(binary_masks)
        )

        N, H, W = binary_masks.shape

        masks_min_i = self.scatter_reduce(
            number_of_masks=N,
            initial_value=H - 1,
            device=device,
            index=batch_indices,
            values=non_zero_i_indices,
            reduce="amin",
        ).unsqueeze(1)
        masks_max_i = self.scatter_reduce(
            number_of_masks=N,
            initial_value=0,
            device=device,
            index=batch_indices,
            values=non_zero_i_indices,
            reduce="amax",
        ).unsqueeze(1)
        masks_min_j = self.scatter_reduce(
            number_of_masks=N,
            initial_value=W - 1,
            device=device,
            index=batch_indices,
            values=non_zero_j_indices,
            reduce="amin",
        ).unsqueeze(1)
        masks_max_j = self.scatter_reduce(
            number_of_masks=N,
            initial_value=0,
            device=device,
            index=batch_indices,
            values=non_zero_j_indices,
            reduce="amax",
        ).unsqueeze(1)

        xmin = masks_min_j
        ymin = masks_min_i
        xmax = masks_max_j
        ymax = masks_max_i

        return TensorDict(
            {
                "masks_absolute_bboxes": torch.cat([xmin, ymin, xmax, ymax], dim=1),
                "masks_normalized_bboxes": torch.cat(
                    [xmin / (W - 1), ymin / (H - 1), xmax / (W - 1), ymax / (H - 1)],
                    dim=1,
                ),
            },
            batch_size=torch.Size([N]),
            device=device,
        )

    def get_masks_non_zero_indices(
        self, binary_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input: binary_masks shape : (N,H,W) One binary mask per segment, there are N segments.

        This function finds the non-zero indices (i,j) for all the masks.

        Ex:
            Mask 1 [0, 0, 0] Mask 2 [0, 1, 1]
                   [0, 1, 1]        [1, 1, 1]
                   [1, 1, 0]        [0, 0, 0]
            Output:
                batch_indices:      [0, 0, 0, 0, 1, 1, 1, 1, 1]
                non_zero_i_indices: [1, 1, 2, 2, 0, 0, 1, 1, 1]
                non_zero_j_indices: [1, 2, 0, 1, 1, 2, 0, 1, 2]

        Given there are Z non-zero pixels in binary_masks (eg: binary_masks.sum() == Z),

        Returns :
            batch_indices shape : (Z), each element has a value between [0, N-1] representing which mask the index belongs to.
            non_zero_i_indices shape : (Z), each element has a value between [0, H-1] representing the rows of the non-zero mask values.
            non_zero_j_indices shape : (Z), each element has a value between [0, W-1] representing the columns of the non-zero mask values.
        """
        return torch.where(binary_masks.bool())

    def scatter_reduce(
        self,
        number_of_masks: int,
        initial_value: int,
        device: str,
        index: torch.Tensor,
        values: torch.Tensor,
        reduce: str,
    ):
        """
        reduce=amin
            Input:
                index:  [0, 0, 1, 1, 1]
                values: [1, 0, 2, 3, 1]
            Output:
                [min([1,0]), min([2, 3, 1])] = [0, 1]

        reduce=amax
            Input:
                index:  [0, 0, 1, 1, 1]
                values: [1, 0, 2, 3, 1]
            Output:
                [max([1,0]), max([2, 3, 1])] = [1, 3]
        """
        result = torch.full(
            (number_of_masks,), initial_value, device=device, dtype=torch.long
        )
        result.scatter_reduce_(dim=0, index=index, src=values, reduce=reduce)
        return result
