import torch
from tensordict import tensorclass
from tensordict import TensorDict
from typing import Optional


@tensorclass
class MdpData:
    """
    data: TensorDict with all the data for the current MDP step (eg: observations, actions, rewards, etc), batch size is the number of images in the batch
    segmentation_data: Optional[TensorDict] with the segmentation data for the current MDP step, batch size is the number of segments in the batch
    next: Optional[MdpData] with the data for the next MDP step
    """

    data: TensorDict
    segmentation_data: Optional[TensorDict] = None
    next: Optional["MdpData"] = None

    @property
    def device(self) -> str:
        return self.data.device

    def step(self) -> "MdpData":
        if self.next is None:
            return self

        data = self.next.data.exclude("reward")

        if self.segmentation_data is not None:
            segmentation_data = self.next.segmentation_data.copy()
        else:
            segmentation_data = None

        next = self.next.next

        return MdpData(data=data, segmentation_data=segmentation_data, next=next)

    @staticmethod
    def stack(mdp_data_list: list["MdpData"], dim: int = 0) -> "MdpData":
        """
        Stacks a list of MdpData objects along a new dimension.

        Assumes that if optional fields (segmentation_data, next.segmentation_data)
        are present in the first element, they are consistently present (non-None)
        in all other elements (user guarantee).

        Args:
            mdp_data_list: The list of MdpData objects to stack.
            dim: The dimension along which to stack.

        Returns:
            A new MdpData object with stacked tensors, or None if the input list is empty.
        """
        assert len(mdp_data_list) > 0

        data_list = [item.data for item in mdp_data_list]
        stacked_data = torch.stack(data_list, dim=dim)

        stacked_segmentation_data = None
        if mdp_data_list[0].segmentation_data is not None:
            seg_data_list = [item.segmentation_data for item in mdp_data_list]
            stacked_segmentation_data = torch.stack(seg_data_list, dim=dim)

        stacked_next_mdp = None
        if mdp_data_list[0].next is not None:
            next_data_list = [item.next.data for item in mdp_data_list]
            stacked_next_data = torch.stack(next_data_list, dim=dim)

            stacked_next_segmentation_data = None
            if mdp_data_list[0].next.segmentation_data is not None:
                next_seg_data_list = [
                    item.next.segmentation_data for item in mdp_data_list
                ]
                stacked_next_segmentation_data = torch.stack(
                    next_seg_data_list, dim=dim
                )

            stacked_next_mdp = MdpData(
                data=stacked_next_data, segmentation_data=stacked_next_segmentation_data
            )

        return MdpData(
            data=stacked_data,
            segmentation_data=stacked_segmentation_data,
            next=stacked_next_mdp,
        )

    @staticmethod
    def cat(mdp_data_list: list["MdpData"], dim: int = 0) -> "MdpData":
        """
        Concatenates a list of MdpData objects along an existing dimension.

        Assumes that if optional fields (segmentation_data, next.segmentation_data)
        are present in the first element, they are consistently present (non-None)
        in all other elements (user guarantee).

        Args:
            mdp_data_list: The list of MdpData objects to concatenate.
            dim: The dimension along which to concatenate.

        Returns:
            A new MdpData object with concatenated tensors, or None if the input list is empty.
        """
        assert len(mdp_data_list) > 0

        data_list = [item.data for item in mdp_data_list]
        concatenated_data = torch.cat(data_list, dim=dim)

        concatenated_segmentation_data = None
        if mdp_data_list[0].segmentation_data is not None:
            seg_data_list = [item.segmentation_data for item in mdp_data_list]
            concatenated_segmentation_data = torch.cat(seg_data_list, dim=dim)

        concatenated_next_mdp = None
        if mdp_data_list[0].next is not None:  # Assumes consistency
            next_data_list = [item.next.data for item in mdp_data_list]
            concatenated_next_data = torch.cat(next_data_list, dim=dim)

            concatenated_next_segmentation_data = None
            if mdp_data_list[0].next.segmentation_data is not None:
                next_seg_data_list = [
                    item.next.segmentation_data for item in mdp_data_list
                ]
                concatenated_next_segmentation_data = torch.cat(
                    next_seg_data_list, dim=dim
                )

            concatenated_next_mdp = MdpData(
                data=concatenated_next_data,
                segmentation_data=concatenated_next_segmentation_data,
            )

        return MdpData(
            data=concatenated_data,
            segmentation_data=concatenated_segmentation_data,
            next=concatenated_next_mdp,
        )

    def to(self, device: str, non_blocking: bool = True) -> "MdpData":
        data = self.data.to(device=device, non_blocking=non_blocking)
        if self.segmentation_data is not None:
            segmentation_data = self.segmentation_data.to(
                device=device, non_blocking=non_blocking
            )
        else:
            segmentation_data = None

        if self.next is not None:
            next = self.next.to(device=device, non_blocking=non_blocking)
        else:
            next = None

        torch.cuda.synchronize()

        return MdpData(data=data, segmentation_data=segmentation_data, next=next)
