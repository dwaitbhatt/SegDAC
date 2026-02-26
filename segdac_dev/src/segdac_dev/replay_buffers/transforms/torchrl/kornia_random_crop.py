import kornia
import torch
import torch.nn.functional as F
from torchrl.envs.transforms import Transform


class KorniaRandomCropTransform(Transform):
    def __init__(
        self,
        in_keys: list[str],
        out_keys: list[str],
        pad: int,
        include_next: bool = False,
    ):
        if include_next:
            in_keys_formatted = []
            for k in in_keys:
                in_keys_formatted.append(k)
                in_keys_formatted.append(("next", k))
            out_keys_formatted = []
            for k in out_keys:
                out_keys_formatted.append(k)
                out_keys_formatted.append(("next", k))
        else:
            in_keys_formatted = in_keys
            out_keys_formatted = out_keys
        super().__init__(in_keys=in_keys_formatted, out_keys=out_keys_formatted)
        self.pad = pad

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 7:  # (b, 1, frame_stack, nb_segs, c, h, w)
            b, _, frame_stack, nb_segs, c, h, w = obs.shape
            original_b = b
            obs = obs.reshape(b * nb_segs, 1, frame_stack, c, h, w)
            has_segs_dim = True
        else:
            has_segs_dim = False

        h, w = obs.shape[-2], obs.shape[-1]
        b = obs.shape[0]
        frame_stack = obs.shape[2]
        c = obs.shape[3]
        obs = F.pad(
            obs.squeeze(1).reshape(b, frame_stack * c, h, w),
            (self.pad, self.pad, self.pad, self.pad),
            mode="replicate",
        )
        obs = (kornia.augmentation.RandomCrop((h, w))(obs / 255.0) * 255.0).to(
            torch.uint8
        )
        new_h = obs.shape[-2]
        new_w = obs.shape[-1]
        if has_segs_dim:
            obs = obs.reshape(original_b, 1, frame_stack, nb_segs, c, new_h, new_w)
        else:
            obs = obs.reshape(b, 1, frame_stack, c, new_h, new_w)
        return obs
