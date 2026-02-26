import torch
import math
import numpy as np
import cv2
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from einops import rearrange
from tensordict import TensorDict
from torchvision.utils import make_grid
from torchvision.utils import draw_bounding_boxes
from matplotlib.patches import Rectangle

def draw_segments_contours(
    predictions: TensorDict,
    image_id: int,
    no_prediction_fill_value: int = 255,
    unscale: bool = True,
    draw_bboxes: bool = False,
) -> np.ndarray:

    image_id_selection_mask = predictions["image_ids"] == image_id

    image_predictions = predictions[image_id_selection_mask]

    image_size = image_predictions["rgb_segments"][0].shape[-1]

    output_image = (
        np.ones((image_size, image_size, 3), dtype=np.uint8) * no_prediction_fill_value
    )

    for i in range(image_predictions.shape[0]):

        binary_mask = (
            rearrange(image_predictions["binary_masks"][i], "c h w -> h w c")
            .cpu()
            .numpy()
        )

        rgb_segment = (
            rearrange(image_predictions["rgb_segments"][i], "c h w -> h w c")
            .cpu()
            .numpy()
        )
        if unscale:
            rgb_segment = (rgb_segment * 255).astype(np.uint8)

        output_image = np.where(binary_mask == True, rgb_segment, output_image)

        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        cv2.drawContours(output_image, contours, -1, (255, 0, 0), thickness=1)

    if draw_bboxes:
        output_image = (
            draw_bounding_boxes(
                torch.as_tensor(output_image).permute(2, 0, 1),
                image_predictions["coords"]["masks_absolute_bboxes"].cpu(),
                width=1,
            )
            .permute(1, 2, 0)
            .numpy()
        )

    return output_image


def unnormalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).reshape(
        1, 3, 1, 1
    )
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).reshape(
        1, 3, 1, 1
    )
    unnormalized = (tensor * imagenet_std) + imagenet_mean
    unnormalized = torch.clamp(unnormalized, 0, 1)
    return unnormalized


def draw_rgb_segments_grid(
    predictions: TensorDict,
    image_id: int,
    figsize: tuple = (16, 16),
    show: bool = True,
    unnormalize_fn=None,
    unscale: bool = False,
) -> list[Figure]:
    image_id_selection_mask = predictions["image_ids"] == image_id
    image_predictions = predictions[image_id_selection_mask]

    figures = []

    rgb_segments = image_predictions["rgb_segments"]

    if unnormalize_fn is not None:
        rgb_segments = unnormalize_fn(rgb_segments)

    if unscale:
        rgb_segments = (rgb_segments * 255).to(torch.uint8)

    pad_size = 4

    N, C, H, W = rgb_segments.shape
    H_padded = H + 2 * pad_size
    W_padded = W + 2 * pad_size

    r = 1
    g = 0
    b = 0

    if unscale:
        r *= 255
        g *= 255
        b *= 255

    imgs_padded = torch.zeros(
        (N, C, H_padded, W_padded), dtype=rgb_segments.dtype, device=rgb_segments.device
    )
    imgs_padded[:, 0, :, :] = r
    imgs_padded[:, 1, :, :] = g
    imgs_padded[:, 2, :, :] = b

    # Insert the original segment in the middle of the bordered image
    imgs_padded[:, :, pad_size:-pad_size, pad_size:-pad_size] = rgb_segments

    ncols = int(math.ceil(math.sqrt(N)))

    grid_img = make_grid(imgs_padded, nrow=ncols, padding=0)

    grid_img = grid_img.detach().cpu()
    grid_img = F.to_pil_image(grid_img)
    grid_img = np.array(grid_img)

    if show:
        fig = plt.figure(figsize=figsize)
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(grid_img)

        figures.append(fig)

        plt.close(fig)

        return figures
    else:
        return grid_img


def draw_q_value_segments_contribs_quantiles_heatmap(
    q_value_segments_contribs: torch.Tensor,
    segments_binary_masks: torch.Tensor,
    num_bins: int,
    thresholds_excluding_min_max: torch.Tensor,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    """
    Inputs:
        q_value_segments_contribs: shape (N,)
        segments_binary_masks: shape (N, H, W)
        num_bins: number of bins to create
        thresholds_excluding_min_max: shape (num_bins - 1), should be sorted ex: [-8.0, -6.0, -4.0, -2.0,  0.0,  2.0,  4.0,  6.0,
            8.0] if we have data in range [-10.0, 10.0]
        lower_bound: lower bound of the data range ex: -10.0
        upper_bound: upper bound of the data range ex: 10.0
    Outputs:
        img: heatmap image with segments contribution values shape (H, W, 3)
    """
    N, H, W = segments_binary_masks.shape

    """
    Example:
    Index 0 Bin 1 :  [-10.0, -8.0[ 
    Index 1 Bin 2 :  [ -8.0, -6.0[
    Index 2 Bin 3 :  [ -6.0, -4.0[
    Index 3 Bin 4 :  [ -4.0, -2.0[
    Index 4 Bin 5 :  [ -2.0,  0.0[
    Index 5 Bin 6 :  [  0.0,  2.0[
    Index 6 Bin 7 :  [  2.0,  4.0[
    Index 7 Bin 8 :  [  4.0,  6.0[
    Index 8 Bin 9 :  [  6.0,  8.0[
    Index 9 Bin 10 : [  8.0, 10.0[

    q_contrib: 
        torch.tensor([0.1, 10.0, 5.0, -6.0, -8.0, 0.5])
        Bin index :    5    9     7     2    1     5
    """
    bin_indices = torch.bucketize(
        q_value_segments_contribs, thresholds_excluding_min_max, right=True
    )

    heatmap_img = torch.full((H, W), float("nan"))
    for seg_idx in range(N):
        mask = segments_binary_masks[seg_idx].bool()
        segment_bin_index = bin_indices[seg_idx].item()
        heatmap_img[mask] = segment_bin_index

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("Q-Value Segments Contributions Heatmap", fontsize=14, weight="bold")
    ax.set_axis_off()

    cmap = plt.get_cmap("RdBu", num_bins)
    cmap.set_bad(color="lightgray")
    norm = Normalize(vmin=-0.5, vmax=num_bins - 0.5)  # To map bin indices to colors

    im = ax.imshow(heatmap_img.numpy(), cmap=cmap, norm=norm, interpolation="nearest")

    # Create Custom Colorbar
    bin_edges = [lower_bound] + thresholds_excluding_min_max.tolist() + [upper_bound]
    tick_labels = []
    for i in range(num_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        if i == num_bins - 1:
            label = f"[{low:.2f}, {high:.2f}]"
        else:
            label = f"[{low:.2f}, {high:.2f}["
        tick_labels.append(label)
    cbar_ticks = np.arange(num_bins)
    cbar = fig.colorbar(im, ticks=cbar_ticks, spacing="proportional", ax=ax)
    cbar.ax.set_yticklabels(tick_labels, fontsize=9)
    cbar.set_label("Q-Value Contribution Range", labelpad=20)

    stats_text = (
        f"Sum: {q_value_segments_contribs.sum():.2f}\n"
        f"Mean: {q_value_segments_contribs.mean():.2f}\n"
        f"Max: {q_value_segments_contribs.max():.2f}\n"
        f"Min: {q_value_segments_contribs.min():.2f}"
    )
    props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.5)
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    for seg_idx in range(N):
        mask = segments_binary_masks[seg_idx].bool().numpy()
        indices = np.argwhere(mask)
        if indices.size > 0:
            y_center, x_center = indices.mean(axis=0)
            ax.text(
                x_center,
                y_center,
                f"{q_value_segments_contribs[seg_idx]:.2f}",
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="black",
                    alpha=0.6,
                    edgecolor="none",
                ),
            )
    fig.tight_layout()
    fig.canvas.draw()
    img_rgba = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img_rgba = img_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img_rgba[..., 1:4]  # Convert ARGB to RGB

    plt.close(fig)

    img_copy = img.copy()

    del img

    return img_copy


def draw_q_value_segments_attn_heatmap(
    q_value: torch.Tensor,
    q_value_segments_attn: torch.Tensor,
    q_value_proprio_attn: torch.Tensor,
    segments_binary_masks: torch.Tensor,
    figsize: tuple = (6.5, 5.0)
) -> np.ndarray:
    """
    Inputs:
        q_value: (1, 1) or scalar - Q-value for the current state/observation.
        q_value_segments_attn: (N,) - Attention scores for N segments.
        q_value_proprio_attn: (1, 1), (1,), or scalar or None - Attention score for proprioception.
        segments_binary_masks: (N, H, W) - Binary masks for N segments.
        figsize: tuple - Figure size in inches (width, height).
    Outputs:
        img: heatmap image with shape (H_fig, W_fig, 3) representing the plotted figure.
    """
    N, H, W = segments_binary_masks.shape

    q_value_scalar = q_value.item() if torch.is_tensor(q_value) else q_value

    num_bins = 20
    thresholds_excluding_min_max = torch.linspace(0.0, 1.0, num_bins + 1)[1:num_bins]

    bin_indices_segments = torch.bucketize(
        q_value_segments_attn, thresholds_excluding_min_max, right=True
    )

    if q_value_proprio_attn is not None:
        proprio_attn_scalar = q_value_proprio_attn.item() if torch.is_tensor(q_value_proprio_attn) else q_value_proprio_attn
        proprio_attn_tensor = torch.tensor([proprio_attn_scalar]) if not isinstance(proprio_attn_scalar, torch.Tensor) else proprio_attn_scalar.flatten()
        bin_index_proprio = torch.bucketize(
            proprio_attn_tensor, thresholds_excluding_min_max, right=True
        ).item()

    heatmap_img_tensor = torch.full((H, W), float("nan"))
    for seg_idx in range(N):
        mask = segments_binary_masks[seg_idx].bool()
        segment_bin_index = bin_indices_segments[seg_idx].item()
        heatmap_img_tensor[mask] = segment_bin_index

    fig = plt.figure(figsize=figsize)

    left_margin = 0.0
    right_margin = 0.0
    bottom_margin = 0.01
    top_margin_for_suptitle_region = 0.01
    padding_infobar_suptitle_fig = 0.006
    top_margin_for_infobar_region = 0.05
    right_panel_strip_width = 0.10

    # Heatmap dimensions
    ax_heatmap_bottom = bottom_margin
    ax_heatmap_height = 1.0 - bottom_margin - top_margin_for_suptitle_region - padding_infobar_suptitle_fig - top_margin_for_infobar_region
    ax_heatmap_left = left_margin
    ax_heatmap_width = 1.0 - left_margin - right_panel_strip_width - right_margin
    heatmap_top_y = ax_heatmap_bottom + ax_heatmap_height

    # Main Title (Suptitle)
    suptitle_y_pos = heatmap_top_y + top_margin_for_infobar_region + padding_infobar_suptitle_fig + (top_margin_for_suptitle_region / 2.0)
    fig.suptitle("Q-Value Attention Heatmap", fontsize=12, weight="bold", y=suptitle_y_pos)

    # Info Bar
    info_text = (
        f"Q-Value: {q_value_scalar:.3f}  |  "
        f"Seg Attn (n={q_value_segments_attn.numel()}): "
        f"Mean: {q_value_segments_attn.mean():.2f} "
        f"Std: {q_value_segments_attn.std():.2f} "
        f"Max: {q_value_segments_attn.max():.2f} "
        f"Min: {q_value_segments_attn.min():.2f}"
    )
    info_props = dict(boxstyle="round,pad=0.15", facecolor="whitesmoke", alpha=0.95, edgecolor='darkgray')
    info_bar_y_pos = heatmap_top_y + 0.002
    fig.text(
        ax_heatmap_left + ax_heatmap_width / 2, info_bar_y_pos, info_text,
        ha="center", va="bottom", fontsize=6.5, bbox=info_props
    )

    # Heatmap Axes
    ax_heatmap = fig.add_axes([ax_heatmap_left, ax_heatmap_bottom, ax_heatmap_width, ax_heatmap_height])
    ax_heatmap.set_axis_off()
    cmap = plt.get_cmap("viridis", num_bins)
    cmap.set_bad(color="lightgray")
    norm = Normalize(vmin=-0.5, vmax=num_bins - 0.5)
    im = ax_heatmap.imshow(heatmap_img_tensor.numpy(), cmap=cmap, norm=norm, interpolation="nearest")

    # Text for individual segment attention values
    for seg_idx in range(N):
        mask_np = segments_binary_masks[seg_idx].bool().numpy()
        indices = np.argwhere(mask_np)
        if indices.size > 0:
            y_center, x_center = indices.mean(axis=0)
            ax_heatmap.text(
                x_center, y_center, f"{q_value_segments_attn[seg_idx]:.2f}", color="white",
                fontsize=6, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.6, edgecolor="none")
            )

    right_panel_x_start = ax_heatmap_left + ax_heatmap_width - 0.1
    right_panel_usable_height = ax_heatmap_height
    right_panel_usable_bottom = ax_heatmap_bottom

    # Colorbar
    colorbar_rel_h_in_strip = 0.58
    cbar_h = right_panel_usable_height * colorbar_rel_h_in_strip
    cbar_y = right_panel_usable_bottom + right_panel_usable_height * (1 - colorbar_rel_h_in_strip)
    cbar_w = 0.03 # Width of colorbar itself in figure fraction
    cbar_x = right_panel_x_start
    cax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
    bin_edges = [0.0] + thresholds_excluding_min_max.tolist() + [1.0]
    tick_labels = [f"[{le:.2f},{he:.2f}{']' if i == num_bins-1 else ')'}" for i, (le, he) in enumerate(zip(bin_edges[:-1], bin_edges[1:]))]
    cbar_ticks = np.arange(num_bins)
    cbar = fig.colorbar(im, cax=cax, ticks=cbar_ticks, spacing="proportional", orientation="vertical")
    cbar.ax.set_yticklabels(tick_labels, fontsize=6)
    cbar.set_label("Attention Bins", labelpad=5, fontsize=7, loc="center")
    cbar.ax.minorticks_off()

    # Axes for Proprioception
    bars_rel_h_in_strip = 1.0 - colorbar_rel_h_in_strip
    bars_h = right_panel_usable_height * bars_rel_h_in_strip * 0.96
    bars_y = right_panel_usable_bottom
    ax_bars = fig.add_axes([right_panel_x_start, bars_y, right_panel_strip_width, bars_h])
    ax_bars.set_axis_off()
    ax_bars.set_xlim(0,1)
    ax_bars.set_ylim(0,1)

    bar_h_in_ax = 0.25
    txt_fs = 6.0
    lbl_fs = 6.5
    lbl_y_off = 0.03

    # Desired absolute width of the bar's colored rectangle (same as colorbar)
    bar_color_actual_width_fig = cbar_w
    # Convert this to be relative to ax_bars's width
    bar_color_width_rel_to_ax_bars = bar_color_actual_width_fig / right_panel_strip_width + 0.4

    # Desired left padding for the bar within ax_bars (in figure coordinates)
    bar_left_padding_fig_coords = 0.025 # Align with the colorbar's left edge

    # Convert this padding to be relative to ax_bars's width
    bar_x_pos_rel_to_ax_bars = bar_left_padding_fig_coords / right_panel_strip_width

    # Horizontal center of the bar, relative to ax_bars
    bar_center_x_rel_to_ax_bars = bar_x_pos_rel_to_ax_bars + (bar_color_width_rel_to_ax_bars / 2.0)

    # Proprioception Bar
    if q_value_proprio_attn is not None:
        prop_y_center = 0.68
        prop_col = cmap(norm(bin_index_proprio))
        prop_rect = Rectangle(
            (bar_x_pos_rel_to_ax_bars, prop_y_center - bar_h_in_ax/2),
            bar_color_width_rel_to_ax_bars,
            bar_h_in_ax,
            color=prop_col, ec='black', transform=ax_bars.transAxes
        )
        ax_bars.add_patch(prop_rect)
        ax_bars.text(
            bar_center_x_rel_to_ax_bars, prop_y_center, f"{proprio_attn_scalar:.2f}",
            color="white" if sum(prop_col[:3]) < 1.5 else "black",
            fontsize=txt_fs, ha="center", va="center", weight="bold", transform=ax_bars.transAxes
        )
        ax_bars.text(
            bar_center_x_rel_to_ax_bars, prop_y_center + bar_h_in_ax/2 + lbl_y_off, "Proprio Attn",
            fontsize=lbl_fs, ha="center", va="bottom", weight="bold", transform=ax_bars.transAxes
        )

    fig.canvas.draw()
    img_rgba = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img_rgba = img_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_rgb = img_rgba[..., 1:4]

    plt.close(fig)
    return img_rgb.copy()