# Run from the SegDAC repository root so ./weights/ resolves (see install_dependencies.sh).
"""
ManiSkill vs SegDAC, aligned with mani_skill/examples/demo_vis_segmentation.py:

- The RGB + simulator segmentation are always the current state of the running env.
- **Interactive (default):** a Matplotlib window shows a 4-in-1 panel (like tiling in the
  demo). Press **SPACE** to apply a **random** `env.action_space.sample()` step
  (same as the ManiSkill demo) and update the view with SegDAC on the **new** frame.
  **Q** or **Esc** quits. Run with `--one-shot` for a single static frame, save, exit.
- **Benchmark:** use `--fps` to measure throughput of `env.step` + full SegDAC (grounded
  segment + SAM-encoder object tokens) with no matplotlib.

SegDAC runs on RGB at the **segmenter spatial size** (512 for EfficientViT-SAM
l0–l2, see `GroundedEfficientVitSam.get_segmenter_image_size`). The default
camera is **512×512** so RGB matches that resolution; other sizes are bilinearly
resized for SegDAC. YOLO-World still uses 640×640 internally. ManiSkill panels
use native camera H×W; SegDAC panels match when H=W=512, otherwise outputs are
resampled for the mosaic.
"""

import argparse
import signal
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mani_skill.utils import common
from mani_skill.utils.structs import Actor, Link
from torchvision.utils import draw_bounding_boxes

from segdac.networks.image_segmentation_models.grounded_efficientvit_sam import (
    GroundedEfficientVitSam,
    get_image_covered_by_predicted_masks,
)
from segdac.networks.segments_encoders.sam_encoder_segments_encoder import (
    SamEncoderEmbeddingsSegmentsEncoder,
)

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c, like the ManiSkill demo

# color palette from ManiSkill demo_vis_segmentation.py
color_pallete = np.array(
    [
        [164, 74, 82],
        [85, 200, 95],
        [149, 88, 210],
        [111, 185, 57],
        [89, 112, 223],
        [194, 181, 43],
        [219, 116, 216],
        [71, 146, 48],
        [214, 70, 164],
        [157, 183, 57],
        [154, 68, 158],
        [82, 196, 133],
        [225, 64, 121],
        [50, 141, 77],
        [224, 59, 84],
        [74, 201, 189],
        [237, 93, 68],
        [77, 188, 225],
        [182, 58, 29],
        [77, 137, 200],
        [230, 155, 53],
        [93, 90, 162],
        [213, 106, 38],
        [150, 153, 224],
        [120, 134, 37],
        [186, 135, 220],
        [78, 110, 27],
        [182, 61, 117],
        [106, 184, 145],
        [184, 62, 65],
        [44, 144, 124],
        [229, 140, 186],
        [48, 106, 60],
        [167, 102, 155],
        [160, 187, 114],
        [150, 74, 107],
        [204, 177, 86],
        [34, 106, 77],
        [226, 129, 94],
        [72, 106, 45],
        [222, 125, 129],
        [101, 146, 86],
        [150, 89, 44],
        [147, 138, 73],
        [210, 156, 106],
        [102, 96, 32],
        [168, 124, 34],
    ],
    np.uint8,
)


def maniskill_seg_to_color_rgb(
    obs_seg_1env: np.ndarray, selected_id: int | None
) -> np.ndarray:
    """
    (H, W, 1) int ids, same as demo. If selected_id, mask to that id first (see demo).
    """
    seg = obs_seg_1env.astype(np.int64)
    if selected_id is not None:
        seg = seg == int(selected_id)
    seg = np.remainder(seg, len(color_pallete))
    seg_rgb = np.zeros(
        (obs_seg_1env.shape[0], obs_seg_1env.shape[1], 3), dtype=np.uint8
    )
    for seg_id, color in enumerate(color_pallete):
        seg_rgb[seg[..., 0] == seg_id] = color
    return seg_rgb


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ManiSkill vs SegDAC. Interactive: SPACE=random step like demo, Q=quit. "
        "Or --one-shot."
    )
    p.add_argument("-e", "--env-id", type=str, default="PickCube-v1")
    p.add_argument(
        "--id",
        type=str,
        default=None,
        help="Only highlight this id (int) or actor/link name (see printed map).",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--cam-width",
        type=int,
        default=512,
        help="Sensor width (default 512 = native segmenter res for efficientvit_sam l0-l2).",
    )
    p.add_argument(
        "--cam-height",
        type=int,
        default=512,
        help="Sensor height (default 512 = native segmenter res for efficientvit_sam l0-l2).",
    )
    p.add_argument("-s", "--seed", type=int, default=0)
    p.add_argument(
        "--grounding",
        type=str,
        action="append",
        default=None,
    )
    p.add_argument(
        "--save",
        type=str,
        default="test_maniskill_seg_compare.png",
        help="--one-shot only: output image path",
    )
    p.add_argument(
        "--one-shot",
        action="store_true",
        help="No loop: one optional random step, build figure, save, exit",
    )
    p.add_argument(
        "--no-random-step",
        action="store_true",
        help="--one-shot: use only reset frame (no extra env.step)",
    )
    p.add_argument(
        "--fps",
        action="store_true",
        help="Benchmark only: time env.step + SegDAC (segment + object-token encoder); no plots",
    )
    p.add_argument(
        "--fps-steps",
        type=int,
        default=100,
        help="Number of timed iterations for --fps (default 100)",
    )
    p.add_argument(
        "--fps-warmup",
        type=int,
        default=5,
        help="Untimed warmup steps before --fps measurement (default 5)",
    )
    return p


@torch.inference_mode()
def extract_object_tokens(
    pixels_01: torch.Tensor,
    device: str,
    segmentation_model: GroundedEfficientVitSam,
    segments_encoder: SamEncoderEmbeddingsSegmentsEncoder,
) -> tuple:
    pixels_01 = pixels_01.to(device)
    segments_data, sam_enc = segmentation_model.segment(
        pixels_01, return_sam_encoder_embeddings=True
    )
    sam_for_print = sam_enc
    sam_enc = sam_enc.unsqueeze(1)
    segments_encoder_out = segments_encoder(segments_data, sam_enc)
    return segments_data, segments_encoder_out, sam_for_print


@torch.inference_mode()
def forward_segdac_object_tokens(
    obs: dict,
    cam: str,
    device: str,
    seg_model: GroundedEfficientVitSam,
    seg_enc: SamEncoderEmbeddingsSegmentsEncoder,
):
    """RGB from obs -> segmenter res -> Grounded SAM + token encoder (no viz / numpy)."""
    rgb = obs["sensor_data"][cam]["rgb"]
    pixels_01 = rgb.permute(0, 3, 1, 2).contiguous().float() / 255.0
    pixels_seg = pixels_to_segmenter_res(pixels_01, device, seg_model)
    _sd, out, _sraw = extract_object_tokens(
        pixels_seg, device, seg_model, seg_enc
    )
    return out["embeddings"]


def run_fps_benchmark(
    env: gym.Env,
    obs: dict,
    cam: str,
    device: str,
    seg_model: GroundedEfficientVitSam,
    seg_enc: SamEncoderEmbeddingsSegmentsEncoder,
    n_warmup: int,
    n_steps: int,
) -> None:
    """Time: random env.step, then full SegDAC token extraction; prints FPS."""
    for _ in range(n_warmup):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        _ = forward_segdac_object_tokens(obs, cam, device, seg_model, seg_enc)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        emb = forward_segdac_object_tokens(
            obs, cam, device, seg_model, seg_enc
        )
        _ = emb.shape  # keep result live
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    if elapsed <= 0:
        print("FPS: (elapsed time too small to measure)")
        return
    fps = n_steps / elapsed
    ms = 1000.0 * elapsed / n_steps
    print(
        f"env.step + SegDAC (grounded YOLO + SAM + object-token head): {fps:.2f} FPS  "
        f"({n_steps} timed steps in {elapsed:.3f} s, {ms:.2f} ms / frame)"
    )


def segdac_masks_to_instance_rgb(
    binary_masks: np.ndarray, height: int, width: int
) -> np.ndarray:
    n = binary_masks.shape[0]
    out = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(n):
        c = color_pallete[i % len(color_pallete)].astype(np.float32) / 255.0
        m = binary_masks[i, 0] > 0.5
        out[m] = c
    return out


def find_first_usable_camera(obs) -> str | None:
    for cam in obs["sensor_data"].keys():
        s = obs["sensor_data"][cam]
        if "rgb" in s and "segmentation" in s:
            return cam
    return None


def to_uint8_image(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def pixels_to_segmenter_res(
    pixels_01: torch.Tensor, device: str, seg_model: GroundedEfficientVitSam
) -> torch.Tensor:
    """(B,3,h,w) in [0,1] -> (B,3,S,S) where S = seg_model.segmenter_image_size (e.g. 512)."""
    s = int(seg_model.segmenter_image_size)
    x = pixels_01.to(device)
    if int(x.shape[2]) == s and int(x.shape[3]) == s:
        return x
    return F.interpolate(
        x, size=(s, s), mode="bilinear", align_corners=False
    )


def render_yolo_world_detections(
    rgb_u8: np.ndarray,
    xyxy_list: list,
    classes_list: list,
    batch_index: int,
    grounding_tags: list[str],
) -> np.ndarray:
    """
    YOLO-World boxes are in 640x640 preprocessed image space; map to native (H,W) of rgb_u8.
    """
    h, w = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
    xyxy = xyxy_list[batch_index]
    cls_t = classes_list[batch_index]
    if xyxy is None or xyxy.numel() == 0:
        return rgb_u8.copy()
    # YOLO-World is run on 640x640 in GroundedEfficientVitSam (object_detection_img_size)
    scale = torch.tensor([w, h, w, h], dtype=xyxy.dtype, device=xyxy.device) / 640.0
    boxes = (xyxy * scale).round()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, float(w - 1))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, float(h - 1))
    labels: list[str] = []
    for c in cls_t.long().flatten():
        ci = int(c.item())
        if 0 <= ci < len(grounding_tags):
            labels.append(grounding_tags[ci])
        else:
            labels.append(f"class_{ci}")
    img = torch.from_numpy(rgb_u8).permute(2, 0, 1).contiguous()
    out = draw_bounding_boxes(
        img,
        boxes.cpu(),
        labels=labels,
        width=max(1, min(3, w // 128)),
    )
    return out.permute(1, 2, 0).contiguous().numpy().astype(np.uint8)


@torch.inference_mode()
def run_pipeline_on_obs(
    obs: dict,
    cam: str,
    device: str,
    seg_model: GroundedEfficientVitSam,
    seg_enc: SamEncoderEmbeddingsSegmentsEncoder,
    selected_id: int | None,
    grounding_tags: list[str],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int, int
]:
    """
    Return rgb_u8, ms_u8, yolo_viz_u8, segdac_inst_u8, union_u8, print_block, h, w.
    SegDAC forward uses RGB upsampled to seg_model.segmenter_image_size; outputs for
    the last two panels are resampled to native (h, w) to align with ManiSkill columns.
    """
    rgb = obs["sensor_data"][cam]["rgb"]
    h, w = int(rgb.shape[1]), int(rgb.shape[2])
    pixels_01 = rgb.permute(0, 3, 1, 2).contiguous().float() / 255.0

    seg_t = obs["sensor_data"][cam]["segmentation"][0]
    seg_np = common.to_numpy(seg_t)
    ms = maniskill_seg_to_color_rgb(seg_np, selected_id)
    rgb_n = common.to_numpy(rgb[0])

    pixels_seg = pixels_to_segmenter_res(pixels_01, device, seg_model)
    sd, out, sraw = extract_object_tokens(
        pixels_seg, device, seg_model, seg_enc
    )
    yolo_viz = render_yolo_world_detections(
        rgb_n,
        seg_model.last_yolo_xyxy,
        seg_model.last_yolo_classes,
        0,
        grounding_tags,
    )
    u = get_image_covered_by_predicted_masks(pixels_seg, sd)
    if u.shape[2] != h or u.shape[3] != w:
        u = F.interpolate(u, size=(h, w), mode="bilinear", align_corners=False)
    un = common.to_numpy(u[0].permute(1, 2, 0).clamp(0, 1))

    bm_t = sd["binary_masks"].float()
    if int(bm_t.shape[-2]) != h or int(bm_t.shape[-1]) != w:
        bm_t = F.interpolate(bm_t, size=(h, w), mode="nearest")
    bm = common.to_numpy(bm_t)
    sgi = to_uint8_image(segdac_masks_to_instance_rgb(bm, h, w))

    emb = out["embeddings"]
    s_sz = int(seg_model.segmenter_image_size)
    lines = [
        "  --- SegDAC object token / segment tensor shapes ---",
        f"  (forward at segmenter res {s_sz}x{s_sz}; ManiSkill panels are {h}x{w})",
        f"  segments_encoder_output['embeddings'] (object tokens): {tuple(emb.shape)}  (N x D)",
        f"  binary_masks: {tuple(sd['binary_masks'].shape)}",
        f"  image_ids: {tuple(sd['image_ids'].shape)}",
        f"  classes (YOLO per mask): {tuple(sd['classes'].shape)}",
        f"  sam_encoder raw features: {tuple(sraw.shape)}",
    ]
    info = "\n".join(lines)
    return (
        rgb_n,
        ms,
        yolo_viz,
        sgi,
        to_uint8_image(un),
        info,
        h,
        w,
    )


def get_panel_titles(cam: str) -> list[str]:
    return [
        f"RGB ({cam})",
        "ManiSkill seg (id-colored)",
        "YOLO-World (text classes + boxes)",
        "SegDAC (instance masks)",
        "SegDAC union of masks × RGB",
    ]


def resolve_filter_id(
    id_arg: str | None, env, reverse_name_to_id: dict[str, int]
) -> int | None:
    if id_arg is None:
        return None
    if str(id_arg).isdigit():
        return int(id_arg)
    if id_arg not in reverse_name_to_id:
        raise SystemExit(
            f"--id {id_arg!r} not in actor/link name map. Use a printed name or an int id."
        )
    return int(reverse_name_to_id[id_arg])


def main() -> None:
    args = build_parser().parse_args()
    if args.grounding is None or len(args.grounding) == 0:
        grounding = [
            "robot arm",
            "red cube",
        ]
    else:
        grounding = list(args.grounding)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU (SegDAC will be slow).")
        device = "cpu"

    if args.seed is not None:
        np.random.seed(args.seed)

    sensor_configs: dict = {}
    if args.cam_width is not None:
        sensor_configs["width"] = args.cam_width
    if args.cam_height is not None:
        sensor_configs["height"] = args.cam_height

    env = gym.make(
        args.env_id,
        robot_uids="xarm6_robotiq",
        obs_mode="rgb+depth+segmentation",
        num_envs=1,
        sensor_configs=sensor_configs,
    )
    try:
        obs, _ = env.reset(seed=args.seed)
        cam = find_first_usable_camera(obs)
        if cam is None:
            raise RuntimeError("No camera with rgb+segmentation.")

        reverse: dict[str, int] = {}
        for obj_id, obj in sorted(env.unwrapped.segmentation_id_map.items()):
            if isinstance(obj, (Actor, Link)):
                reverse[obj.name] = int(obj_id)
        if not args.fps:
            print("ID to Actor/Link (ManiSkill); 0: Background")
            for obj_id, obj in sorted(env.unwrapped.segmentation_id_map.items()):
                if isinstance(obj, (Actor, Link)):
                    print(f"{obj_id}: {type(obj).__name__} name = {obj.name!r}")
        fil = None if args.fps else resolve_filter_id(args.id, env, reverse)

        weights = Path("weights")
        yolo = weights / "yolov8s-worldv2.pt"
        sam_w = weights / "efficientvit_sam_l0.pt"
        if not yolo.is_file() or not sam_w.is_file():
            print(
                f"Expected {yolo} and {sam_w}. "
                "Run segdac/install_dependencies.sh from the repo root."
            )

        seg_model = GroundedEfficientVitSam(
            device=device,
            grounding_text_tags=grounding,
            object_detector_weights_path=str(yolo),
            segmenter_weights_path=str(sam_w),
        )
        seg_enc = SamEncoderEmbeddingsSegmentsEncoder(512, min_pixels=4)
        if device == "cuda":
            seg_enc = seg_enc.to(device)

        # ----- FPS benchmark: no figures -----
        if args.fps:
            print(
                f"FPS mode: {args.fps_warmup} warmup steps, {args.fps_steps} timed steps, "
                f"device={device!r}, env_id={args.env_id!r}"
            )
            run_fps_benchmark(
                env,
                obs,
                cam,
                device,
                seg_model,
                seg_enc,
                n_warmup=int(args.fps_warmup),
                n_steps=int(args.fps_steps),
            )
            return

        # ----- one-shot: single static figure -----
        if args.one_shot:
            if not args.no_random_step:
                obs, _, _, _, _ = env.step(env.action_space.sample())
            r, m, yv, s, u, block, _h, _w = run_pipeline_on_obs(
                obs, cam, device, seg_model, seg_enc, fil, grounding
            )
            print("\n" + block)
            titles = get_panel_titles(cam)
            fig, axes = plt.subplots(1, 5, figsize=(26, 4.5))
            for ax, img, t in zip(axes, (r, m, yv, s, u), titles, strict=True):
                ax.imshow(img)
                ax.set_title(t, fontsize=10, pad=8)
                ax.set_axis_off()
            plt.suptitle(
                f"{args.env_id}  one-shot  seed={args.seed}", fontsize=11, y=1.04
            )
            if args.save:
                p = Path(args.save)
                p.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(p), dpi=150, bbox_inches="tight")
                print(f"Saved: {p.resolve()}")
            plt.show()
            return

        # ----- interactive: like demo, but show after SPACE and use random action on each step -----
        print(
            "Window focus required.  "
            "SPACE: random action + advance sim, then run SegDAC on the new frame.  "
            "Q / Esc: quit.  (Same random-action policy as demo_vis_segmentation.)"
        )
        plt.ion()
        panel_titles = get_panel_titles(cam)
        fig, axes = plt.subplots(1, 5, figsize=(26, 5.5))
        im_arts: list = [None] * 5
        # [0]: unblocked; [1]: should quit
        kstate = [False, False]

        def on_key(ev) -> None:
            if ev.key in (" ", "space"):
                kstate[0] = True
            if ev.key in ("q", "escape"):
                kstate[0] = True
                kstate[1] = True

        fig.canvas.mpl_connect("key_press_event", on_key)

        step_n = 0
        first = True
        while not kstate[1]:
            r, m, yv, s, u, block, _h, _w = run_pipeline_on_obs(
                obs, cam, device, seg_model, seg_enc, fil, grounding
            )
            panels = (r, m, yv, s, u)
            print(
                f"\n--- frame @ sim counter {step_n} (SegDAC + ManiSkill) ---\n" + block,
            )
            if im_arts[0] is None:
                for i, (ax, img) in enumerate(zip(axes, panels, strict=True)):
                    im_arts[i] = ax.imshow(img, aspect="auto")
                    ax.set_title(panel_titles[i], fontsize=10, pad=8)
                    ax.set_axis_off()
            else:
                for i, img in enumerate(panels):
                    im_arts[i].set_data(img)
            title = (
                f"{args.env_id}  |  SPACE: random sim.step, then refresh  |  Q: quit\n"
                f"Showing env state at internal step counter = {step_n}."
            )
            fig.suptitle(title, fontsize=9)
            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            kstate[0] = False
            if first:
                print(
                    "First view = state right after env.reset. "
                    "Press SPACE to run a random action (as in demo_vis_segmentation)."
                )
                first = False
            while not kstate[0] and not kstate[1]:
                plt.pause(0.02)
            if kstate[1]:
                break
            obs, _, term, trun, _ = env.step(env.action_space.sample())
            step_n += 1
            if bool(term) or bool(trun):
                obs, _ = env.reset(seed=args.seed)
                print("(episode end — env reset)")

        plt.ioff()
        plt.close("all")
    finally:
        env.close()


if __name__ == "__main__":
    main()
