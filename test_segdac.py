#!/usr/bin/env python3
"""
SegDACProcessor + ManiSkill demo (aligned with ``test.py``).

* **Default:** interactive Matplotlib window — **SPACE**: random ``env.step`` then refresh
  SegDAC panels; **Q** / **Esc**: quit (same stepping policy as ``demo_vis_segmentation``).
* **``--one-shot``:** single frame, save PNG, exit.
* **``--profiling``:** benchmark only (no Matplotlib): ``env.step`` + full ``SegDACProcessor``
  forward, then print mean phase times and FPS (same style as ``test.py --fps``).

Run from the SegDAC repo root so ``./weights/`` resolves::

    PYTHONPATH=segdac/src python test_segdac.py
    PYTHONPATH=segdac/src python test_segdac.py --one-shot --save out.png
    PYTHONPATH=segdac/src python test_segdac.py --profiling
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from mani_skill.utils import common

from segdac.processor import SegDACProcessor
from segdac.processor_viz import maniskill_seg_to_color_rgb

signal.signal(signal.SIGINT, signal.SIG_DFL)


def find_first_usable_camera(obs) -> str | None:
    for cam in obs["sensor_data"].keys():
        s = obs["sensor_data"][cam]
        if "rgb" in s and "segmentation" in s:
            return cam
    return None


def get_panel_titles(cam: str) -> list[str]:
    return [
        f"RGB ({cam})",
        "ManiSkill seg (id-colored)",
        "YOLO-World (text classes + boxes)",
        "SegDAC (instance masks)",
        "SegDAC union of masks × RGB",
        "Token encoder (PCA + per-object patch outlines)",
    ]


def build_panels(
    obs: dict,
    cam: str,
    proc: SegDACProcessor,
    grounding: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Run ``SegDACProcessor`` on the current observation; return six uint8 panels + text."""
    rgb = obs["sensor_data"][cam]["rgb"]
    pixels_01 = rgb.permute(0, 3, 1, 2).contiguous().float() / 255.0
    res = proc.process(pixels_01)
    _ = res.object_tokens.shape

    assert len(res.seg_mask_classes) == int(res.seg_masks.shape[0]), (
        "seg_mask_classes must align 1:1 with seg_masks"
    )

    rgb_u8 = (
        (pixels_01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0)
        .round()
        .astype(np.uint8)
    )
    seg_t = obs["sensor_data"][cam]["segmentation"][0]
    ms = maniskill_seg_to_color_rgb(common.to_numpy(seg_t), selected_id=None)

    yv = proc.viz_bboxes(res, grounding)
    sm = proc.viz_seg_masks(res)
    un = proc.viz_mask_union(res)
    tok = proc.viz_token_encoder(res)

    emb = res.object_tokens
    s_sz = int(proc.seg_model.segmenter_image_size)
    h, w = res.native_hw
    lines = [
        "  --- SegDACProcessor (lazy result) ---",
        f"  object_tokens: {tuple(emb.shape)}  (N × D)",
        f"  seg_masks: {tuple(res.seg_masks.shape)}",
        f"  seg_mask_classes: {res.seg_mask_classes}",
        f"  img_features: {tuple(res.img_features.shape)}",
        f"  bboxes.xyxy: {tuple(res.bboxes.xyxy.shape)}",
        f"  (forward at segmenter {s_sz}×{s_sz}; mosaic native {h}×{w})",
    ]
    block = "\n".join(lines)
    return rgb_u8, ms, yv, sm, un, tok, block


def _timings_to_acc(
    t,
    acc_pipe: dict[str, float],
) -> None:
    if t is None:
        return
    for k in acc_pipe:
        v = getattr(t, k, None)
        if v is not None:
            acc_pipe[k] += float(v)


@torch.inference_mode()
def run_profiling_benchmark(
    env: gym.Env,
    cam: str,
    device: str,
    proc: SegDACProcessor,
    n_warmup: int,
    n_steps: int,
    token_encoder: str,
) -> None:
    """Time random ``env.step`` + full processor forward; print FPS and mean phase times."""
    for _ in range(n_warmup):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        rgb = obs["sensor_data"][cam]["rgb"]
        pixels_01 = rgb.permute(0, 3, 1, 2).contiguous().float() / 255.0
        res = proc.process(pixels_01)
        _ = res.object_tokens.shape
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    acc_step = 0.0
    acc_pipe: dict[str, float] = {
        "rgb_to_segmenter_s": 0.0,
        "object_detection_s": 0.0,
        "segmentation_s": 0.0,
        "token_encoder_s": 0.0,
    }
    for _ in range(n_steps):
        action = env.action_space.sample()
        if device == "cuda":
            torch.cuda.synchronize()
        t_env0 = time.perf_counter()
        obs, _, _, _, _ = env.step(action)
        if device == "cuda":
            torch.cuda.synchronize()
        t_env1 = time.perf_counter()
        acc_step += t_env1 - t_env0
        rgb = obs["sensor_data"][cam]["rgb"]
        pixels_01 = rgb.permute(0, 3, 1, 2).contiguous().float() / 255.0
        res = proc.process(pixels_01)
        emb = res.object_tokens
        _ = emb.shape
        _timings_to_acc(proc.last_frame_processing_times(), acc_pipe)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    if elapsed <= 0:
        print("FPS: (elapsed time too small to measure)")
        return
    fps = n_steps / elapsed
    ms_frame = 1000.0 * elapsed / n_steps
    n = float(n_steps)
    m_env = 1000.0 * acc_step / n
    m_rgb = 1000.0 * acc_pipe["rgb_to_segmenter_s"] / n
    m_od = 1000.0 * acc_pipe["object_detection_s"] / n
    m_seg = 1000.0 * acc_pipe["segmentation_s"] / n
    m_tok = 1000.0 * acc_pipe["token_encoder_s"] / n
    m_segdac = m_rgb + m_od + m_seg + m_tok
    print(
        f"env.step + SegDAC (grounded YOLO + SAM + object-token head): {fps:.2f} FPS  "
        f"({n_steps} timed steps in {elapsed:.3f} s, {ms_frame:.2f} ms / full iteration)"
    )
    tok_lbl = {
        "sam": "SAM-feature pool (SegDAC head)",
        "dinov2": "DINO dense patch map + mask pool (spatial adapter)",
        "random": "RandomImageEncoder + segment adapter (global)",
        "sam_image": "SamImageEncoder + segment adapter (spatial, extra ViT run)",
    }.get(token_encoder, token_encoder)
    print("  Per-iteration means (ms):")
    print(f"    env.step: {m_env:.2f}")
    print(f"    RGB to segmenter res: {m_rgb:.2f}")
    print(f"    Object detection (YOLO-World): {m_od:.2f}")
    print(f"    Segmentation (SAM + masks, post-YOLO): {m_seg:.2f}")
    print(f"    Object-token encoding ({tok_lbl}): {m_tok:.2f}")
    print(
        f"    Subtotal (RGB + YOLO + SAM + token head): {m_segdac:.2f} ms  "
        f"(+ env.step = ~{m_env + m_segdac:.2f} ms, vs full loop {ms_frame:.2f} ms; "
        "overhead = extra sync, Python, etc.)"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="SegDACProcessor + ManiSkill: interactive stepping or --one-shot"
    )
    p.add_argument("-e", "--env-id", type=str, default="PickCube-v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--token-encoder",
        type=str,
        choices=("sam", "dinov2", "random", "sam_image"),
        default="sam",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--save",
        type=str,
        default="test_segdac.png",
        help="Only used with --one-shot: output image path",
    )
    p.add_argument(
        "--one-shot",
        action="store_true",
        help="No loop: optional random step, build figure, save, exit",
    )
    p.add_argument(
        "--no-random-step",
        action="store_true",
        help="With --one-shot: use only reset frame (no extra env.step)",
    )
    p.add_argument(
        "--profiling",
        action="store_true",
        help="Benchmark only: time env.step + SegDACProcessor; print FPS and mean ms (no plots)",
    )
    p.add_argument(
        "--fps-steps",
        type=int,
        default=100,
        help="Number of timed iterations for --profiling (default 100)",
    )
    p.add_argument(
        "--fps-warmup",
        type=int,
        default=5,
        help="Untimed warmup steps before --profiling measurement (default 5)",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    weights = root / "weights"
    yolo = weights / "yolov8s-worldv2.pt"
    sam_w = weights / "efficientvit_sam_l0.pt"
    dinov2_w = weights / "dinov2" / "dinov2_vits14.pth"
    if not yolo.is_file() or not sam_w.is_file() or not dinov2_w.is_file():
        print(
            f"Expected {yolo}, {sam_w}, and {dinov2_w}. "
            "Run segdac/install_dependencies.sh from the repo root.",
            file=sys.stderr,
        )
        sys.exit(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.", file=sys.stderr)
        device = "cpu"

    env = gym.make(
        args.env_id,
        robot_uids="xarm6_robotiq",
        obs_mode="rgb+depth+segmentation",
        num_envs=1,
        sensor_configs={"width": 512, "height": 512},
    )
    try:
        obs, _ = env.reset(seed=args.seed)
        cam = find_first_usable_camera(obs)
        if cam is None:
            raise RuntimeError("No camera with rgb+segmentation.")

        grounding = ["robot arm", "red cube"]
        proc = SegDACProcessor(
            token_encoder=args.token_encoder,
            device=device,
            grounding_text_tags=grounding,
            object_detector_weights_path=str(yolo),
            segmenter_weights_path=str(sam_w),
            profiling=bool(args.profiling),
        )

        if args.profiling:
            print(
                f"Profiling mode: {args.fps_warmup} warmup steps, {args.fps_steps} timed steps, "
                f"device={device!r}, env_id={args.env_id!r}, "
                f"token_encoder={args.token_encoder!r}"
            )
            run_profiling_benchmark(
                env,
                cam,
                device,
                proc,
                n_warmup=int(args.fps_warmup),
                n_steps=int(args.fps_steps),
                token_encoder=str(args.token_encoder),
            )
            return

        if args.one_shot:
            if not args.no_random_step:
                obs, _, _, _, _ = env.step(env.action_space.sample())
            panels = build_panels(obs, cam, proc, grounding)
            r, m, yv, s, u, tok, block = panels
            print("\n" + block)
            titles = get_panel_titles(cam)
            fig, axes = plt.subplots(1, 6, figsize=(31, 4.5))
            for ax, img, t in zip(axes, (r, m, yv, s, u, tok), titles, strict=True):
                ax.imshow(img)
                ax.set_title(t, fontsize=10, pad=8)
                ax.set_axis_off()
            plt.suptitle(
                f"{args.env_id}  one-shot  seed={args.seed}", fontsize=11, y=1.04
            )
            out = Path(args.save)
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out), dpi=150, bbox_inches="tight")
            print(f"Saved: {out.resolve()}")
            plt.show()
            return

        print(
            "Window focus required.  "
            "SPACE: random action + advance sim, then run SegDACProcessor on the new frame.  "
            "Q / Esc: quit.  (Same random-action policy as demo_vis_segmentation.)"
        )
        plt.ion()
        panel_titles = get_panel_titles(cam)
        fig, axes = plt.subplots(1, 6, figsize=(31, 5.5))
        im_arts: list = [None] * 6
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
            r, m, yv, s, u, tok, block = build_panels(obs, cam, proc, grounding)
            panels = (r, m, yv, s, u, tok)
            print(
                f"\n--- frame @ sim counter {step_n} (SegDACProcessor + ManiSkill) ---\n"
                + block
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
