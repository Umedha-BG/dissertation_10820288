#!/usr/bin/env python3
"""
Monocular depth extractor using MiDaS (DPT).
- Input: video file (.mp4/.mov), single image, or folder of images
- Output: results/depth/<basename>/
    - depth_XXXXX.npy     (float32 depth, arbitrary scale)
    - depth_XXXXX.png     (16-bit PNG, normalized)
    - preview.mp4         (colored depth preview for quick checks)

Example:
    python src/mono_depth.py --input data_phone/raw/scene01_clip01.mp4 --device cpu --stride 1 --max_frames 300 --model DPT_Hybrid
"""

import os
import cv2
import sys
import glob
import time
import argparse
import numpy as np
from pathlib import Path

import torch

# -------------------------- utils --------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def is_image(p):
    ext = str(p).lower()
    return ext.endswith((".png", ".jpg", ".jpeg", ".bmp"))

def is_video(p):
    ext = str(p).lower()
    return ext.endswith((".mp4", ".mov", ".avi", ".mkv", ".m4v"))

def colorize_depth(depth_f32, percentile_clip=(1, 99)):
    """
    Map depth (float32, arbitrary scale) to a colored uint8 BGR image for preview.
    Uses percentile clipping to stabilize visualization across frames.
    """
    d = depth_f32.copy()
    if np.isnan(d).any() or np.isinf(d).any():
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    lo = np.percentile(d, percentile_clip[0])
    hi = np.percentile(d, percentile_clip[1])
    if hi <= lo:
        hi = lo + 1e-6
    d = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    d_u8 = (d * 255.0).astype(np.uint8)
    d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)
    return d_color

def save_depth_pair(out_dir, idx, depth_f32):
    """
    Save:
      - depth_XXXXX.npy  (float32)
      - depth_XXXXX.png  (uint16 normalized to [0..65535])
    """
    npy_path = os.path.join(out_dir, f"depth_{idx:05d}.npy")
    png_path = os.path.join(out_dir, f"depth_{idx:05d}.png")

    # Save float32 array
    np.save(npy_path, depth_f32.astype(np.float32))

    # Normalize robustly and save 16-bit PNG (preserves gradients)
    d = depth_f32.copy()
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    if hi <= lo:
        hi = lo + 1e-6
    d_norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    d_u16 = (d_norm * 65535.0).astype(np.uint16)
    cv2.imwrite(png_path, d_u16)

# -------------------------- MiDaS loader --------------------------

# Map friendly/old aliases to exact torch.hub entry points
_ALIAS_TO_HUB = {
    "dpt_large": "DPT_Large",
    "dpt-hybrid": "DPT_Hybrid",
    "dpt-large": "DPT_Large",
    "dpt_hybrid": "DPT_Hybrid",
    "midas_v21_small": "MiDaS_small",
    "midas_small": "MiDaS_small",
    # already-canonical names map to themselves
    "dpt_large".upper(): "DPT_Large",
    "dpt_hybrid".upper(): "DPT_Hybrid",
    "midas_small".capitalize(): "MiDaS_small",
}

_CANONICAL = {"DPT_Large", "DPT_Hybrid", "MiDaS_small"}

def _normalize_model_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return "DPT_Hybrid"
    # direct canonical?
    if n in _CANONICAL:
        return n
    # case-insensitive alias lookup
    hub = _ALIAS_TO_HUB.get(n.lower())
    if hub:
        return hub
    return n  # pass through; will be validated in load

def load_midas(model_type="DPT_Hybrid", device="cpu"):
    """
    model_type: 'DPT_Large', 'DPT_Hybrid', or 'MiDaS_small' (lowercase aliases supported)
    """
    model_type = _normalize_model_name(model_type)
    try:
        # Requires: pip install torch torchvision timm
        model = torch.hub.load("intel-isl/MiDaS", model_type)
    except Exception as e:
        # Give a clearer hint if the name is wrong
        msg = (
            f"Failed to load MiDaS model '{model_type}'. "
            f"Use one of: {sorted(_CANONICAL)} or aliases: dpt_large, dpt_hybrid, midas_v21_small."
        )
        raise RuntimeError(msg) from e

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type in ("DPT_Large", "DPT_Hybrid"):
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    model.to(device)
    model.eval()
    return model, transform

@torch.inference_mode()
def infer_depth(model, transform, image_bgr, device="cpu", max_side=768):
    """
    image_bgr: uint8 HxWx3, BGR
    Returns: float32 depth map (HxW), arbitrary scale (MiDaS relative depth)
    """
    # Optional: resize so longer side <= max_side (keeps speed reasonable)
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convert BGR->RGB for model
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    pred = model(input_batch)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    depth = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    # depth is relative; keep raw MiDaS scale
    return depth

# -------------------------- IO pipelines --------------------------

def process_images_list(paths, out_dir, model, transform, device, max_size, write_preview=True):
    ensure_dir(out_dir)
    writer = None
    preview_path = os.path.join(out_dir, "preview.mp4") if write_preview else None

    t0 = time.time()
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] Could not read image: {p}")
            continue

        depth = infer_depth(model, transform, img, device=device, max_side=max_size)
        save_depth_pair(out_dir, i, depth)

        if write_preview:
            vis = colorize_depth(depth)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(preview_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(vis)

        if (i + 1) % 20 == 0:
            dt = time.time() - t0
            fps = (i + 1) / max(dt, 1e-6)
            print(f"[info] images processed: {i+1}  FPS~{fps:.1f}")

    if writer is not None:
        writer.release()
    print(f"[done] images -> {out_dir}")

def process_video(path, out_dir, model, transform, device, max_size, stride=1, max_frames=None, write_preview=True):
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[error] Failed to open video: {path}")
        return

    writer = None
    preview_path = os.path.join(out_dir, "preview.mp4") if write_preview else None

    idx = 0
    frame_id = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % stride != 0:
            frame_id += 1
            continue

        depth = infer_depth(model, transform, frame, device=device, max_side=max_size)
        save_depth_pair(out_dir, idx, depth)

        if write_preview:
            vis = colorize_depth(depth)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(preview_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(vis)

        idx += 1
        frame_id += 1

        if max_frames is not None and idx >= max_frames:
            break

        if idx % 30 == 0:
            dt = time.time() - t0
            fps = idx / max(dt, 1e-6)
            print(f"[info] frames processed: {idx}  (stride={stride})  FPS~{fps:.1f}")

    cap.release()
    if writer is not None:
        writer.release()
    print(f"[done] video -> {out_dir}  frames={idx}")

# -------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser(description="Monocular depth extraction with MiDaS (DPT).")
    ap.add_argument("--input", required=True, help="Video file, image file, or folder of images.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device.")
    # Accept canonical names + old aliases (we normalize internally)
    ap.add_argument(
        "--model",
        default="DPT_Hybrid",
        choices=["DPT_Hybrid", "DPT_Large", "MiDaS_small", "dpt_hybrid", "dpt_large", "midas_v21_small", "midas_small"],
        help="MiDaS model type."
    )
    ap.add_argument("--max_size", type=int, default=768, help="Resize longer image side to this (keeps speed/memory sane).")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth frame for videos.")
    ap.add_argument("--max_frames", type=int, default=None, help="Process at most N frames from a video.")
    ap.add_argument("--no_preview", action="store_true", help="Disable preview.mp4 writing.")
    args = ap.parse_args()

    inp = args.input
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    # Load model
    norm_model = _normalize_model_name(args.model)
    print(f"[info] loading MiDaS model: {norm_model} (device={device})")
    model, transform = load_midas(model_type=norm_model, device=device)

    # Resolve output dir
    base = Path(inp).stem if os.path.isfile(inp) else Path(inp).name
    out_dir = os.path.join("results", "depth", base)
    ensure_dir(out_dir)

    # Route by input type
    if os.path.isdir(inp):
        # Collect images
        imgs = sorted(glob.glob(os.path.join(inp, "*.*")))
        imgs = [p for p in imgs if is_image(p)]
        if len(imgs) == 0:
            print(f"[error] No images in folder: {inp}")
            sys.exit(1)
        process_images_list(imgs, out_dir, model, transform, device, args.max_size, write_preview=not args.no_preview)

    elif os.path.isfile(inp) and is_image(inp):
        process_images_list([inp], out_dir, model, transform, device, args.max_size, write_preview=not args.no_preview)

    elif os.path.isfile(inp) and is_video(inp):
        process_video(inp, out_dir, model, transform, device, args.max_size, stride=args.stride, max_frames=args.max_frames, write_preview=not args.no_preview)

    else:
        print(f"[error] Unsupported input: {inp}")
        sys.exit(1)

if __name__ == "__main__":
    main()


# python src/mono_depth.py --input "data_phone/raw/phaseA_180.mp4" --device cuda --stride 1 --max_frames 300 --model dpt_hybrid
