# src/detect.py
# ------------------------------------------------------------
# YOLOv8 detection wrapper for your phone clips/photos.
# - Reads configs/default.yaml
# - Processes a video or image folder
# - Writes CSV with per-frame detections
# - Saves overlay frames (optional)
# ------------------------------------------------------------

import os
import sys
import csv
import glob
import time
import math
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# --------------------------- utils ---------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def draw_box(img, xyxy, color=(0, 200, 100), label: Optional[str] = None):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def is_video_file(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return ext in [".mp4", ".mov", ".avi", ".mkv", ".m4v"]

def is_image_file(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


# --------------------------- config ---------------------------

@dataclass
class DetectConfig:
    weights: str
    classes: Optional[List[str]]
    conf_thres: float
    iou_thres: float
    device: str
    overlay: bool
    max_frames: Optional[int]
    stride: int

    # paths
    input_path: str
    out_csv: str
    out_overlays: str

def build_config(cfg_path: str, input_path: Optional[str], csv_out_dir: Optional[str], overlays_dir: Optional[str]) -> DetectConfig:
    cfg = load_yaml(cfg_path)

    paths = cfg.get("paths", {})
    detect_cfg = cfg.get("detect", {})
    eval_cfg = cfg.get("eval", {})

    data_root = paths.get("data", "data_phone/raw")
    csv_dir = paths.get("csv_out", "results/csv")
    overlays = paths.get("overlays", "results/overlays")

    return DetectConfig(
        weights = detect_cfg.get("weights", "models/yolov8n.pt"),
        classes = detect_cfg.get("classes", None),
        conf_thres = float(detect_cfg.get("conf", 0.25)),
        iou_thres = float(detect_cfg.get("iou", 0.45)),
        device = detect_cfg.get("device", "cpu"),      # "cpu" or "cuda"
        overlay = bool(detect_cfg.get("save_overlays", True)),
        max_frames = eval_cfg.get("max_frames", None),
        stride = int(eval_cfg.get("frame_stride", 1)),

        input_path = input_path or data_root,
        out_csv = csv_out_dir or csv_dir,
        out_overlays = overlays_dir or overlays
    )


# --------------------------- core ---------------------------

def resolve_inputs(input_path: str) -> List[str]:
    """Return a list of files to process (video(s) or image(s))."""
    if os.path.isdir(input_path):
        # all media files under folder
        files = sorted([p for p in glob.glob(os.path.join(input_path, "*"))
                        if is_video_file(p) or is_image_file(p)])
        return files
    else:
        return [input_path]

def class_filter_from_model(model: YOLO, allow_classes: Optional[List[str]]) -> Optional[List[int]]:
    if allow_classes is None:
        return None
    # map class names to ids according to model.names
    name_to_id = {v: k for k, v in model.model.names.items()}
    ids = [name_to_id[c] for c in allow_classes if c in name_to_id]
    return ids

def write_csv_header(csv_path: str):
    ensure_dir(os.path.dirname(csv_path))
    new_file = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow([
            "source_file", "frame_idx", "timestamp_ms",
            "cls_id", "cls_name", "conf",
            "x1","y1","x2","y2","w","h"
        ])
    return f, writer

def process_video(model: YOLO, path: str, cfg: DetectConfig, allowed_cls_ids: Optional[List[int]]):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[error] cannot open video: {path}")
        return

    base = os.path.splitext(os.path.basename(path))[0]
    csv_path = os.path.join(cfg.out_csv, f"{base}_detections.csv")
    f, writer = write_csv_header(csv_path)

    overlay_dir = os.path.join(cfg.out_overlays, base)
    if cfg.overlay:
        ensure_dir(overlay_dir)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t0 = time.time()

    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cfg.stride > 1 and (frame_idx % cfg.stride != 0):
            frame_idx += 1
            continue

        # run detection
        res = model.predict(
            source=frame,
            conf=cfg.conf_thres,
            iou=cfg.iou_thres,
            device=cfg.device,
            verbose=False
        )[0]

        # draw & save rows
        overlay_img = frame.copy()
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                cls_id = int(b.cls.item())
                if (allowed_cls_ids is not None) and (cls_id not in allowed_cls_ids):
                    continue
                conf = float(b.conf.item())
                x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].tolist()]
                w = x2 - x1
                h = y2 - y1
                cls_name = model.model.names.get(cls_id, str(cls_id))
                ts_ms = int((frame_idx / fps) * 1000)

                writer.writerow([os.path.basename(path), frame_idx, ts_ms,
                                 cls_id, cls_name, f"{conf:.4f}",
                                 f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", f"{w:.1f}", f"{h:.1f}"])
                written += 1

                if cfg.overlay:
                    draw_box(overlay_img, (x1,y1,x2,y2),
                             color=(0,200,100),
                             label=f"{cls_name} {conf:.2f}")

        if cfg.overlay:
            out_name = os.path.join(overlay_dir, f"{base}_{frame_idx:06d}.jpg")
            cv2.imwrite(out_name, overlay_img)

        frame_idx += 1
        if cfg.max_frames is not None and frame_idx >= cfg.max_frames:
            break

    f.close()
    cap.release()
    dt = time.time() - t0
    used_frames = frame_idx if cfg.max_frames is None else min(frame_idx, cfg.max_frames)
    print(f"[done] {path} -> {csv_path}  rows={written}  frames={used_frames}  time={dt:.2f}s")

def process_image(model: YOLO, path: str, cfg: DetectConfig, allowed_cls_ids: Optional[List[int]]):
    img = cv2.imread(path)
    if img is None:
        print(f"[error] cannot read image: {path}")
        return

    base = os.path.splitext(os.path.basename(path))[0]
    csv_path = os.path.join(cfg.out_csv, f"{base}_detections.csv")
    f, writer = write_csv_header(csv_path)

    overlay_dir = os.path.join(cfg.out_overlays, base)
    if cfg.overlay:
        ensure_dir(overlay_dir)

    res = model.predict(
        source=img,
        conf=cfg.conf_thres,
        iou=cfg.iou_thres,
        device=cfg.device,
        verbose=False,
        imgsz=1280
    )[0]

    written = 0
    overlay_img = img.copy()
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            cls_id = int(b.cls.item())
            if (allowed_cls_ids is not None) and (cls_id not in allowed_cls_ids):
                continue
            conf = float(b.conf.item())
            x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].tolist()]
            w = x2 - x1
            h = y2 - y1
            cls_name = model.model.names.get(cls_id, str(cls_id))
            # timestamp 0 for images
            writer.writerow([os.path.basename(path), 0, 0,
                             cls_id, cls_name, f"{conf:.4f}",
                             f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", f"{w:.1f}", f"{h:.1f}"])
            written += 1

            if cfg.overlay:
                draw_box(overlay_img, (x1,y1,x2,y2),
                         color=(0,200,100),
                         label=f"{cls_name} {conf:.2f}")

    f.close()
    if cfg.overlay:
        out_name = os.path.join(overlay_dir, f"{base}_000000.jpg")
        ensure_dir(overlay_dir)
        cv2.imwrite(out_name, overlay_img)

    print(f"[done] {path} -> {csv_path}  rows={written}")

def run_detection(cfg: DetectConfig):
    ensure_dir(cfg.out_csv)
    if cfg.overlay:
        ensure_dir(cfg.out_overlays)

    print(f"[info] loading model: {cfg.weights} (device={cfg.device})")
    model = YOLO(cfg.weights)
    allowed_ids = class_filter_from_model(model, cfg.classes)

    inputs = resolve_inputs(cfg.input_path)
    if len(inputs) == 0:
        print(f"[warn] no media found at {cfg.input_path}")
        return

    for p in inputs:
        if is_video_file(p):
            process_video(model, p, cfg, allowed_ids)
        elif is_image_file(p):
            process_image(model, p, cfg, allowed_ids)
        else:
            print(f"[skip] unsupported file: {p}")


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="YOLOv8 detection runner")
    ap.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    ap.add_argument("--input", default=None, help="Video/image file or folder (overrides config paths.data)")
    ap.add_argument("--csv_out", default=None, help="CSV output folder (overrides paths.csv_out)")
    ap.add_argument("--overlays", default=None, help="Overlay output folder (overrides paths.overlays)")
    ap.add_argument("--device", default=None, help="'cpu' or 'cuda' (overrides detect.device)")
    ap.add_argument("--max_frames", type=int, default=None, help="Limit frames per video for quick tests")
    ap.add_argument("--stride", type=int, default=None, help="Process every Nth frame (default from config)")

    # NEW: runtime overrides for detection thresholds & model
    ap.add_argument("--conf", type=float, default=None, help="Confidence threshold (overrides detect.conf)")
    ap.add_argument("--iou", type=float, default=None, help="IoU threshold (overrides detect.iou)")
    ap.add_argument("--weights", default=None, help="Path to YOLO weights (overrides detect.weights)")
    ap.add_argument("--classes", default=None,
                    help="Comma-separated class names to keep (overrides detect.classes). "
                         "Example: 'person,car,dog'")

    args = ap.parse_args()

    cfg = build_config(args.config, args.input, args.csv_out, args.overlays)
    # cfg.classes = ["cup", "laptop", "keyboard"]

    # Apply CLI overrides to config
    if args.device is not None:
        cfg.device = args.device
    if args.max_frames is not None:
        cfg.max_frames = args.max_frames
    if args.stride is not None:
        cfg.stride = args.stride
    if args.conf is not None:
        cfg.conf_thres = float(args.conf)
    if args.iou is not None:
        cfg.iou_thres = float(args.iou)
    if args.weights is not None:
        cfg.weights = args.weights
    if args.classes is not None:
        # Parse comma-separated class names; empty string -> None
        parsed = [c.strip() for c in args.classes.split(",") if c.strip()]
        cfg.classes = parsed if parsed else None

    run_detection(cfg)

if __name__ == "__main__":
    main()


# python src/detect.py --input "data_phone/raw/phaseA_180.mp4" --device cuda --max_frames 300 --stride 1 --conf 0.25  --weights "yolo11s.pt"                             