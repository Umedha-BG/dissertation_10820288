#!/usr/bin/env python3
"""
Scale MiDaS relative depth to metric using an A4 sheet as reference
or reuse an existing calibration file.

Two modes:
  A) Calibrate using A4 (requires --calibrate_from_frame and --known_distance_cm)
  B) Reuse existing calibration file (--use_calibration data_phone/calib/scale_meta.json)

Outputs:
  - depth_metric_XXXXX.npy  (float32, metres)
  - depth_metric_XXXXX.png  (16-bit PNG)
  - scale_meta.json         (focal_px, mapping a,b, stats)

Example calibration:
python src/scale_fix.py \
  --depth_dir results/depth/scene01_clip01 \
  --rgb data_phone/raw/scene01_clip01.mp4 \
  --out_dir results/depth_metric/scene01_clip01 \
  --calibrate_from_frame 42 --known_distance_cm 120

Example reuse:
python src/scale_fix.py \
  --depth_dir results/depth/phaseA_60 \
  --rgb data_phone/raw/phaseA_60.mp4 \
  --out_dir results/depth_metric/phaseA_60 \
  --use_calibration data_phone/calib/scale_meta.json
"""

import os, json, glob, math, argparse, time, shutil
from pathlib import Path
import cv2
import numpy as np

A4_WIDTH_CM  = 21.0
A4_HEIGHT_CM = 29.7
A4_RATIO     = A4_HEIGHT_CM / A4_WIDTH_CM

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def list_depth_npy(depth_dir):
    files = sorted(glob.glob(os.path.join(depth_dir, "depth_*.npy")))
    if not files:
        raise FileNotFoundError(f"No depth_XXXXX.npy in {depth_dir}")
    return files

def open_rgb_source(rgb_path):
    if os.path.isdir(rgb_path):
        frames = sorted([p for p in glob.glob(os.path.join(rgb_path, "*.*"))
                         if p.lower().endswith((".png",".jpg",".jpeg",".bmp"))])
        if not frames:
            raise FileNotFoundError(f"No images in folder {rgb_path}")
        return ("images", frames, None)
    else:
        cap = cv2.VideoCapture(rgb_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {rgb_path}")
        return ("video", None, cap)

def read_rgb_at(index, src_kind, frames, cap):
    if src_kind == "images":
        if index < 0 or index >= len(frames): return None
        img = cv2.imread(frames[index], cv2.IMREAD_COLOR)
        return img
    else:
        return None

def iter_video_frames(cap):
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        yield i, frame
        i += 1

def detect_a4_sheet(img_bgr, canny1=60, canny2=160, ratio_tol=0.12):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, canny1, canny2)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        if area < 0.002 * (h*w):
            continue
        pts = approx.reshape(-1,2).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        (cx,cy), (w_box,h_box), ang = rect
        if w_box == 0 or h_box == 0: continue
        r = max(w_box,h_box)/max(1.0,min(w_box,h_box))
        if abs(r - A4_RATIO) > A4_RATIO * ratio_tol:
            continue
        if area > best_area:
            best_area = area
            box = cv2.boxPoints(rect)
            s = box.sum(axis=1); diff = np.diff(box, axis=1).ravel()
            tl = box[np.argmin(s)]; br = box[np.argmax(s)]
            tr = box[np.argmin(diff)]; bl = box[np.argmax(diff)]
            best = np.array([tl,tr,br,bl], dtype=np.float32)
    if best is None:
        return None, None
    side1 = np.linalg.norm(best[1]-best[0])
    side2 = np.linalg.norm(best[3]-best[0])
    short_px = min(side1, side2)
    return best, short_px

def focal_from_known_distance(known_Z_cm, width_px, width_cm=A4_WIDTH_CM):
    return (known_Z_cm * (width_px / max(1e-6, width_cm)))

def sheet_distance_from_focal(f_px, width_px, width_cm=A4_WIDTH_CM):
    return (f_px * (width_cm / max(1e-6, width_px)))

def fit_mapping(D_rel_list, Z_m_list, affine=False):
    D = np.array(D_rel_list, dtype=np.float64)
    Z = np.array(Z_m_list,    dtype=np.float64)
    msk = np.isfinite(D) & np.isfinite(Z)
    D, Z = D[msk], Z[msk]
    if len(D) < 3:
        a = np.median(Z) / max(1e-6, np.median(D))
        return a, 0.0
    if affine:
        A = np.vstack([D, np.ones_like(D)]).T
        sol, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        a0, b0 = sol
        res = Z - (a0*D + b0)
    else:
        a0 = np.median(Z/D)
        b0 = 0.0
        res = Z - a0*D
    med = np.median(res)
    mad = np.median(np.abs(res - med)) + 1e-9
    keep = np.abs(res - med) < 3.0 * 1.4826 * mad
    D2, Z2 = D[keep], Z[keep]
    if len(D2) < 3:
        return a0, b0
    if affine:
        A2 = np.vstack([D2, np.ones_like(D2)]).T
        sol, _, _, _ = np.linalg.lstsq(A2, Z2, rcond=None)
        return float(sol[0]), float(sol[1])
    else:
        a = np.median(Z2/D2)
        return float(a), 0.0

def save_depth_metric(out_dir, idx, Z_m):
    npy_path = os.path.join(out_dir, f"depth_metric_{idx:05d}.npy")
    png_path = os.path.join(out_dir, f"depth_metric_{idx:05d}.png")
    np.save(npy_path, Z_m.astype(np.float32))
    z = np.nan_to_num(Z_m, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(z, 1), np.percentile(z, 99)
    if hi <= lo: hi = lo + 1e-6
    zn = np.clip((z - lo)/(hi - lo), 0, 1)
    z16 = (zn * 65535.0).astype(np.uint16)
    cv2.imwrite(png_path, z16)

def main():
    import shutil  # ✅ make sure this is imported at the top

    ap = argparse.ArgumentParser("Scale MiDaS depths to metric using A4 or existing calibration")
    ap.add_argument("--use_calibration", default=None,
                    help="Path to an existing calibration JSON file (e.g., data_phone/calib/scale_meta.json)")
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--calibrate_from_frame", type=int, default=None)
    ap.add_argument("--known_distance_cm", type=float, default=None)
    ap.add_argument("--focal_px", type=float, default=None)
    ap.add_argument("--affine_fit", action="store_true")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ratio_tol", type=float, default=0.12)
    ap.add_argument("--no_preview", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # ----------------------------------------------------------------
    # 🟢 1️⃣ USE EXISTING CALIBRATION (skip A4 detection)
    # ----------------------------------------------------------------
    if args.use_calibration:
        print(f"[info] Using existing calibration file: {args.use_calibration}")

        if not os.path.exists(args.use_calibration):
            raise FileNotFoundError(f"Calibration file not found: {args.use_calibration}")

        os.makedirs(args.out_dir, exist_ok=True)
        dst_meta = os.path.join(args.out_dir, "scale_meta.json")

        # Copy calibration file to this video’s metric depth folder
        shutil.copy(args.use_calibration, dst_meta)
        print(f"[done] Copied existing calibration to {dst_meta}")

        # 🔹 Apply the calibration mapping directly to convert relative → metric
        with open(dst_meta, "r") as f:
            meta = json.load(f)
        a = meta["mapping"]["a"]
        b = meta["mapping"].get("b", 0.0)
        print(f"[apply] Using calibration mapping: Z_metric = {a:.6f} * D_rel + {b:.6f}")

        depth_files = list_depth_npy(args.depth_dir)
        print("[apply] Writing metric depths...")
        for j, dpath in enumerate(depth_files):
            Zm = a * np.load(dpath).astype(np.float32) + b
            save_depth_metric(args.out_dir, j, Zm)

        print(f"[done] Metric depths generated using existing calibration.")
        return

    # ----------------------------------------------------------------
    # 🧾 2️⃣ NORMAL CALIBRATION MODE (A4 DETECTION)
    # ----------------------------------------------------------------
    depth_files = list_depth_npy(args.depth_dir)
    src_kind, frames, cap = open_rgb_source(args.rgb)
    total_frames = len(frames) if src_kind == "images" else int(cv2.VideoCapture(args.rgb).get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = len(depth_files)

    # --- Determine focal length from known A4 distance ---
    focal_px = args.focal_px
    if focal_px is None:
        if args.calibrate_from_frame is None or args.known_distance_cm is None:
            raise ValueError("Provide either --focal_px OR (--calibrate_from_frame AND --known_distance_cm).")
        if src_kind == "images":
            img = read_rgb_at(args.calibrate_from_frame, src_kind, frames, cap=None)
        else:
            cap_cal = cv2.VideoCapture(args.rgb)
            cap_cal.set(cv2.CAP_PROP_POS_FRAMES, args.calibrate_from_frame)
            ok, img = cap_cal.read()
            cap_cal.release()
            if not ok:
                raise ValueError("Failed to read calibration frame.")

        poly, width_px = detect_a4_sheet(img, ratio_tol=args.ratio_tol)
        if width_px is None:
            raise RuntimeError("A4 detection failed — ensure sheet visible.")
        focal_px = focal_from_known_distance(args.known_distance_cm, width_px, A4_WIDTH_CM)
        print(f"[calib] focal_px ≈ {focal_px:.2f} from frame {args.calibrate_from_frame}")

    # --- Fit scale using the A4 sheet ---
    D_rel_medians, Z_sheet_m, used_idx = [], [], []
    fit_gathered, processed = 0, 0

    if src_kind == "images":
        src_iter = enumerate(frames)
        get_img = lambda path: cv2.imread(path, cv2.IMREAD_COLOR)
    else:
        cap_main = cv2.VideoCapture(args.rgb)
        src_iter = iter_video_frames(cap_main)
        get_img = lambda frame: frame

    for i, src in src_iter:
        if i % args.stride != 0:
            continue
        img = get_img(src) if src_kind == "images" else src
        if i >= len(depth_files):
            break
        depth_rel = np.load(depth_files[i]).astype(np.float32)
        if img is None:
            continue
        poly, width_px = detect_a4_sheet(img, ratio_tol=args.ratio_tol)
        if width_px is not None:
            z_sheet_cm = sheet_distance_from_focal(focal_px, width_px, A4_WIDTH_CM)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, poly.astype(np.int32), 255)
            if mask.shape != depth_rel.shape:
                mask = cv2.resize(mask, (depth_rel.shape[1], depth_rel.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = np.median(depth_rel[mask > 0])
            if np.isfinite(m):
                D_rel_medians.append(float(m))
                Z_sheet_m.append(float(z_sheet_cm / 100.0))
                used_idx.append(i)
                fit_gathered += 1
        processed += 1

    if src_kind == "video":
        cap_main.release()

    if len(D_rel_medians) == 0:
        raise RuntimeError("No A4 detected during calibration.")

    a, b = fit_mapping(D_rel_medians, Z_sheet_m, affine=args.affine_fit)
    print(f"[fit] Z_metric = {a:.6f} * D_rel + {b:.6f}")

    # --- Apply mapping to all frames ---
    for j, dpath in enumerate(depth_files):
        Zm = a * np.load(dpath).astype(np.float32) + b
        save_depth_metric(args.out_dir, j, Zm)

    # --- Save metadata ---
    meta = {
        "focal_px": float(focal_px),
        "mapping": {"a": float(a), "b": float(b)},
        "fit_samples": len(D_rel_medians),
        "depth_dir": args.depth_dir
    }
    with open(os.path.join(args.out_dir, "scale_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] Metric depths -> {args.out_dir}")
    print(f"[done] Metadata saved to scale_meta.json")

if __name__ == "__main__":
    main()
