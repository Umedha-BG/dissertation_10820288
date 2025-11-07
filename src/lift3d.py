#!/usr/bin/env python3
"""
Lift 2D detections to 3D (metric) using calibrated depth maps.
Now with temporal smoothing: EMA / median window / Kalman (1-D).

Examples
--------
# Kalman smoothing + reference lock to A4/laptop @ 1.00 m
python src/lift3d.py --rgb "data_phone/raw/scene01_clip01.mp4" ^
  --detections_csv "results/csv/scene01_clip01_detections.csv" ^
  --depth_metric_dir "results/depth_metric/scene01_clip01" ^
  --scale_meta "results/depth_metric/scene01_clip01/scale_meta.json" ^
  --inner_roi 0.6 --smooth kalman --kf_q 1e-3 --kf_r 1e-2 ^
  --ref_class laptop --ref_dist_m 1.0 --round_cm 1

# Lightweight: rolling median over 5 frames
python src/lift3d.py --rgb "...mp4" --detections_csv "...csv" ^
  --depth_metric_dir ".../depth_metric/scene01_clip01" ^
  --scale_meta ".../scale_meta.json" --inner_roi 0.6 ^
  --smooth median --median_window 5 --round_cm 1
"""

import os, csv, json, argparse, collections
from pathlib import Path
import numpy as np
import cv2

# -------------------- utils --------------------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def exists_nonempty(path: str) -> bool:
    """Return True if a file exists and is non-empty."""
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False

def load_focal(scale_meta_path: str | None, focal_override: float | None) -> float:
    if focal_override is not None:
        return float(focal_override)
    if not scale_meta_path:
        raise ValueError("Provide --scale_meta or --focal_px.")
    with open(scale_meta_path, "r") as f:
        meta = json.load(f)
    if "focal_px" not in meta:
        raise ValueError("scale_meta.json missing 'focal_px'.")
    return float(meta["focal_px"])

def read_detections(csv_path: str):
    """
    Accepts either schema:
      A) frame,x1,y1,x2,y2,cls,conf
      B) source_file,frame_idx,timestamp,cls_id,cls_name,conf,x1,y1,x2,y2,w,h
    Returns unified: frame,x1,y1,x2,y2,cls,conf
    """
    rows = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        headers = [h.strip() for h in r.fieldnames]
        fmtA = {"frame","x1","y1","x2","y2","cls","conf"}.issubset(headers)
        fmtB = {"frame_idx","x1","y1","x2","y2","conf"}.issubset(headers)
        if not (fmtA or fmtB):
            raise ValueError(f"Unsupported headers in {csv_path}: {headers}")
        for d in r:
            try:
                if fmtA:
                    frame = int(float(d["frame"]))
                    cls = str(d["cls"])
                else:
                    frame = int(float(d["frame_idx"]))
                    cls = d.get("cls_name") or f"id_{int(float(d.get('cls_id','0')))}"
                conf = float(d["conf"])
                x1,y1,x2,y2 = map(float, (d["x1"], d["y1"], d["x2"], d["y2"]))
                rows.append({"frame":frame,"x1":x1,"y1":y1,"x2":x2,"y2":y2,"cls":cls,"conf":conf})
            except Exception:
                continue
    return rows

def depth_path(depth_dir: str, idx: int) -> str:
    return os.path.join(depth_dir, f"depth_metric_{idx:05d}.npy")

def clip_int(a, lo, hi): return int(max(lo, min(hi, a)))

def roi_center_and_inner(box, W, H, inner=0.6):
    x1,y1,x2,y2 = box
    x1i,y1i = clip_int(min(x1,x2), 0, W-1), clip_int(min(y1,y2), 0, H-1)
    x2i,y2i = clip_int(max(x1,x2), 0, W-1), clip_int(max(y1,y2), 0, H-1)
    if x2i <= x1i or y2i <= y1i: return None, None, None
    u,v = (x1i+x2i)//2, (y1i+y2i)//2
    w,h = x2i-x1i, y2i-y1i
    iw,ih = max(3,int(round(w*inner))), max(3,int(round(h*inner)))
    cx1,cy1 = clip_int(u - iw//2, 0, W-1), clip_int(v - ih//2, 0, H-1)
    cx2,cy2 = clip_int(u + iw//2, 0, W),   clip_int(v + ih//2, 0, H)
    return (u,v), (cx1,cy1,cx2,cy2), (x1i,y1i,x2i,y2i)

def robust_depth_from_roi(depth, roi):
    """30th percentile after 5–95% trimming for dark/low-texture surfaces."""
    cx1, cy1, cx2, cy2 = roi
    Z = depth[cy1:cy2, cx1:cx2]
    Z = Z[np.isfinite(Z)]
    if Z.size < 10: return float("nan")
    lo, hi = np.percentile(Z, [5, 95])
    Z = Z[(Z >= lo) & (Z <= hi)]
    if Z.size == 0: return float("nan")
    return float(np.percentile(Z, 30))

def lift(u,v,Z,fx,fy,cx,cy):
    X = (u - cx)/fx * Z
    Y = (v - cy)/fy * Z
    return X,Y,Z

def measure_depth_for_box(depth, box, W, H, inner=0.6):
    """Same statistic as objects; used for per-frame reference lock."""
    x1,y1,x2,y2 = box
    x1i,y1i = clip_int(min(x1,x2), 0, W-1), clip_int(min(y1,y2), 0, H-1)
    x2i,y2i = clip_int(max(x1,x2), 0, W-1), clip_int(max(y1,y2), 0, H-1)
    if x2i <= x1i or y2i <= y1i: return float("nan")
    u,v = (x1i+x2i)//2, (y1i+y2i)//2
    w,h = x2i-x1i, y2i-y1i
    iw,ih = max(3,int(round(w*inner))), max(3,int(round(h*inner)))
    cx1,cy1 = clip_int(u - iw//2, 0, W-1), clip_int(v - ih//2, 0, H-1)
    cx2,cy2 = clip_int(u + iw//2, 0, W),   clip_int(v + ih//2, 0, H)
    Z = depth[cy1:cy2, cx1:cx2]
    Z = Z[np.isfinite(Z)]
    if Z.size < 10: return float("nan")
    lo, hi = np.percentile(Z, [5, 95])
    Z = Z[(Z >= lo) & (Z <= hi)]
    if Z.size == 0: return float("nan")
    return float(np.percentile(Z, 30))

# --------- smoothing helpers ---------

class Kalman1D:
    """Scalar KF for range smoothing: x=r, P variance, Q process, R meas."""
    def __init__(self, x0=None, P0=1.0, Q=1e-3, R=1e-2):
        self.x = x0
        self.P = P0
        self.Q = float(Q)
        self.R = float(R)

    def update(self, z):
        if self.x is None:
            self.x = float(z)
            self.P = max(self.P, self.R)
            return self.x
        # predict
        self.P = self.P + self.Q
        # update
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (float(z) - self.x)
        self.P = (1.0 - K) * self.P
        return self.x

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser("Lift 2D detections to 3D (metric) with temporal smoothing")
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--detections_csv", required=True)
    ap.add_argument("--depth_metric_dir", required=True)
    ap.add_argument("--scale_meta", default=None)
    ap.add_argument("--focal_px", type=float, default=None)
    ap.add_argument("--fy_px", type=float, default=None)
    ap.add_argument("--inner_roi", type=float, default=0.6)
    ap.add_argument("--min_depth", type=float, default=0.05)
    ap.add_argument("--max_depth", type=float, default=10.0)
    ap.add_argument("--post_scale", type=float, default=1.0, help="Global multiplier on Z after per-frame scaling.")
    # per-frame reference lock
    ap.add_argument("--ref_class", type=str, default=None, help="Class name to use as per-frame reference (e.g., 'laptop').")
    ap.add_argument("--ref_dist_m", type=float, default=None, help="Known metric distance for the reference (e.g., 1.0).")
    # smoothing options
    ap.add_argument("--smooth", choices=["none","ema","median","kalman"], default="kalman")
    ap.add_argument("--ema", type=float, default=0.2, help="EMA alpha, if --smooth ema.")
    ap.add_argument("--median_window", type=int, default=5, help="Window size for --smooth median.")
    ap.add_argument("--kf_q", type=float, default=1e-3, help="Kalman process noise Q for --smooth kalman.")
    ap.add_argument("--kf_r", type=float, default=1e-2, help="Kalman measurement noise R for --smooth kalman.")
    # cosmetics
    ap.add_argument("--round_cm", type=int, default=0, help="Round label to nearest N cm (0=off).")
    ap.add_argument("--no_overlay", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.rgb)
    if not cap.isOpened(): raise FileNotFoundError(args.rgb)
    W, H = int(cap.get(3)), int(cap.get(4))
    FPS = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    fx = load_focal(args.scale_meta, args.focal_px)
    fy = float(args.fy_px) if args.fy_px else fx
    cx, cy = W/2.0, H/2.0

    dets = read_detections(args.detections_csv)
    # optional class renaming example (cup->bottle). Add more if you like.
    # rename_map = {"cup": "bottle"}
    # for d in dets:
    #     name = str(d["cls"]).lower().strip()
    #     if name in rename_map: d["cls"] = rename_map[name]

    by_frame = {}
    for d in dets:
        by_frame.setdefault(d["frame"], []).append(d)

    base = Path(args.detections_csv).stem.replace("_detections","")
    out_dir = os.path.join("results","poses3d"); ensure_dir(out_dir)
    out_csv = os.path.join(out_dir,f"{base}_3d.csv")

    # Overlay MP4 handling: only generate if missing (unless --no_overlay)
    overlay_path = os.path.join(out_dir, f"{base}_overlay.mp4")
    writer = None
    if args.no_overlay:
        print("[info] --no_overlay set; overlay MP4 will not be written.")
    else:
        if exists_nonempty(overlay_path):
            print(f"[skip] Overlay already exists: {overlay_path}")
        else:
            writer = cv2.VideoWriter(overlay_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W,H))
            print(f"[viz ] Writing overlay to: {overlay_path}")

    # smoothing state per class label (simple scene assumption: one instance per class)
    ema_alpha = max(0.0, min(1.0, args.ema))
    ema_state = {}
    med_state = collections.defaultdict(lambda: collections.deque(maxlen=max(3, args.median_window)))
    kf_state  = {}

    def smooth_range(label, raw_r):
        if args.smooth == "none":
            return raw_r
        if args.smooth == "ema":
            prev = ema_state.get(label, raw_r)
            sm = ema_alpha * raw_r + (1.0 - ema_alpha) * prev
            ema_state[label] = sm
            return sm
        if args.smooth == "median":
            dq = med_state[label]
            dq.append(raw_r)
            return float(np.median(dq))
        if args.smooth == "kalman":
            kf = kf_state.get(label)
            if kf is None:
                kf = Kalman1D(x0=raw_r, Q=args.kf_q, R=args.kf_r)
                kf_state[label] = kf
            return float(kf.update(raw_r))
        return raw_r

    with open(out_csv,"w",newline="") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(["frame","cls","conf","x1","y1","x2","y2","u","v","X","Y","Z","range_m"])

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            dpath = depth_path(args.depth_metric_dir, frame_idx)
            depth = None
            if os.path.exists(dpath):
                depth = np.load(dpath).astype(np.float32)
                if depth.shape[:2] != (H,W):
                    depth = cv2.resize(depth,(W,H),interpolation=cv2.INTER_NEAREST)

            if depth is not None and frame_idx in by_frame:
                # per-frame reference lock
                frame_scale = 1.0
                if args.ref_class and args.ref_dist_m:
                    ref_det = next((dd for dd in by_frame[frame_idx]
                                    if str(dd["cls"]).lower().strip() == args.ref_class.lower().strip()), None)
                    if ref_det is not None:
                        ref_meas = measure_depth_for_box(
                            depth, (ref_det["x1"],ref_det["y1"],ref_det["x2"],ref_det["y2"]), W,H, inner=args.inner_roi
                        )
                        if np.isfinite(ref_meas) and ref_meas > 0.02:
                            frame_scale = float(args.ref_dist_m / ref_meas)

                for d in by_frame[frame_idx]:
                    center, roi, box = roi_center_and_inner(
                        (d["x1"],d["y1"],d["x2"],d["y2"]), W,H, inner=args.inner_roi)
                    if center is None: continue
                    u,v = center
                    Z = robust_depth_from_roi(depth, roi)
                    if not np.isfinite(Z) or Z < args.min_depth or Z > args.max_depth:
                        continue

                    # apply scales
                    # If scale_meta is already metric-calibrated, don't double-apply frame_scale
                    if args.scale_meta is None:
                        Z *= frame_scale
                    Z *= args.post_scale

                    X,Y,Z = lift(u,v,Z,fx,fy,cx,cy)
                    raw_r = float(np.sqrt(X*X + Y*Y + Z*Z))
                    sm_r  = smooth_range(d["cls"], raw_r)

                    # scale vector to match smoothed range (keeps direction)
                    if raw_r > 1e-6:
                        s = sm_r / raw_r
                        X, Y, Z = X*s, Y*s, Z*s

                    # label rounding for less flicker
                    disp_r = sm_r
                    if args.round_cm and args.round_cm > 0:
                        disp_r = round(disp_r * 100.0 / args.round_cm) * (args.round_cm / 100.0)

                    wcsv.writerow([frame_idx,d["cls"],d["conf"],d["x1"],d["y1"],d["x2"],d["y2"],u,v,X,Y,Z,sm_r])

                    if writer:
                        x1i,y1i,x2i,y2i = map(int, box)
                        cv2.rectangle(frame,(x1i,y1i),(x2i,y2i),(0,255,0),2)
                        cv2.circle(frame,(int(u),int(v)),3,(0,255,0),-1)
                        cv2.putText(frame, f"{d['cls']} {disp_r:.2f} m",
                                    (x1i, max(0,y1i-8)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0,255,0), 2, cv2.LINE_AA)

            if writer: writer.write(frame)
            frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    print(f"[done] 3D poses -> {out_csv}")
    # Report overlay status clearly
    if not args.no_overlay:
        if exists_nonempty(overlay_path):
            print(f"[viz ] Overlay available -> {overlay_path}")
        else:
            print("[viz ] Overlay not created (no frames written or skipped because file already existed).")

if __name__ == "__main__":
    main()


#  python src/lift3d.py --rgb "data_phone/raw/scene01_clip01.mp4" --detections_csv "results/csv/scene01_clip01_detections.csv" --depth_metric_dir "results/depth_metric/scene01_clip01" --scale_meta "results/depth_metric/scene01_clip01/scale_meta.json"
