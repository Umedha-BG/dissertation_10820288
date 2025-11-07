#!/usr/bin/env python3
"""
Master pipeline runner for Single-RGB 3D Perception Experiments.
Now includes:
  - Auto calibration management (create once, then reuse)
  - Runtime profiling per stage: seconds, ms/frame, FPS
  - Peak RAM and GPU usage (via psutil + pynvml / nvidia-smi fallback)
Stages:
  1) detect.py
  2) mono_depth.py
  3) scale_fix.py (create/reuse calibration, apply to main video)
  4) lift3d.py
  5) make_groundtruth.py
  6) evaluate.py
Outputs:
  - results/profile/<video>_runtime.csv
"""

import os
import time
import csv
import subprocess
import torch
import pandas as pd
import cv2
import psutil

# optional GPU sampling
try:
    import pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def run(cmd):
    print(f"\n[RUN] {cmd}\n{'-'*80}")
    subprocess.run(cmd, shell=True, check=True)

def exists(path):
    return os.path.exists(path) and (os.path.getsize(path) > 0 if os.path.isfile(path) else True)

def skip(msg):
    print(f"[skip] {msg} (already done)")

class ResourceMonitor:
    """
    Periodically samples process RAM (RSS) and GPU usage while a stage runs.
    Usage:
        mon = ResourceMonitor(sample_sec=0.5, gpu_index=0).start()
        ... run stage ...
        mon.stop()
        -> mon.peak_rss_mb, mon.peak_gpu_mem_mb, mon.peak_gpu_util
    """
    def __init__(self, sample_sec=0.5, gpu_index=0):
        self.sample_sec = float(sample_sec)
        self.gpu_index = int(gpu_index)
        self._running = False
        self._thread = None
        self.peak_rss_mb = 0.0
        self.peak_gpu_mem_mb = None
        self.peak_gpu_util = None
        self._proc = psutil.Process()

        # NVML
        self._nvml_ok = False
        if _NVML_OK:
            try:
                pynvml.nvmlInit()
                self._nvml_ok = True
                self._h = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except Exception:
                self._nvml_ok = False

    def _sample_once(self):
        # RAM
        try:
            rss = self._proc.memory_info().rss / (1024 * 1024)
            if rss > self.peak_rss_mb:
                self.peak_rss_mb = rss
        except Exception:
            pass

        # GPU
        try:
            if self._nvml_ok:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._h)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._h)
                mem_mb = mem.used / (1024 * 1024)
                if (self.peak_gpu_mem_mb is None) or (mem_mb > self.peak_gpu_mem_mb):
                    self.peak_gpu_mem_mb = mem_mb
                if (self.peak_gpu_util is None) or (util.gpu > self.peak_gpu_util):
                    self.peak_gpu_util = float(util.gpu)
            else:
                # nvidia-smi fallback
                out = subprocess.run(
                    ["nvidia-smi", f"--id={self.gpu_index}",
                     "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True
                )
                if out.returncode == 0:
                    line = out.stdout.strip().split("\n")[0]
                    mem_mb_str, util_str = [t.strip() for t in line.split(",")]
                    mem_mb = float(mem_mb_str)
                    util = float(util_str)
                    if (self.peak_gpu_mem_mb is None) or (mem_mb > self.peak_gpu_mem_mb):
                        self.peak_gpu_mem_mb = mem_mb
                    if (self.peak_gpu_util is None) or (util > self.peak_gpu_util):
                        self.peak_gpu_util = util
        except Exception:
            pass

    def _loop(self):
        while self._running:
            self._sample_once()
            time.sleep(self.sample_sec)

    def start(self):
        self._running = True
        import threading
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._nvml_ok:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        if self.peak_gpu_mem_mb is not None:
            self.peak_gpu_mem_mb = round(self.peak_gpu_mem_mb, 1)
        if self.peak_gpu_util is not None:
            self.peak_gpu_util = round(self.peak_gpu_util, 1)
        self.peak_rss_mb = round(self.peak_rss_mb, 1)

def profile_stage(name, cmd, frame_count, results, gpu_index=0):
    print(f"\n[PROFILE] Starting: {name}")
    mon = ResourceMonitor(sample_sec=0.5, gpu_index=gpu_index).start()
    t0 = time.perf_counter()
    subprocess.run(cmd, shell=True, check=True)
    dt = time.perf_counter() - t0
    mon.stop()

    msf = (dt / max(frame_count, 1)) * 1000.0
    fps = frame_count / max(dt, 1e-9)

    row = {
        "stage": name,
        "seconds": round(dt, 3),
        "ms_per_frame": round(msf, 3),
        "fps": round(fps, 3),
        "peak_rss_mb": mon.peak_rss_mb,
        "peak_gpu_mem_mb": mon.peak_gpu_mem_mb if mon.peak_gpu_mem_mb is not None else "NA",
        "peak_gpu_util_pct": mon.peak_gpu_util if mon.peak_gpu_util is not None else "NA",
    }
    results.append(row)

    print(f"[PROFILE] {name}: {dt:.2f}s  ({msf:.2f} ms/frame, {fps:.2f} FPS)  "
          f"RAM_peak={row['peak_rss_mb']} MB  "
          f"GPU_mem_peak={row['peak_gpu_mem_mb']} MB  "
          f"GPU_util_peak={row['peak_gpu_util_pct']}%")

# ------------------------------------------------------------
# Main orchestrator
# ------------------------------------------------------------
def main():
    print("=== Single RGB 3D Perception - Full Pipeline ===\n")

    # 1) Ask for video name
    video_name = input("Enter video filename (e.g. phaseA_clip02.mp4): ").strip()
    base = os.path.splitext(video_name)[0]

    # 2) Choose device
    auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_choice = input(f"Use GPU or CPU? [cuda/cpu, default={auto_device}]: ").strip().lower()
    if device_choice not in ["cuda", "cpu"]:
        device_choice = auto_device
    print(f"→ Using device: {device_choice}")

    # 3) Paths
    raw_video = f"data_phone/raw/{video_name}"
    depth_dir = f"results/depth/{base}"
    depth_metric_dir = f"results/depth_metric/{base}"
    csv_path = f"results/csv/{base}_detections.csv"
    scale_meta = f"{depth_metric_dir}/scale_meta.json"
    calib_file = "data_phone/calib/scale_meta.json"
    poses3d_dir = "results/poses3d"
    gt_csv = f"data_phone/groundtruth_{base}.csv"
    eval_dir = f"results/eval/{base}"
    profile_dir = "results/profile"
    os.makedirs("results/csv", exist_ok=True)
    os.makedirs("results/depth", exist_ok=True)
    os.makedirs("results/depth_metric", exist_ok=True)
    os.makedirs("results/poses3d", exist_ok=True)
    os.makedirs("results/eval", exist_ok=True)
    os.makedirs("data_phone/calib", exist_ok=True)
    os.makedirs(profile_dir, exist_ok=True)

    # count frames for profiling denominators
    cap = cv2.VideoCapture(raw_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()

    results = []
    gpu_index = 0  # change if needed

    # --------------------------------------------------------
    # STEP 1: YOLO Object Detection
    # --------------------------------------------------------
    if not exists(csv_path):
        cmd = (f'python src/detect.py --input "{raw_video}" '
               f'--device {device_choice} --stride 1 --conf 0.15 --weights "yolo11s.pt"')
        profile_stage("detect", cmd, total_frames, results, gpu_index=gpu_index)
    else:
        skip("Detection CSV found")

    # --------------------------------------------------------
    # STEP 2: Monocular Depth (MiDaS)
    # --------------------------------------------------------
    if not exists(depth_dir):
        cmd = (f'python src/mono_depth.py --input "{raw_video}" '
               f'--device {device_choice} --stride 1 --model dpt_hybrid')
        profile_stage("mono_depth", cmd, total_frames, results, gpu_index=gpu_index)
    else:
        skip("Depth results found")

    # --------------------------------------------------------
    # STEP 3: Calibration Management (A4 or reuse) + APPLY to main video
    # --------------------------------------------------------
    if not exists(calib_file):
        print("\n[STEP 3] No calibration found — let's create one using an A4 sheet.")
        print("Record a short calibration video (e.g., data_phone/raw/calibration_A4.mp4) "
              "with an A4 sheet visible and mostly frontal.")
        calib_video = input("Enter calibration video filename (e.g. calibration_A4.mp4): ").strip()
        calib_base = os.path.splitext(calib_video)[0]
        calib_raw = f"data_phone/raw/{calib_video}"

        known_distance_cm = input("Enter known A4 sheet distance in cm (e.g. 60): ").strip()
        try:
            known_distance_cm = float(known_distance_cm)
        except ValueError:
            known_distance_cm = 60.0
        print(f"→ Calibrating using {calib_video} at {known_distance_cm:.1f} cm")

        # Depth for calibration clip
        if not exists(f"results/depth/{calib_base}"):
            cmd = (f'python src/mono_depth.py --input "{calib_raw}" '
                   f'--device {device_choice} --stride 1 --model dpt_hybrid')
            profile_stage("calib_depth", cmd, total_frames, results, gpu_index=gpu_index)
        else:
            skip("Calibration depth found")

        # Perform calibration (writes to data_phone/calib/scale_meta.json)
        cmd = (f'python src/scale_fix.py --depth_dir "results/depth/{calib_base}" '
               f'--rgb "{calib_raw}" --out_dir "data_phone/calib" '
               f'--calibrate_from_frame 1 --known_distance_cm {known_distance_cm} --affine_fit')
        profile_stage("calib_fit", cmd, total_frames, results, gpu_index=gpu_index)
        print(f"[done] Calibration saved to {calib_file}")

    else:
        print(f"\n[STEP 3] Using existing calibration from {calib_file}")

    # Apply (or re-apply) calibration to MAIN video depths -> metric depths
    if not exists(scale_meta):
        cmd = (f'python src/scale_fix.py --depth_dir "{depth_dir}" '
               f'--rgb "{raw_video}" --out_dir "{depth_metric_dir}" '
               f'--use_calibration "{calib_file}"')
        profile_stage("scale_apply", cmd, total_frames, results, gpu_index=gpu_index)
    else:
        skip("Metric depth + scale_meta found")

    # --------------------------------------------------------
    # STEP 4: Lift to 3D
    # --------------------------------------------------------
    pred_csv = f"{poses3d_dir}/{base}_3d.csv"
    if not exists(pred_csv):
        cmd = (f'python src/lift3d.py --rgb "{raw_video}" '
               f'--detections_csv "{csv_path}" '
               f'--depth_metric_dir "{depth_metric_dir}" '
               f'--scale_meta "{scale_meta}" '
               f'--ref_class laptop --ref_dist_m 1.0 '
               f'--inner_roi 0.45 '
               f'--smooth kalman --kf_q 1e-3 --kf_r 1e-2 --round_cm 1')
        profile_stage("lift3d", cmd, total_frames, results, gpu_index=gpu_index)
    else:
        skip("3D pose results found")

    # --------------------------------------------------------
    # STEP 5: Ground Truth Creation
    # --------------------------------------------------------
    print("\n[STEP 5] Ground-truth generation or loading...")

    try:
        df = pd.read_csv(csv_path)
        if 'cls' in df.columns:
            classes = sorted(df['cls'].dropna().unique().tolist())
        elif 'cls_name' in df.columns:
            classes = sorted(df['cls_name'].dropna().unique().tolist())
        else:
            classes = []
    except Exception as e:
        print(f"[warn] Could not auto-detect classes from {csv_path}: {e}")
        classes = []

    if not classes:
        classes = input("Enter object class names (space-separated): ").strip().split()

    cap = cv2.VideoCapture(raw_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()

    if not exists(gt_csv):
        mode = input("Is the ground truth fixed or movement? [fixed/move]: ").strip().lower()
        if mode.startswith("f"):
            dists = input(f"Enter ground-truth distances (cm) for {classes}, separated by commas: ").strip()
            distances = [d.strip() for d in dists.split(",")]
            if len(distances) != len(classes):
                print("[error] Number of distances must match number of classes!")
                exit(1)
            safe_objects = [f'"{c}"' if " " in c else c for c in classes]
            run(f'python src/make_groundtruth.py --out "{gt_csv}" '
                f'--frames {total_frames} '
                f'--objects {" ".join(safe_objects)} '
                f'--distances {" ".join(distances)}')
        else:
            csv_in = input("Enter the path or filename of the moving ground-truth CSV: ").strip()
            run(f'python src/make_groundtruth.py --out "{gt_csv}" --from_csv "{csv_in}"')
    else:
        skip("Ground-truth file found")

    # --------------------------------------------------------
    # STEP 6: Evaluation
    # --------------------------------------------------------
    if not exists(f"{eval_dir}/summary.csv"):
        cmd = (f'python src/evaluate.py --pred_csv "{pred_csv}" '
               f'--truth_csv "{gt_csv}" --out_dir "{eval_dir}"')
        profile_stage("evaluate", cmd, total_frames, results, gpu_index=gpu_index)
    else:
        skip("Evaluation results found")

    # --------------------------------------------------------
    # Runtime profile summary
    # --------------------------------------------------------
    print("\n=== Runtime Profile ===")
    print("Stage          | Seconds | ms/frame |  FPS  | Peak RAM(MB) | Peak GPU Mem(MB) | Peak GPU Util(%)")
    print("------------------------------------------------------------------------------------------------")
    for r in results:
        print(f"{r['stage']:<14} | {r['seconds']:7.2f} | {r['ms_per_frame']:8.2f} | {r['fps']:5.1f} | "
              f"{r['peak_rss_mb']:12} | {str(r['peak_gpu_mem_mb']):16} | {str(r['peak_gpu_util_pct']):16}")
    print("------------------------------------------------------------------------------------------------")

    out_csv = os.path.join(profile_dir, f"{base}_runtime.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "stage","seconds","ms_per_frame","fps",
            "peak_rss_mb","peak_gpu_mem_mb","peak_gpu_util_pct",
            "frames","device"
        ])
        w.writeheader()
        for r in results:
            row = dict(r)
            row["frames"] = total_frames
            row["device"] = device_choice
            w.writerow(row)
    print(f"[done] Runtime summary saved to {out_csv}")

    # --------------------------------------------------------
    print("\n✅ All processing steps completed successfully!")
    print(f"📊 Evaluation results saved to: {eval_dir}/")
    print(f"🎥 Overlay video: results/poses3d/{base}_overlay.mp4")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
