#!/usr/bin/env python3
"""
Compare predicted 3D ranges vs. ground-truth tape measurements.

Inputs
------
--pred_csv   : results/poses3d/<clip>_3d.csv   (from lift3d.py)
--truth_csv  : data_phone/groundtruth_<clip>.csv
               columns: frame,cls,range_cm
--out_dir    : results/eval/<clip>

Notes
-----
- Matching key = (frame, cls). Class names are lowercased & stripped.
- You can alias classes (e.g., cup->bottle) with --rename.
"""

import os, argparse, csv, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_preds(path):
    df = pd.read_csv(path)
    # expected columns from lift3d: frame,cls,conf,x1,y1,x2,y2,u,v,X,Y,Z,range_m
    need = {"frame","cls","range_m"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} missing columns {need}")
    df = df.copy()
    df["cls"] = df["cls"].astype(str).str.strip().str.lower()
    df["frame"] = df["frame"].astype(int)
    df["pred_m"] = df["range_m"].astype(float)
    df["pred_cm"] = df["pred_m"] * 100.0
    return df[["frame","cls","pred_cm"]]

def load_truth(path):
    df = pd.read_csv(path)
    need = {"frame","cls","range_cm"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must have columns: frame,cls,range_cm")
    df = df.copy()
    df["cls"] = df["cls"].astype(str).str.strip().str.lower()
    df["frame"] = df["frame"].astype(int)
    df["gt_cm"] = df["range_cm"].astype(float)
    return df[["frame","cls","gt_cm"]]

def apply_renames(df, rename):
    if not rename:
        return df
    m = {k.lower(): v.lower() for k,v in rename.items()}
    df = df.copy()
    df["cls"] = df["cls"].apply(lambda c: m.get(str(c).lower(), str(c).lower()))
    return df

def metrics(cm_err):
    cm_err = np.asarray(cm_err, dtype=float)
    cm_ae = np.abs(cm_err)
    return {
        "N": int((~np.isnan(cm_err)).sum()),
        "MAE_cm": float(np.nanmean(cm_ae)) if cm_ae.size else np.nan,
        "MedianAE_cm": float(np.nanmedian(cm_ae)) if cm_ae.size else np.nan,
        "RMSE_cm": float(np.sqrt(np.nanmean(cm_err**2))) if cm_err.size else np.nan,
    }

def moving_std(x, w=5):
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    if n == 0:
        return np.array([])
    out = []
    for i in range(n):
        a = max(0, i - (w - 1))
        seg = arr[a:i + 1]
        out.append(float(np.nanstd(seg)))
    return np.asarray(out)

def plot_series(out_dir, cls_name, df_cls):
    ensure_dir(out_dir)
    t = df_cls["frame"].to_numpy()
    gt = df_cls["gt_cm"].to_numpy()
    pr = df_cls["pred_cm"].to_numpy()
    err = pr - gt

    # Error over time
    plt.figure()
    plt.plot(t, gt, label="GT (cm)")
    plt.plot(t, pr, label="Pred (cm)")
    plt.xlabel("frame"); plt.ylabel("distance (cm)"); plt.title(f"{cls_name}: distance vs frame")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cls_name}_series.png")); plt.close()

    # Error histogram
    plt.figure()
    plt.hist(err[~np.isnan(err)], bins=30)
    plt.xlabel("error (cm)"); plt.ylabel("count"); plt.title(f"{cls_name}: error histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cls_name}_hist.png")); plt.close()

    # Jitter plot (moving std of pred)
    plt.figure()
    jit = moving_std(pr, w=5)
    plt.plot(t, jit)
    plt.xlabel("frame"); plt.ylabel("moving std (cm)"); plt.title(f"{cls_name}: temporal jitter (5-frame std)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cls_name}_jitter.png")); plt.close()

def main():
    ap = argparse.ArgumentParser("Evaluate predicted ranges vs ground truth")
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--truth_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rename", nargs="+", default=[],
                    help="Optional class renames like cup:bottle mug:bottle")
    args = ap.parse_args()

    rename = {}
    for p in args.rename:
        if ":" in p:
            k,v = p.split(":",1)
            rename[k.strip().lower()] = v.strip().lower()

    ensure_dir(args.out_dir)
    preds = load_preds(args.pred_csv)
    truth = load_truth(args.truth_csv)
    preds = apply_renames(preds, rename)
    truth = apply_renames(truth, rename)

    # join on (frame, cls)
    joined = pd.merge(truth, preds, on=["frame","cls"], how="inner")
    if joined.empty:
        raise SystemExit("No matching (frame, cls) between prediction and ground truth.")

    joined["err_cm"] = joined["pred_cm"] - joined["gt_cm"]
    joined.to_csv(Path(args.out_dir, "by_frame.csv"), index=False)

    # per-class metrics
    rows = []
    for cls_name, dfc in joined.groupby("cls"):
        m = metrics(dfc["err_cm"].to_numpy())
        # add jitter stat (std of pred over time)
        jitter_cm = float(np.nanstd(dfc["pred_cm"].to_numpy())) if len(dfc) > 1 else np.nan
        m["JitterSTD_cm"] = jitter_cm
        m["cls"] = cls_name
        rows.append(m)
        # plots
        plot_series(args.out_dir, cls_name.replace(" ","_"), dfc.sort_values("frame"))

    # overall metrics
    m_all = metrics(joined["err_cm"].to_numpy())
    m_all["JitterSTD_cm"] = float(np.nanstd(joined["pred_cm"].to_numpy()))
    m_all["cls"] = "OVERALL"
    rows.append(m_all)

    out_sum = pd.DataFrame(rows)[["cls","N","MAE_cm","MedianAE_cm","RMSE_cm","JitterSTD_cm"]]
    out_sum.to_csv(Path(args.out_dir, "summary.csv"), index=False)

    print(f"[done] wrote {Path(args.out_dir, 'by_frame.csv')}")
    print(f"[done] wrote {Path(args.out_dir, 'summary.csv')}")
    print(f"[done] plots saved under {args.out_dir}/")

if __name__ == "__main__":
    main()


# python src/evaluate.py --pred_csv "results/poses3d/phaseA_mix.csv" --truth_csv "data_phone/ground_truth_sample.csv" --out_dir "results/eval/phaseA_mix" --rename cup:bottle
