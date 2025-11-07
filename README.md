# Single-RGB 3D Perception for Indoor Robotics

## This project implements a full experimental pipeline for 3D object localization and range estimation using a **single RGB camera**.

## Requirements

- Python 3.10+
- Anaconda or venv environment recommended

### Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, the main packages are:

```bash
pip install torch torchvision ultralytics opencv-python pandas matplotlib open3d
```

---

## Project Structure

```
src/
 ├── detect.py          # YOLO-based object detection
 ├── mono_depth.py      # Monocular depth estimation
 ├── scale_fix.py       # Scale calibration between scenes
 ├── lift3d.py          # 3D point lifting and range computation
 ├── make_groundtruth.py# Ground truth generation
 ├── evaluate.py        # Evaluation metrics and plotting
 └── run_all.py         # Full pipeline orchestrator
data_phone/
 └── raw/               # Input RGB videos
results/
 └── csv/               # Output detections and evaluation results
```

---

## How to Run

### 1. Activate environment

```bash
conda activate phonevision-gpu
```

### 2. Run the full pipeline

```bash
python src/run_all.py
```

### 3. Or run individual stages

```bash
python src/detect.py --input data_phone/raw/scene01_clip01.mp4
python src/mono_depth.py
python src/scale_fix.py
python src/lift3d.py
python src/evaluate.py
```

---

## Notes

- All paths are relative to the project root.
- Input videos go in `data_phone/raw/`.
- Results and plots are saved automatically under `results/`.

---

## Author

**Ethugal Umedha Rajaratne**  
MSc Robotics, University of Plymouth  
2025
