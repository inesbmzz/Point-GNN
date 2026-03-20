# visualize_results.py — Usage Guide

## Basic Syntax

```bash
python3 visualize_results.py RESULTS_DIR \
    --dataset_root_dir DATASET_DIR \
    [options]
```

---

## Modes (`--mode`)

### `image` — Camera image with 2D boxes *(best for reports)*

```bash
# With LiDAR depth overlay
python3 visualize_results.py results/car_T3_val/ \
    --dataset_root_dir ../data/ --mode image --frame 4

# Clean boxes only (no LiDAR dots)
python3 visualize_results.py results/car_T3_val/ \
    --dataset_root_dir ../data/ --mode image --frame 4 --no-lidar

# All frames at once
python3 visualize_results.py results/car_T3_val/ \
    --dataset_root_dir ../data/ --mode image --frame -1 --no-lidar
```

Output: `viz/000005_image.png` — open anywhere.

---

### `bev` — Bird's-eye view PNG *(good overview)*

```bash
python3 visualize_results.py results/car_T3_val/ \
    --dataset_root_dir ../data/ --mode bev --frame 4
```

Output: `viz/000005_bev.png` — top-down view, arrows show car orientation.

---

### `ply` — 3D point cloud *(open in CloudCompare)*

```bash
python3 visualize_results.py results/car_T3_val/ \
    --dataset_root_dir ../data/ --mode ply --frame 4
```

Output: `viz/000005.ply` — camera-FOV LiDAR points + colored box wireframes.

---

### `html` — Interactive 3D *(open in any browser)*

```bash
# Requires: pip install plotly
python3 visualize_results.py results/car_T3_val/ \
    --dataset_root_dir ../data/ --mode html --frame 4
```

Output: `viz/000005.html` — rotate/zoom in browser.

---

## All Options

| Option | Default | Description |
|---|---|---|
| `--mode` | `html` | Output format: `image`, `bev`, `ply`, `html` |
| `--frame` | `0` | Frame index to export. Use `-1` for all frames |
| `--score_thresh` | `0.1` | Minimum detection score to display |
| `--no-lidar` | off | Hide LiDAR dot overlay (`image` mode only) |
| `--no-gt` | off | Hide ground truth boxes |
| `--no-pred` | off | Hide predicted boxes |
| `--dataset_root_dir` | `../dataset/kitti/` | Path to KITTI dataset root |
| `--dataset_split_file` | `3DOP_splits/val.txt` | Custom split file |
| `--output_dir` | `RESULTS_DIR/viz/` | Where to save output files |

---

## Color Legend (all modes)

| Color | Meaning |
|---|---|
| Green | Ground truth label |
| Red | Car prediction |
| Blue | Pedestrian prediction |
| Orange | Cyclist prediction |

---

## Controlling Box Visibility

```bash
# Point cloud only — no boxes
python3 visualize_results.py results/car_T3_val/ \
    --dataset_root_dir ../data/ --mode ply --frame 4 --no-gt --no-pred

# Ground truth only
python3 visualize_results.py results/car_T3_val/ \
    --dataset_root_dir ../data/ --mode bev --frame 4 --no-pred

# Predictions only
python3 visualize_results.py results/car_T3_val/ \
    --dataset_root_dir ../data/ --mode image --frame 4 --no-gt
```

---

## Typical Workflow for a Report

```bash
# 1. Quick scan — generate all BEV images cheaply to find interesting frames
python3 visualize_results.py results/car_T3_val/ \
    --dataset_root_dir ../data/ --mode bev --frame -1

# 2. Pick interesting frames, then generate high-quality outputs for those
python3 visualize_results.py results/car_T3_val/ \
    --dataset_root_dir ../data/ --mode image --frame 42 --no-lidar

python3 visualize_results.py results/car_T3_val/ \
    --dataset_root_dir ../data/ --mode html --frame 42

# 3. Download results to local machine
scp -r user@hpc:/path/to/results/car_T3_val/viz/ ./viz_local/
```
