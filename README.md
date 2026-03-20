# Point-GNN:Reproduction Study

> **NPM3D Project:Master IASD, March 2026**
>
> This fork reproduces the results of Point-GNN (CVPR 2020) on the KITTI benchmark,
> performs an ablation study, and evaluates cross-dataset generalization on nuScenes mini.
>
> **Key results:**
> - Reproduced KITTI val Car 3D AP: 87.88 / 78.34 / 77.39 (Easy / Moderate / Hard):matching paper numbers to within rounding error
> - Ablation study fully reproduced (box merge, box score, auto-registration)
> - Zero detections on nuScenes 32-beam data:confirming sensor-specific training dependency
>
> **Added scripts:**
> - [`visualize_results.py`](visualize_results.py):headless visualization (no display needed). Outputs camera images, BEV PNGs, PLY files for CloudCompare, and interactive HTML. See [VISUALIZE_GUIDE.md](VISUALIZE_GUIDE.md) for usage.

---

## Cross-Dataset Evaluation on nuScenes

This section explains how to evaluate the pretrained Point-GNN checkpoint on the
nuScenes dataset by converting it to KITTI format.

> **Note:** Our experiment shows that the model produces **zero detections** on
> nuScenes 32-beam data without retraining, due to the 7× reduction in point
> density compared to KITTI's 64-beam sensor. This is documented as a generalization
> failure in our report.

### 1:Install nuscenes-devkit

```bash
pip install nuscenes-devkit --no-deps
```

`--no-deps` avoids reinstalling packages you already have.

### 2:Download nuScenes mini

Register and download from [https://www.nuscenes.org/download](https://www.nuscenes.org/download).
Select **nuScenes mini** (~4 GB). Extract to a folder, e.g. `/data/nuscenes/`.

The extracted structure should be:
```
/data/nuscenes/
├── maps/
├── samples/
├── sweeps/
└── v1.0-mini/
```

### 3:Fix the export script

The installed `export_kitti.py` has a bug where `dataroot` is not passed to the
`NuScenes` constructor. Find and fix it:

```bash
# Find the installed script
python3 -c "import nuscenes; print(nuscenes.__file__)"
# Navigate to nuscenes/scripts/export_kitti.py and find the line:
#   self.nusc = NuScenes(version=nusc_version)
# Replace with:
#   self.nusc = NuScenes(version=nusc_version, dataroot=dataroot)
```

### 4:Convert nuScenes mini to KITTI format

```bash
python3 -m nuscenes.scripts.export_kitti nuscenes_gt_to_kitti \
    --nusc_version v1.0-mini \
    --nusc_kitti_dir /data/nuscenes_kitti/ \
    --dataroot /data/nuscenes/ \
    --split mini_val
```

### 5:Rename UUID filenames to sequential numbers

nuScenes exports files with UUID names; Point-GNN expects sequential 6-digit names:

```bash
python3 - <<'EOF'
import os, glob

src = "/data/nuscenes_kitti/mini_val"
subdirs = {"velodyne": ".bin", "image_2": ".png", "label_2": ".txt", "calib": ".txt"}

bins = sorted(glob.glob(os.path.join(src, "velodyne", "*.bin")))
stems = [os.path.splitext(os.path.basename(b))[0] for b in bins]

for subdir, ext in subdirs.items():
    for i, stem in enumerate(stems):
        s = os.path.join(src, subdir, stem + ext)
        d = os.path.join(src, subdir, f"{i:06d}" + ext)
        if os.path.exists(s) and not os.path.exists(d):
            os.rename(s, d)

split = "/data/nuscenes_kitti/3DOP_splits/mini_val.txt"
os.makedirs(os.path.dirname(split), exist_ok=True)
with open(split, "w") as f:
    f.writelines(f"{i:06d}\n" for i in range(len(stems)))

print(f"Done: {len(stems)} frames")
EOF
```

### 6:Set up the directory structure

Point-GNN expects a specific directory layout. Create symlinks to match it:

```bash
BASE=/data/nuscenes_kitti
SRC=$BASE/mini_val

mkdir -p $BASE/calib/training/calib
mkdir -p $BASE/velodyne/training/velodyne
mkdir -p $BASE/labels/training/label_2
mkdir -p $BASE/image/training/image_2
mkdir -p $BASE/3DOP_splits

ln -s $SRC/calib/*    $BASE/calib/training/calib/
ln -s $SRC/velodyne/* $BASE/velodyne/training/velodyne/
ln -s $SRC/label_2/*  $BASE/labels/training/label_2/
ln -s $SRC/image_2/*  $BASE/image/training/image_2/
```

### 7:Run inference

```bash
python3 run.py checkpoints/car_auto_T3_train/ \
    --dataset_root_dir /data/nuscenes_kitti/ \
    --dataset_split_file /data/nuscenes_kitti/3DOP_splits/mini_val.txt \
    --output_dir results/car_nuscenes_mini/
```

### 8:Evaluate

```bash
./kitti_native_evaluation/evaluate_object_offline \
    /data/nuscenes_kitti/labels/training/label_2/ \
    results/car_nuscenes_mini/
```

---

# Point-GNN (Original)

This repository contains a reference implementation of our [Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Point-GNN_Graph_Neural_Network_for_3D_Object_Detection_in_a_CVPR_2020_paper.pdf), CVPR 2020.

If you find this code useful in your research, please consider citing our work:
```
@InProceedings{Point-GNN,
author = {Shi, Weijing and Rajkumar, Ragunathan (Raj)},
title = {Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Getting Started

### Prerequisites

We use Tensorflow 1.15 for this implementation. Please [install CUDA](https://developer.nvidia.com/cuda-10.0-download-archive) if you want GPU support.   
```
pip3 install --user tensorflow-gpu==1.15.0
```

To install other dependencies: 
```
pip3 install --user opencv-python
pip3 install --user open3d-python==0.7.0.0
pip3 install --user scikit-learn
pip3 install --user tqdm
pip3 install --user shapely
```

### KITTI Dataset

We use the KITTI 3D Object Detection dataset. Please download the dataset from the KITTI website and also download the 3DOP train/val split [here](https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz). We provide extra split files for seperated classes in [splits/](splits). We recommand the following file structure:

    DATASET_ROOT_DIR
    ├── image                    #  Left color images
    │   ├── training
    |   |   └── image_2            
    │   └── testing
    |       └── image_2 
    ├── velodyne                 # Velodyne point cloud files
    │   ├── training
    |   |   └── velodyne            
    │   └── testing
    |       └── velodyne 
    ├── calib                    # Calibration files
    │   ├── training
    |   |   └──calib            
    │   └── testing
    |       └── calib 
    ├── labels                   # Training labels
    │   └── training
    |       └── label_2
    └── 3DOP_splits              # split files.
        ├── train.txt
        ├── train_car.txt
        └── ...

### Download Point-GNN

Clone the repository recursively:
```
git clone https://github.com/WeijingShi/Point-GNN.git --recursive
```

## Inference
### Run a checkpoint
Test on the validation split:
```
python3 run.py checkpoints/car_auto_T3_train/ --dataset_root_dir DATASET_ROOT_DIR --output_dir DIR_TO_SAVE_RESULTS
```
Test on the test dataset:
```
python3 run.py checkpoints/car_auto_T3_trainval/ --test --dataset_root_dir DATASET_ROOT_DIR --output_dir DIR_TO_SAVE_RESULTS
```

```
usage: run.py [-h] [-l LEVEL] [--test] [--no-box-merge] [--no-box-score]
              [--dataset_root_dir DATASET_ROOT_DIR]
              [--dataset_split_file DATASET_SPLIT_FILE]
              [--output_dir OUTPUT_DIR]
              checkpoint_path

Point-GNN inference on KITTI

positional arguments:
  checkpoint_path       Path to checkpoint

optional arguments:
  -h, --help            show this help message and exit
  -l LEVEL, --level LEVEL
                        Visualization level, 0 to disable,1 to nonblocking
                        visualization, 2 to block.Default=0
  --test                Enable test model
  --no-box-merge        Disable box merge.
  --no-box-score        Disable box score.
  --dataset_root_dir DATASET_ROOT_DIR
                        Path to KITTI dataset. Default="../dataset/kitti/"
  --dataset_split_file DATASET_SPLIT_FILE
                        Path to KITTI dataset split
                        file.Default="DATASET_ROOT_DIR/3DOP_splits/val.txt"
  --output_dir OUTPUT_DIR
                        Path to save the detection
                        resultsDefault="CHECKPOINT_PATH/eval/"
```
### Performance
Install kitti_native_evaluation offline evaluation:
```
cd kitti_native_evaluation
cmake ./
make
```
Evaluate output results on the validation split:
```
evaluate_object_offline DATASET_ROOT_DIR/labels/training/label_2/ DIR_TO_SAVE_RESULTS
```

## Training
We put training parameters in a train_config file. To start training, we need both the train_config and config.
```
usage: train.py [-h] [--dataset_root_dir DATASET_ROOT_DIR]
                [--dataset_split_file DATASET_SPLIT_FILE]
                train_config_path config_path

Training of PointGNN

positional arguments:
  train_config_path     Path to train_config
  config_path           Path to config

optional arguments:
  -h, --help            show this help message and exit
  --dataset_root_dir DATASET_ROOT_DIR
                        Path to KITTI dataset. Default="../dataset/kitti/"
  --dataset_split_file DATASET_SPLIT_FILE
                        Path to KITTI dataset split file.Default="DATASET_ROOT
                        _DIR/3DOP_splits/train_config["train_dataset"]"
```
For example:
```
python3 train.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config
```
We strongly recommand readers to view the train_config before starting the training. 
Some common parameters which you might want to change first:
```
train_dir     The directory where checkpoints and logs are stored.
train_dataset The dataset split file for training. 
NUM_GPU       The number of GPUs to use. We used two GPUs for the reference model. 
              If you want to use a single GPU, you might also need to reduce the batch size by half to save GPU memory.
              Similarly, you might want to increase the batch size if you want to utilize more GPUs. 
              Check the train.py for details.               
```
We also provide an evaluation script to evaluate the checkpoints periodically. For example:
```
python3 eval.py configs/car_auto_T3_train_eval_config 
```
You can use tensorboard to view the training and evaluation status. 
```
tensorboard --logdir=./train_dir
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


