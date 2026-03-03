# TempoVLM

Temporal adapter for Vision-Language Models with GRU memory, depth regression, and motion prediction.

TempoVLM extends `Qwen2-VL` visual features with a lightweight multi-task head to improve robustness under occlusion and support geometry-aware tasks.

## Highlights

- GRU-based long-term temporal memory
- Adaptive occlusion handling with memory-based feature injection
- Absolute depth regression (left/center/right regions)
- 6-DoF camera motion prediction
- End-to-end demos and benchmark scripts for ScanNet, TUM RGB-D, and NYU Depth V2

## Repository Structure

```text
.
├── models_unified.py             # Main model + loss
├── models_ablation.py            # Ablation/Model-C variant (no depth scale head)
├── train.py                      # Main training pipeline
├── train_model_c.py              # Model-C training pipeline
├── train_finetune.py             # Geometry-focused fine-tuning
├── complete_demo.py              # End-to-end visual demo + report
```

## Requirements

- Python 3.10+
- PyTorch 2.x
- CUDA-capable GPU recommended

Install dependencies:

```bash
pip install -r requirements.txt
pip install qwen-vl-utils
```

## Quick Start

1. Prepare data (ScanNet or TUM; NYU optional).
2. Train (or use an existing checkpoint).
3. Run evaluation and/or demo scripts.

### 1) Data Preparation

ScanNet:

```bash
python download_scannet_dataset.py
```

Expected ScanNet layout:

```text
scannet_data/
├── scannet_frames_25k/
│   └── sceneXXXX_00/
│       ├── color/*.jpg
│       ├── depth/*.png
│       └── pose/*.txt
└── scannet_frames_test/
```

TUM RGB-D:

```bash
bash download_tum.sh
```

NYU Depth V2 (Hugging Face conversion):

```bash
python prepare_nyu_from_hf.py
```

### 2) Training

Main training (`train.py`):

- without `--use_gru`: baseline unified trainer
- with `--use_gru`: GRU sequence trainer (recommended)

```bash
python train.py \
  --data_root ./scannet_data \
  --output_dir ./checkpoints_unified \
  --use_gru \
  --tasks temporal depth_regression motion \
  --epochs 10 \
  --batch_size 2 \
  --lr 1e-4 \
  --max_scenes 100
```

Model-C / ablation training:

```bash
python train_model_c.py \
  --data_root ./scannet_data \
  --output_dir ./checkpoints_model_c \
  --tasks temporal depth_regression motion \
  --epochs 10 \
  --batch_size 2
```

Geometry fine-tuning from an existing checkpoint:

```bash
python train_finetune.py \
  --resume <base_checkpoint.pt> \
  --data_root ./scannet_data \
  --output_dir ./checkpoints_geometry \
  --epochs 5
```

## Evaluation

### TUM Depth

```bash
python test_tum_regression.py \
  --model_path <checkpoint.pt> \
  --sequence_path ./tum_data/rgbd_dataset_freiburg1_desk
```

- Default mode: median-scale aligned evaluation
- Absolute-scale mode: add `--no_align`

### NYU Depth

```bash
python test_nyu_regression.py --model_path <checkpoint.pt>
```

- Default mode: aligned
- Absolute-scale mode: add `--no_align`

### Base VLM vs TempoVLM (TUM)

```bash
python test_tum_baseline.py \
  --model_path <checkpoint.pt> \
  --sequence_path ./tum_data/rgbd_dataset_freiburg1_desk \
  --max_frames 100
```

### Batch Benchmark Scripts

- `run_tmu.sh` -> batch TUM regression evaluation
- `run_tum_comparison.sh` -> batch base-vs-ours comparison

Set `MODEL_PATH` inside scripts before running.

## Demo and Occlusion Experiments

### Complete Demo

```bash
python complete_demo.py \
  --model_path <checkpoint.pt> \
  --data_root ./scannet_data \
  --dataset scannet \
  --output_dir ./complete_demo_output \
  --demos all \
  --max_scenes 3
```

`--demos` options: `temporal`, `depth`, `motion`, `occlusion`, `all`

Common options:

- `--occlusion_mode continuous|interval|random`
- `--calibration_frames`
- `--occlusion_type`
- `--injection_method`
- `--anomaly_threshold`

### Adaptive Memory Injection Test

```bash
python test_adaptive.py \
  --model_path <checkpoint.pt> \
  --scene_dir <scene_dir_or_video> \
  --add_occlusion \
  --occlusion_start 15 \
  --occlusion_end 25 \
  --occlusion_type black \
  --injection_method full
```

Alternative input options:

- `--video <video_path>`
- `--data_root <scannet_root> --auto_best`

### Occlusion Robustness (Quantitative)

```bash
python test_occlusion_robustness.py \
  --model_path <checkpoint.pt> \
  --data_root ./scannet_data/scannet_frames_test \
  --max_scenes 10
```

## Outputs

Typical checkpoint/output locations:

- `models/*.pt`
- `checkpoints_unified/*.pt`
- `checkpoints_model_c/*.pt`
- `checkpoints_geometry/*.pt`
- `complete_demo_output/`
- `demo_results_depth_motion/`
- `logs/`
- `tum_*_report*.txt`

