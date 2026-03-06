# TempoDepth-VLM

A parameter-efficient temporal and depth adapter for frozen Vision-Language Models (VLMs), based on **Qwen2-VL-2B**.

This repository corresponds to the project report **"TempoDepth-VLM: A Parameter-Efficient Temporal and Depth Adapter for Vision-Language Models"** (`report_TempoDepthVLM.pdf`).

## 1. Project Goal

Base VLMs perform well on single images but are weak on video streams:
- temporal flicker across frames,
- poor metric depth consistency,
- fragile behavior under occlusion.

TempoDepth-VLM keeps the base VLM frozen and trains a lightweight adapter to improve temporal consistency, depth quality, and occlusion robustness.

## 2. Method Overview

The pipeline follows the report:
1. Frozen `Qwen/Qwen2-VL-2B-Instruct` extracts per-frame visual features (`1536-dim`).
2. A shared adapter (`2-layer Transformer`, hidden dim `768`) processes features.
3. GRU long-term memory integrates temporal context across frames.
4. Multi-task heads optimize:
- depth regression (3-region depth: left/center/right),
- temporal consistency (contrastive objective),
- motion prediction,
- occlusion reconstruction.

For ablation, `models_ablation.py` implements a variant without explicit scale decoupling (Model-C / w/o Scale).

## 3. Datasets

Used in the report and code:
- **ScanNet**: primary training/evaluation dataset.
- **TUM RGB-D**: zero-shot depth/generalization tests.
- **NYU Depth V2**: zero-shot depth tests.

## 4. Repository Structure

```text
.
├── models_unified.py            # Full TempoDepth-VLM (with GRU memory + task heads)
├── models_ablation.py           # Ablation model (w/o scale design)
├── train.py                     # Main training script (with optional --use_gru)
├── train_model_c.py             # Ablation training script (Model-C)
├── complete_demo.py             # End-to-end demo (temporal/depth/motion/occlusion)
├── test_tum_regression.py       # TUM depth evaluation (aligned or absolute)
├── test_nyu_regression.py       # NYU depth evaluation (aligned or absolute)
├── test_tum_baseline.py         # Base VLM (prompting) vs Ours comparison on TUM
├── test_occlusion_robustness.py # Occlusion robustness quantitative test
└── report_TempoDepthVLM.pdf     # Project report
```

## 5. Installation

- Python 3.10+
- CUDA GPU recommended (Qwen2-VL inference/training is heavy)

```bash
pip install -r requirements.txt
```

## 6. Data Preparation

### 6.1 ScanNet

```bash
python download_scannet_dataset.py
```

Expected path (default in training scripts):

```text
./scannet_data/
├── scannet_frames_25k/
└── scannet_frames_test/
```

### 6.2 TUM RGB-D

```bash
bash download_tum.sh
```

Default output path: `./tum_data/`.

### 6.3 NYU Depth V2 (Hugging Face conversion)

```bash
python prepare_nyu_from_hf.py
```

Default output path: `data/nyu_hf_test/`.

## 7. Training

### 7.1 Main model (recommended: GRU enabled)

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

### 7.2 Ablation (Model-C / w/o Scale)

```bash
python train_model_c.py \
  --data_root ./scannet_data \
  --output_dir ./checkpoints_model_c \
  --tasks temporal depth_regression motion \
  --epochs 10 \
  --batch_size 2
```

## 8. Evaluation

### 8.1 TUM depth regression

```bash
python test_tum_regression.py \
  --model_path <checkpoint.pt> \
  --sequence_path ./tum_data/rgbd_dataset_freiburg1_desk
```

- Default: median scaling alignment (standard protocol)
- Absolute mode: add `--no_align`

### 8.2 NYU depth regression

```bash
python test_nyu_regression.py --model_path <checkpoint.pt>
```

- Default: aligned evaluation
- Absolute mode: add `--no_align`

### 8.3 Base VLM vs Ours on TUM

```bash
python test_tum_baseline.py \
  --model_path <checkpoint.pt> \
  --sequence_path ./tum_data/rgbd_dataset_freiburg1_desk \
  --max_frames 100
```

### 8.4 Occlusion robustness (quantitative)

```bash
python test_occlusion_robustness.py \
  --model_path <checkpoint.pt> \
  --data_root ./scannet_data/scannet_frames_test \
  --max_scenes 10
```

## 9. Demo

```bash
python complete_demo.py \
  --model_path <checkpoint.pt> \
  --data_root ./scannet_data \
  --dataset scannet \
  --output_dir ./complete_demo_output \
  --demos all \
  --max_scenes 3
```

`--demos` supports comma-separated options:
- `temporal`
- `depth`
- `motion`
- `occlusion`
- `all`

Useful options:
- `--occlusion_mode continuous|interval|random`
- `--calibration_frames`
- `--occlusion_type`
- `--injection_method`
- `--anomaly_threshold`

## 10. Key Results (from report)

### Standard evaluation (median scaled)

| Dataset | AbsRel ↓ | RMSE ↓ | δ1 ↑ |
|---|---:|---:|---:|
| ScanNet | 0.123 | 0.352 | 81.66% |
| NYU Depth V2 | 0.324 | 1.006 | 52.62% |
| TUM RGB-D | 0.240 | 0.898 | 62.12% |

### Absolute metric evaluation (no test-time scaling)

| Dataset | AbsRel ↓ | RMSE ↓ | δ1 ↑ |
|---|---:|---:|---:|
| ScanNet* | 0.242 | 0.535 | 59.78% |
| NYU Depth V2 | 0.349 | 1.107 | 40.94% |
| TUM RGB-D | 0.322 | 0.957 | 45.10% |

`*` ScanNet absolute evaluation in the report uses temporal initialization for scale calibration.

### Ablation (w/o Scale vs Full Model)

Report shows strong gains for the full model in absolute depth on zero-shot datasets:
- NYU AbsRel: `0.555 -> 0.349`
- TUM AbsRel: `0.611 -> 0.322`

## 11. Output Artifacts

Common outputs:
- `checkpoints_unified/*.pt`
- `checkpoints_model_c/*.pt`
- `checkpoints_geometry/*.pt`
- `complete_demo_output/`
- `demo_results_depth_motion/`
- `tum_benchmark_absolute_results.txt`
- `tum_benchmark_aligned_results.txt`
- `tum_baseline_vs_ours_report.txt`
- `tum_baseline_vs_ours_report_align.txt`

## 12. Notes and Limitations

As discussed in the report:
- Depth output is currently 3-region regression (not dense per-pixel depth).
- Performance still trails specialized depth-only SOTA models in raw precision.
- Memory injection needs warm-up frames and can degrade near abrupt scene switches.

## 13. Citation

If you use this codebase, please cite the project report:

```text
TempoDepth-VLM: A Parameter-Efficient Temporal and Depth Adapter for Vision-Language Models,
CVPDL Final Project, Group 19, 2025.
```
