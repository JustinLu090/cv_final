# TempoVLM: Temporal Adapter for Vision-Language Models

A temporal adapter that enhances Vision-Language Models (Qwen2-VL) with **temporal consistency**, **depth perception**, and **motion prediction** capabilities for visually impaired navigation assistance.

## Features

- **Temporal Consistency**: Maintains stable feature representations across video frames
- **Depth Ordering**: Predicts relative depth ordering between scene regions (which is closer)
- **Depth Regression**: Predicts absolute depth values for scene regions
- **Motion Prediction**: Predicts camera motion (6-DoF) between consecutive frames
- **Occlusion Robustness**: Maintains scene understanding during temporary occlusions

## Project Structure

```
├── models_unified.py        # UnifiedTempoVLM model definition
├── train_unified.py         # Multi-task training script
├── visualization_demo.py    # Visualization demos
├── requirements.txt         # Dependencies
└── README.md
```

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Additional Setup

Install Qwen2-VL utilities:
```bash
pip install qwen-vl-utils
```

## Dataset

This project uses [ScanNet]dataset. Organize your data as:

```
scannet_data/
├── scannet_frames_25k/     # Training scenes
│   ├── scene0000_00/
│   │   ├── color/          # RGB images (*.jpg)
│   │   ├── depth/          # Depth maps (*.png, 16-bit, mm)
│   │   └── pose/           # Camera poses (*.txt, 4x4 matrix)
│   └── ...
└── scannet_frames_test/    # Test scenes
    └── ...
```

## Training


### Basic Training

```bash
python train_unified.py \
    --data_root ./scannet_data \
    --output_dir ./checkpoints_depth \
    --tasks temporal depth_order depth_regression motion \
    --epochs 20 \
    --batch_size 2 \
    --lr 1e-4 \
    --max_scenes 100
```

### Resume Training

```bash
python train_unified.py \
    --data_root ./scannet_data \
    --output_dir ./checkpoints \
    --resume ./checkpoints/best_unified_model.pt \
    --epochs 30 \
    --lr 5e-5
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `./scannet_data` | Path to ScanNet data |
| `--output_dir` | `./checkpoints_unified` | Output directory |
| `--tasks` | `temporal depth_order motion` | Tasks to train |
| `--epochs` | `10` | Number of epochs |
| `--batch_size` | `2` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--max_scenes` | `50` | Maximum scenes to use |
| `--resume` | `None` | Checkpoint to resume from |
| `--save_every` | `2` | Save checkpoint every N epochs |

## Visualization Demo

Generate visualization videos for trained model:

```bash
python visualization_demo.py \
    --model_path ./checkpoints/best_unified_model.pt \
    --data_root ./scannet_data \
    --output_dir ./viz_output \
    --split test \
    --max_scenes 5
```

### Output Videos

For each scene, the following videos are generated:

1. **temporal_consistency.mp4** - Base vs TempoVLM feature similarity over time
2. **depth_ordering.mp4** - Depth ordering predictions vs ground truth
3. **depth_regression.mp4** - Absolute depth predictions vs ground truth (if trained)
4. **trajectory.mp4** - Camera trajectory prediction vs ground truth



##  Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UnifiedTempoVLM                          │
├─────────────────────────────────────────────────────────────┤
│  Input: Qwen2-VL features (1536-dim)                        │
│                                                             │
│  ┌─────────────────┐                                        │
│  │ Shared Encoder  │  1536 → 768                            │
│  └────────┬────────┘                                        │
│           │                                                 │
│  ┌────────┴────────┬──────────────┬──────────────┐          │
│  │                 │              │              │          │
│  ▼                 ▼              ▼              ▼          │
│ Temporal       Depth Order   Depth Regress   Motion         │
│ Branch         Branch        Branch          Branch         │
│ (768→1536)     (768*2→2)     (768→1)         (768*2→6)      │
│                                                             │
│ Output:        Output:       Output:         Output:        │
│ Enhanced       A/B closer    Depth (m)       6-DoF          │
│ Features       probability   0.5~5.0m        tx,ty,tz,      │
│                                              rx,ry,rz       │
└─────────────────────────────────────────────────────────────┘
```

## References

- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) - Base vision-language model
- [ScanNet](http://www.scan-net.org/) - Indoor scene dataset

## License

This project is for academic use only.

## Author

CVPDL Final Project - TempoVLM for Visually Impaired Navigation
