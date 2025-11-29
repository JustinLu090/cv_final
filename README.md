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

---

## Occlusion Experiment: Object Permanence Test

This experiment demonstrates TempoVLM's **"memory"** capability - the ability to remember scene content during temporary occlusions.

### Concept

```
Normal frame:                 Occluded frame:
┌─────────────────┐          ┌─────────────────┐
│ ╔═══╗ ╔═══╗     │          │                 │
│ ║   ║ ║   ║     │    →     │   ██████████    │
│ ╠═══╬═╬═══╣     │          │   ██ BLACK ██   │
│ ║   ║ ║   ║     │          │   ██████████    │
│ ╚═══╝ ╚═══╝     │          │                 │
└─────────────────┘          └─────────────────┘
  (wooden lattice)              (center blocked)
```

**Question**: Can the model "remember" what's behind the black box?

### Method: Direct Feature Injection

We inject TempoVLM's temporally-enhanced features back into Qwen2-VL's vision encoder to make the "memory" visible in language output.

```
Standard VLM (Base Model):
┌──────────┐    ┌─────────────────┐    ┌──────────────┐    ┌─────┐
│ Occluded │ →  │ Vision Encoder  │ →  │ Features     │ →  │ LLM │ → "black square"
│ Image    │    └─────────────────┘    │ (corrupted)  │    └─────┘
└──────────┘                           └──────────────┘

Direct Feature Injection:
┌──────────┐    ┌─────────────────┐    ┌──────────────┐
│ Occluded │ →  │ Vision Encoder  │ →  │ Original     │─┐
│ Image    │    └─────────────────┘    │ Features     │ │
└──────────┘                           └──────────────┘ │
                                                        ├→ Fusion → LLM → "wooden lattice"
┌──────────────────────────────────┐                    │
│ TempoVLM Enhanced Features       │────────────────────┘
│ (contains memory of prev frames) │
└──────────────────────────────────┘
```

### Injection Formula

```python
# Interpolate method (safest)
modified_features = (1 - α) × original_features + α × memory_features

# Where α = 0.1 (10% memory injection is usually sufficient)
```

### Run Occlusion Test

```bash
python test_occlusion.py \
    --model_path ./checkpoints/best_unified_model.pt \
    --data_root ./scannet_data \
    --output_dir ./occlusion_results \
    --split test \
    --num_scenes 3 \
    --occlusion_type box \
    --occlusion_ratio 0.5
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | Required | Path to trained UnifiedTempoVLM |
| `--data_root` | `./scannet_data` | Path to ScanNet data |
| `--output_dir` | `./occlusion_results` | Output directory |
| `--split` | `test` | Dataset split (train/test) |
| `--num_scenes` | `3` | Number of scenes to test |
| `--occlusion_type` | `box` | Occlusion type: `box`, `noise` |
| `--occlusion_ratio` | `0.5` | Ratio of image to occlude (0-1) |

### Output Files

```
occlusion_results/
├── scene_xxx/
│   ├── occlusion_test.mp4      # Video with similarity curves
│   ├── results.json            # Detailed metrics
│   ├── similarity_plot.png     # Feature similarity over time
│   └── report.md               # Markdown report
└── summary.json                # Cross-scene summary
```

### Example Results

```json
{
  "frame_idx": 15,
  "is_occluded": true,
  "memory_test": {
    "base_model": {
      "text_response": "There is a black square in the center of the image.",
      "feature_similarity_to_pre_occlusion": 0.758
    },
    "unified_model": {
      "text_response": "There is a black square in the center of the image.",
      "feature_similarity_to_pre_occlusion": 0.935
    },
    "direct_injection": {
      "text_response": "The image shows a wooden structure with a lattice design. The lattice is made up of horizontal and vertical wooden slats...",
      "method": "interpolate_0.1"
    },
    "ground_truth": "wooden structure... lattice-like pattern... horizontal and vertical wooden beams..."
  }
}
```

### Key Findings

| Model | Can See Through Occlusion? | Feature Memory Score |
|-------|---------------------------|---------------------|
| Base Model | "black square" | 0.758 |
| TempoVLM (feature only) |  Same text output | **0.935** (+0.177) |
| Direct Injection |  **"wooden lattice"** | - |

**Conclusion**: 
- TempoVLM maintains **+17.7%** higher feature similarity during occlusion
- Direct Feature Injection successfully translates this memory into language output
- The model can "see through" the occlusion and describe what was there before

---

## Model Architecture

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
