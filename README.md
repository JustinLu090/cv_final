# TempoVLM: Temporal Adapter with GRU Long-term Memory

A temporal adapter that enhances Vision-Language Models (Qwen2-VL) with **GRU-based long-term memory**, **depth perception**, and **motion prediction** capabilities for visually impaired navigation assistance.

## Features

### 🧠 Core Capabilities

- **GRU Long-term Memory**: Maintains scene understanding across extended occlusions using recurrent memory
- **Adaptive Memory Quality Gate**: Automatically detects occlusion quality and adjusts memory update rate
- **Direct Feature Injection**: Injects temporal memory directly into VLM's vision encoder for language-grounded recall
- **Depth Regression**: Predicts absolute depth values (0-10m) for left/center/right regions
- **Motion Prediction**: Predicts camera 6-DoF motion with uncertainty estimation and trajectory optimization
- **Occlusion Robustness**: Maintains scene understanding during temporary occlusions

### 🎯 Advanced Features

- **Motion Uncertainty Estimation**: Predicts per-frame motion confidence for robust trajectory estimation
- **Global Scale Correction**: Learns scene-adaptive scale factors to reduce trajectory drift
- **Motion Quality Detection**: Identifies motion blur and fast movements
- **Place Recognition**: Embedding for loop closure detection

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
| `--tasks` | `temporal depth_regression motion` | Tasks to train (see below) |
| `--epochs` | `10` | Number of epochs |
| `--batch_size` | `2` | Batch size (reduce if OOM) |
| `--lr` | `1e-4` | Learning rate |
| `--max_scenes` | `50` | Maximum scenes to use |
| `--resume` | `None` | Checkpoint to resume from |
| `--save_every` | `2` | Save checkpoint every N epochs |

**Available Tasks:**
- `temporal`: GRU-based temporal consistency (with memory quality gate)
- `depth_regression`: Absolute depth prediction for 3 regions (left/center/right)
- `motion`: 6-DoF camera motion with uncertainty and quality estimation

**Note:** The model automatically uses GRU long-term memory for the temporal task. Hidden states are maintained across frames within each scene and reset between scenes.


---

## Quick Testing and Demos

### 1) test_adaptive.py - Automatic Occlusion Detection + Memory Injection
Takes video as input, detects anomalies (occlusions), selects memory, dynamically calculates injection strength, and generates comparison logs.

**Core Features:**
- **Automatic Memory Bank**: Stores GRU-enhanced features of high-quality frames
-  **Anomaly Detection**: No need to predict occlusion timing, automatically detects quality degradation
- **Smart Memory Selection**: Selects best memory based on similarity, temporal distance, and scene matching
- **Adaptive Strength**: Dynamically adjusts injection strength based on anomaly score, scene match, and GRU memory quality

```bash
# Basic example: Black center occlusion 55%
python test_adaptive.py \
  --model_path ./checkpoints_unified/best_unified_model.pt \
  --add_occlusion \
  --occlusion_type black \
  --occlusion_ratio 0.55
```

**Common Parameters:**
- `--scene_dir PATH`: Specify scene folder (will try to find example path by default)
- `--add_occlusion`: Whether to add artificial occlusion in specified frame range
- `--occlusion_type`: `black`/`white`/`noise`/`blur`
- `--occlusion_ratio`: Occlusion size ratio (0-1)
- `--injection_method`: `full`/`raw`/`adaptive`/`strong`, default `full`
- `--memory_size`: Memory bank size (default: 8)
- `--anomaly_threshold`: Anomaly detection threshold (default: 0.25)

**Output:**
- **Terminal Output**: Quality score, anomaly score, memory bank status, scene matching for each frame
- **Detailed Logs**: Complete GT, occlusion, and post-injection responses (untruncated)
- **Experiment Summary**: Success rate, injection strength distribution, memory usage statistics

### 2) complete_demo.py - Generate Occlusion Test Video
Converts entire scene into visualization video, including original images, occluded images, injection results, and statistics.

```bash
python complete_demo.py \
  --model_path ./checkpoints_unified/best_unified_model.pt \
  --scene_dir scannet_data/scannet_frames_25k/sceneXXXX_00 \
  --output demo_output/sceneXXXX_00.mp4 \
  --occlusion_start 10 \
  --occlusion_end 20 \
  --occlusion_type black \
  --occlusion_ratio 0.55 \
  --injection_method full
```

**Common Parameters:**
- `--occlusion_start / --occlusion_end`: Set occlusion frame range
- `--occlusion_type / --occlusion_ratio`: Occlusion type and size
- `--injection_method`: `full`/`raw`/`adaptive`/`strong`
- `--max_frames`: Limit number of frames to process

**Output:** MP4 video (containing original images, occluded images, post-injection descriptions, statistics panel)

**Notes:**
- First-time loading of Qwen2-VL may take several minutes, please be patient
- If "fast processor" message appears, it's normal; no additional action needed

---

## Occlusion Experiment: GRU Memory + Direct Injection

This experiment demonstrates TempoVLM's **"memory"** capability - the ability to remember scene content during temporary occlusions using **GRU long-term memory** and **direct feature injection**.

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

**Answer**: Yes! Using two complementary mechanisms:
1. **GRU Long-term Memory**: Preserves scene understanding at the feature level
2. **Direct Feature Injection**: Translates preserved features into language output

### Two-Stage Memory Architecture

**Stage 1: GRU Long-term Memory (Feature-level Preservation)**
```
Frame t-5 (clear)  →  [GRU Memory] → Hidden State h₅
Frame t-4 (clear)  →  [GRU Memory] → Hidden State h₄
Frame t-3 (clear)  →  [GRU Memory] → Hidden State h₃
Frame t (occluded) →  [GRU Memory] → Hidden State hₜ
                          ↑
                   Quality Gate detects occlusion
                   → Preserves h₃ instead of updating
```

**Key Feature**: GRU automatically learns to:
- **Preserve** memory when quality gate detects occlusion (low quality score)
- **Update** memory when input is trustworthy (high quality score)
- **Balance** between current observation and long-term memory

**Stage 2: Direct Feature Injection (Language-grounded Recall)**
```
Standard VLM (Base Model):
┌──────────┐    ┌─────────────────┐    ┌──────────────┐    ┌─────┐
│ Occluded │ →  │ Vision Encoder  │ →  │ Features     │ →  │ LLM │ → "black square"
│ Image    │    └─────────────────┘    │ (corrupted)  │    └─────┘
└──────────┘                           └──────────────┘

TempoVLM with GRU Memory + Direct Injection:
┌──────────┐    ┌─────────────────┐    ┌──────────────┐
│ Occluded │ →  │ Vision Encoder  │ →  │ Corrupted    │─┐
│ Image    │    └─────────────────┘    │ Features     │ │
└──────────┘                           └──────────────┘ │
                                                        ├→ Injection → LLM
┌────────────────────────────────────────┐              │      ↓
│ GRU-Enhanced Memory (from Frame t-3)   │──────────────┘  "wooden lattice"
│ • Quality-gated preservation           │
│ • Learned to retain scene understanding│
└────────────────────────────────────────┘
```

### Injection Methods

```python
# 1. raw: Conservative interpolation
modified_features = (1 - α) × original + α × memory
# α ≈ 0.3-0.5, safe but moderate effect

# 2. full: Center-masked injection
mask = create_center_mask(occlusion_ratio)
modified = original * (1 - mask) + memory * mask * α
# Only injects where occluded, preserves clear regions

# 3. strong: Aggressive center injection
modified = full_injection_with_higher_alpha
# α ≈ 0.4-0.5, stronger recall but risk of artifacts

# 4. adaptive: Dynamic strength based on quality
α = compute_adaptive_strength(
    anomaly_score,      # How bad is the occlusion?
    scene_match,        # Is memory from similar scene?
    memory_quality      # GRU gate confidence
)
# Smart adjustment based on multiple factors
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


**Key Insights**: 
1. **GRU Memory Works**: TempoVLM maintains **+17.7%** higher feature similarity during occlusion
   - Quality gate successfully detects occlusion and preserves memory
   - Feature-level understanding is retained across multiple frames
   
2. **Direct Injection Translates Memory to Language**: Without injection, VLM's language decoder only sees corrupted features
   - Injection bridges the gap between preserved features and language output
   - Enables the model to "see through" occlusion and verbalize what it remembers

3. **Two-Stage Design is Essential**: 
   - Stage 1 (GRU): Learns what to remember
   - Stage 2 (Injection): Makes memory actionable for language tasks

---

## Model Architecture

### UnifiedTempoVLM with GRU Long-term Memory

```
┌──────────────────────────────────────────────────────────────────────┐
│                    UnifiedTempoVLM (GRU-enhanced)                    │
├──────────────────────────────────────────────────────────────────────┤
│  Input: Qwen2-VL visual features (1536-dim)                          │
│                                                                      │
│  ┌────────────────────┐                                              │
│  │ Shared Encoder     │  1536 → 768                                  │
│  │ (Linear + LN + GELU)│                                              │
│  └─────────┬──────────┘                                              │
│            │                                                         │
│            ├──────────────┬────────────────┬────────────────────┐    │
│            │              │                │                    │    │
│            ▼              ▼                ▼                    ▼    │
│  ┌─────────────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐ │
│  │ Temporal Branch │  │  Depth   │  │  Motion   │  │ Motion Utils │ │
│  │  (GRU Memory)   │  │Regression│  │Prediction │  │              │ │
│  └─────────────────┘  └──────────┘  └───────────┘  └──────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 🧠 Temporal Branch with GRU Long-term Memory                 │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │                                                             │    │
│  │  Current Frame (768) ──┐                                    │    │
│  │                        │                                    │    │
│  │  Hidden State (768) ───┼──→ [Memory Quality Gate] ──→ α     │    │
│  │      (Long-term)       │         ↓                          │    │
│  │                        └──→ GRU Cell ──→ New Memory         │    │
│  │                                   ↓                         │    │
│  │                        [Memory Output Gate] ──→ β           │    │
│  │                                   ↓                         │    │
│  │                     Fusion: β·memory + (1-β)·current        │    │
│  │                                   ↓                         │    │
│  │                        Temporal Output (1536)               │    │
│  │                                   ↓                         │    │
│  │                     Residual: input + output                │    │
│  │                                                             │    │
│  │  🔑 Key Components:                                          │    │
│  │  • GRU Cell: Learns to retain/forget long-term memory       │    │
│  │  • Quality Gate (α): Detects if current frame is occluded   │    │
│  │    - High α → trust current, update memory                  │    │
│  │    - Low α → distrust current, preserve old memory          │    │
│  │  • Output Gate (β): Balances memory vs current observation  │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 📏 Depth Regression Branch                                   │     │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  768 → 384 → 3 regions [left, center, right]                │    │
│  │  Output: Absolute depth in meters (0-10m)                   │    │
│  │  Activation: Softplus (smooth positive values)              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 🎥 Motion Prediction Branch                                 │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  Current + Previous (768*2) → Fusion (768)                  │    │
│  │                              ↓                              │    │
│  │            ┌─────────────────┼─────────────────┬────────┐   │    │
│  │            ▼                 ▼                 ▼        ▼   │    │
│  │     [Motion Head]    [Uncertainty]   [Global    [Quality]   │    │
│  │          6-DoF          Head           Scale]    Detector   │    │
│  │     tx,ty,tz,rx,ry,rz   (6-dim)       (1-dim)   (1-dim)     │    │
│  │                                                             │    │
│  │  🔑 Advanced Features:                                       │   │
│  │  • Learnable Scale Parameters: Separate for trans/rot       │    │
│  │  • Uncertainty Estimation: Per-DOF confidence scores        │    │
│  │  • Global Scale Correction: Adaptive trajectory scaling     │    │
│  │  • Motion Quality: Detects blur/fast motion                 │    │
│  │  • Place Embedding: For loop closure detection              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Final Outputs:                                                   │
│  • temporal: Enhanced features (1536-dim) + memory metadata         │
│  • depth_regression: [left, center, right] depths in meters         │
│  • motion: 6-DoF [tx, ty, tz, rx, ry, rz] + uncertainty + quality   │
│  • hidden_state: Updated GRU memory for next frame                  │
└──────────────────────────────────────────────────────────────────────┘
```

### Direct Feature Injection Mechanism

```
Memory Injection Process (for Occlusion Handling)
═══════════════════════════════════════════════════════════════════

Step 1: Extract Features
┌──────────────┐         ┌──────────────────┐         ┌──────────────┐
│ Clear Frame  │────────→│ Qwen2-VL Encoder │────────→│ Base Feature │
│   (RGB)      │         │    (frozen)      │         │   (1536-d)   │
└──────────────┘         └──────────────────┘         └──────┬───────┘
                                                             │
                                                             ▼
                                                    ┌─────────────────┐
                                                    │ UnifiedTempoVLM │
                                                    │   (Adapter)     │
                                                    └────────┬────────┘
                                                             │
                                                             ▼
                                                    ┌─────────────────┐
                                                    │Enhanced Feature │
                                                    │  + GRU Memory   │
                                                    └─────────────────┘
                                                     Store in Memory Buffer

Step 2: Detect Occlusion
┌──────────────┐         ┌──────────────────┐         ┌──────────────┐
│Occluded Frame│────────→│ Quality Analysis │────────→│  Anomaly?    │
│  (Blocked)   │         │ • Variance       │         │  Yes → Inject│
└──────────────┘         │ • Consistency    │         │  No → Normal │
                         └──────────────────┘         └──────────────┘

Step 3: Retrieve Best Memory
┌────────────────┐
│ Memory Buffer  │
│ [Feat₁, t=5]   │         Select Best Match
│ [Feat₂, t=8]   │    ┌────────────────────────┐
│ [Feat₃, t=12]  │───→│ • Cosine Similarity    │───→ Best Memory
│ [Feat₄, t=17]  │    │ • Temporal Distance    │     Feature
│    ...         │    │ • Scene Match (edges)  │
└────────────────┘    └────────────────────────┘

Step 4: Direct Feature Injection
┌──────────────┐         ┌────────────────────────────────┐
│Occluded Image│────────→│   Qwen2-VL Vision Encoder      │
└──────────────┘         └────────────┬───────────────────┘
                                      │ Hook intercepts here
                                      ▼
                         ┌─────────────────────────────┐
                         │    Original Features (X)    │
                         └─────────────┬───────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
     ┌────────────────┐     ┌───────────────────┐    ┌─────────────────┐
     │  Method: raw   │     │  Method: full     │    │ Method: strong  │
     │ Interpolate    │     │ Center mask only  │    │ Strong center   │
     │ X' = (1-α)X    │     │ Apply to masked   │    │ Higher strength │
     │      + αM      │     │ regions only      │    │ in center       │
     └────────┬───────┘     └─────────┬─────────┘    └────────┬────────┘
              │                       │                        │
              └───────────────────────┼────────────────────────┘
                                      ▼
                         ┌─────────────────────────────┐
                         │   Modified Features (X')    │
                         └─────────────┬───────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────────┐
                         │ Qwen2-VL Language Model     │
                         └─────────────┬───────────────┘
                                       ▼
                         ┌─────────────────────────────┐
                         │ "wooden lattice structure"  │
                         │ (Sees through occlusion!)   │
                         └─────────────────────────────┘

Injection Methods:
• raw: Conservative interpolation (α ≈ 0.3-0.5)
• full: Masked-region injection with center focus
• strong: Aggressive center injection (α ≈ 0.4-0.5)
• adaptive: Dynamic strength based on anomaly score

Adaptive Strength Calculation:
    α = base_strength × scene_match × memory_quality
    where:
    - base_strength: From anomaly score (0.2-0.5)
    - scene_match: Edge-based similarity (0-1)
    - memory_quality: GRU gate output (0-1)
```

## References

- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) - Base vision-language model (2B parameters)
- [ScanNet](http://www.scan-net.org/) - Indoor scene dataset with RGB-D and pose annotations
- GRU (Gated Recurrent Unit) - Recurrent architecture for long-term memory

## Citation

If you use this work, please cite:

```bibtex
@misc{tempovlm2024,
  title={TempoVLM: Temporal Adapter with GRU Long-term Memory for Vision-Language Models},
  author={Your Name},
  year={2024},
  note={CVPDL Final Project}
}
```

## License

This project is for academic use only.

## Author

CVPDL Final Project - TempoVLM for Visually Impaired Navigation
