# High-Density Object Segmentation

> **B.Tech Final Year Project** -- Applied Machine Learning & Deep Learning
> **Authors:** Raman Luhach (230107), Rachit Kumar (230128)
> **Department of Computer Science**
> **Dataset:** [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) (dense retail shelf scenes)

---

## Overview

Detecting and segmenting objects in high-density retail environments presents significant challenges due to severe inter-object occlusion, near-identical appearance, and extreme variation in object counts (10--700+ per image). This project implements a **three-phase hybrid framework** that progressively builds from classical machine learning to a neuro-symbolic system:

| Phase | Approach | Method | Key Result |
|-------|----------|--------|------------|
| **Phase 1** | Classical ML | HOG + SVM + Sliding Window | 86.4% precision, 2.1% recall |
| **Phase 2** | Deep Learning | YOLACT + MobileNetV3 + Soft-NMS | Best val loss: 3.145, 8.3 FPS (ONNX) |
| **Phase 3** | Hybrid Fusion | YOLACT + GMM + KDE Spatial Reasoning | Best val loss: 3.097, 98.4% mAP drop without Soft-NMS |

The key finding from our ablation study: **Soft-NMS is essential** for dense detection -- replacing it with Hard-NMS causes a 98.4% relative mAP drop.

---

## Architecture

### Phase 2: YOLACT (Core Detector)

```
Input Image (3 x 550 x 550)
         |
  MobileNetV3-Large (ImageNet pretrained, 5.4M params)
    |           |           |
  C3 (40ch)  C4 (112ch)  C5 (960ch)
    |           |           |
  Feature Pyramid Network (256ch, 3 levels) + CBAM Attention
    |     |     |
   P3    P4    P5
    |     |     |
    +-----+-----+
    |             |
 ProtoNet    Prediction Head (shared)
 (32 masks)   cls | box | mask coeffs
    |             |
  Assembly: masks = sigmoid(proto @ coeffs^T)
                |
         Soft-NMS (Gaussian, sigma=0.5)
                |
         Final Detections
```

### Phase 3: Hybrid Fusion

```
YOLACT Detections
       |
  +----+----+
  |         |
  v         v
GMM         KDE
(7 row     (5K point
components) density)
  |         |
  +----+----+
       |
  8-dim Spatial Feature Vector
       |
  +----+----+
  |         |
Gated      Confidence
Spatial    Recalibrator
Attention  (17.8K params)
(gate=0.408)
  |         |
  +----+----+
       |
  Refined Detections
```

### Parameter Breakdown

| Component | Parameters |
|-----------|-----------|
| MobileNetV3-Large (backbone) | ~5.4M |
| FPN + CBAM Attention | ~3.3M |
| ProtoNet (32 prototypes) | ~1.0M |
| Prediction Head | ~0.3M |
| **YOLACT Total** | **~10.0M** |
| Spatial Attention (gate) | 2.6K |
| Confidence Recalibrator | 17.8K |
| Visual Projector | 16.4K |
| **Hybrid Overhead** | **~37K (0.4%)** |

---

## Results

### Phase 1: HOG + SVM Baseline

| Metric | Value |
|--------|-------|
| mAP@0.5 | 3.09% |
| Precision | 86.36% |
| Recall | 2.09% |
| Detections | 22 / 907 GT boxes |
| Inference | 0.251 s/image |

**Verdict:** High precision but near-zero recall. Sliding window + Hard-NMS cannot scale to 100+ objects per image.

### Phase 2: YOLACT Training (20 epochs, H100 GPU)

| Epoch | Train Loss | Val Loss | Cls | Box | Mask |
|-------|-----------|----------|-----|-----|------|
| 1 | 8.620 | -- | .283 | 3.479 | .509 |
| 10 | 4.216 | 3.374 | .081 | 1.137 | .259 |
| 20 | 3.808 | **3.145** | .075 | 1.040 | .247 |

- **Training time:** ~137 minutes (mixed precision, batch size 8)
- **No overfitting:** Validation loss decreases monotonically
- **Classification loss:** 73% reduction (0.283 -> 0.075)

### Phase 3: Hybrid Training (7 epochs)

| Stage | Epochs | Best Val Loss | Gate Value |
|-------|--------|--------------|------------|
| 3a (frozen backbone) | 5 | **3.097** | 0.408 |
| 3b (joint fine-tune) | 2 | 3.583 | 0.408 |

### Ablation Study (588 val images, 8 variants)

| Variant | mAP@0.5 | Delta mAP |
|---------|---------|-----------|
| Full Hybrid | 2.73% | baseline |
| DL Only (no spatial) | 3.03% | +0.30 |
| No Recalibrator | 2.73% | 0.00 |
| No Spatial Attention | 3.03% | +0.30 |
| No Row Model (GMM) | 2.73% | 0.00 |
| No Density Field (KDE) | 2.73% | 0.00 |
| **Hard NMS** | **0.04%** | **-2.68** |
| No CBAM | 2.73% | 0.00 |

**Key finding:** Hard-NMS causes **98.4% relative mAP drop** -- Soft-NMS is the single most critical component for dense detection.

### Per-Density Analysis

| Density | Images | mAP@0.5 | F1 | Recall |
|---------|--------|---------|-----|--------|
| Low (<50) | 1 | 1.19% | 5.06% | 15.00% |
| Medium (50-150) | 356 | 1.21% | 3.39% | 4.38% |
| High (>150) | 231 | 9.09% | 2.42% | 2.42% |

### Deployment

| Backend | Device | Latency | FPS | Model Size |
|---------|--------|---------|-----|------------|
| PyTorch FP32 | MPS (GPU) | 318.3 ms | 3.1 | 38.2 MB |
| ONNX FP32 | CPU | 120.3 ms | 8.3 | ~38 MB |

### Robustness (AP@0.5 under corruptions)

| Corruption | None | Low | Medium | High |
|-----------|------|-----|--------|------|
| Gaussian Noise | 0.071% | 0.078% | 0.077% | 0.095% |
| Gaussian Blur | 0.071% | 0.084% | 0.086% | 0.096% |
| Brightness Shift | 0.071% | 0.075% | 0.076% | 0.075% |

---

## Project Structure

```
AMLDLProject1/
|
|-- configs/
|   |-- default.yaml              # Phase 2 training config
|   +-- hybrid.yaml               # Phase 3 hybrid config
|
|-- scripts/
|   |-- download_data.sh          # Download SKU-110K dataset
|   |-- run_eda.py                # Exploratory data analysis
|   |-- run_baseline.py           # Phase 1: HOG+SVM baseline
|   |-- train.py                  # Phase 2: Train YOLACT
|   |-- train_hybrid.py           # Phase 3: Train hybrid detector
|   |-- evaluate.py               # COCO-style evaluation
|   |-- run_ablation.py           # Ablation study (8 variants)
|   |-- export.py                 # ONNX export + quantization
|   |-- inference_api.py          # YOLACT inference (web API)
|   |-- inference_baseline.py     # HOG+SVM inference (web API)
|   +-- inference_hybrid.py       # Hybrid inference (web API)
|
|-- src/
|   |-- models/
|   |   |-- yolact.py             # YOLACT model assembly
|   |   |-- hybrid.py             # Hybrid detector (Phase 3)
|   |   |-- spatial_reasoning.py  # GMM + KDE spatial engine
|   |   |-- confidence_recalibrator.py  # Score recalibration MLP
|   |   |-- backbone.py           # MobileNetV3-Large
|   |   |-- fpn.py                # Feature Pyramid Network + CBAM
|   |   |-- detection.py          # Post-processing (Soft-NMS)
|   |   +-- protonet.py           # Prototype mask generator
|   |-- training/
|   |   |-- trainer.py            # Phase 2 training loop
|   |   +-- hybrid_trainer.py     # Phase 3 training loop
|   |-- evaluation/
|   |   |-- ablation.py           # Ablation framework
|   |   +-- metrics.py            # mAP, precision, recall
|   |-- data/
|   |   |-- dataset.py            # SKU-110K dataset loader
|   |   +-- augmentations.py      # MixUp, albumentations
|   +-- utils/
|       |-- soft_nms.py           # Soft-NMS implementation
|       +-- helpers.py            # Utilities
|
|-- web/                          # Next.js web demo (3 model comparison)
|   |-- src/app/
|   |   |-- page.tsx              # Landing page
|   |   |-- demo/page.tsx         # Interactive inference demo
|   |   +-- api/
|   |       |-- inference/        # YOLACT API endpoint
|   |       |-- inference-baseline/  # HOG+SVM API endpoint
|   |       +-- inference-hybrid/    # Hybrid API endpoint
|
|-- results/
|   |-- training/checkpoints/     # YOLACT checkpoints (.pth)
|   |-- hybrid/checkpoints/       # Hybrid checkpoints (.pth)
|   |-- hybrid/spatial_models/    # GMM + KDE fitted models (.pkl)
|   |-- ablation/                 # Ablation results + charts
|   |-- eval/                     # Evaluation metrics + visualizations
|   |-- eda/                      # EDA plots
|   |-- baseline/                 # HOG+SVM results
|   |-- figures/                  # Architecture diagrams
|   +-- deployment/               # ONNX models + benchmarks
|
|-- report/
|   +-- phase3.tex                # IEEE-format LaTeX report
|
|-- Dockerfile                    # Reproducible environment
|-- requirements.txt
|-- setup.py
+-- README.md
```

---

## Reproduction Guide

### Prerequisites

- Python >= 3.8
- NVIDIA GPU with CUDA (recommended) or Apple Silicon Mac (MPS) or CPU
- ~10 GB disk space (dataset + checkpoints)
- Node.js >= 18 (for web demo only)

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/Raman-Luhach/AMLDLProject1.git
cd AMLDLProject1

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

Or use Docker:

```bash
docker build -t sku110k-detector .
docker run -it --gpus all sku110k-detector bash
```

### Step 2: Download Dataset

```bash
bash scripts/download_data.sh
```

This downloads and extracts the SKU-110K dataset (~2.4 GB) to `data/`.

### Step 3: Run Full Pipeline

```bash
# Phase 1: Exploratory Data Analysis
python scripts/run_eda.py

# Phase 1: Classical ML Baseline (HOG+SVM)
python scripts/run_baseline.py

# Phase 2: Train YOLACT (20 epochs)
python scripts/train.py --config configs/default.yaml

# Phase 2: Evaluate + generate visualizations
python scripts/evaluate.py

# Phase 2: Export to ONNX
python scripts/export.py

# Phase 3: Train Hybrid Detector (Stage 2 + Stage 3)
python scripts/train_hybrid.py --config configs/hybrid.yaml

# Phase 3: Run Ablation Study (8 variants)
python scripts/run_ablation.py --config configs/hybrid.yaml
```

**Expected training times (NVIDIA H100):**
- Phase 2 (YOLACT, 20 epochs): ~137 minutes
- Phase 3 Stage 2 (fit GMM+KDE): ~3 minutes
- Phase 3 Stage 3 (fusion, 7 epochs): ~3 hours
- Ablation study: ~35 minutes

### Step 4: Web Demo (Optional)

```bash
cd web
npm install
npm run dev
# Open http://localhost:3000
```

The demo supports all 3 models: YOLACT, Hybrid, and HOG+SVM.

### Step 5: Compile Report (Optional)

```bash
cd report
pdflatex phase3.tex && pdflatex phase3.tex
```

Requires LaTeX (BasicTeX/TexLive) with the `ieeetran` package.

---

## Pre-trained Checkpoints

The repository includes pre-trained checkpoints:

| File | Size | Description |
|------|------|-------------|
| `results/training/checkpoints/best_model.pth` | 76 MB | YOLACT (20 epochs, val_loss=3.145) |
| `results/hybrid/checkpoints/hybrid_best.pth` | 76 MB | Hybrid detector (val_loss=3.097) |
| `results/hybrid/spatial_models/spatial_engine.pkl` | 127 KB | Fitted GMM + KDE models |
| `results/baseline/hog_svm_model.pkl` | ~1 MB | Trained HOG+SVM classifier |

To skip training and use pre-trained models directly:

```bash
# Run evaluation with pre-trained YOLACT
python scripts/evaluate.py

# Run ablation with pre-trained hybrid
python scripts/run_ablation.py --config configs/hybrid.yaml

# Start web demo
cd web && npm install && npm run dev
```

---

## Training Configuration

### Phase 2 (`configs/default.yaml`)

```yaml
training:
  epochs: 20
  batch_size: 8
  optimizer: sgd
  lr: 0.001
  momentum: 0.9
  weight_decay: 0.0005
  warmup_epochs: 3
  scheduler: cosine
  gradient_clip: 10.0
  use_amp: true          # Mixed precision

augmentation:
  mixup: true
  mixup_alpha: 0.2

loss:
  focal_alpha: 0.25
  focal_gamma: 2.0

softnms:
  method: gaussian
  sigma: 0.5
```

### Phase 3 (`configs/hybrid.yaml`)

```yaml
hybrid:
  spatial_reasoning:
    num_row_components: 8    # GMM (BIC selects 7)
    kde_bandwidth: 0.05
    max_kde_samples: 5000    # Subsampled from 1.1M

  training:
    frozen_epochs: 5         # Stage 3a
    finetune_epochs: 2       # Stage 3b
    frozen_lr: 0.001
    finetune_lr: 0.0001
```

---

## Technical Details

### Soft-NMS (Gaussian Decay)

Standard NMS hard-suppresses all detections overlapping above an IoU threshold, causing missed detections in dense scenes. Soft-NMS decays scores:

```
score_i = score_i * exp(-IoU(M, b_i)^2 / sigma)
```

Our ablation proves this is the single most critical component: Hard-NMS causes 98.4% mAP drop (2.73% -> 0.04%).

### Spatial Reasoning Engine

- **GMM (7 components):** Fitted on y-centers of all training boxes to detect shelf-row structure via BIC model selection.
- **KDE (5,000 points):** Gaussian KDE on normalized (x,y) centers generates a 2D density field as spatial prior.
- **Gated Attention:** Learnable gate `g` (converges to 0.408) controls spatial influence: `F' = F + sigmoid(g) * D_spatial`

### Device Support

| Backend | Training | Inference | AMP |
|---------|----------|-----------|-----|
| CUDA | Yes | Yes | Yes |
| MPS (Apple Silicon) | Yes | Yes | No |
| CPU | Yes | Yes | No |

---

## References

1. Goldman et al., "Precise Detection in Densely Packed Scenes," CVPR 2019
2. Bolya et al., "YOLACT: Real-time Instance Segmentation," ICCV 2019
3. Howard et al., "Searching for MobileNetV3," ICCV 2019
4. He et al., "Mask R-CNN," ICCV 2017
5. Woo et al., "CBAM: Convolutional Block Attention Module," ECCV 2018
6. Bodla et al., "Soft-NMS -- Improving Object Detection With One Line of Code," ICCV 2017
7. Lin et al., "Feature Pyramid Networks for Object Detection," CVPR 2017
8. Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017
9. Dalal and Triggs, "Histograms of Oriented Gradients for Human Detection," CVPR 2005
10. Zhang et al., "mixup: Beyond Empirical Risk Minimization," ICLR 2018

---

## License

MIT License -- Copyright (c) 2026 Raman Luhach, Rachit Kumar
