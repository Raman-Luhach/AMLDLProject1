# Part 1: Project Overview, Motivation & Why Everything Was Chosen

## Table of Contents
1. [What Is This Project?](#what-is-this-project)
2. [Why Was This Project Selected?](#why-was-this-project-selected)
3. [What Real-World Problem Does It Solve?](#what-real-world-problem-does-it-solve)
4. [Why Is This Useful?](#why-is-this-useful)
5. [Who Built This Project?](#who-built-this-project)
6. [Complete Project Structure](#complete-project-structure)
7. [The Full Pipeline at a Glance](#the-full-pipeline-at-a-glance)
8. [Key Technologies Used and Why](#key-technologies-used-and-why)
9. [Important Questions & Answers](#important-questions--answers)

---

## What Is This Project?

This is a **B.Tech Final Year Applied Machine Learning & Deep Learning project** that implements a complete **instance segmentation system** for detecting and segmenting individual products on densely packed retail store shelves.

In simple terms: imagine you take a photograph of a supermarket shelf filled with 100-400 products crammed together, overlapping each other, and you want a computer to:
1. **Find every single product** in the image (even if products are partially hidden behind others)
2. **Draw a box** around each product (bounding box detection)
3. **Create a pixel-level mask** showing exactly where each product is (instance segmentation)
4. **Do all this quickly** so it could work in real-time on a phone or edge device

This project builds a **lightweight deep learning system** using the **YOLACT architecture** with a **MobileNetV3-Large backbone**, enhanced with **CBAM attention modules** and **Soft-NMS post-processing**, trained on the **SKU-110K dataset** (11,762 images of retail shelves with 1.73 million annotations).

---

## Why Was This Project Selected?

### 1. It's a Real, Unsolved Problem in Computer Vision
Standard object detection models (like basic YOLO or SSD) work great when there are 5-20 objects in an image. But retail shelves have **100-400+ products per image**, all tightly packed together. This "high-density" scenario causes standard detectors to fail because:
- Products overlap heavily (high IoU between ground-truth boxes)
- Standard NMS (Non-Maximum Suppression) incorrectly removes valid detections
- The model sees thousands of background regions for every actual product (extreme class imbalance)

### 2. It Combines Multiple Advanced Concepts
This project isn't just "train a model on data." It covers:
- **Classical ML baseline** (HOG + SVM) for comparison
- **Deep learning architecture design** (YOLACT + custom modifications)
- **Attention mechanisms** (CBAM - Channel and Spatial Attention)
- **Advanced loss functions** (Focal Loss for class imbalance)
- **Custom post-processing** (Soft-NMS for dense scenes)
- **Model deployment** (ONNX export, INT8 quantization)
- **Web application** (Next.js interactive demo)
- **Comprehensive evaluation** (COCO metrics, Grad-CAM, ablation studies, robustness testing)

### 3. Industry Relevance
Retail shelf monitoring is a **multi-billion dollar industry problem**:
- Automated inventory management
- Out-of-stock detection
- Planogram compliance checking
- Price tag verification
- Supply chain optimization

### 4. Academic Rigor
The project follows a complete research methodology:
- Literature review (18 academic references)
- Hypothesis (Soft-NMS + lightweight backbone can handle dense scenes)
- Experimental design with controls (HOG+SVM baseline)
- Quantitative evaluation with standard metrics (COCO AP/AR)
- Ablation studies (Soft-NMS vs Hard-NMS)
- Robustness analysis (noise, blur, brightness)
- Interpretability (Grad-CAM visualizations)

---

## What Real-World Problem Does It Solve?

### The Density Problem
Take a photo of a supermarket shelf. A human can count maybe 50-200 products visible. Now imagine a computer trying to do this:

```
Standard Detection (e.g., YOLOv5 on COCO):
- Designed for: 5-20 objects per image
- Average IoU between objects: < 0.1 (objects well-separated)
- NMS threshold: 0.5 (aggressively remove overlaps)
- Result: Works perfectly

Dense Retail Detection (SKU-110K):
- Reality: 100-400+ objects per image
- Average IoU between objects: > 0.3 (products touching/overlapping)
- NMS threshold: 0.5 (incorrectly removes valid detections!)
- Result: FAILS - "recall collapse"
```

### The Recall Collapse Problem
When you apply standard hard-NMS (Non-Maximum Suppression) to a dense shelf image:
1. Model detects Product A with score 0.95
2. Model detects Product B (right next to A) with score 0.90
3. IoU between A and B = 0.55 (they overlap because they're adjacent)
4. Hard-NMS says: "IoU > 0.5, so B must be a duplicate of A" → **removes B**
5. Product B is now a **false negative** (missed detection)
6. Multiply this across 147 products → massive recall loss

### The Solution: Soft-NMS
Instead of binary removal (keep/remove), Soft-NMS gradually reduces confidence:
```
Hard-NMS:  score_B = 0  (if IoU > threshold)  → LOST FOREVER
Soft-NMS:  score_B *= exp(-IoU²/σ)            → REDUCED BUT PRESERVED
           score_B = 0.90 * exp(-0.55²/0.5)
           score_B = 0.90 * 0.543 = 0.489     → STILL DETECTED
```

---

## Why Is This Useful?

### Immediate Applications
| Application | How This Project Helps |
|-------------|----------------------|
| **Automated Inventory** | Count every product on every shelf automatically |
| **Out-of-Stock Detection** | Identify empty spaces where products should be |
| **Planogram Compliance** | Verify products are placed in correct positions |
| **Price Monitoring** | Match products to their price tags |
| **Theft Prevention** | Detect when products are removed from shelves |
| **Supply Chain** | Real-time inventory data for restocking decisions |

### Why Lightweight Matters
The model is only **9.98M parameters** and **0.6MB ONNX size**:
- Can run on a **Raspberry Pi** or **phone camera**
- Doesn't need expensive cloud GPUs for inference
- Can be deployed in-store on edge devices
- Battery-friendly for mobile applications

### Why Instance Segmentation (Not Just Detection)
- **Bounding boxes** overlap heavily in dense scenes → hard to count
- **Instance masks** give pixel-precise boundaries → accurate counting
- Masks enable better **occlusion reasoning** (which product is in front?)
- Future applications: **augmented reality** overlays on products

---

## Who Built This Project?

- **Raman Luhach** (Roll No: 230107)
- **Rachit Kumar** (Roll No: 230128)
- **Course:** Applied Machine Learning & Deep Learning
- **Level:** B.Tech Final Year Project (2026)
- **License:** MIT (open source)

---

## Complete Project Structure

```
AMLDLProject1/
│
├── configs/                          # Configuration files
│   ├── default.yaml                  # ALL hyperparameters (81 lines)
│   └── custom_anchors.yaml           # K-means optimized anchor sizes
│
├── notebooks/                        # Interactive Jupyter notebooks
│   ├── 01_EDA_and_Data_Analysis.ipynb    # Dataset exploration
│   ├── 02_Classic_ML_Baseline.ipynb      # HOG+SVM training
│   ├── 03_DL_Training_and_Evaluation.ipynb  # YOLACT training
│   └── 04_ONNX_Deployment.ipynb          # Model export & benchmark
│
├── scripts/                          # Entry-point Python scripts
│   ├── train.py                      # Train YOLACT model
│   ├── evaluate.py                   # Run COCO evaluation
│   ├── run_eda.py                    # Generate EDA plots
│   ├── run_baseline.py               # Train HOG+SVM baseline
│   ├── export.py                     # Export to ONNX + benchmark
│   ├── advanced_evaluation.py        # Grad-CAM, ablation, robustness
│   ├── demo.py                       # Single image inference
│   ├── inference_api.py              # API inference for web app
│   ├── inference_baseline.py         # Baseline inference for web app
│   ├── generate_presentation.py      # Auto-generate slides
│   └── download_data.sh              # Download SKU-110K dataset
│
├── src/                              # Core source code (7,318 lines)
│   ├── models/                       # Neural network architecture
│   │   ├── yolact.py                 # Main model assembly (337 lines)
│   │   ├── backbone.py               # MobileNetV3-Large extractor (124 lines)
│   │   ├── fpn.py                    # Feature Pyramid Network (147 lines)
│   │   ├── protonet.py              # Prototype mask generator (127 lines)
│   │   ├── prediction_head.py        # Multi-task head (194 lines)
│   │   ├── detection.py              # Post-processing (379 lines)
│   │   └── cbam.py                   # CBAM attention module (124 lines)
│   │
│   ├── data/                         # Data handling
│   │   ├── dataset.py                # SKU110K PyTorch Dataset (609 lines)
│   │   ├── augmentations.py          # Training augmentations (631 lines)
│   │   └── anchors.py               # Anchor generation & encoding (549 lines)
│   │
│   ├── training/                     # Training pipeline
│   │   ├── trainer.py                # Training loop (545 lines)
│   │   └── losses.py                # Multi-task loss functions (516 lines)
│   │
│   ├── evaluation/                   # Evaluation pipeline
│   │   ├── evaluator.py              # Inference & metrics (394 lines)
│   │   └── metrics.py               # COCO-style AP/AR (321 lines)
│   │
│   ├── deployment/                   # Production deployment
│   │   ├── export_onnx.py            # ONNX conversion (293 lines)
│   │   ├── quantize.py              # INT8 quantization (127 lines)
│   │   └── benchmark.py             # Latency benchmarking (355 lines)
│   │
│   ├── baseline/                     # Classical ML baseline
│   │   └── hog_svm.py               # HOG + Linear SVM (515 lines)
│   │
│   └── utils/                        # Utilities
│       ├── soft_nms.py               # Gaussian Soft-NMS (334 lines)
│       ├── visualization.py          # Plotting & drawing (442 lines)
│       └── helpers.py               # Config loading, device setup (215 lines)
│
├── web/                              # Next.js web application
│   ├── src/app/
│   │   ├── page.tsx                  # Landing page (6 sections)
│   │   ├── demo/page.tsx             # Interactive live demo (840 lines)
│   │   ├── layout.tsx                # Root layout
│   │   ├── globals.css               # Global styles
│   │   └── api/
│   │       ├── inference/route.ts    # YOLACT API endpoint
│   │       └── inference-baseline/route.ts  # HOG+SVM API endpoint
│   ├── public/
│   │   ├── results/                  # 8 analysis chart images
│   │   └── samples/                  # 4 sample shelf images
│   └── package.json                  # Next.js 16.2, React 19, Tailwind
│
├── results/                          # All outputs (~426 MB)
│   ├── eda/                          # 8 EDA visualizations
│   ├── training/                     # Checkpoints + loss logs
│   ├── eval/                         # COCO metrics + Grad-CAM
│   ├── baseline/                     # HOG+SVM results
│   └── deployment/                   # ONNX models + benchmarks
│
├── report/                           # Academic report
│   ├── main.tex                      # IEEE-format LaTeX source
│   ├── main.pdf                      # Compiled PDF
│   └── references.bib               # 18 academic references
│
├── reference/                        # Grading rubrics & requirements
├── Makefile                          # 10 automation targets
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package installation
├── README.md                         # Project documentation
├── PROJECT_STRUCTURE.md              # File-by-file guide
└── LICENSE                           # MIT License
```

### File Count Summary
| Category | Count | Lines of Code |
|----------|-------|---------------|
| Python source files | 29 | 7,318 LOC |
| Scripts | 11 | ~2,000 LOC |
| Jupyter notebooks | 4 | Interactive |
| TypeScript/TSX files | 7 | ~1,500 LOC |
| Configuration files | 5 | ~200 lines |
| Documentation | 3 markdown | ~37 KB |
| LaTeX report | 1 PDF | ~10 pages |

---

## The Full Pipeline at a Glance

```
PHASE 1: Data Understanding
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Download SKU-110K → Run EDA → Generate 8 analysis plots
                              ├── Object density histogram
                              ├── Bounding box dimensions
                              ├── Aspect ratio distribution
                              ├── Pairwise IoU analysis
                              ├── K-means anchor optimization
                              └── Sample visualizations

PHASE 2: Classical Baseline
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract HOG features → Train Linear SVM → Sliding window detection
Result: mAP@0.5 = 3.09%, Precision = 86.36%, Recall = 2.09%

PHASE 3: Deep Learning Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Build YOLACT architecture → Train 8 epochs on 3,000 images
├── MobileNetV3-Large backbone (3.0M params)
├── FPN + CBAM attention (3.3M params)
├── ProtoNet 32 masks (2.4M params)
├── Prediction Head (1.4M params)
└── Total: 9.98M params

PHASE 4: Evaluation
━━━━━━━━━━━━━━━━━━━
COCO metrics → Grad-CAM → Ablation → Robustness → Error analysis
Result: AP@0.50 = 0.076% (undertrained, loss still decreasing)

PHASE 5: Deployment
━━━━━━━━━━━━━━━━━━━
PyTorch → ONNX FP32 → ONNX INT8 → Benchmark
├── PyTorch: 38.2 MB, 318ms, 3.1 FPS (MPS)
├── ONNX FP32: 0.6 MB, 120ms, 8.3 FPS (CPU)
└── ONNX INT8: 9.9 MB, ~115ms, ~8.7 FPS (CPU)

PHASE 6: Web Application
━━━━━━━━━━━━━━━━━━━━━━━━
Next.js landing page + Interactive live demo
├── Upload shelf images
├── Run YOLACT or HOG+SVM inference
├── Adjust confidence threshold
├── View/download annotated results
└── Fullscreen zoom/pan viewer
```

---

## Key Technologies Used and Why

### Why YOLACT? (Not Mask R-CNN, DETR, or YOLOv8)
| Consideration | YOLACT | Mask R-CNN | DETR | YOLOv8-seg |
|--------------|--------|------------|------|------------|
| **Speed** | Fast (single-stage) | Slow (two-stage) | Slow (transformer) | Fast |
| **Mask Quality** | Good (prototype-based) | Best (per-RoI) | Good | Good |
| **Simplicity** | Moderate | Complex | Complex | Simple |
| **Dense Scene** | Good with Soft-NMS | Good | Poor (fixed queries) | Good |
| **Customizability** | High (modular design) | Medium | Low | Low |
| **Educational Value** | High (learn all components) | High | Medium | Low (black box) |

**Why YOLACT was chosen:**
- It generates masks **without per-RoI computation** (much faster)
- The prototype + coefficient approach is **elegant and efficient**
- It's modular enough to add CBAM attention and swap backbones
- It's a great architecture to **learn** because every component is visible and understandable
- It provides a balance of speed, accuracy, and complexity for an academic project

### Why MobileNetV3-Large? (Not ResNet-101 or EfficientNet)
- **ResNet-101:** 44.5M params → too heavy for edge deployment
- **MobileNetV3-Large:** 3.0M params → **93% fewer parameters**
- Still powerful enough (SE attention blocks, NAS-optimized)
- ImageNet pretrained → strong initialization
- Designed specifically for mobile/edge deployment
- Demonstrates the project's focus on **lightweight, deployable** models

### Why CBAM Attention? (Not SE-Net, BAM, or Self-Attention)
- **CBAM** = Channel Attention + Spatial Attention (sequential)
- Only ~10K additional parameters per FPN level (minimal overhead)
- Helps the model focus on "what" (product features) and "where" (product locations)
- Well-studied, proven effective for object detection
- Easy to integrate into existing FPN architecture

### Why Focal Loss? (Not Cross-Entropy or OHEM)
- **The problem:** 147 products vs ~137,000 anchors per image = 99.9% background
- **Cross-entropy:** Treats all samples equally → overwhelmed by easy negatives
- **Focal Loss:** Down-weights easy examples, focuses on hard ones
  - FL(p) = -α(1-p)^γ log(p) where α=0.25, γ=2.0
- The single most important loss function for dense detection

### Why Soft-NMS? (Not Hard-NMS or Matrix NMS)
- **Hard-NMS:** Binary removal → loses valid overlapping detections
- **Soft-NMS:** Gaussian decay → preserves detections with reduced confidence
- Specifically designed for scenes with naturally overlapping objects
- Theoretically justified by the EDA finding that GT boxes have high pairwise IoU

### Why SKU-110K? (Not COCO, VOC, or custom dataset)
- **COCO/VOC:** Average 7-10 objects per image → not a dense detection challenge
- **SKU-110K:** Average 147 objects per image → exactly the dense scenario we need
- Published at **CVPR 2019** (top computer vision conference) → credible and well-studied
- 11,762 images with 1.73M annotations → sufficient for training
- Single-class simplification → focus on the density problem, not classification
- Freely available on GitHub

---

## Important Questions & Answers

### Q: Why did I choose this specific project?
**A:** This project was chosen because it addresses a genuinely challenging computer vision problem (high-density object segmentation) that standard methods fail at, while covering the entire ML pipeline from data analysis to deployment. It demonstrates advanced concepts (attention mechanisms, focal loss, soft-NMS, ONNX quantization) that are highly relevant to industry applications like retail automation.

### Q: What will I do with this project?
**A:** The project serves as:
1. A B.Tech final year project demonstrating mastery of ML/DL concepts
2. A foundation for a deployable retail shelf monitoring system
3. A comprehensive portfolio piece showing end-to-end ML engineering
4. A research contribution exploring lightweight architectures for dense detection

### Q: Why is this project better than just training a YOLO model?
**A:** Training a YOLO model on COCO is a standard exercise. This project:
- Addresses a **non-standard problem** (dense detection) requiring custom solutions
- Implements **custom architectural modifications** (CBAM, Soft-NMS)
- Includes a **classical baseline** for comparison
- Features **deployment optimization** (ONNX, INT8 quantization)
- Has a **web application** for interactive demonstration
- Includes **advanced evaluation** (Grad-CAM, ablation, robustness)
- Produces an **academic paper** in IEEE format

### Q: What makes this project complete?
**A:** It covers **every stage** of an ML project:
1. Data analysis (EDA with 8 visualizations)
2. Classical baseline (HOG+SVM)
3. Deep learning model design and training
4. Comprehensive evaluation (6 types of analysis)
5. Model optimization and deployment (ONNX, INT8)
6. Web application for demonstration
7. Academic documentation (IEEE paper)
8. Full reproducibility (Makefile, configs, notebooks)

### Q: How do I run this project from scratch?
**A:** Using the Makefile:
```bash
make install          # Install all dependencies
make download-data    # Download SKU-110K dataset (~2.4GB)
make eda              # Run exploratory data analysis
make baseline         # Train HOG+SVM baseline
make train            # Train YOLACT model (8 epochs)
make evaluate         # Evaluate on test set
make export           # Export to ONNX + benchmark
make all              # Run everything end-to-end
```

### Q: What hardware do I need?
**A:**
- **Minimum:** Any modern laptop with 8GB RAM (CPU training will be slow)
- **Recommended:** Apple Silicon Mac (M1/M2/M3) with MPS support
- **Ideal:** NVIDIA GPU with CUDA support (RTX 3060+)
- **Training time:** ~3.5 hours on Apple Silicon MPS for 8 epochs on 3,000 images

---

**Next:** [Part 2 - Dataset Deep Dive](./LEARN_PART2_Dataset_and_EDA.md) →
