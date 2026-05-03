# Part 3: Architecture Deep Dive - Every Component Explained

## Table of Contents
1. [The Big Picture: Why This Architecture?](#the-big-picture-why-this-architecture)
2. [YOLACT: The Foundation](#yolact-the-foundation)
3. [Component 1: MobileNetV3-Large Backbone](#component-1-mobilenetv3-large-backbone)
4. [Component 2: Feature Pyramid Network (FPN)](#component-2-feature-pyramid-network-fpn)
5. [Component 3: CBAM Attention Modules](#component-3-cbam-attention-modules)
6. [Component 4: Prototype Network (ProtoNet)](#component-4-prototype-network-protonet)
7. [Component 5: Prediction Head](#component-5-prediction-head)
8. [Component 6: Mask Assembly](#component-6-mask-assembly)
9. [Component 7: Soft-NMS Post-Processing](#component-7-soft-nms-post-processing)
10. [Anchor System Explained](#anchor-system-explained)
11. [Complete Forward Pass Walkthrough](#complete-forward-pass-walkthrough)
12. [Parameter Budget Breakdown](#parameter-budget-breakdown)
13. [Important Questions & Answers](#important-questions--answers)

---

## The Big Picture: Why This Architecture?

The architecture was designed around **three core requirements**:

1. **Handle extreme density** (147 objects/image with heavy overlap)
2. **Be lightweight** (deployable on phones/edge devices)
3. **Generate instance masks** (not just bounding boxes)

Every component serves one or more of these requirements:

```
Requirement → Component → How It Helps
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dense detection → Soft-NMS         → Preserves overlapping detections
Dense detection → Focal Loss       → Handles 1:930 class imbalance
Dense detection → CBAM Attention   → Focuses on product regions
Lightweight     → MobileNetV3      → 93% fewer params than ResNet-101
Lightweight     → Shared Head      → One head for all FPN levels
Lightweight     → ProtoNet (32)    → Only 32 shared prototypes
Instance masks  → YOLACT design    → Prototype + coefficient assembly
Instance masks  → ProtoNet         → Generates mask basis
Multi-scale     → FPN (P3-P7)      → Detects objects of all sizes
```

---

## YOLACT: The Foundation

### What Is YOLACT?
**YOLACT** stands for **"You Only Look At CoefficienTs"** — it's a real-time instance segmentation method published at **ICCV 2019** by Bolya et al.

### Why Is It Special?
Previous instance segmentation methods (like Mask R-CNN) worked in two stages:
1. **Stage 1:** Propose regions of interest (RoIs) — slow
2. **Stage 2:** For each RoI, predict a mask — even slower (one CNN pass per detection!)

YOLACT completely changes this:
1. Generate **K shared prototype masks** for the entire image (one CNN pass, period)
2. For each detection, predict **K coefficients** (linear combination weights)
3. Combine: `mask_i = sigmoid(coefficients_i @ prototypes)` → instant!

```
Mask R-CNN: 100 detections × 1 CNN pass each = 100 CNN passes  (SLOW)
YOLACT:     1 ProtoNet pass + 100 dot products                  (FAST)
```

### How Masks Work in YOLACT

Imagine the ProtoNet produces 32 "prototype masks." Think of them as **building blocks**:
- Prototype 1: Highlights left edges
- Prototype 2: Highlights right edges
- Prototype 3: Highlights top regions
- Prototype 4: Highlights small objects
- ...
- Prototype 32: Highlights some other spatial pattern

For each detection, the model predicts 32 coefficients. For example:
```
Detection "Bottle at position (100, 200)":
  Coefficients: [+0.8, +0.7, -0.3, +0.5, ..., -0.2]

  Final mask = sigmoid(
    +0.8 × Prototype_1    (strong left edge)
    +0.7 × Prototype_2    (strong right edge)
    -0.3 × Prototype_3    (suppress top region)
    +0.5 × Prototype_4    (highlight small object)
    ...
    -0.2 × Prototype_32   (slight suppression)
  )
```

The **linear combination** of prototypes creates an instance-specific mask without needing per-instance computation!

---

## Component 1: MobileNetV3-Large Backbone

### What Is a Backbone?
The backbone is the **feature extractor** — it takes a raw image (pixels) and produces **feature maps** (learned representations). Think of it as translating the image from "pixel language" to "feature language" that the detection head can understand.

### Why MobileNetV3-Large?

| Backbone | Parameters | ImageNet Top-1 | Suitable for Edge? |
|----------|-----------|----------------|-------------------|
| ResNet-18 | 11.7M | 69.8% | Borderline |
| ResNet-50 | 25.6M | 76.2% | No |
| ResNet-101 | 44.5M | 77.4% | No |
| **MobileNetV3-Large** | **3.0M** | **75.2%** | **Yes** |
| MobileNetV2 | 3.4M | 72.0% | Yes |
| EfficientNet-B0 | 5.3M | 77.1% | Borderline |

MobileNetV3-Large achieves **75.2% ImageNet accuracy with only 3.0M parameters** — nearly matching ResNet-50's accuracy with **88% fewer parameters**.

### How MobileNetV3-Large Works

#### Building Block: Inverted Residual + SE Attention
```
Input tensor
    ↓
[1×1 Conv] Expand channels (e.g., 16 → 64)   ← "Expand"
    ↓
[3×3 Depthwise Conv] Process each channel independently ← "Filter"
    ↓
[SE Block] Channel attention (squeeze & excite) ← "Attend"
    ↓
[1×1 Conv] Reduce channels (e.g., 64 → 24)    ← "Project"
    ↓
(+ skip connection from input if dimensions match)
```

**Why "Inverted"?** Traditional residual blocks shrink then expand channels. MobileNetV3 does the opposite: expand first (to a high-dimensional space where depthwise conv is effective), then project back down.

**Why Depthwise Separable Convolutions?**
```
Standard 3×3 Conv:    input_channels × 3 × 3 × output_channels parameters
                      Example: 64 × 3 × 3 × 64 = 36,864 parameters

Depthwise + Pointwise: input_channels × 3 × 3 + input_channels × output_channels
                      Example: 64 × 3 × 3 + 64 × 64 = 4,672 parameters
                      → 7.9× fewer parameters!
```

**Why Squeeze-and-Excitation (SE)?**
SE blocks learn **channel-wise attention** — they figure out which feature channels are important for each input:
```
Feature map: (B, C, H, W)
    ↓
Global Average Pool → (B, C, 1, 1)     ← "Squeeze" spatial info
    ↓
FC → ReLU → FC → Sigmoid → (B, C, 1, 1) ← "Excite" channel weights
    ↓
Multiply × original feature map          ← Reweight channels
```

### What the Backbone Outputs
MobileNetV3-Large produces feature maps at three scales:

```
Input: (B, 3, 550, 550)     ← RGB image
    ↓
C3: (B, 40, 69, 69)         ← stride-8  (fine details, small objects)
C4: (B, 112, 35, 35)        ← stride-16 (medium features)
C5: (B, 960, 18, 18)        ← stride-32 (coarse, semantic features)
```

**Why three scales?**
- **C3 (stride-8):** Preserves fine spatial details → detects small products
- **C4 (stride-16):** Captures medium-scale features → detects normal products
- **C5 (stride-32):** Captures high-level semantic features → detects large products & context

### ImageNet Pretraining
The backbone is **initialized with ImageNet-pretrained weights**:
- ImageNet has 1.28 million images across 1,000 categories
- Pretrained features already understand edges, textures, shapes, colors
- Without pretraining: model starts from random features → needs much more training
- With pretraining: model starts with good features → converges faster

**Code location:** `src/models/backbone.py` (124 lines)

---

## Component 2: Feature Pyramid Network (FPN)

### What Is FPN?
FPN takes the multi-scale features from the backbone (C3, C4, C5) and creates a **top-down feature pyramid** where every level has the same number of channels (256) and benefits from both **fine spatial detail** and **semantic richness**.

### Why Is FPN Needed?
Without FPN:
- C3 (stride-8): Good spatial detail but poor semantics (doesn't "understand" what products are)
- C5 (stride-32): Rich semantics but poor spatial detail (knows "there's a product" but not exactly where)

With FPN:
- P3: Has **both** fine spatial detail AND rich semantics (best of both worlds)
- This fusion is critical for detecting small, densely packed products

### How FPN Works

```
Top-Down Pathway (information flows from coarse to fine):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

C5 (960ch, 18×18) → [1×1 Conv: 960→256] → P5 (256ch, 18×18)
                                              ↓ upsample 2×
                                              ↓ + element-wise add
C4 (112ch, 35×35) → [1×1 Conv: 112→256] →──→ P4 (256ch, 35×35)
                                              ↓ upsample 2×
                                              ↓ + element-wise add
C3 (40ch, 69×69)  → [1×1 Conv: 40→256]  →──→ P3 (256ch, 69×69)

Extra levels (for detecting larger objects):
P5 → [3×3 Conv, stride 2] → P6 (256ch, 9×9)
P6 → [3×3 Conv, stride 2] → P7 (256ch, 5×5)
```

### Step-by-Step:
1. **Lateral connections (1×1 convs):** Project C3, C4, C5 to 256 channels each
2. **Top-down merging:** Upsample P5→P4 size, add to lateral C4 output → P4
3. **Continue:** Upsample P4→P3 size, add to lateral C3 output → P3
4. **Smoothing (3×3 convs):** Remove aliasing from upsampling
5. **Extra levels:** Downsample P5 → P6, P6 → P7 for large object detection
6. **CBAM attention:** Applied on P3, P4, P5 (see next section)

### FPN Output Summary
| Level | Resolution | Stride | Best For |
|-------|-----------|--------|----------|
| **P3** | 69×69 | 8 | Small products (most detections happen here) |
| **P4** | 35×35 | 16 | Medium products |
| **P5** | 18×18 | 32 | Large products |
| **P6** | 9×9 | 64 | Very large objects |
| **P7** | 5×5 | 128 | Scene-level features |

**Code location:** `src/models/fpn.py` (147 lines)

---

## Component 3: CBAM Attention Modules

### What Is CBAM?
**CBAM** stands for **Convolutional Block Attention Module** (Woo et al., ECCV 2018). It's a lightweight attention mechanism that learns to focus on important features both **channel-wise** ("what to focus on") and **spatially** ("where to focus").

### Why CBAM for Dense Detection?
In dense shelf images, the model needs to:
- **Channel attention:** Focus on product-relevant features (edges, textures) while suppressing background features (shelf color, lighting)
- **Spatial attention:** Focus on regions where products are while ignoring empty shelf space

### How CBAM Works

```
Input Feature Map F: (B, 256, H, W)
    ↓
┌─────────────────────────────────────────┐
│ CHANNEL ATTENTION ("What to focus on")  │
│                                         │
│ F → Global Avg Pool → (B, 256, 1, 1)   │
│   → FC(256→16) → ReLU → FC(16→256)     │
│                         ↓               │
│ F → Global Max Pool → (B, 256, 1, 1)   │
│   → FC(256→16) → ReLU → FC(16→256)     │
│                         ↓               │
│ Add → Sigmoid → M_c: (B, 256, 1, 1)    │
│                                         │
│ F' = F × M_c  (broadcast multiply)     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ SPATIAL ATTENTION ("Where to focus")    │
│                                         │
│ F' → Channel Avg Pool → (B, 1, H, W)   │
│ F' → Channel Max Pool → (B, 1, H, W)   │
│ Concat → (B, 2, H, W)                  │
│ → [7×7 Conv] → Sigmoid → M_s: (B,1,H,W)│
│                                         │
│ F'' = F' × M_s  (broadcast multiply)   │
└─────────────────────────────────────────┘
    ↓
Output F'': (B, 256, H, W)  ← Attention-weighted features
```

### Channel Attention: "What"
- Uses **both average pooling and max pooling** to capture different statistics
- Shared MLP with **reduction ratio 16** (256→16→256) to reduce parameters
- Learns to amplify channels that correspond to product features
- Learns to suppress channels that respond to background

### Spatial Attention: "Where"
- Compresses channel dimension with avg/max pooling
- Uses **7×7 convolution** for large receptive field
- Learns spatial attention map highlighting product locations
- Suppresses empty shelf areas

### Parameter Overhead
```
Per CBAM module:
  Channel attention MLP: 256×16 + 16×256 = 8,192 params
  Spatial attention conv: 2 × 7 × 7 × 1 = 98 params
  Total per module: ~8,290 params

Applied to P3, P4, P5: 3 × 8,290 = ~24,870 params
Percentage of model: 24,870 / 9,980,000 = 0.25%
```

CBAM adds only **0.25% more parameters** but provides meaningful attention weighting for dense scenes.

**Code location:** `src/models/cbam.py` (124 lines)

---

## Component 4: Prototype Network (ProtoNet)

### What Is ProtoNet?
ProtoNet generates **K=32 prototype masks** from the finest FPN level (P3). These prototypes are **shared basis masks** that all detections combine to create their individual instance masks.

### Architecture
```
Input: P3 feature map (B, 256, 69, 69)
    ↓
Conv3×3(256→256) + ReLU    ← Extract fine features
    ↓
Conv3×3(256→256) + ReLU    ← Deepen representation
    ↓
Conv3×3(256→256) + ReLU    ← Further refinement
    ↓
Bilinear Upsample 2×       ← Increase resolution: 69→138
    ↓
Conv3×3(256→256) + ReLU    ← Smooth upsampled features
    ↓
Conv1×1(256→32) + ReLU     ← Project to 32 prototypes
    ↓
Output: (B, 32, 138, 138)  ← 32 prototype masks at 1/4 resolution
```

### Why 32 Prototypes?
- **Too few (e.g., 4):** Can't represent diverse mask shapes → poor quality
- **Too many (e.g., 128):** Each detection needs 128 coefficients → more parameters
- **32 is the sweet spot:** Enough to represent diverse masks, few enough to be efficient
- YOLACT paper found 32 prototypes sufficient for COCO (which has more shape variety than retail products)

### Why P3 (Not P4 or P5)?
- P3 is the **finest resolution** (69×69) → produces detailed masks
- P5 (18×18) would produce coarse masks (blurry boundaries)
- The upsample to 138×138 gives masks at 1/4 of input resolution (550/4 = 137.5 ≈ 138)

### What Do Prototypes Look Like?
Prototypes are not directly interpretable, but conceptually:
```
Prototype 1:  [bright in top-left, dark elsewhere]     → encodes position
Prototype 2:  [bright in center, dark at edges]         → encodes centrality
Prototype 3:  [vertical bright stripe]                  → encodes tall objects
Prototype 4:  [horizontal bright stripe]                → encodes wide objects
Prototype 5:  [bright at edges, dark in center]         → encodes boundaries
...
Prototype 32: [complex pattern]                         → encodes some other spatial relationship
```

The model **learns** what prototypes to create during training — they emerge from the mask loss.

**Code location:** `src/models/protonet.py` (127 lines)

---

## Component 5: Prediction Head

### What Does It Predict?
For each anchor at every FPN level, the prediction head outputs three things:

```
Per anchor predictions:
1. Classification:     [background_score, object_score]  ← "Is this a product?"
2. Box regression:     [Δcx, Δcy, Δlog(w), Δlog(h)]    ← "Where exactly?"
3. Mask coefficients:  [c1, c2, ..., c32]               ← "What does its mask look like?"
```

### Architecture
```
For each FPN level (P3, P4, P5, P6, P7):
    Input: P_i (B, 256, H_i, W_i)

    Shared feature extraction:
    ├── Conv3×3(256→256) + ReLU     ← Feature layer 1
    └── Conv3×3(256→256) + ReLU     ← Feature layer 2

    Three parallel branches:
    ├── Class branch:  Conv3×3(256 → num_anchors × 2)       ← Classification
    ├── Box branch:    Conv3×3(256 → num_anchors × 4)       ← Regression
    └── Coeff branch:  Conv3×3(256 → num_anchors × 32) + Tanh  ← Mask coefficients
```

### Why Shared Weights Across FPN Levels?
The **same head** is applied to P3, P4, P5, P6, P7:
- **Fewer parameters:** 1 head instead of 5 → saves ~5.6M params
- **Better generalization:** Features at different scales should use similar detection logic
- **Standard practice:** Used in RetinaNet, FCOS, and most modern detectors

### Why Tanh for Mask Coefficients?
- Coefficients can be **positive or negative** (to add/subtract prototypes)
- Tanh outputs in [-1, +1] range → stable combination weights
- Sigmoid would only give [0, 1] → can't subtract prototypes

### Output Shapes Per Level
| FPN Level | Spatial Size | Anchors/Cell | Classification | Box Regression | Mask Coefficients |
|-----------|-------------|-------------|----------------|----------------|-------------------|
| P3 | 69×69 | 9 | 69×69×9×2 | 69×69×9×4 | 69×69×9×32 |
| P4 | 35×35 | 9 | 35×35×9×2 | 35×35×9×4 | 35×35×9×32 |
| P5 | 18×18 | 9 | 18×18×9×2 | 18×18×9×4 | 18×18×9×32 |
| P6 | 9×9 | 9 | 9×9×9×2 | 9×9×9×4 | 9×9×9×32 |
| P7 | 5×5 | 9 | 5×5×9×2 | 5×5×9×4 | 5×5×9×32 |

**Total anchors:** (69² + 35² + 18² + 9² + 5²) × 9 = (4761 + 1225 + 324 + 81 + 25) × 9 = **57,744 anchors**

(Note: the exact count depends on implementation details; the README mentions ~137,000 which may include additional anchor configurations)

**Code location:** `src/models/prediction_head.py` (194 lines)

---

## Component 6: Mask Assembly

### How Instance Masks Are Created
This is the elegant core of YOLACT. After NMS selects the top detections:

```python
# Prototypes: (B, 32, 138, 138) from ProtoNet
# Coefficients: (B, N, 32) from Prediction Head (N = num detections after NMS)

# Matrix multiplication + sigmoid:
masks = sigmoid(coefficients @ prototypes.reshape(32, 138*138))
# masks shape: (B, N, 138, 138)

# Crop to bounding box:
for each detection i:
    masks[i] = masks[i] * crop_to_bbox(detection_i.box)
```

### Visual Example
```
Detection: "Product at box [100, 200, 150, 350]"
Coefficients: [+0.9, -0.3, +0.5, ..., +0.1]

Prototype 1 (strong positive → ADD):
████████████░░░░
████████████░░░░
████████████░░░░
░░░░░░░░░░░░░░░░

Prototype 2 (negative → SUBTRACT):
░░░░░░░░████████
░░░░░░░░████████
░░░░░░░░████████
░░░░░░░░░░░░░░░░

Combined & Sigmoid:
████████░░░░░░░░
████████░░░░░░░░
████████░░░░░░░░
░░░░░░░░░░░░░░░░

Cropped to BBox [100,200,150,350]:
    ████████
    ████████
    ████████
```

The mask perfectly outlines the product within its bounding box!

---

## Component 7: Soft-NMS Post-Processing

### What Is NMS?
After the model predicts scores for all ~57,744 anchors, many nearby anchors will detect the same product. NMS removes these duplicates.

### Hard-NMS (Standard)
```
Algorithm:
1. Sort all detections by score (highest first)
2. Take highest-scoring detection M
3. For every other detection B:
   if IoU(M, B) > threshold (0.5):
       REMOVE B completely (score = 0)
4. Repeat with next highest remaining detection

Problem in dense scenes:
  Product A: score=0.95, box=[100, 200, 180, 350]
  Product B: score=0.90, box=[170, 200, 260, 350]  ← DIFFERENT product, adjacent
  IoU(A, B) = 0.55 > 0.5
  Hard-NMS: REMOVES B! → False negative → Recall collapse!
```

### Soft-NMS (This Project's Solution)
```
Algorithm:
1. Sort all detections by score (highest first)
2. Take highest-scoring detection M
3. For every other detection B:
   B.score *= exp(-IoU(M, B)² / σ)    where σ=0.5
   (Gaussian decay — gradually reduce score based on overlap)
4. Remove only if B.score < threshold (0.001)
5. Repeat with next highest remaining detection

Same example:
  Product A: score=0.95
  Product B: score=0.90, IoU with A = 0.55
  New score for B: 0.90 × exp(-0.55²/0.5) = 0.90 × 0.543 = 0.489
  0.489 > 0.001 → B is PRESERVED with reduced confidence
```

### Why Gaussian Decay?
```
Three options for Soft-NMS:

1. Linear:    score *= max(0, 1 - IoU/threshold)
   Problem:   Discontinuous at threshold → unstable behavior

2. Gaussian:  score *= exp(-IoU²/σ)           ← THIS PROJECT
   Advantage: Smooth, continuous decay
              No hard threshold needed
              σ controls decay rate

3. None:      Keep all detections
   Problem:   Too many false positives
```

### The σ=0.5 Parameter
- **Small σ (e.g., 0.1):** Aggressive decay → behaves like hard-NMS → loses overlapping detections
- **Large σ (e.g., 2.0):** Gentle decay → keeps too many duplicates → more false positives
- **σ=0.5:** Balanced — preserves legitimate overlaps while reducing duplicates
- This value was chosen empirically based on the EDA finding that GT box pairs commonly have IoU in the 0.3-0.5 range

**Code location:** `src/utils/soft_nms.py` (334 lines)

---

## Anchor System Explained

### What Are Anchors?
Think of anchors as "guesses" the model makes at every position in the feature map. Instead of predicting boxes from scratch, the model predicts **adjustments** to these pre-positioned anchors.

### How Anchors Are Generated
At each spatial position of each FPN level, we place multiple anchors:

```
P3 (69×69 grid, stride 8):
  At position (0,0):
    Anchor 1: center=(4,4), size=24×48  (aspect ratio 0.5)
    Anchor 2: center=(4,4), size=24×24  (aspect ratio 1.0)
    Anchor 3: center=(4,4), size=48×24  (aspect ratio 2.0)
    Anchor 4: center=(4,4), size=30×60  (scale 1.26)
    ... (9 anchors total per position)

  At position (0,1):
    Anchor 1: center=(12,4), size=24×48
    ... (9 more anchors)

  ... continues for all 69×69 = 4,761 positions
  Total P3 anchors: 4,761 × 9 = 42,849
```

### Anchor-Target Matching
During training, each anchor is matched to a GT box (or background):

```
For each anchor:
  1. Compute IoU with ALL GT boxes
  2. If max IoU ≥ 0.5: POSITIVE (matched to that GT box)
     → This anchor should predict: class=1, box_offset=delta, mask_coeff=values
  3. If max IoU < 0.4: NEGATIVE (background)
     → This anchor should predict: class=0
  4. If 0.4 ≤ max IoU < 0.5: IGNORE (ambiguous)
     → Not used in loss computation

Additionally:
  - Each GT box is guaranteed at least one matching anchor (bipartite matching)
  - Hard negative mining: limit negative-to-positive ratio to 3:1
```

### Why This Matters for Dense Scenes
With 147 objects and ~57,744 anchors:
- ~147 positive anchors (0.25%)
- ~440 hard negative anchors (3:1 ratio)
- ~57,157 ignored/easy negatives (99.75%)

This extreme imbalance is why **Focal Loss** is essential.

**Code location:** `src/data/anchors.py` (549 lines)

---

## Complete Forward Pass Walkthrough

Let's trace a single image through the entire model:

### Step 1: Input
```
Image: shelf photograph, resized to (3, 550, 550)
Normalized: (pixel - mean) / std
Batched: (1, 3, 550, 550)  [batch size 1 for illustration]
```

### Step 2: Backbone (MobileNetV3-Large)
```
(1, 3, 550, 550)
  → ... 16 inverted residual blocks ...
  → C3: (1, 40, 69, 69)     ← stride 8
  → C4: (1, 112, 35, 35)    ← stride 16
  → C5: (1, 960, 18, 18)    ← stride 32
```

### Step 3: FPN + CBAM
```
C5 → 1×1 conv → (1, 256, 18, 18)                              = P5_raw
P5_raw → upsample → (1, 256, 35, 35) + C4 lateral              = P4_raw
P4_raw → upsample → (1, 256, 69, 69) + C3 lateral              = P3_raw

P3_raw → 3×3 smooth → CBAM attention → P3: (1, 256, 69, 69)
P4_raw → 3×3 smooth → CBAM attention → P4: (1, 256, 35, 35)
P5_raw → 3×3 smooth → CBAM attention → P5: (1, 256, 18, 18)
P5 → 3×3 stride-2 conv                → P6: (1, 256, 9, 9)
P6 → 3×3 stride-2 conv                → P7: (1, 256, 5, 5)
```

### Step 4: ProtoNet (from P3)
```
P3: (1, 256, 69, 69)
  → 3×3 conv + ReLU → (1, 256, 69, 69)
  → 3×3 conv + ReLU → (1, 256, 69, 69)
  → 3×3 conv + ReLU → (1, 256, 69, 69)
  → upsample 2× → (1, 256, 138, 138)
  → 3×3 conv + ReLU → (1, 256, 138, 138)
  → 1×1 conv + ReLU → (1, 32, 138, 138)   ← 32 prototype masks!
```

### Step 5: Prediction Head (for each level)
```
P3: (1, 256, 69, 69) → Head →
  class: (1, 69, 69, 18)   = (B, H, W, 9_anchors × 2_classes)
  box:   (1, 69, 69, 36)   = (B, H, W, 9_anchors × 4_offsets)
  coeff: (1, 69, 69, 288)  = (B, H, W, 9_anchors × 32_coefficients)

P4: (1, 256, 35, 35) → same Head → class, box, coeff
P5: (1, 256, 18, 18) → same Head → class, box, coeff
P6: (1, 256, 9, 9)   → same Head → class, box, coeff
P7: (1, 256, 5, 5)   → same Head → class, box, coeff
```

### Step 6: Concatenate All Predictions
```
All class predictions:  (1, 57744, 2)    ← 57,744 total anchors
All box predictions:    (1, 57744, 4)
All coeff predictions:  (1, 57744, 32)
```

### Step 7: During Inference — Decode & Filter
```
1. Apply softmax to class predictions → confidence scores
2. Filter: keep anchors with object_score > 0.05
   → e.g., 500 candidates remain

3. Decode box offsets:
   predicted_box = anchor_box + predicted_offset
   (cx, cy, w, h) → (x1, y1, x2, y2)

4. Apply Soft-NMS (Gaussian, σ=0.5):
   → e.g., 200 detections remain with adjusted scores

5. Keep top-200 by score
```

### Step 8: Mask Assembly
```
For 200 remaining detections:
  coefficients: (200, 32)
  prototypes: (32, 138×138) = (32, 19044)

  raw_masks = coefficients @ prototypes = (200, 19044) → (200, 138, 138)
  masks = sigmoid(raw_masks)   → values in [0, 1]
  masks = crop_to_bbox(masks, boxes)  → zero outside each detection's box
```

### Step 9: Final Output
```
boxes:  (200, 4)     ← [x1, y1, x2, y2] for each detection
scores: (200,)       ← confidence score for each detection
masks:  (200, 138, 138) ← instance mask for each detection
labels: (200,)       ← class label (always "object" in this dataset)
```

---

## Parameter Budget Breakdown

```
TOTAL MODEL: 9,980,000 parameters (~10M)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌────────────────────────┬────────────┬─────────┐
│ Component              │ Parameters │ % Total │
├────────────────────────┼────────────┼─────────┤
│ MobileNetV3-Large      │  3,000,000 │  30.1%  │
│  ├─ Inverted residuals │  2,800,000 │         │
│  └─ SE attention       │    200,000 │         │
├────────────────────────┼────────────┼─────────┤
│ FPN + CBAM             │  3,300,000 │  33.1%  │
│  ├─ Lateral convs      │    350,000 │         │
│  ├─ Smooth convs       │  1,200,000 │         │
│  ├─ Extra down convs   │  1,200,000 │         │
│  └─ CBAM attention     │     25,000 │  0.25%  │
├────────────────────────┼────────────┼─────────┤
│ ProtoNet (32 masks)    │  2,400,000 │  24.0%  │
│  ├─ 3 conv layers      │  1,800,000 │         │
│  ├─ 1 smooth conv      │    590,000 │         │
│  └─ 1×1 projection     │      8,200 │         │
├────────────────────────┼────────────┼─────────┤
│ Prediction Head        │  1,400,000 │  14.0%  │
│  ├─ Feature convs (×2) │    600,000 │         │
│  ├─ Class branch       │    100,000 │         │
│  ├─ Box branch         │    200,000 │         │
│  └─ Coeff branch       │    500,000 │         │
└────────────────────────┴────────────┴─────────┘

For comparison:
  ResNet-101 backbone alone:     44,500,000 params
  Mask R-CNN (ResNet-101):       ~63,000,000 params
  This model (full):              9,980,000 params → 6.3× smaller!
```

---

## Important Questions & Answers

### Q: Why did I decide on this specific architecture?
**A:** The architecture was designed to solve three simultaneous challenges:
1. **Density:** CBAM attention + Soft-NMS + Focal Loss specifically address dense detection
2. **Efficiency:** MobileNetV3 + shared prediction head keep it lightweight (10M params)
3. **Segmentation:** YOLACT's prototype approach enables fast instance segmentation without per-RoI computation

No single off-the-shelf model addresses all three. This architecture is a **custom combination** of proven components, each chosen for a specific reason.

### Q: Why not use the original YOLACT backbone (ResNet-101)?
**A:** ResNet-101 has 44.5M parameters — the backbone alone is 4.5× larger than our entire model. It would:
- Not fit on mobile/edge devices
- Be slower to train and inference
- Provide diminishing returns (most accuracy comes from the detection pipeline, not backbone size)
- MobileNetV3 achieves competitive accuracy with 93% fewer parameters

### Q: Why not use a transformer backbone (ViT, Swin)?
**A:** Transformers are powerful but:
- Much more parameters than MobileNetV3
- Require more training data and epochs
- Not designed for edge deployment
- Would defeat the "lightweight" goal of the project

### Q: Why 32 prototypes? What if I changed it?
**A:**
- **Fewer (e.g., 8):** Masks would be coarser, less able to represent complex shapes
- **More (e.g., 128):** Each detection needs 128 coefficients → larger prediction head, slower
- **32** is the YOLACT paper's recommended default, validated on COCO's diverse shapes
- For retail products (mostly rectangular), 32 is likely more than enough

### Q: Why does CBAM help specifically for dense scenes?
**A:** In dense scenes, the model must distinguish between:
- Products that look very similar (same shape, similar colors)
- Products vs shelf background (both have edges and textures)
- Products that are heavily occluded

CBAM's channel attention helps **amplify discriminative features** (unique textures, logos), while spatial attention helps **locate product boundaries** in cluttered regions. Without attention, all features are weighted equally, including noise and background.

### Q: What is the receptive field of the model?
**A:**
- **P3 (stride-8):** Each cell "sees" a ~50×50 pixel region → good for small products
- **P5 (stride-32):** Each cell "sees" a ~200×200 pixel region → captures context
- **CBAM's 7×7 spatial conv:** Expands effective receptive field
- **MobileNetV3 SE blocks:** Global receptive field for channel attention

### Q: Could this architecture be improved?
**A:** Yes, several enhancements are possible:
1. **Deformable convolutions** in FPN (better alignment for irregular shapes)
2. **More prototypes** (64-128) for finer masks
3. **Cascade detection** (iteratively refine predictions)
4. **Feature alignment** (DCN or adaptive pooling)
5. **Stronger backbone** (EfficientNet-B3, ResNet-50) if edge deployment isn't required
6. **Multi-scale testing** (image pyramids at inference)

---

**← Previous:** [Part 2 - Dataset & EDA](./LEARN_PART2_Dataset_and_EDA.md) | **Next:** [Part 4 - Training & Evaluation](./LEARN_PART4_Training_and_Evaluation.md) →
