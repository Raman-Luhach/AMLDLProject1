# Part 2: Dataset Deep Dive, EDA & Classical Baseline

## Table of Contents
1. [The SKU-110K Dataset](#the-sku-110k-dataset)
2. [Why This Dataset?](#why-this-dataset)
3. [Where Did the Dataset Come From?](#where-did-the-dataset-come-from)
4. [Dataset Statistics](#dataset-statistics)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [K-Means Anchor Optimization](#k-means-anchor-optimization)
7. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
8. [Data Augmentation Strategies](#data-augmentation-strategies)
9. [Classical ML Baseline: HOG + SVM](#classical-ml-baseline-hog--svm)
10. [Why the Baseline Matters](#why-the-baseline-matters)
11. [Important Questions & Answers](#important-questions--answers)

---

## The SKU-110K Dataset

### What Is It?
SKU-110K is a large-scale dataset of **densely packed retail shelf images** published at **CVPR 2019** (one of the top 3 computer vision conferences in the world). Each image is a photograph of a supermarket shelf, and every single product on that shelf is annotated with a bounding box.

### Dataset at a Glance
| Property | Value |
|----------|-------|
| **Total Images** | 11,762 |
| **Total Annotations** | ~1,730,000 bounding boxes |
| **Average Objects/Image** | 147.4 |
| **Min Objects/Image** | ~1 |
| **Max Objects/Image** | 400+ |
| **Number of Classes** | 1 (just "object") |
| **Image Format** | JPEG (variable resolution) |
| **Annotation Format** | CSV (image_name, x1, y1, x2, y2, class, image_width, image_height) |
| **Train Split** | 8,233 images |
| **Validation Split** | 588 images |
| **Test Split** | 2,941 images |
| **Download Size** | ~2.4 GB |

### What Does the Data Look Like?
Each image shows a section of a retail store shelf with products arranged in rows. The annotations are **axis-aligned bounding boxes** around each product:

```
Image: shelf_001.jpg (2048 x 1536 pixels)
Annotations:
  Product 1: [x1=100, y1=50, x2=180, y2=200]   # top-left shelf
  Product 2: [x1=170, y1=50, x2=260, y2=200]   # right next to Product 1
  Product 3: [x1=160, y1=55, x2=245, y2=195]   # OVERLAPPING with 1 & 2!
  ...
  Product 147: [x1=1800, y1=1200, x2=1900, y2=1400]  # bottom-right
```

Notice how Products 1, 2, and 3 have **overlapping bounding boxes** — this is the fundamental challenge of dense detection.

---

## Why This Dataset?

### Comparison with Other Datasets

| Dataset | Avg Objects/Image | Classes | Challenge Level | Dense? |
|---------|-------------------|---------|----------------|--------|
| **COCO** | 7.7 | 80 | Standard | No |
| **Pascal VOC** | 2.5 | 20 | Easy | No |
| **Open Images** | ~8 | 600 | Standard | No |
| **CrowdHuman** | 22.6 | 1 (person) | Hard | Moderate |
| **SKU-110K** | **147.4** | 1 (object) | **Extreme** | **Yes** |

### Why Not COCO or VOC?
- COCO has only 7.7 objects per image on average — this is NOT a dense detection problem
- Standard models already achieve 50%+ mAP on COCO — the problem is "solved"
- We specifically need a dataset where objects are **tightly packed** and **overlapping**

### Why Not a Custom Dataset?
- Creating annotations for 147 objects per image × 11,762 images = **1.73 million annotations** would take months
- SKU-110K is **professionally annotated** and **peer-reviewed** (published at CVPR)
- Using a published benchmark allows **fair comparison** with other research

### Why Single-Class?
- The dataset only has one class: "object" (not specific product types)
- This is **intentional**: the research question is about **density**, not **classification**
- If the model can detect 147 "objects" in a dense scene, adding classification is a simpler extension
- Multi-class detection on shelves would require product-specific labels (extremely expensive to annotate)

---

## Where Did the Dataset Come From?

### Origin
- **Paper:** "Precise Detection in Densely Packed Scenes" by Goldman et al.
- **Conference:** CVPR 2019 (IEEE Conference on Computer Vision and Pattern Recognition)
- **Authors:** From Tel Aviv University and Trax (retail AI company)
- **GitHub:** https://github.com/eg4000/SKU110K_CVPR19
- **Download:** Hosted on AWS S3 (the `download_data.sh` script handles this)

### How Were Images Collected?
- Images were taken in **real retail stores** (supermarkets, convenience stores)
- Multiple countries and store formats for diversity
- Various lighting conditions, angles, and shelf configurations
- Both close-up and wide-angle shots

### How Were Images Annotated?
- **Professional annotation teams** drew bounding boxes around every visible product
- Quality control with multiple annotators per image
- Strict guidelines: annotate every product even if partially occluded
- Result: 1.73 million high-quality bounding boxes

### Important Note: No Instance Masks
- SKU-110K provides **bounding boxes only**, NOT pixel-level instance masks
- For the instance segmentation part of this project, bounding boxes are used as pseudo-masks
- This means the mask quality is limited by the rectangular box approximation
- Full pixel-level annotations would require additional annotation effort

---

## Dataset Statistics

### Object Count Distribution
```
Objects/Image Distribution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
  0-50:    ████░░░░░░░░░░░  ~15% of images (sparse shelves)
 50-100:   █████████░░░░░░  ~25% of images
100-150:   ████████████░░░  ~30% of images (most common)
150-200:   ████████░░░░░░░  ~18% of images
200-300:   ███░░░░░░░░░░░░  ~10% of images
300-400+:  █░░░░░░░░░░░░░░  ~2% of images (ultra-dense)

Mean: 147.4 | Median: ~120 | Std: ~95
```

**Key Insight:** The distribution is **right-skewed** — most images have 50-200 objects, but some extreme cases have 400+. The model must handle this wide variance.

### Bounding Box Dimensions
```
Typical Box Sizes (normalized to image dimensions):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Width:  0.01 ─────────── 0.04 ─── 0.10 ─── 0.40
        (tiny)          (typical) (medium)  (large)

Height: 0.01 ─────────── 0.06 ─── 0.15 ─── 0.50
        (tiny)          (typical) (medium)  (large)

Mean Area: 0.285% of image area (objects are VERY small)
```

**Key Insight:** Products are tiny relative to the image. This means:
- Fine-resolution features (stride-8 P3 level) are critical
- Anchor sizes must be carefully tuned for small objects
- ProtoNet operates at 1/4 resolution for mask detail

### Aspect Ratio Distribution
```
Aspect Ratios (width/height):
━━━━━━━━━━━━━━━━━━━━━━━━━━━
  < 0.3:  █░░░░░░░░░  ~5% (very tall/narrow - bottles)
0.3-0.5:  ██████░░░░  ~25% (tall products)
0.5-0.7:  █████████░  ~35% (typical products, MEDIAN=0.534)
0.7-1.0:  ████░░░░░░  ~20% (near-square products)
  > 1.0:  ███░░░░░░░  ~15% (wide products - boxes)
```

**Key Insight:** Products are mostly **portrait-oriented** (taller than wide, median AR=0.534). This informs anchor aspect ratio design — we need more tall/narrow anchors.

### Pairwise IoU Between Ground-Truth Boxes
This is the **most important finding** from EDA:
```
Pairwise IoU Distribution (between GT boxes in same image):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IoU = 0.0:    ██████████████████  ~60% (non-overlapping)
0.0 < IoU ≤ 0.1:  █████████░░░  ~20%
0.1 < IoU ≤ 0.3:  ████░░░░░░░░  ~12%
0.3 < IoU ≤ 0.5:  ██░░░░░░░░░░  ~5%  ← Hard-NMS danger zone!
IoU > 0.5:        █░░░░░░░░░░░░  ~3%  ← Hard-NMS KILLS these!
```

**Critical Finding:** A significant fraction of **legitimate** ground-truth box pairs have IoU > 0.3. Hard-NMS with threshold 0.5 would suppress these as "duplicates" even though they are **different products**. This is the **quantitative justification** for using Soft-NMS.

---

## Exploratory Data Analysis (EDA)

The EDA phase generates 8 visualization plots saved in `results/eda/`. Each one reveals something important about the data that directly informed architecture decisions.

### 1. `objects_per_image_histogram.png`
**What it shows:** Histogram of object counts per image
**Key finding:** Right-skewed distribution, mean=147.4, high variance (std=95)
**Design implication:** Model must handle 10-400+ objects without fixed upper limit
**Code:** `scripts/run_eda.py` → `src/utils/visualization.py`

### 2. `box_dimensions_scatter.png`
**What it shows:** Scatter plot of box width vs height (normalized)
**Key finding:** Most boxes are small (< 5% of image area), portrait-oriented
**Design implication:** Need fine-resolution feature maps (P3 at stride-8)

### 3. `aspect_ratio_distribution.png`
**What it shows:** Histogram of bounding box aspect ratios
**Key finding:** Median AR=0.534 (portrait), wide range from 0.1 to 10
**Design implication:** Anchors must cover diverse aspect ratios (0.5, 1.0, 2.0)

### 4. `box_area_distribution.png`
**What it shows:** Log-scale distribution of bounding box areas
**Key finding:** Log-normal distribution, spanning 3 orders of magnitude
**Design implication:** Multi-scale detection essential (FPN P3-P7)

### 5. `pairwise_iou_histogram.png`
**What it shows:** Distribution of IoU between all pairs of GT boxes per image
**Key finding:** Many legitimate box pairs have IoU > 0.3
**Design implication:** Soft-NMS required; hard-NMS would cause recall collapse

### 6. `anchor_kmeans_analysis.png`
**What it shows:** K-means clustering of GT box dimensions + elbow curve
**Key finding:** K=9 clusters with mean IoU=0.7156 (vs ~0.5 for default anchors)
**Design implication:** Custom anchors improve matching → faster convergence

### 7. `sample_images_with_annotations.png`
**What it shows:** Example images with ground-truth boxes drawn
**Key finding:** Visual confirmation of extreme density and occlusion
**Design implication:** Confirms this is genuinely a challenging problem

### 8. `data_split_statistics.png`
**What it shows:** Statistics breakdown for train/val/test splits
**Key finding:** 8,233/588/2,941 split (70/5/25), consistent distributions
**Design implication:** Standard split, adequate for training and evaluation

---

## K-Means Anchor Optimization

### What Are Anchors?
Anchors (also called prior boxes) are **predefined reference boxes** at each spatial location in the feature map. The model doesn't predict bounding boxes from scratch — it predicts **offsets** from the nearest anchor.

```
Without good anchors:  Model predicts: "move 200px right, 150px down,
                       shrink 80%, stretch 120%"  → hard to learn!

With good anchors:     Model predicts: "move 5px right, 3px down,
                       shrink 5%, stretch 2%"     → easy to learn!
```

### Why K-Means on Ground Truth?
Standard COCO anchors are optimized for COCO dataset object sizes. SKU-110K products have very different size distributions. By running K-means clustering on all GT box dimensions:

1. Extract all GT box (width, height) pairs from training set
2. Normalize to [0, 1] range
3. Run K-means with K=1 to 15
4. Measure mean IoU between GT boxes and nearest cluster center
5. Choose K that gives best IoU/complexity tradeoff

### Results
```
K=1:  mean IoU = 0.42  (too few anchors)
K=3:  mean IoU = 0.58
K=5:  mean IoU = 0.64
K=9:  mean IoU = 0.72  ← CHOSEN (elbow point)
K=12: mean IoU = 0.75  (diminishing returns)
K=15: mean IoU = 0.77  (too many, slows inference)
```

### Final Anchor Configuration
```yaml
# configs/custom_anchors.yaml
anchor_sizes: [24, 48, 96, 192, 384]      # 5 base sizes (pixels)
aspect_ratios: [0.5, 1.0, 2.0]            # 3 ratios per size
scales: [1.0, 1.2599, 1.5874]             # 3 scales = 2^(0/3), 2^(1/3), 2^(2/3)

# Total: 5 FPN levels × 3 ratios × 3 scales = 45 anchor types
# Per image: 45 × (69×69 + 35×35 + 18×18 + 9×9 + 5×5) ≈ 137,000 anchors
```

**Why 137,000 anchors for 147 objects?** Because each spatial location needs multiple anchor types to match different product sizes. Only ~147 anchors will match actual products (positive anchors) — the rest are background (negative anchors). This creates a **1:930 class imbalance**, which is why Focal Loss is critical.

---

## Data Preprocessing Pipeline

### Loading (`src/data/dataset.py`)
```python
class SKU110KDataset(torch.utils.data.Dataset):
    """
    Loads SKU-110K dataset from CSV annotations.

    For each image:
    1. Parse CSV row: image_name, x1, y1, x2, y2, class, width, height
    2. Group all boxes for same image
    3. Load image from disk (JPEG)
    4. Apply augmentations
    5. Generate anchor targets
    6. Return (image, targets) pair
    """
```

### Preprocessing Steps
1. **Image Loading:** Read JPEG, convert to RGB
2. **Resize:** Resize to 550×550 (YOLACT standard input size)
3. **Normalization:** Subtract ImageNet mean, divide by ImageNet std
   ```
   mean = [0.485, 0.456, 0.406]
   std  = [0.229, 0.224, 0.225]
   ```
4. **Box Rescaling:** Scale bounding boxes proportionally with image resize
5. **Anchor Matching:** Assign each GT box to best matching anchor(s)
   - Positive: IoU ≥ 0.5 with any GT box
   - Negative: IoU < 0.4 with all GT boxes
   - Ignore: 0.4 ≤ IoU < 0.5 (ambiguous, not used in loss)

### Why 550×550?
- YOLACT paper standard (designed for this resolution)
- Divisible by 32 (required for stride-32 feature maps)
- Balance between resolution and memory/speed
- Produces P3 feature maps of 69×69 (fine enough for small products)

---

## Data Augmentation Strategies

### Why Augmentation?
With only 3,000 training images (and even the full 8,233), the model can easily overfit. Augmentation artificially increases data diversity:

### 1. MixUp (α=0.2)
```
What:  Blend two training images: image_new = λ*image_A + (1-λ)*image_B
       where λ ~ Beta(0.2, 0.2)
Why:   Creates "virtual" training examples between real samples
       Smooths decision boundaries
       Reduces overconfidence
Effect: Model sees ghostly overlaps of two shelf images → learns robust features
```

### 2. Label Smoothing (ε=0.1)
```
What:  Instead of hard labels [0, 1], use soft labels [0.05, 0.95]
       target = (1 - ε) * one_hot + ε / num_classes
Why:   Prevents model from being overconfident
       Acts as regularization
       Improves calibration of predicted scores
Effect: Model assigns slightly less extreme probabilities → better generalization
```

### 3. Random Horizontal Flip
```
What:  Flip image left-right with 50% probability
Why:   Shelves look the same from left or right
       Doubles effective training data
Effect: Model doesn't learn left/right bias
```

### 4. Photometric Distortion
```
What:  Random adjustments to brightness, contrast, saturation, hue
Why:   Different stores have different lighting
       Makes model robust to camera/lighting variations
Effect: Model focuses on shape/texture, not color/brightness
```

### 5. Random Crop with IoU Constraint
```
What:  Randomly crop a region of the image
       BUT: ensure crop overlaps with GT boxes by at least IoU ≥ 0.3
Why:   Creates different viewpoints of the shelf
       Ensures we don't accidentally crop out all products
Effect: Model learns to detect products at various positions
```

### 6. Resize to 550×550
```
What:  Resize all images to fixed input size
Why:   Batch processing requires fixed dimensions
       550 is YOLACT's standard input resolution
Effect: Consistent feature map sizes across all images
```

---

## Classical ML Baseline: HOG + SVM

### What Is the Baseline?
Before training a deep learning model, we establish a **classical machine learning baseline** using:
- **HOG** (Histogram of Oriented Gradients) for feature extraction
- **Linear SVM** (Support Vector Machine) for classification
- **Sliding Window** for detection

This provides a **lower bound** for performance and validates that the problem is genuinely hard.

### How HOG + SVM Works

#### Step 1: HOG Feature Extraction
```
Input: 64×64 image patch

1. Compute gradients:
   Gx = image ∗ [-1, 0, 1]  (horizontal gradient)
   Gy = image ∗ [-1, 0, 1]^T (vertical gradient)

2. Compute magnitude and orientation:
   magnitude = sqrt(Gx² + Gy²)
   orientation = atan2(Gy, Gx)  → quantize to 9 bins (0°-180°)

3. Divide patch into 8×8 cells
4. For each cell: create 9-bin histogram of gradient orientations
5. Normalize histograms in overlapping 2×2 blocks

Output: 1,764-dimensional feature vector per patch
```

**Why HOG?** It captures **edge and texture patterns** (product labels, packaging edges) without learning — it's a hand-crafted feature that has been a standard in classical detection since 2005.

#### Step 2: SVM Training
```
Input: HOG features from positive (product) and negative (background) patches

1. Extract ~5,000 positive patches (centered on GT boxes)
2. Extract ~15,000 negative patches (random background regions)
3. Compute HOG features for all patches: X ∈ R^(20000 × 1764)
4. Train Linear SVM: find hyperplane that separates products from background
   minimize: ||w||² + C * Σ max(0, 1 - y_i * (w·x_i + b))

Output: Weight vector w ∈ R^1764, bias b
```

#### Step 3: Sliding Window Detection
```
For each image:
1. Create image pyramid (multiple scales: 0.5x, 0.75x, 1.0x, 1.25x, 1.5x)
2. At each scale, slide 64×64 window across image:
   - Step size: 16 pixels (stride)
   - At each position: extract HOG, classify with SVM
   - If SVM score > threshold: record as detection
3. Apply NMS to remove duplicate detections
4. Map detections back to original image coordinates

Output: List of (x1, y1, x2, y2, score) detections
```

### Baseline Results

| Metric | Value |
|--------|-------|
| **mAP@0.50** | 3.09% |
| **Precision** | 86.36% |
| **Recall** | 2.09% |
| **Total Detections** | 22 (across 50 test images) |
| **True Positives** | 19 |
| **False Positives** | 3 |
| **Avg Time/Image** | 0.251 seconds |

### Understanding the Baseline Numbers

**High Precision (86.36%) + Low Recall (2.09%):**
- The SVM is **extremely conservative**: when it detects something, it's almost always correct
- But it only detects **22 products across 50 images** (vs ~7,350 ground truth products)
- It misses 98% of products

**Why?**
1. **Sliding window is inefficient:** Fixed 64×64 patch can't match varying product sizes
2. **HOG features are limited:** Can't capture complex product appearances
3. **No context:** HOG looks at each patch independently, ignoring surrounding shelf context
4. **Scale mismatch:** Image pyramid covers limited scale range
5. **NMS problems:** Even the few detections get suppressed by NMS

**Why 3.09% mAP is still meaningful:**
- It proves the problem is **solvable** (non-random)
- It sets a **floor** that deep learning should exceed
- It shows that **even perfect precision is useless without recall** in dense scenes

---

## Why the Baseline Matters

### Scientific Method
Every experiment needs a **control group**. The HOG+SVM baseline serves as:
1. **Lower bound:** Any reasonable deep learning model should beat 3.09% mAP
2. **Methodology validation:** Proves evaluation pipeline works correctly
3. **Classical vs Deep Learning comparison:** Quantifies the benefit of learned features
4. **Feature understanding:** HOG visualizations show what hand-crafted features capture vs what deep features learn

### What Baseline Reveals About the Problem
| Insight | Evidence | Implication |
|---------|----------|-------------|
| Problem is genuinely hard | 2.09% recall with classical methods | Deep learning approach is justified |
| Feature learning is critical | HOG captures edges but not product identity | Need learned representations (CNN) |
| Multi-scale detection needed | Fixed window misses size variation | FPN architecture is justified |
| Dense NMS is a bottleneck | Even 22 detections have NMS issues | Soft-NMS approach is justified |
| Context matters | Independent patches miss spatial relationships | FPN/attention mechanisms are justified |

---

## Important Questions & Answers

### Q: Why didn't you use a newer/bigger dataset?
**A:** SKU-110K is the **gold standard** for dense retail detection. It has the highest object density of any public detection dataset (147 avg vs ~8 for COCO). Using it allows direct comparison with other research papers. There simply isn't a better public dataset for this specific problem.

### Q: Why only use 3,000 of 8,233 training images?
**A:** Computational constraints. Training on Apple Silicon MPS with batch size 4, each epoch on 3,000 images takes ~30 minutes. Full dataset training (8,233 images) for 80 epochs would take ~50+ hours. The 3,000 image subset was chosen to demonstrate the complete pipeline within reasonable time constraints while still showing clear learning signals.

### Q: Why is the dataset single-class? Isn't that too simple?
**A:** Actually, it's the **opposite** — single-class makes the **detection** problem harder, not easier. With multiple classes, the model can use class differences to distinguish overlapping products. With single-class, the model must rely entirely on **spatial** cues to separate identical-class objects that overlap. The research question is specifically about **density**, not **classification**.

### Q: Could I use this dataset for multi-class product recognition?
**A:** Not directly — SKU-110K only labels "object." But the detection pipeline from this project could be combined with:
- A separate product classification model
- Transfer learning from product recognition datasets (e.g., Grozi-120, RP2K)
- Fine-tuning with custom product labels

### Q: Why HOG+SVM for the baseline? Why not a simpler/different baseline?
**A:** HOG+SVM is the **canonical classical detection method** (Dalal & Triggs, 2005). It's the most well-established non-deep-learning approach for object detection. Using it provides:
- A well-understood comparison point
- Clear demonstration of deep learning's advantages
- Educational value (understanding classical CV)
- Other baselines (e.g., template matching, SIFT+BoW) would give similar or worse results

### Q: What would the EDA look different if I used a non-dense dataset like COCO?
**A:** Dramatically:
- Object count histogram: peak at 5-10 (not 100-150)
- Pairwise IoU: almost all near 0 (objects well-separated)
- Box area: much larger (objects are bigger in COCO)
- K-means anchors: would match COCO defaults (already optimized for COCO)
- The entire justification for Soft-NMS would disappear

### Q: How confident are we in the annotation quality?
**A:** Very confident. SKU-110K was:
- Published at CVPR 2019 (rigorous peer review)
- Created by Trax, a professional retail AI company
- Annotated by trained professionals with quality control
- Used in dozens of follow-up papers as a benchmark
- Any annotation noise is consistent across train/test splits

### Q: Why not use instance segmentation annotations (polygon masks)?
**A:** SKU-110K only provides bounding boxes. Creating polygon masks for 1.73 million products would cost tens of thousands of dollars in annotation effort. The project uses bounding boxes as **pseudo-masks** — a practical compromise that still demonstrates the instance segmentation pipeline. With real masks, performance would be even better.

---

**← Previous:** [Part 1 - Project Overview](./LEARN_PART1_Project_Overview_and_Motivation.md) | **Next:** [Part 3 - Architecture Deep Dive](./LEARN_PART3_Architecture_and_Model.md) →
