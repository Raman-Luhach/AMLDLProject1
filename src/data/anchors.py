"""
Anchor Generation for FPN-based Dense Object Detection.

Provides:
    - generate_anchors: Default anchor boxes for 5 FPN levels (P3-P7)
    - kmeans_anchors: K-means clustering on GT boxes using IoU distance
    - compute_anchor_coverage: Statistics on anchor-GT box matching quality

Designed for SSD/RetinaNet-style detectors operating on 550x550 input.
FPN levels P3-P7 have strides [8, 16, 32, 64, 128] respectively.

All operations are MPS-compatible (no CUDA-specific code).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def generate_anchors(
    config: Optional[Dict] = None,
) -> torch.Tensor:
    """
    Generate default anchor boxes for 5 FPN levels (P3-P7).

    For each FPN level, anchors are generated at every spatial location
    with 3 aspect ratios x 3 scales = 9 anchors per location.

    Default config:
        - input_size: 550
        - fpn_strides: [8, 16, 32, 64, 128]
        - aspect_ratios: [0.5, 1.0, 2.0]
        - scales: [1.0, 1.26, 1.587]  (approximately 2^0, 2^(1/3), 2^(2/3))

    Args:
        config: Optional configuration dict with keys:
            - input_size (int): Input image size (default: 550)
            - fpn_strides (list): Feature map strides (default: [8, 16, 32, 64, 128])
            - aspect_ratios (list): Anchor aspect ratios (default: [0.5, 1.0, 2.0])
            - anchor_scales (list): Anchor scale multipliers (default: [1.0, 1.26, 1.587])

    Returns:
        anchors: Tensor of shape (num_anchors, 4) in [cx, cy, w, h] format,
                 where coordinates are in absolute pixels relative to input_size.
    """
    if config is None:
        config = {}

    input_size = config.get("input_size", 550)
    fpn_strides = config.get("fpn_strides", [8, 16, 32, 64, 128])
    aspect_ratios = config.get("aspect_ratios", [0.5, 1.0, 2.0])
    scales = config.get("anchor_scales", [1.0, 1.26, 1.587])

    all_anchors = []

    for stride in fpn_strides:
        # Feature map size at this level
        feat_size = int(np.ceil(input_size / stride))

        # Base anchor size = stride * 4 (common convention)
        base_size = stride * 4

        # Generate grid of anchor centers
        # Centers are at (stride/2 + i*stride, stride/2 + j*stride)
        shifts_x = np.arange(feat_size) * stride + stride / 2.0
        shifts_y = np.arange(feat_size) * stride + stride / 2.0
        cx, cy = np.meshgrid(shifts_x, shifts_y)
        cx = cx.flatten()
        cy = cy.flatten()
        num_positions = len(cx)

        # Generate anchor sizes for each (aspect_ratio, scale) combination
        for scale in scales:
            for ar in aspect_ratios:
                # Width and height from base_size, scale, and aspect_ratio
                # ar = w/h, so w = base * sqrt(ar), h = base / sqrt(ar)
                anchor_w = base_size * scale * np.sqrt(ar)
                anchor_h = base_size * scale / np.sqrt(ar)

                # Create anchors: (num_positions, 4) in [cx, cy, w, h]
                level_anchors = np.stack(
                    [
                        cx,
                        cy,
                        np.full(num_positions, anchor_w),
                        np.full(num_positions, anchor_h),
                    ],
                    axis=1,
                ).astype(np.float32)

                all_anchors.append(level_anchors)

    # Concatenate all anchors
    anchors = np.concatenate(all_anchors, axis=0)
    anchors = torch.from_numpy(anchors).float()

    logger.info(
        f"Generated {len(anchors)} anchors across {len(fpn_strides)} FPN levels "
        f"(input_size={input_size}, strides={fpn_strides})"
    )

    return anchors


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2] format."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h] format."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.stack([cx, cy, w, h], axis=1)


def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes in [x1, y1, x2, y2] format.

    Args:
        boxes_a: (N, 4) array
        boxes_b: (M, 4) array

    Returns:
        iou: (N, M) array of IoU values
    """
    # Intersection
    max_xy = np.minimum(
        boxes_a[:, 2:4][:, np.newaxis, :],
        boxes_b[:, 2:4][np.newaxis, :, :],
    )
    min_xy = np.maximum(
        boxes_a[:, 0:2][:, np.newaxis, :],
        boxes_b[:, 0:2][np.newaxis, :, :],
    )
    inter = np.clip(max_xy - min_xy, 0, None)
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    # Union
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter_area

    return inter_area / np.maximum(union, 1e-8)


def kmeans_anchors(
    boxes: np.ndarray,
    k: int = 9,
    iou_distance: bool = True,
    max_iter: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    K-means clustering on ground truth box dimensions to find optimal anchors.

    Uses 1-IoU as the distance metric (when iou_distance=True) which is more
    appropriate for bounding boxes than Euclidean distance.

    Args:
        boxes: (N, 4) array of GT boxes in [x1, y1, x2, y2] format.
               Only widths and heights are used for clustering.
        k: Number of clusters (default: 9 for 3 aspect ratios x 3 scales)
        iou_distance: If True, use 1-IoU as distance metric. Otherwise L2.
        max_iter: Maximum iterations for convergence.
        seed: Random seed for reproducibility.

    Returns:
        centroids: (k, 2) array of [width, height] cluster centers, sorted by area.
        mean_iou: Mean IoU between each GT box and its nearest centroid.
    """
    rng = np.random.RandomState(seed)

    # Extract widths and heights
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    wh = np.stack([widths, heights], axis=1).astype(np.float32)  # (N, 2)

    n = len(wh)
    if n < k:
        logger.warning(f"Fewer boxes ({n}) than clusters ({k}). Using all boxes.")
        k = n

    # Initialize centroids by random selection (k-means++)
    indices = rng.choice(n, size=1, replace=False)
    centroids = wh[indices].copy()

    for _ in range(k - 1):
        # Compute distances to nearest centroid
        dists = _wh_distance(wh, centroids, use_iou=iou_distance)
        min_dists = dists.min(axis=1)

        # Probability proportional to distance squared
        probs = min_dists**2
        probs /= probs.sum()

        idx = rng.choice(n, p=probs)
        centroids = np.vstack([centroids, wh[idx]])

    # K-means iterations
    prev_assignments = None
    for iteration in range(max_iter):
        # Assign each box to nearest centroid
        dists = _wh_distance(wh, centroids, use_iou=iou_distance)  # (N, k)
        assignments = dists.argmin(axis=1)  # (N,)

        # Check convergence
        if prev_assignments is not None and np.all(assignments == prev_assignments):
            logger.info(f"K-means converged after {iteration + 1} iterations")
            break
        prev_assignments = assignments

        # Update centroids
        for j in range(k):
            mask = assignments == j
            if mask.any():
                centroids[j] = wh[mask].mean(axis=0)

    # Sort centroids by area (width * height)
    areas = centroids[:, 0] * centroids[:, 1]
    sort_idx = np.argsort(areas)
    centroids = centroids[sort_idx]

    # Compute mean IoU
    mean_iou = _compute_mean_iou_wh(wh, centroids)

    logger.info(
        f"K-means anchors (k={k}): mean IoU = {mean_iou:.4f}, "
        f"sizes range [{centroids[0, 0]:.1f}x{centroids[0, 1]:.1f}] "
        f"to [{centroids[-1, 0]:.1f}x{centroids[-1, 1]:.1f}]"
    )

    return centroids, mean_iou


def _wh_distance(wh: np.ndarray, centroids: np.ndarray, use_iou: bool = True) -> np.ndarray:
    """
    Compute distance between width-height pairs and centroids.

    For IoU distance, boxes are centered at origin for IoU computation.

    Args:
        wh: (N, 2) array of [width, height]
        centroids: (k, 2) array of [width, height]
        use_iou: Use 1-IoU distance instead of L2

    Returns:
        distances: (N, k) array
    """
    if not use_iou:
        # Euclidean distance
        diff = wh[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        return np.sqrt((diff**2).sum(axis=2))

    # IoU distance: center all boxes at origin
    n = len(wh)
    k = len(centroids)

    # Create [x1, y1, x2, y2] boxes centered at (0, 0)
    boxes_a = np.zeros((n, 4), dtype=np.float32)
    boxes_a[:, 0] = -wh[:, 0] / 2
    boxes_a[:, 1] = -wh[:, 1] / 2
    boxes_a[:, 2] = wh[:, 0] / 2
    boxes_a[:, 3] = wh[:, 1] / 2

    boxes_b = np.zeros((k, 4), dtype=np.float32)
    boxes_b[:, 0] = -centroids[:, 0] / 2
    boxes_b[:, 1] = -centroids[:, 1] / 2
    boxes_b[:, 2] = centroids[:, 0] / 2
    boxes_b[:, 3] = centroids[:, 1] / 2

    iou = _compute_iou_matrix(boxes_a, boxes_b)  # (N, k)
    return 1.0 - iou


def _compute_mean_iou_wh(wh: np.ndarray, centroids: np.ndarray) -> float:
    """Compute mean best-match IoU between GT boxes and centroids."""
    dists = _wh_distance(wh, centroids, use_iou=True)
    min_dists = dists.min(axis=1)
    mean_iou = 1.0 - min_dists.mean()
    return float(mean_iou)


def compute_anchor_coverage(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresholds: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> Dict[str, float]:
    """
    Compute statistics on how well anchors cover ground truth boxes.

    For each GT box, finds the best-matching anchor (highest IoU).
    Reports the fraction of GT boxes with best-match IoU above various thresholds.

    Args:
        anchors: (A, 4) tensor in [cx, cy, w, h] format
        gt_boxes: (G, 4) tensor in [x1, y1, x2, y2] format
        iou_thresholds: Tuple of IoU thresholds to report coverage at

    Returns:
        stats: Dict with keys:
            - mean_best_iou: Average best-match IoU across all GT boxes
            - median_best_iou: Median best-match IoU
            - coverage_at_{t}: Fraction of GT boxes with best IoU >= t
            - num_anchors: Total number of anchors
            - num_gt_boxes: Total number of GT boxes
    """
    if len(anchors) == 0 or len(gt_boxes) == 0:
        return {"mean_best_iou": 0.0, "num_anchors": len(anchors), "num_gt_boxes": len(gt_boxes)}

    # Convert anchors from [cx, cy, w, h] to [x1, y1, x2, y2]
    anchors_np = anchors.numpy() if isinstance(anchors, torch.Tensor) else anchors
    gt_np = gt_boxes.numpy() if isinstance(gt_boxes, torch.Tensor) else gt_boxes

    anchors_xyxy = _cxcywh_to_xyxy(anchors_np)

    # Compute IoU matrix: (G, A) - may be large, process in chunks
    num_gt = len(gt_np)
    chunk_size = 1000  # Process GT boxes in chunks to manage memory

    best_ious = np.zeros(num_gt, dtype=np.float32)

    for start in range(0, num_gt, chunk_size):
        end = min(start + chunk_size, num_gt)
        iou_chunk = _compute_iou_matrix(gt_np[start:end], anchors_xyxy)  # (chunk, A)
        best_ious[start:end] = iou_chunk.max(axis=1)

    stats: Dict[str, float] = {
        "mean_best_iou": float(best_ious.mean()),
        "median_best_iou": float(np.median(best_ious)),
        "min_best_iou": float(best_ious.min()),
        "max_best_iou": float(best_ious.max()),
        "num_anchors": len(anchors_np),
        "num_gt_boxes": num_gt,
    }

    for t in iou_thresholds:
        coverage = float((best_ious >= t).mean())
        stats[f"coverage_at_{t:.1f}"] = coverage

    return stats


def anchors_to_xyxy(anchors: torch.Tensor) -> torch.Tensor:
    """
    Convert anchors from [cx, cy, w, h] to [x1, y1, x2, y2] format.

    Args:
        anchors: (N, 4) tensor in [cx, cy, w, h] format

    Returns:
        anchors_xyxy: (N, 4) tensor in [x1, y1, x2, y2] format
    """
    cx, cy, w, h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def anchors_to_cxcywh(anchors: torch.Tensor) -> torch.Tensor:
    """
    Convert anchors from [x1, y1, x2, y2] to [cx, cy, w, h] format.

    Args:
        anchors: (N, 4) tensor in [x1, y1, x2, y2] format

    Returns:
        anchors_cxcywh: (N, 4) tensor in [cx, cy, w, h] format
    """
    x1, y1, x2, y2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=1)


def encode_boxes(
    gt_boxes: torch.Tensor,
    anchors: torch.Tensor,
    variances: Tuple[float, float] = (0.1, 0.2),
) -> torch.Tensor:
    """
    Encode ground truth boxes relative to anchor boxes (SSD-style encoding).

    Args:
        gt_boxes: (N, 4) tensor in [x1, y1, x2, y2] format
        anchors: (N, 4) tensor in [cx, cy, w, h] format
        variances: (var_xy, var_wh) for normalizing the offsets

    Returns:
        encoded: (N, 4) tensor of encoded offsets [dx, dy, dw, dh]
    """
    # Convert GT to center format
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]

    # Encode as offsets from anchors
    dx = (gt_cx - anchors[:, 0]) / (anchors[:, 2] * variances[0])
    dy = (gt_cy - anchors[:, 1]) / (anchors[:, 3] * variances[0])
    dw = torch.log(gt_w / anchors[:, 2].clamp(min=1e-6)) / variances[1]
    dh = torch.log(gt_h / anchors[:, 3].clamp(min=1e-6)) / variances[1]

    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(
    offsets: torch.Tensor,
    anchors: torch.Tensor,
    variances: Tuple[float, float] = (0.1, 0.2),
) -> torch.Tensor:
    """
    Decode predicted offsets back to [x1, y1, x2, y2] boxes.

    Args:
        offsets: (N, 4) tensor of encoded offsets [dx, dy, dw, dh]
        anchors: (N, 4) tensor in [cx, cy, w, h] format
        variances: (var_xy, var_wh) used during encoding

    Returns:
        decoded: (N, 4) tensor in [x1, y1, x2, y2] format
    """
    cx = offsets[:, 0] * variances[0] * anchors[:, 2] + anchors[:, 0]
    cy = offsets[:, 1] * variances[0] * anchors[:, 3] + anchors[:, 1]
    w = torch.exp(offsets[:, 2] * variances[1]) * anchors[:, 2]
    h = torch.exp(offsets[:, 3] * variances[1]) * anchors[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


if __name__ == "__main__":
    """Test anchor generation and analysis."""
    logging.basicConfig(level=logging.INFO)

    print("=== Anchor Generation Test ===\n")

    # 1. Generate default anchors
    print("--- Default FPN Anchors ---")
    anchors = generate_anchors()
    print(f"Total anchors: {anchors.shape[0]}")
    print(f"Anchor shape: {anchors.shape}")
    print(f"Format: [cx, cy, w, h]")

    # Print per-level statistics
    config = {
        "input_size": 550,
        "fpn_strides": [8, 16, 32, 64, 128],
        "aspect_ratios": [0.5, 1.0, 2.0],
        "anchor_scales": [1.0, 1.26, 1.587],
    }
    strides = config["fpn_strides"]
    n_ratios = len(config["aspect_ratios"])
    n_scales = len(config["anchor_scales"])
    anchors_per_loc = n_ratios * n_scales

    offset = 0
    for stride in strides:
        feat_size = int(np.ceil(550 / stride))
        n_anchors = feat_size * feat_size * anchors_per_loc
        level_anchors = anchors[offset : offset + n_anchors]
        widths = level_anchors[:, 2]
        heights = level_anchors[:, 3]
        print(
            f"  Stride {stride:3d}: {feat_size:3d}x{feat_size:3d} grid, "
            f"{n_anchors:6d} anchors, "
            f"w=[{widths.min():.1f}, {widths.max():.1f}], "
            f"h=[{heights.min():.1f}, {heights.max():.1f}]"
        )
        offset += n_anchors

    # 2. Convert formats
    print("\n--- Format Conversion ---")
    anchors_xyxy = anchors_to_xyxy(anchors)
    print(f"[cx,cy,w,h] -> [x1,y1,x2,y2]: {anchors_xyxy.shape}")
    anchors_back = anchors_to_cxcywh(anchors_xyxy)
    assert torch.allclose(anchors, anchors_back, atol=1e-5), "Roundtrip failed"
    print("Format roundtrip: OK")

    # 3. Test encode/decode
    print("\n--- Encode/Decode ---")
    test_gt = torch.tensor([[100, 100, 200, 200], [300, 300, 400, 450]], dtype=torch.float32)
    test_anchors = torch.tensor([[150, 150, 100, 100], [350, 375, 100, 150]], dtype=torch.float32)
    encoded = encode_boxes(test_gt, test_anchors)
    decoded = decode_boxes(encoded, test_anchors)
    assert torch.allclose(test_gt, decoded, atol=1e-4), "Encode/decode roundtrip failed"
    print("Encode/decode roundtrip: OK")

    # 4. Test K-means anchors on synthetic data
    print("\n--- K-means Anchors ---")
    rng = np.random.RandomState(42)
    n_synthetic = 5000

    # Simulate SKU-110K-like box distribution (many small, dense objects)
    widths_syn = rng.lognormal(mean=3.5, sigma=0.6, size=n_synthetic).astype(np.float32)
    heights_syn = rng.lognormal(mean=3.5, sigma=0.6, size=n_synthetic).astype(np.float32)
    x1_syn = rng.uniform(0, 400, n_synthetic).astype(np.float32)
    y1_syn = rng.uniform(0, 400, n_synthetic).astype(np.float32)
    synthetic_boxes = np.stack([x1_syn, y1_syn, x1_syn + widths_syn, y1_syn + heights_syn], axis=1)

    centroids, mean_iou = kmeans_anchors(synthetic_boxes, k=9)
    print(f"K-means centroids (k=9):")
    for i, (w, h) in enumerate(centroids):
        print(f"  Anchor {i}: {w:.1f} x {h:.1f} (area={w * h:.0f})")
    print(f"Mean IoU with GT: {mean_iou:.4f}")

    # 5. Test anchor coverage
    print("\n--- Anchor Coverage ---")
    gt_sample = torch.from_numpy(synthetic_boxes[:500])
    coverage = compute_anchor_coverage(anchors, gt_sample)
    print(f"Mean best IoU: {coverage['mean_best_iou']:.4f}")
    print(f"Median best IoU: {coverage['median_best_iou']:.4f}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        key = f"coverage_at_{t:.1f}"
        if key in coverage:
            print(f"Coverage at IoU >= {t:.1f}: {coverage[key] * 100:.1f}%")

    print("\n=== All Tests Passed ===")
