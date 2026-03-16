"""
SSD-style Data Augmentation Pipeline for Dense Object Detection.

Implements augmentations using OpenCV and NumPy (no albumentations dependency).
Each augmentation handles both image and bounding boxes jointly to maintain
spatial consistency.

Augmentations for training:
    - Random horizontal flip
    - Random photometric distortion (brightness, contrast, saturation, hue)
    - Random expand (zoom out with mean pixel fill)
    - Random IoU-preserving crop
    - Resize to target size
    - ImageNet normalization

Validation uses only resize + normalize.

All bounding boxes are in [x1, y1, x2, y2] absolute coordinate format.
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Mean pixel value (ImageNet mean * 255) for padding during expand
MEAN_PIXEL = (IMAGENET_MEAN * 255.0).astype(np.uint8)


def intersect(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise intersection areas between two sets of boxes.

    Args:
        boxes_a: (N, 4) array in [x1, y1, x2, y2] format
        boxes_b: (M, 4) array in [x1, y1, x2, y2] format

    Returns:
        intersection: (N, M) array of intersection areas
    """
    n = boxes_a.shape[0]
    m = boxes_b.shape[0]

    max_xy = np.minimum(
        boxes_a[:, 2:4][:, np.newaxis, :],  # (N, 1, 2)
        boxes_b[:, 2:4][np.newaxis, :, :],  # (1, M, 2)
    )
    min_xy = np.maximum(
        boxes_a[:, 0:2][:, np.newaxis, :],  # (N, 1, 2)
        boxes_b[:, 0:2][np.newaxis, :, :],  # (1, M, 2)
    )

    inter = np.clip(max_xy - min_xy, 0, None)
    return inter[:, :, 0] * inter[:, :, 1]  # (N, M)


def jaccard_numpy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU (Jaccard index) between two sets of boxes.

    Args:
        boxes_a: (N, 4) array in [x1, y1, x2, y2] format
        boxes_b: (M, 4) array in [x1, y1, x2, y2] format

    Returns:
        iou: (N, M) array of IoU values
    """
    inter = intersect(boxes_a, boxes_b)
    area_a = (
        (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    )  # (N,)
    area_b = (
        (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    )  # (M,)

    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter
    return inter / np.maximum(union, 1e-6)


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------


def random_horizontal_flip(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly flip image and boxes horizontally.

    Args:
        image: (H, W, 3) uint8 array
        boxes: (N, 4) float32 array in [x1, y1, x2, y2]
        labels: (N,) int64 array
        p: Probability of flipping

    Returns:
        Flipped (image, boxes, labels)
    """
    if np.random.random() < p:
        h, w = image.shape[:2]
        image = image[:, ::-1, :].copy()
        if len(boxes) > 0:
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = w - x2
            boxes[:, 2] = w - x1
    return image, boxes, labels


def random_brightness(
    image: np.ndarray, delta: float = 32.0
) -> np.ndarray:
    """Randomly adjust brightness by adding a uniform random value."""
    if np.random.random() < 0.5:
        d = np.random.uniform(-delta, delta)
        image = image.astype(np.float32) + d
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_contrast(
    image: np.ndarray, lower: float = 0.5, upper: float = 1.5
) -> np.ndarray:
    """Randomly adjust contrast by multiplying by a uniform random factor."""
    if np.random.random() < 0.5:
        factor = np.random.uniform(lower, upper)
        image = image.astype(np.float32) * factor
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_saturation(
    image: np.ndarray, lower: float = 0.5, upper: float = 1.5
) -> np.ndarray:
    """Randomly adjust saturation in HSV space."""
    if np.random.random() < 0.5:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        factor = np.random.uniform(lower, upper)
        hsv[:, :, 1] *= factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return image


def random_hue(image: np.ndarray, delta: float = 18.0) -> np.ndarray:
    """Randomly adjust hue in HSV space."""
    if np.random.random() < 0.5:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        d = np.random.uniform(-delta, delta)
        hsv[:, :, 0] += d
        hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180.0)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return image


def photometric_distortion(image: np.ndarray) -> np.ndarray:
    """
    Apply random photometric distortions (SSD-style).

    Randomly applies brightness, contrast, saturation, and hue adjustments.
    Contrast can be applied before or after saturation/hue (random order).
    """
    image = random_brightness(image)

    if np.random.random() < 0.5:
        # Contrast first
        image = random_contrast(image)
        image = random_saturation(image)
        image = random_hue(image)
    else:
        # Saturation/hue first
        image = random_saturation(image)
        image = random_hue(image)
        image = random_contrast(image)

    return image


def random_expand(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    max_ratio: float = 4.0,
    p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly expand (zoom out) the image by placing it on a larger canvas
    filled with mean pixel values. This simulates objects appearing smaller.

    Args:
        image: (H, W, 3) uint8 array
        boxes: (N, 4) float32 array in [x1, y1, x2, y2]
        labels: (N,) int64 array
        max_ratio: Maximum expansion ratio
        p: Probability of applying expansion

    Returns:
        Expanded (image, boxes, labels)
    """
    if np.random.random() > p:
        return image, boxes, labels

    h, w, c = image.shape
    ratio = np.random.uniform(1.0, max_ratio)
    new_h = int(h * ratio)
    new_w = int(w * ratio)

    # Create canvas filled with mean pixel color
    canvas = np.zeros((new_h, new_w, c), dtype=np.uint8)
    canvas[:, :] = (IMAGENET_MEAN * 255.0).astype(np.uint8)

    # Random placement
    top = int(np.random.uniform(0, new_h - h))
    left = int(np.random.uniform(0, new_w - w))

    canvas[top : top + h, left : left + w, :] = image

    # Shift boxes
    if len(boxes) > 0:
        boxes = boxes.copy()
        boxes[:, 0] += left
        boxes[:, 2] += left
        boxes[:, 1] += top
        boxes[:, 3] += top

    return canvas, boxes, labels


def random_crop(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    min_iou_choices: Tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    min_object_covered: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random IoU-preserving crop (SSD-style).

    Samples a random crop such that the IoU between the crop region and
    at least one ground truth box exceeds a randomly chosen threshold.
    Boxes with centers outside the crop are removed.

    Args:
        image: (H, W, 3) uint8 array
        boxes: (N, 4) float32 array in [x1, y1, x2, y2]
        labels: (N,) int64 array
        min_iou_choices: Tuple of minimum IoU thresholds to sample from
        min_object_covered: Minimum fraction of object that must be in crop

    Returns:
        Cropped (image, boxes, labels). If no valid crop found, returns originals.
    """
    if len(boxes) == 0:
        return image, boxes, labels

    h, w = image.shape[:2]
    original = (image, boxes, labels)

    # Randomly select minimum IoU threshold
    min_iou = np.random.choice(min_iou_choices)

    # If min_iou is 1.0, return the original (no crop)
    if min_iou >= 1.0:
        return original

    # If min_iou is 0.0 (or close), allow any crop
    max_attempts = 50
    for _ in range(max_attempts):
        # Random crop dimensions
        crop_w = np.random.uniform(0.3 * w, w)
        crop_h = np.random.uniform(0.3 * h, h)

        # Maintain reasonable aspect ratio
        aspect_ratio = crop_h / max(crop_w, 1e-6)
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue

        # Random crop position
        left = np.random.uniform(0, w - crop_w)
        top = np.random.uniform(0, h - crop_h)

        crop_rect = np.array(
            [[left, top, left + crop_w, top + crop_h]], dtype=np.float32
        )

        # Compute IoU between crop and all boxes
        iou = jaccard_numpy(crop_rect, boxes)  # (1, N)

        # Check if at least one box has sufficient overlap
        if min_iou > 0.0 and iou.max() < min_iou:
            continue

        # Crop the image
        left_i = int(left)
        top_i = int(top)
        right_i = int(left + crop_w)
        bottom_i = int(top + crop_h)

        cropped_image = image[top_i:bottom_i, left_i:right_i, :].copy()

        # Keep boxes whose centers are inside the crop
        centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0
        mask = (
            (centers[:, 0] > left)
            & (centers[:, 0] < left + crop_w)
            & (centers[:, 1] > top)
            & (centers[:, 1] < top + crop_h)
        )

        if not mask.any():
            continue

        # Filter and clip boxes
        new_boxes = boxes[mask].copy()
        new_labels = labels[mask].copy()

        # Clip to crop boundaries
        new_boxes[:, 0] = np.clip(new_boxes[:, 0] - left, 0, crop_w)
        new_boxes[:, 1] = np.clip(new_boxes[:, 1] - top, 0, crop_h)
        new_boxes[:, 2] = np.clip(new_boxes[:, 2] - left, 0, crop_w)
        new_boxes[:, 3] = np.clip(new_boxes[:, 3] - top, 0, crop_h)

        # Remove degenerate boxes
        valid = (new_boxes[:, 2] > new_boxes[:, 0] + 1) & (
            new_boxes[:, 3] > new_boxes[:, 1] + 1
        )
        if not valid.any():
            continue

        new_boxes = new_boxes[valid]
        new_labels = new_labels[valid]

        return cropped_image, new_boxes, new_labels

    # If no valid crop found after max_attempts, return original
    return original


def resize(
    image: np.ndarray,
    boxes: np.ndarray,
    size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize image and scale boxes accordingly.

    Args:
        image: (H, W, 3) uint8 array
        boxes: (N, 4) float32 array in [x1, y1, x2, y2]
        size: Target size (both height and width)

    Returns:
        Resized (image, boxes)
    """
    h, w = image.shape[:2]

    if len(boxes) > 0:
        scale_x = size / w
        scale_y = size / h
        boxes = boxes.copy()
        boxes[:, 0] *= scale_x
        boxes[:, 2] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 3] *= scale_y

    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return image, boxes


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize image with ImageNet mean and std.

    Args:
        image: (H, W, 3) uint8 array (0-255)

    Returns:
        Normalized float32 array
    """
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image


# ---------------------------------------------------------------------------
# Composed augmentation classes
# ---------------------------------------------------------------------------


class TrainAugmentation:
    """
    Full SSD-style training augmentation pipeline.

    Applied in order:
        1. Photometric distortion (brightness, contrast, saturation, hue)
        2. Random expand (zoom out)
        3. Random IoU-preserving crop
        4. Random horizontal flip
        5. Resize to target size
        6. ImageNet normalization

    Args:
        size: Target input size (default 550)
    """

    def __init__(self, size: int = 550):
        self.size = size

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply training augmentations.

        Args:
            image: (H, W, 3) uint8 RGB array
            boxes: (N, 4) float32 array in [x1, y1, x2, y2]
            labels: (N,) int64 array

        Returns:
            Augmented (image, boxes, labels)
        """
        # Ensure we are working with copies
        image = image.copy()
        if len(boxes) > 0:
            boxes = boxes.copy()
            labels = labels.copy()

        # 1. Photometric distortion
        image = photometric_distortion(image)

        # 2. Random expand (zoom out)
        image, boxes, labels = random_expand(image, boxes, labels)

        # 3. Random crop
        image, boxes, labels = random_crop(image, boxes, labels)

        # 4. Random horizontal flip
        image, boxes, labels = random_horizontal_flip(image, boxes, labels)

        # 5. Resize to target size
        image, boxes = resize(image, boxes, self.size)

        # 6. Normalize
        image = normalize(image)

        return image, boxes, labels


class ValAugmentation:
    """
    Validation/test augmentation pipeline (no data augmentation).

    Applied in order:
        1. Resize to target size
        2. ImageNet normalization

    Args:
        size: Target input size (default 550)
    """

    def __init__(self, size: int = 550):
        self.size = size

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply validation augmentations (resize + normalize only).

        Args:
            image: (H, W, 3) uint8 RGB array
            boxes: (N, 4) float32 array in [x1, y1, x2, y2]
            labels: (N,) int64 array

        Returns:
            Transformed (image, boxes, labels)
        """
        image = image.copy()
        if len(boxes) > 0:
            boxes = boxes.copy()
            labels = labels.copy()

        # 1. Resize
        image, boxes = resize(image, boxes, self.size)

        # 2. Normalize
        image = normalize(image)

        return image, boxes, labels


if __name__ == "__main__":
    """Test the augmentation pipeline."""
    import time

    print("=== Augmentation Pipeline Test ===\n")

    # Create a synthetic test image
    h, w = 480, 640
    image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    # Create test boxes
    boxes = np.array(
        [
            [50, 50, 200, 150],
            [300, 100, 500, 300],
            [100, 200, 250, 400],
            [400, 50, 600, 200],
        ],
        dtype=np.float32,
    )
    labels = np.ones(len(boxes), dtype=np.int64)

    print(f"Input image: {image.shape}, dtype={image.dtype}")
    print(f"Input boxes: {boxes.shape}")
    print(f"Input labels: {labels.shape}")

    # Test TrainAugmentation
    print("\n--- TrainAugmentation ---")
    train_aug = TrainAugmentation(size=550)

    start = time.time()
    n_runs = 100
    for _ in range(n_runs):
        aug_image, aug_boxes, aug_labels = train_aug(image, boxes, labels)
    elapsed = time.time() - start

    print(f"Output image: {aug_image.shape}, dtype={aug_image.dtype}")
    print(f"Output boxes: {aug_boxes.shape}")
    print(f"Output labels: {aug_labels.shape}")
    print(f"Image range: [{aug_image.min():.3f}, {aug_image.max():.3f}]")
    print(f"Speed: {n_runs / elapsed:.1f} augmentations/sec ({elapsed / n_runs * 1000:.1f} ms each)")

    # Verify boxes are valid
    if len(aug_boxes) > 0:
        assert np.all(aug_boxes[:, 2] > aug_boxes[:, 0]), "Invalid box: x2 <= x1"
        assert np.all(aug_boxes[:, 3] > aug_boxes[:, 1]), "Invalid box: y2 <= y1"
        assert np.all(aug_boxes >= 0), "Negative box coordinates"
        assert np.all(aug_boxes[:, [0, 2]] <= 550), "Box x out of bounds"
        assert np.all(aug_boxes[:, [1, 3]] <= 550), "Box y out of bounds"
        print("Box validity checks passed")

    # Test ValAugmentation
    print("\n--- ValAugmentation ---")
    val_aug = ValAugmentation(size=550)
    val_image, val_boxes, val_labels = val_aug(image, boxes, labels)

    print(f"Output image: {val_image.shape}, dtype={val_image.dtype}")
    print(f"Output boxes: {val_boxes.shape}")
    print(f"Output labels: {val_labels.shape}")
    print(f"All {len(val_boxes)} boxes preserved (no aug)")

    # Verify val boxes are all preserved (no dropout)
    assert len(val_boxes) == len(boxes), "Val augmentation should preserve all boxes"
    print("Val augmentation preserves all boxes: OK")

    # Test individual augmentations
    print("\n--- Individual Augmentation Tests ---")

    # Test flip
    flipped_img, flipped_boxes, _ = random_horizontal_flip(
        image.copy(), boxes.copy(), labels.copy(), p=1.0
    )
    print(f"Flip: output shape {flipped_img.shape}")

    # Test expand
    expanded_img, expanded_boxes, _ = random_expand(
        image.copy(), boxes.copy(), labels.copy(), p=1.0
    )
    print(f"Expand: {image.shape} -> {expanded_img.shape}")

    # Test crop
    cropped_img, cropped_boxes, cropped_labels = random_crop(
        image.copy(), boxes.copy(), labels.copy()
    )
    print(f"Crop: {image.shape} -> {cropped_img.shape}, "
          f"{len(boxes)} -> {len(cropped_boxes)} boxes")

    # Test IoU computation
    iou = jaccard_numpy(boxes[:2], boxes[2:])
    print(f"\nIoU matrix shape: {iou.shape}")

    print("\n=== All Tests Passed ===")
