"""
SKU-110K Dataset Loader for Dense Object Detection.

SKU-110K contains 11,762 images of densely packed retail shelf scenes with
~1.73M bounding box annotations (single class: "object", avg 147 objects/image).

CSV format: image_name, x1, y1, x2, y2, class, image_width, image_height
Official split: 8,233 train / 588 val / 2,941 test

This module provides:
    - SKU110KDataset: PyTorch Dataset class with pseudo-mask generation
    - get_dataloaders: Factory function returning train/val DataLoaders
    - convert_to_coco_format: Export annotations to COCO JSON format
    - sku110k_collate_fn: Custom collate for variable-length targets
"""

import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


class SKU110KDataset(Dataset):
    """
    PyTorch Dataset for SKU-110K dense object detection.

    Parses CSV annotations, generates pseudo-masks from bounding boxes,
    and supports train/val/test splits with optional subsetting.

    Args:
        data_dir: Root directory containing SKU110K_fixed/ folder.
        split: One of 'train', 'val', or 'test'.
        transform: Callable that takes (image, boxes, labels) and returns
                   transformed (image, boxes, labels).
        max_images: If set, limit dataset to first N images (for fast iteration).
        input_size: Target input resolution (default 550x550 for SSD).
    """

    # Single class dataset: 0 = background, 1 = object
    NUM_CLASSES = 2
    CLASS_NAMES = ["background", "object"]

    # ImageNet normalization constants
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_dir: str = "data",
        split: str = "train",
        transform: Optional[Callable] = None,
        max_images: Optional[int] = None,
        input_size: int = 550,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.max_images = max_images
        self.input_size = input_size

        # Paths
        self.image_dir = self.data_dir / "SKU110K_fixed" / "images"
        self.annotation_dir = self.data_dir / "SKU110K_fixed" / "annotations"
        self.csv_path = self.annotation_dir / f"annotations_{split}.csv"

        # Parse annotations
        self.image_names: List[str] = []
        self.annotations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.image_sizes: Dict[str, Tuple[int, int]] = {}

        self._parse_csv()

        # Apply max_images limit
        if self.max_images is not None and self.max_images < len(self.image_names):
            self.image_names = self.image_names[: self.max_images]
            logger.info(f"Subset: using {self.max_images} images out of total")

        logger.info(
            f"SKU110K {split}: {len(self.image_names)} images, "
            f"{sum(len(self.annotations[n]) for n in self.image_names)} boxes"
        )

    def _parse_csv(self) -> None:
        """Parse SKU-110K CSV annotation file with data cleaning."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {self.csv_path}\n"
                f"Run scripts/download_data.sh to download the dataset."
            )

        seen_images = set()
        skipped_boxes = 0
        total_boxes = 0

        with open(self.csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 8:
                    continue

                image_name = row[0].strip()
                try:
                    x1 = float(row[1])
                    y1 = float(row[2])
                    x2 = float(row[3])
                    y2 = float(row[4])
                    # row[5] is class (always "object")
                    img_w = int(float(row[6]))
                    img_h = int(float(row[7]))
                except (ValueError, IndexError):
                    skipped_boxes += 1
                    continue

                total_boxes += 1

                # Clip coordinates to image bounds
                x1 = max(0.0, min(x1, img_w))
                y1 = max(0.0, min(y1, img_h))
                x2 = max(0.0, min(x2, img_w))
                y2 = max(0.0, min(y2, img_h))

                # Ensure x1 < x2 and y1 < y2
                if x1 >= x2 or y1 >= y2:
                    skipped_boxes += 1
                    continue

                # Skip zero-area or near-zero-area boxes
                area = (x2 - x1) * (y2 - y1)
                if area < 1.0:
                    skipped_boxes += 1
                    continue

                # Track unique images in order
                if image_name not in seen_images:
                    seen_images.add(image_name)
                    self.image_names.append(image_name)
                    self.image_sizes[image_name] = (img_w, img_h)

                self.annotations[image_name].append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "class_id": 1,  # single class: "object"
                    }
                )

        # Remove duplicate boxes per image
        for img_name in self.image_names:
            boxes = self.annotations[img_name]
            unique_boxes = []
            seen_bboxes = set()
            for box in boxes:
                key = tuple(box["bbox"])
                if key not in seen_bboxes:
                    seen_bboxes.add(key)
                    unique_boxes.append(box)
            dup_count = len(boxes) - len(unique_boxes)
            if dup_count > 0:
                skipped_boxes += dup_count
            self.annotations[img_name] = unique_boxes

        if skipped_boxes > 0:
            logger.info(
                f"Data cleaning: skipped {skipped_boxes}/{total_boxes} boxes "
                f"(zero-area, out-of-bounds, or duplicates)"
            )

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            image: Tensor of shape (3, H, W) normalized with ImageNet stats.
            target: Dict with keys:
                - boxes: (N, 4) tensor in [x1, y1, x2, y2] format (absolute coords)
                - labels: (N,) tensor of class ids (all 1 for SKU-110K)
                - masks: (N, input_size, input_size) pseudo-masks from bounding boxes
                - image_id: scalar tensor
                - area: (N,) tensor of box areas
                - iscrowd: (N,) tensor of zeros
                - orig_size: (2,) tensor [height, width] of original image
        """
        image_name = self.image_names[idx]
        anns = self.annotations[image_name]

        # Load image
        image_path = self.image_dir / image_name
        image = self._load_image(image_path)
        if image is None:
            # Return a blank sample for corrupt images
            logger.warning(f"Corrupt image: {image_path}, returning blank sample")
            return self._blank_sample(idx)

        h, w = image.shape[:2]

        # Extract boxes and labels
        boxes = np.array([ann["bbox"] for ann in anns], dtype=np.float32)
        labels = np.array([ann["class_id"] for ann in anns], dtype=np.int64)

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        # Apply augmentations (handles image + boxes jointly)
        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)
        else:
            # Default: resize and normalize
            image, boxes = self._default_transform(image, boxes)

        # Convert image to tensor: (H, W, C) -> (C, H, W)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Generate pseudo-masks from bounding boxes
        masks = self._generate_pseudo_masks(boxes, self.input_size, self.input_size)

        # Compute area
        if len(boxes) > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor(idx, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            "orig_size": torch.tensor([h, w], dtype=torch.int64),
        }

        return image, target

    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        """Load image from disk, returning None if corrupt."""
        try:
            image = cv2.imread(str(path))
            if image is None:
                return None
            # Convert BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            return None

    def _default_transform(
        self, image: np.ndarray, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Default transform: resize + normalize. Used when no transform is provided."""
        h, w = image.shape[:2]
        target_size = self.input_size

        # Scale boxes to target size
        if len(boxes) > 0:
            scale_x = target_size / w
            scale_y = target_size / h
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y

        # Resize image
        image = cv2.resize(image, (target_size, target_size))

        # Normalize to [0, 1] then apply ImageNet normalization
        image = image.astype(np.float32) / 255.0
        mean = np.array(self.MEAN, dtype=np.float32)
        std = np.array(self.STD, dtype=np.float32)
        image = (image - mean) / std

        return image, boxes

    def _generate_pseudo_masks(
        self, boxes: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        Generate pseudo segmentation masks from bounding boxes.

        Since SKU-110K only has bounding box annotations, we create
        filled rectangle masks as pseudo ground truth for any
        instance segmentation components.

        Args:
            boxes: (N, 4) tensor in [x1, y1, x2, y2] format
            height: Mask height
            width: Mask width

        Returns:
            masks: (N, height, width) binary uint8 tensor
        """
        n = len(boxes)
        if n == 0:
            return torch.zeros((0, height, width), dtype=torch.uint8)

        masks = torch.zeros((n, height, width), dtype=torch.uint8)
        for i in range(n):
            x1 = int(max(0, boxes[i, 0].item()))
            y1 = int(max(0, boxes[i, 1].item()))
            x2 = int(min(width, boxes[i, 2].item()))
            y2 = int(min(height, boxes[i, 3].item()))
            if x2 > x1 and y2 > y1:
                masks[i, y1:y2, x1:x2] = 1

        return masks

    def _blank_sample(
        self, idx: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return a blank sample for corrupt images."""
        image = torch.zeros((3, self.input_size, self.input_size), dtype=torch.float32)
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros(
                (0, self.input_size, self.input_size), dtype=torch.uint8
            ),
            "image_id": torch.tensor(idx, dtype=torch.int64),
            "area": torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64),
            "orig_size": torch.tensor(
                [self.input_size, self.input_size], dtype=torch.int64
            ),
        }
        return image, target

    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata for an image without loading it."""
        name = self.image_names[idx]
        w, h = self.image_sizes[name]
        return {
            "image_name": name,
            "width": w,
            "height": h,
            "num_objects": len(self.annotations[name]),
        }


def sku110k_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Custom collate function for SKU-110K DataLoader.

    Images are stacked into a single tensor, while targets remain
    as a list of dicts (since each image has a variable number of objects).

    Args:
        batch: List of (image, target) tuples from Dataset.__getitem__

    Returns:
        images: (B, 3, H, W) tensor
        targets: List of B target dicts
    """
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)

    images = torch.stack(images, dim=0)
    return images, targets


def get_dataloaders(
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from config.

    Expected config keys:
        data_dir (str): Root data directory (default: "data")
        batch_size (int): Batch size (default: 8)
        num_workers (int): DataLoader workers (default: 4)
        max_images (int or None): Limit dataset size (default: None)
        input_size (int): Input resolution (default: 550)
        pin_memory (bool): Pin memory for faster transfers (default: True)

    Args:
        config: Configuration dictionary.

    Returns:
        train_loader: DataLoader for training split
        val_loader: DataLoader for validation split
    """
    # Import augmentations here to avoid circular imports
    from src.data.augmentations import TrainAugmentation, ValAugmentation

    data_dir = config.get("data_dir", "data")
    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 4)
    max_images = config.get("max_images", None)
    input_size = config.get("input_size", 550)
    pin_memory = config.get("pin_memory", True)

    # Determine device-compatible settings
    # On macOS with MPS, pin_memory should be False
    if not torch.cuda.is_available():
        pin_memory = False

    # Create augmentation pipelines
    train_transform = TrainAugmentation(size=input_size)
    val_transform = ValAugmentation(size=input_size)

    # Create datasets
    train_dataset = SKU110KDataset(
        data_dir=data_dir,
        split="train",
        transform=train_transform,
        max_images=max_images,
        input_size=input_size,
    )

    val_dataset = SKU110KDataset(
        data_dir=data_dir,
        split="val",
        transform=val_transform,
        max_images=max_images,
        input_size=input_size,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=sku110k_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sku110k_collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


def convert_to_coco_format(
    dataset: SKU110KDataset, output_path: str
) -> Dict[str, Any]:
    """
    Convert SKU-110K annotations to COCO JSON format for evaluation.

    This is required for using pycocotools (mAP evaluation).

    Args:
        dataset: SKU110KDataset instance.
        output_path: Path to save the COCO JSON file.

    Returns:
        coco_dict: The COCO-format annotation dictionary.
    """
    coco_dict: Dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "object", "supercategory": "product"}
        ],
    }

    ann_id = 1

    for img_idx in range(len(dataset)):
        image_name = dataset.image_names[img_idx]
        img_w, img_h = dataset.image_sizes[image_name]

        # Image entry
        coco_dict["images"].append(
            {
                "id": img_idx,
                "file_name": image_name,
                "width": img_w,
                "height": img_h,
            }
        )

        # Annotation entries
        for ann in dataset.annotations[image_name]:
            x1, y1, x2, y2 = ann["bbox"]
            w = x2 - x1
            h = y2 - y1
            area = w * h

            coco_dict["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": ann["class_id"],
                    "bbox": [x1, y1, w, h],  # COCO format: [x, y, width, height]
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [
                        [x1, y1, x2, y1, x2, y2, x1, y2]
                    ],  # Pseudo-segmentation from box
                }
            )
            ann_id += 1

    # Save to file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_dict, f)

    logger.info(
        f"COCO annotations saved: {len(coco_dict['images'])} images, "
        f"{len(coco_dict['annotations'])} annotations -> {output_path}"
    )

    return coco_dict


if __name__ == "__main__":
    """Test the dataset loader."""
    logging.basicConfig(level=logging.INFO)

    print("=== SKU-110K Dataset Test ===\n")

    # Check if data exists
    data_dir = "data"
    csv_path = os.path.join(
        data_dir, "SKU110K_fixed", "annotations", "annotations_train.csv"
    )

    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        print("Run: bash scripts/download_data.sh")
        print("\nTesting with mock data instead...")

        # Create mock data for testing
        mock_dir = "/tmp/sku110k_test"
        os.makedirs(f"{mock_dir}/SKU110K_fixed/annotations", exist_ok=True)
        os.makedirs(f"{mock_dir}/SKU110K_fixed/images", exist_ok=True)

        # Create a mock CSV
        with open(
            f"{mock_dir}/SKU110K_fixed/annotations/annotations_train.csv", "w"
        ) as f:
            writer = csv.writer(f)
            for i in range(5):
                img_name = f"test_{i}.jpg"
                # Create a small test image
                test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                cv2.imwrite(f"{mock_dir}/SKU110K_fixed/images/{img_name}", test_img)
                # Write some annotations
                for j in range(3):
                    x1 = 50 + j * 100
                    y1 = 50 + j * 80
                    x2 = x1 + 80
                    y2 = y1 + 60
                    writer.writerow(
                        [img_name, x1, y1, x2, y2, "object", 640, 480]
                    )

        data_dir = mock_dir

    # Test dataset
    dataset = SKU110KDataset(data_dir=data_dir, split="train", max_images=3)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        image, target = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Boxes shape: {target['boxes'].shape}")
        print(f"Labels shape: {target['labels'].shape}")
        print(f"Masks shape: {target['masks'].shape}")
        print(f"Num objects: {len(target['boxes'])}")

        # Test collate function
        batch = [dataset[i] for i in range(min(2, len(dataset)))]
        images, targets = sku110k_collate_fn(batch)
        print(f"\nBatch images shape: {images.shape}")
        print(f"Batch targets count: {len(targets)}")

        # Test COCO conversion
        coco_path = "/tmp/sku110k_coco_test.json"
        coco_dict = convert_to_coco_format(dataset, coco_path)
        print(f"\nCOCO format: {len(coco_dict['images'])} images, "
              f"{len(coco_dict['annotations'])} annotations")

    print("\n=== Test Complete ===")
