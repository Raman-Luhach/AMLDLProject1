#!/usr/bin/env python3
"""Inference script called by the Next.js API route.

Usage:
    python scripts/inference_api.py /path/to/image.jpg

Outputs JSON to stdout with detection results.
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.yolact import YOLACT, DEFAULT_CONFIG
from src.data.augmentations import IMAGENET_MEAN, IMAGENET_STD


def load_model(device):
    """Load the trained YOLACT model."""
    config = {**DEFAULT_CONFIG, "pretrained_backbone": True}
    model = YOLACT(config=config)

    # Try to find the best checkpoint
    checkpoint_paths = [
        os.path.join(project_root, "results/training/checkpoints/best_model.pth"),
        os.path.join(project_root, "results/training/checkpoints/final_model.pth"),
    ]

    for ckpt_path in checkpoint_paths:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            break

    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """Load and preprocess image for YOLACT inference."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # Resize to 550x550
    resized = cv2.resize(img_rgb, (550, 550), interpolation=cv2.INTER_LINEAR)

    # Normalize
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

    # Convert to tensor (C, H, W)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
    tensor = tensor.unsqueeze(0)  # Add batch dim

    return tensor, orig_w, orig_h


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    model = load_model(device)

    # Preprocess
    tensor, orig_w, orig_h = preprocess_image(image_path)
    tensor = tensor.to(device)

    # Run inference
    t0 = time.time()
    with torch.no_grad():
        detections = model(tensor)
    inference_ms = (time.time() - t0) * 1000

    # Process detections
    det = detections[0]
    boxes = det["boxes"].cpu().numpy()  # [x1, y1, x2, y2] normalized [0,1]
    scores = det["scores"].cpu().numpy()
    labels = det["labels"].cpu().numpy()

    # Scale boxes to original image dimensions
    results = []
    for i in range(len(boxes)):
        if scores[i] < 0.05:
            continue
        x1, y1, x2, y2 = boxes[i]
        results.append({
            "box": [
                float(x1 * orig_w),
                float(y1 * orig_h),
                float(x2 * orig_w),
                float(y2 * orig_h),
            ],
            "score": float(scores[i]),
            "label": int(labels[i]),
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    output = {
        "detections": results,
        "num_detections": len(results),
        "inference_time_ms": round(inference_ms, 1),
        "image_width": orig_w,
        "image_height": orig_h,
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
