#!/usr/bin/env python3
"""Hybrid detector inference script called by the Next.js API route.

Usage:
    python scripts/inference_hybrid.py /path/to/image.jpg

Outputs JSON to stdout with detection results.
Logs progress to stderr.
"""

import json
import os
import sys
import time


def log(msg):
    print(f"[hybrid:python] {msg}", file=sys.stderr, flush=True)


log("Script started")
t_start = time.time()

import cv2
import numpy as np
import torch

log(f"Imports done ({time.time() - t_start:.1f}s)")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.hybrid import HybridDetector
from src.data.augmentations import IMAGENET_MEAN, IMAGENET_STD
from src.utils.helpers import load_config

log(f"Project modules imported ({time.time() - t_start:.1f}s)")


def load_model(device):
    """Load the trained hybrid detector."""
    log("Loading config...")
    config_path = os.path.join(project_root, "configs", "hybrid.yaml")
    config = load_config(config_path)

    log("Building Hybrid Detector...")
    model = HybridDetector(config)
    log(f"Model built ({time.time() - t_start:.1f}s)")

    # Load hybrid checkpoint
    checkpoint_paths = [
        os.path.join(project_root, "results/hybrid/checkpoints/hybrid_best.pth"),
        os.path.join(project_root, "results/hybrid/checkpoints/hybrid_final.pth"),
    ]

    loaded = False
    for ckpt_path in checkpoint_paths:
        if os.path.exists(ckpt_path):
            log(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            log(f"Checkpoint loaded ({time.time() - t_start:.1f}s)")
            loaded = True
            break

    if not loaded:
        # Fall back to base YOLACT checkpoint
        base_path = os.path.join(project_root, "results/training/checkpoints/best_model.pth")
        if os.path.exists(base_path):
            log(f"Loading base YOLACT checkpoint: {base_path}")
            ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.yolact.load_state_dict(state_dict, strict=False)
            log(f"Base checkpoint loaded ({time.time() - t_start:.1f}s)")
        else:
            log("WARNING: No checkpoint found, using random weights!")

    log(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()
    log(f"Model ready on {device} ({time.time() - t_start:.1f}s)")
    return model


def preprocess_image(image_path):
    """Load and preprocess image for inference."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    log(f"Image loaded: {orig_w}x{orig_h}")

    resized = cv2.resize(img_rgb, (550, 550), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
    tensor = tensor.unsqueeze(0)

    return tensor, orig_w, orig_h


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    image_path = sys.argv[1]
    log(f"Image path: {image_path}")
    if not os.path.exists(image_path):
        log(f"ERROR: Image not found: {image_path}")
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"Device: {device}")

    model = load_model(device)

    tensor, orig_w, orig_h = preprocess_image(image_path)
    tensor = tensor.to(device)

    log("Running hybrid inference...")
    t0 = time.time()
    with torch.no_grad():
        detections = model(tensor)
    inference_ms = (time.time() - t0) * 1000
    log(f"Inference done: {inference_ms:.0f}ms")

    det = detections[0]
    boxes = det["boxes"].cpu().numpy()
    scores = det["scores"].cpu().numpy()
    labels = det["labels"].cpu().numpy() if "labels" in det else np.zeros(len(scores), dtype=int)

    results = []
    for i in range(len(boxes)):
        if scores[i] < 0.25:
            continue
        x1, y1, x2, y2 = boxes[i]
        results.append(
            {
                "box": [
                    float(x1 * orig_w),
                    float(y1 * orig_h),
                    float(x2 * orig_w),
                    float(y2 * orig_h),
                ],
                "score": float(scores[i]),
                "label": int(labels[i]),
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    log(f"Total detections (score > 0.05): {len(results)}")

    output = {
        "detections": results,
        "num_detections": len(results),
        "inference_time_ms": round(inference_ms, 1),
        "image_width": orig_w,
        "image_height": orig_h,
        "model": "Hybrid (YOLACT + Spatial)",
    }

    log(f"Total time: {time.time() - t_start:.1f}s")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
