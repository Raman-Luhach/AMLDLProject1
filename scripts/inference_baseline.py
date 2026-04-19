#!/usr/bin/env python3
"""Baseline HOG+SVM inference script called by the Next.js API route.

Usage:
    python scripts/inference_baseline.py /path/to/image.jpg

Outputs JSON to stdout with detection results.
Logs progress to stderr.
"""

import json
import os
import sys
import time

def log(msg):
    print(f"[baseline:python] {msg}", file=sys.stderr, flush=True)

log("Script started")
t_start = time.time()

import cv2
import numpy as np
import joblib
from skimage.feature import hog

log(f"Imports done ({time.time() - t_start:.1f}s)")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

MODEL_PATH = os.path.join(project_root, "results", "baseline", "hog_svm_model.pkl")


def load_model():
    """Load the saved HOG+SVM model."""
    if not os.path.exists(MODEL_PATH):
        log(f"ERROR: Model not found at {MODEL_PATH}")
        log("Run 'python scripts/run_baseline.py' first to train and save the model.")
        sys.exit(1)

    log(f"Loading model from {MODEL_PATH}")
    data = joblib.load(MODEL_PATH)
    log(f"Model loaded ({time.time() - t_start:.1f}s)")
    return data


def sliding_window_detect(image, model_data, score_threshold=0.5, nms_threshold=0.3):
    """Run multi-scale sliding window detection with HOG+SVM."""
    svm = model_data["svm"]
    scaler = model_data["scaler"]
    window_size = tuple(model_data["window_size"])
    cell_size = tuple(model_data["cell_size"])
    block_size = tuple(model_data["block_size"])
    nbins = model_data["nbins"]

    win_w, win_h = window_size
    cells_per_block = (
        block_size[0] // cell_size[0],
        block_size[1] // cell_size[1],
    )

    scales = (1.0, 0.75, 0.5, 0.35)
    step_size = 16
    h_orig, w_orig = image.shape[:2]

    detections = []

    for scale in scales:
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        h_r, w_r = resized.shape[:2]

        if h_r < win_h or w_r < win_w:
            continue

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

        for y in range(0, h_r - win_h + 1, step_size):
            for x in range(0, w_r - win_w + 1, step_size):
                patch = gray[y:y + win_h, x:x + win_w]
                feat = hog(
                    patch,
                    orientations=nbins,
                    pixels_per_cell=cell_size,
                    cells_per_block=cells_per_block,
                    block_norm="L2-Hys",
                    visualize=False,
                    feature_vector=True,
                )
                feat_scaled = scaler.transform(feat.reshape(1, -1))
                score = svm.decision_function(feat_scaled)[0]

                if score > score_threshold:
                    x1 = int(x / scale)
                    y1 = int(y / scale)
                    x2 = int((x + win_w) / scale)
                    y2 = int((y + win_h) / scale)
                    detections.append([x1, y1, x2, y2, float(score)])

    if len(detections) == 0:
        return [], []

    detections = np.array(detections)
    boxes = detections[:, :4]
    scores = detections[:, 4]

    # Non-maximum suppression
    keep = nms(boxes, scores, nms_threshold)
    return boxes[keep].tolist(), scores[keep].tolist()


def nms(boxes, scores, iou_threshold=0.3):
    """Greedy non-maximum suppression."""
    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = boxes[:, 2].astype(float)
    y2 = boxes[:, 3].astype(float)
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


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

    # Load model
    model_data = load_model()

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(json.dumps({"error": f"Failed to read image: {image_path}"}))
        sys.exit(1)

    orig_h, orig_w = img.shape[:2]
    log(f"Image loaded: {orig_w}x{orig_h}")

    # Resize for sliding window (keep it manageable)
    scale = 1.0
    if max(orig_h, orig_w) > 600:
        scale = 600.0 / max(orig_h, orig_w)
        img_resized = cv2.resize(img, None, fx=scale, fy=scale)
    else:
        img_resized = img

    log("Running sliding window detection...")
    t0 = time.time()
    boxes, scores = sliding_window_detect(img_resized, model_data, score_threshold=0.3, nms_threshold=0.3)
    inference_ms = (time.time() - t0) * 1000
    log(f"Detection done: {inference_ms:.0f}ms, {len(boxes)} detections")

    # Scale boxes back to original image dimensions
    results = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        results.append({
            "box": [
                float(x1 / scale),
                float(y1 / scale),
                float(x2 / scale),
                float(y2 / scale),
            ],
            "score": float(scores[i]),
            "label": 0,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    log(f"Total detections: {len(results)}")

    output = {
        "detections": results,
        "num_detections": len(results),
        "inference_time_ms": round(inference_ms, 1),
        "image_width": orig_w,
        "image_height": orig_h,
        "model": "HOG+SVM",
    }

    log(f"Total time: {time.time() - t_start:.1f}s")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
