#!/usr/bin/env python3
"""Advanced evaluation: Grad-CAM, ablation studies, robustness testing, and error analysis.

Usage:
    python scripts/advanced_evaluation.py
    python scripts/advanced_evaluation.py --checkpoint results/training/checkpoints/best_model.pth
    python scripts/advanced_evaluation.py --max-images 100

Outputs (saved to results/eval/):
    - gradcam/gradcam_sample_{i}.png  : Individual Grad-CAM heatmaps
    - gradcam_grid.png                : 2x4 grid of Grad-CAM visualizations
    - ablation_nms.json               : Soft-NMS vs Hard-NMS comparison
    - robustness_analysis.png         : AP@0.50 under corruptions
    - robustness_metrics.json         : Detailed robustness metrics
    - error_analysis.json             : TP/FP/FN breakdown by density
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.yolact import YOLACT, DEFAULT_CONFIG
from src.evaluation.evaluator import COCOEvaluator
from src.evaluation.metrics import compute_detection_metrics, match_predictions_single_image
from src.utils.helpers import get_device, set_seed
from src.utils.soft_nms import soft_nms, hard_nms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced evaluation suite.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="results/eval")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_checkpoint() -> Optional[str]:
    search_dirs = [
        str(PROJECT_ROOT / "results" / "training" / "checkpoints"),
        str(PROJECT_ROOT / "results" / "training"),
    ]
    preferred = ["best_model.pth", "final_model.pth"]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for name in preferred:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path
        for f in sorted(os.listdir(d)):
            if f.endswith(".pth"):
                return os.path.join(d, f)
    return None


def load_model(checkpoint_path: Optional[str], device: torch.device) -> YOLACT:
    config = {**DEFAULT_CONFIG, "pretrained_backbone": True}
    model = YOLACT(config=config)
    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_val_data(data_dir: str, max_images: Optional[int] = None):
    from src.data.dataset import SKU110KDataset

    dataset = SKU110KDataset(
        data_dir=data_dir,
        split="val",
        max_images=max_images,
        input_size=550,
    )
    images, targets = [], []
    for i in range(len(dataset)):
        img, tgt = dataset[i]
        images.append(img)
        targets.append(tgt)
    return images, targets


@torch.no_grad()
def run_inference(model, images, device, batch_size=4):
    all_dets = []
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch = torch.stack(images[start:end]).to(device)
        try:
            dets = model(batch)
        except Exception as e:
            logger.warning(f"Inference failed: {e}")
            for _ in range(end - start):
                all_dets.append(
                    {
                        "boxes": torch.zeros(0, 4),
                        "scores": torch.zeros(0),
                        "labels": torch.zeros(0, dtype=torch.long),
                        "masks": torch.zeros(0, 138, 138),
                    }
                )
            continue
        for det in dets:
            all_dets.append({k: v.detach().cpu() for k, v in det.items()})
    return all_dets


def to_eval_format(detections, targets, input_size=550):
    preds_np, gts_np = [], []
    for det, tgt in zip(detections, targets):
        boxes = det["boxes"].numpy().astype(np.float64)
        scores = det["scores"].numpy().astype(np.float64)
        labels = det["labels"].numpy().astype(np.int64)
        if len(boxes) > 0 and boxes.max() <= 1.0:
            boxes = boxes * input_size
        preds_np.append({"boxes": boxes, "scores": scores, "labels": labels})

        gt_boxes = tgt["boxes"].numpy().astype(np.float64)
        gt_labels = tgt["labels"].numpy().astype(np.int64)
        if len(gt_boxes) > 0 and gt_boxes.max() <= 1.0:
            gt_boxes = gt_boxes * input_size
        gts_np.append({"boxes": gt_boxes, "labels": gt_labels})
    return preds_np, gts_np


# ============================================================
# GRAD-CAM
# ============================================================


def run_gradcam(model, images, device, output_dir, num_samples=8):
    logger.info("Running Grad-CAM analysis...")
    gradcam_dir = os.path.join(output_dir, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    # Find the target layer: last conv block in MobileNetV3 (stage3 = features[13:17])
    target_layer = model.backbone.stage3[-1]

    # ImageNet denormalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    sample_indices = list(range(min(num_samples, len(images))))
    gradcam_images = []

    for idx in sample_indices:
        img_tensor = images[idx].unsqueeze(0).to(device).requires_grad_(True)

        activations = {}
        gradients = {}

        def fwd_hook(module, input, output):
            activations["value"] = output.detach()

        def bwd_hook(module, grad_input, grad_output):
            gradients["value"] = grad_output[0].detach()

        fwd_handle = target_layer.register_forward_hook(fwd_hook)
        bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

        # Forward pass in train mode to get raw predictions
        model.train()
        try:
            class_preds, box_preds, mask_coeffs, prototypes, anchors = model(img_tensor)
            # Use max foreground confidence as the target
            fg_scores = F.softmax(class_preds[0], dim=-1)[:, 1:]
            target_score = fg_scores.max()
            model.zero_grad()
            target_score.backward()

            acts = activations["value"][0]  # (C, H, W)
            grads = gradients["value"][0]  # (C, H, W)

            # Global average pool of gradients
            weights = grads.mean(dim=(1, 2))  # (C,)
            cam = (weights[:, None, None] * acts).sum(dim=0)  # (H, W)
            cam = F.relu(cam)
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()

            # Upsample to input size
            cam_up = (
                F.interpolate(
                    cam.unsqueeze(0).unsqueeze(0),
                    size=(550, 550),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
        except Exception as e:
            logger.warning(f"Grad-CAM failed for sample {idx}: {e}")
            cam_up = np.zeros((550, 550))
        finally:
            fwd_handle.remove()
            bwd_handle.remove()
            model.eval()

        # Denormalize image
        img_np = images[idx].numpy().transpose(1, 2, 0)
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        gradcam_images.append((img_np, cam_up))

        # Save individual
        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(img_np)
            ax.imshow(cam_up, cmap="jet", alpha=0.4)
            ax.set_title(f"Grad-CAM Sample {idx}", fontsize=11)
            ax.axis("off")
            fig.savefig(os.path.join(gradcam_dir, f"gradcam_sample_{idx}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Save 2x4 grid
    if HAS_MATPLOTLIB and len(gradcam_images) > 0:
        nrows, ncols = 2, 4
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        for i in range(nrows * ncols):
            ax = axes[i // ncols, i % ncols]
            if i < len(gradcam_images):
                img_np, cam_up = gradcam_images[i]
                ax.imshow(img_np)
                ax.imshow(cam_up, cmap="jet", alpha=0.4)
                ax.set_title(f"Sample {i}", fontsize=10)
            ax.axis("off")
        plt.suptitle("Grad-CAM Visualizations", fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "gradcam_grid.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved Grad-CAM grid to {output_dir}/gradcam_grid.png")


# ============================================================
# ABLATION: Soft-NMS vs Hard-NMS
# ============================================================


def run_ablation_nms(model, images, targets, device, output_dir, max_images=200, batch_size=4):
    logger.info("Running NMS ablation study...")
    n = min(max_images, len(images))
    imgs = images[:n]
    tgts = targets[:n]

    results = {}

    for nms_name, nms_config in [
        ("Soft-NMS (sigma=0.5)", {"method": "soft", "sigma": 0.5}),
        ("Hard-NMS (IoU=0.5)", {"method": "hard", "iou_threshold": 0.5}),
    ]:
        logger.info(f"  Evaluating with {nms_name}...")

        # Temporarily modify the model's detect post-processor
        original_process = model.detect._process_single_image

        if nms_config["method"] == "hard":

            def patched_process(
                self_ref, class_probs, boxes, mask_coeffs, prototypes, _iou_thresh=nms_config["iou_threshold"]
            ):
                device = class_probs.device
                proto_h, proto_w = prototypes.shape[1], prototypes.shape[2]
                empty = {
                    "boxes": torch.zeros(0, 4, device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                    "masks": torch.zeros(0, proto_h, proto_w, device=device),
                }
                fg_probs = class_probs[:, 1:]
                max_scores, max_classes = fg_probs.max(dim=-1)
                keep_mask = max_scores > self_ref.conf_threshold
                if keep_mask.sum() == 0:
                    return empty
                scores = max_scores[keep_mask]
                labels = max_classes[keep_mask]
                det_boxes = boxes[keep_mask]
                det_coeffs = mask_coeffs[keep_mask]
                if scores.size(0) > self_ref.top_k:
                    top_k_scores, top_k_idx = scores.topk(self_ref.top_k)
                    scores, labels = top_k_scores, labels[top_k_idx]
                    det_boxes, det_coeffs = det_boxes[top_k_idx], det_coeffs[top_k_idx]

                all_boxes, all_scores, all_labels, all_coeffs = [], [], [], []
                for cls in labels.unique():
                    m = labels == cls
                    kb, ks, ki = hard_nms(det_boxes[m], scores[m], iou_threshold=_iou_thresh)
                    if kb.size(0) > 0:
                        all_boxes.append(kb)
                        all_scores.append(ks)
                        all_labels.append(torch.full((kb.size(0),), cls.item(), dtype=torch.long, device=device))
                        all_coeffs.append(det_coeffs[m][ki])
                if not all_boxes:
                    return empty
                final_boxes = torch.cat(all_boxes)
                final_scores = torch.cat(all_scores)
                final_labels = torch.cat(all_labels)
                final_coeffs = torch.cat(all_coeffs)
                if final_scores.size(0) > self_ref.max_detections:
                    top_idx = final_scores.topk(self_ref.max_detections).indices
                    final_boxes = final_boxes[top_idx]
                    final_scores = final_scores[top_idx]
                    final_labels = final_labels[top_idx]
                    final_coeffs = final_coeffs[top_idx]
                from src.models.detection import Detect as Det

                final_masks = Det.assemble_masks(prototypes, final_coeffs, final_boxes)
                return {"boxes": final_boxes, "scores": final_scores, "labels": final_labels, "masks": final_masks}

            # Monkey-patch
            import types

            model.detect._process_single_image = types.MethodType(
                lambda self, cp, b, mc, p: patched_process(self, cp, b, mc, p),
                model.detect,
            )

        dets = run_inference(model, imgs, device, batch_size=batch_size)
        preds_np, gts_np = to_eval_format(dets, tgts)
        evaluator = COCOEvaluator()
        metrics = evaluator.evaluate(preds_np, gts_np)
        results[nms_name] = {
            "AP@0.50": metrics.get("AP@0.50", 0),
            "AP@0.75": metrics.get("AP@0.75", 0),
            "AP@[.50:.95]": metrics.get("AP@[.50:.95]", 0),
            "AR@100": metrics.get("AR@100", 0),
        }

        # Restore original
        if nms_config["method"] == "hard":
            model.detect._process_single_image = original_process

    # Save
    ablation_path = os.path.join(output_dir, "ablation_nms.json")
    with open(ablation_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved NMS ablation to {ablation_path}")

    # Print table
    print("\n" + "=" * 60)
    print("NMS ABLATION STUDY")
    print("=" * 60)
    print(f"{'Method':<25s} {'AP@0.50':>8s} {'AP@0.75':>8s} {'AP@[.50:.95]':>12s} {'AR@100':>8s}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<25s} {m['AP@0.50']:>8.4f} {m['AP@0.75']:>8.4f} {m['AP@[.50:.95]']:>12.4f} {m['AR@100']:>8.4f}")
    print("=" * 60)

    return results


# ============================================================
# ROBUSTNESS TESTING
# ============================================================


def apply_corruption(images, corruption_type, level):
    """Apply a corruption to a list of image tensors."""
    corrupted = []
    for img in images:
        img_np = img.numpy().copy()  # (C, H, W)
        if corruption_type == "gaussian_noise" and level > 0:
            noise = np.random.normal(0, level, img_np.shape).astype(np.float32)
            img_np = img_np + noise
        elif corruption_type == "gaussian_blur" and level > 0:
            import cv2

            for c in range(3):
                img_np[c] = cv2.GaussianBlur(img_np[c], (level, level), 0)
        elif corruption_type == "brightness" and level > 0:
            img_np = img_np + level
        img_np = np.clip(img_np, -3.0, 3.0)  # stay in reasonable normalized range
        corrupted.append(torch.from_numpy(img_np))
    return corrupted


def run_robustness(model, images, targets, device, output_dir, max_images=100, batch_size=4):
    logger.info("Running robustness testing...")
    n = min(max_images, len(images))
    imgs = images[:n]
    tgts = targets[:n]

    corruptions = {
        "Gaussian Noise": ("gaussian_noise", [0, 0.05, 0.1, 0.2]),
        "Gaussian Blur": ("gaussian_blur", [0, 3, 5, 9]),
        "Brightness Shift": ("brightness", [0, 0.2, 0.4, 0.6]),
    }

    all_results = {}
    for corr_name, (corr_type, levels) in corruptions.items():
        logger.info(f"  Testing {corr_name}...")
        level_aps = []
        for level in levels:
            corrupted = apply_corruption(imgs, corr_type, level)
            dets = run_inference(model, corrupted, device, batch_size=batch_size)
            preds_np, gts_np = to_eval_format(dets, tgts)
            metrics = compute_detection_metrics(preds_np, gts_np, iou_thresholds=[0.5])
            ap50 = metrics.get("AP@0.50", 0.0)
            level_aps.append(ap50)
            logger.info(f"    Level={level}: AP@0.50={ap50:.4f}")
        all_results[corr_name] = {"levels": levels, "AP@0.50": level_aps}

    # Save metrics
    rob_path = os.path.join(output_dir, "robustness_metrics.json")
    with open(rob_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved robustness metrics to {rob_path}")

    # Plot
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (name, data) in zip(axes, all_results.items()):
            ax.plot(data["levels"], data["AP@0.50"], "o-", linewidth=2, markersize=6)
            ax.set_xlabel(name + " Level", fontsize=11)
            ax.set_ylabel("AP@0.50", fontsize=11)
            ax.set_title(name, fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        plt.suptitle("Robustness Analysis: AP@0.50 Under Corruptions", fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "robustness_analysis.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved robustness plot to {output_dir}/robustness_analysis.png")

    return all_results


# ============================================================
# ERROR ANALYSIS
# ============================================================


def run_error_analysis(predictions_np, ground_truths_np, output_dir, iou_threshold=0.5):
    logger.info("Running error analysis...")

    density_buckets = [
        ("0-30", 0, 30),
        ("30-100", 30, 100),
        ("100-200", 100, 200),
        ("200+", 200, float("inf")),
    ]

    overall = {"TP": 0, "FP": 0, "FN": 0}
    bucket_results = {}
    for name, lo, hi in density_buckets:
        bucket_results[name] = {"TP": 0, "FP": 0, "FN": 0, "num_images": 0}

    for pred, gt in zip(predictions_np, ground_truths_np):
        n_gt = len(gt["boxes"])
        tp_fp = match_predictions_single_image(pred["boxes"], pred["scores"], gt["boxes"], iou_threshold)
        tp = int(tp_fp.sum())
        fp = len(tp_fp) - tp
        fn = n_gt - tp

        overall["TP"] += tp
        overall["FP"] += fp
        overall["FN"] += fn

        for name, lo, hi in density_buckets:
            if lo <= n_gt < hi:
                bucket_results[name]["TP"] += tp
                bucket_results[name]["FP"] += fp
                bucket_results[name]["FN"] += fn
                bucket_results[name]["num_images"] += 1
                break

    # Compute precision/recall for each bucket
    for name in bucket_results:
        b = bucket_results[name]
        b["precision"] = b["TP"] / max(b["TP"] + b["FP"], 1)
        b["recall"] = b["TP"] / max(b["TP"] + b["FN"], 1)

    overall["precision"] = overall["TP"] / max(overall["TP"] + overall["FP"], 1)
    overall["recall"] = overall["TP"] / max(overall["TP"] + overall["FN"], 1)

    result = {"overall": overall, "by_density": bucket_results}
    err_path = os.path.join(output_dir, "error_analysis.json")
    with open(err_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved error analysis to {err_path}")

    # Print
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS (IoU=0.5)")
    print("=" * 60)
    print(
        f"Overall: TP={overall['TP']}, FP={overall['FP']}, FN={overall['FN']}, "
        f"Prec={overall['precision']:.4f}, Rec={overall['recall']:.4f}"
    )
    print(f"\n{'Density':<12s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'Prec':>8s} {'Rec':>8s} {'Images':>8s}")
    print("-" * 60)
    for name, b in bucket_results.items():
        print(
            f"{name:<12s} {b['TP']:>6d} {b['FP']:>6d} {b['FN']:>6d} "
            f"{b['precision']:>8.4f} {b['recall']:>8.4f} {b['num_images']:>8d}"
        )
    print("=" * 60)

    return result


# ============================================================
# MAIN
# ============================================================


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    ckpt = args.checkpoint or find_checkpoint()
    model = load_model(ckpt, device)
    logger.info(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Load data
    logger.info("Loading validation data...")
    images, targets = load_val_data(args.data_dir, max_images=args.max_images)
    logger.info(f"Loaded {len(images)} validation images")

    # 1. Grad-CAM
    run_gradcam(model, images, device, args.output_dir, num_samples=8)

    # 2. NMS Ablation
    run_ablation_nms(
        model, images, targets, device, args.output_dir, max_images=args.max_images, batch_size=args.batch_size
    )

    # 3. Robustness
    run_robustness(
        model, images, targets, device, args.output_dir, max_images=min(100, len(images)), batch_size=args.batch_size
    )

    # 4. Full evaluation for error analysis
    logger.info("Running full inference for error analysis...")
    dets = run_inference(model, images, device, batch_size=args.batch_size)
    preds_np, gts_np = to_eval_format(dets, targets)

    # 5. Error analysis
    run_error_analysis(preds_np, gts_np, args.output_dir)

    # Save combined metrics
    combined = {
        "num_images": len(images),
        "checkpoint": ckpt or "none",
        "device": str(device),
    }
    with open(os.path.join(args.output_dir, "advanced_metrics.json"), "w") as f:
        json.dump(combined, f, indent=2)

    print("\nAdvanced evaluation complete. Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
