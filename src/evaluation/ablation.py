"""Ablation Study Framework for Hybrid Neuro-Symbolic Detector.

Automated component removal and evaluation to quantify the contribution
of each architectural component. Generates comprehensive metrics tables,
per-density analysis, and visualizations.

Ablation Variants:
    1. full_hybrid: Complete system (no components removed)
    2. dl_only: Spatial attention + recalibrator disabled
    3. no_recalibrator: Skip confidence recalibration
    4. no_spatial_attention: Gate = 0 (no feedback loop)
    5. no_row_model: GMM features zeroed
    6. no_density_field: KDE density replaced with uniform
    7. hard_nms: Replace Soft-NMS with standard NMS
    8. no_cbam: Disable CBAM attention in FPN
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# Density buckets for per-density analysis
DENSITY_BUCKETS = {
    "low": (0, 50),
    "medium": (50, 150),
    "high": (150, float("inf")),
}


class AblationFramework:
    """Automated ablation study runner.

    Systematically disables components at inference time (no retraining)
    and evaluates each variant on the validation/test set.

    Args:
        model: Trained HybridDetector instance.
        data_loader: DataLoader for evaluation.
        device: PyTorch device.
        output_dir: Directory to save results.
    """

    # Define ablation variants and what to disable
    VARIANTS = {
        "full_hybrid": {},
        "dl_only": {
            "disable_spatial_attention": True,
            "disable_recalibrator": True,
        },
        "no_recalibrator": {
            "disable_recalibrator": True,
        },
        "no_spatial_attention": {
            "disable_spatial_attention": True,
        },
        "no_row_model": {
            "disable_gmm": True,
        },
        "no_density_field": {
            "disable_kde": True,
        },
        "hard_nms": {
            "use_hard_nms": True,
        },
        "no_cbam": {
            "disable_cbam": True,
        },
    }

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        output_dir: str = "results/ablation",
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self, iou_threshold: float = 0.5) -> Dict[str, Any]:
        """Run all ablation variants and return comprehensive metrics.

        Args:
            iou_threshold: IoU threshold for matching predictions to GT.

        Returns:
            Dict with per-variant metrics, density analysis, and deltas.
        """
        results = {}

        for name, config in self.VARIANTS.items():
            logger.info("Running ablation variant: %s", name)
            start = time.time()

            metrics = self._evaluate_variant(name, config, iou_threshold)
            elapsed = time.time() - start

            metrics["eval_time"] = elapsed
            results[name] = metrics

            logger.info(
                "  %s: mAP@0.5=%.4f, AR@100=%.4f, F1=%.4f (%.1fs)",
                name,
                metrics["mAP_50"],
                metrics["AR_100"],
                metrics["F1"],
                elapsed,
            )

        # Compute deltas relative to full hybrid
        if "full_hybrid" in results:
            baseline = results["full_hybrid"]
            for name, metrics in results.items():
                if name == "full_hybrid":
                    continue
                results[name]["delta_mAP_50"] = metrics["mAP_50"] - baseline["mAP_50"]
                results[name]["delta_AR_100"] = metrics["AR_100"] - baseline["AR_100"]
                results[name]["delta_F1"] = metrics["F1"] - baseline["F1"]

        # Save results
        self._save_results(results)

        return results

    def _evaluate_variant(
        self,
        name: str,
        config: Dict[str, bool],
        iou_threshold: float,
    ) -> Dict[str, Any]:
        """Evaluate a single ablation variant.

        Modifies model behavior at inference time without retraining.

        Args:
            name: Variant name.
            config: Dict of flags to disable components.
            iou_threshold: IoU threshold.

        Returns:
            Metrics dict.
        """
        self.model.eval()

        # Apply ablation modifications
        original_state = self._apply_ablation(config)

        all_predictions = []
        all_targets = []
        per_image_density = []

        with torch.no_grad():
            for images, targets in self.data_loader:
                images = images.to(self.device)

                # Run inference
                detections = self.model(images)

                for i, (det, target) in enumerate(zip(detections, targets)):
                    gt_boxes = target["boxes"].numpy()
                    n_gt = len(gt_boxes)
                    per_image_density.append(n_gt)

                    pred_boxes = det["boxes"].cpu().numpy()
                    # Detection boxes are normalized [0,1]; GT boxes are in pixel coords
                    input_size = images.size(2)  # H == W == input_size
                    pred_boxes = pred_boxes * input_size
                    pred_scores = det["scores"].cpu().numpy()

                    all_predictions.append(
                        {
                            "boxes": pred_boxes,
                            "scores": pred_scores,
                        }
                    )
                    all_targets.append(
                        {
                            "boxes": gt_boxes,
                            "n_objects": n_gt,
                        }
                    )

        # Restore original state
        self._restore_state(original_state)

        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_targets, iou_threshold)

        # Per-density analysis
        density_metrics = self._per_density_analysis(all_predictions, all_targets, per_image_density, iou_threshold)
        metrics["density_analysis"] = density_metrics

        return metrics

    def _apply_ablation(self, config: Dict[str, bool]) -> Dict[str, Any]:
        """Apply ablation by modifying model components. Returns original state."""
        original = {}

        if config.get("disable_spatial_attention"):
            original["gate"] = self.model.spatial_attention.gate.data.clone()
            # Set gate to large negative → sigmoid(gate) ≈ 0
            self.model.spatial_attention.gate.data = torch.tensor(-100.0)

        if config.get("disable_recalibrator"):
            original["recalibrator_state"] = True
            # Monkey-patch recalibrator to return original scores
            self.model._recalibrator_disabled = True

        if config.get("disable_gmm"):
            original["gmm_fitted"] = self.model.spatial_engine.is_fitted
            # Override spatial features to zeros
            self.model._gmm_disabled = True

        if config.get("disable_kde"):
            original["kde_disabled"] = False
            self.model._kde_disabled = True

        if config.get("use_hard_nms"):
            if hasattr(self.model.yolact, "detect"):
                original["nms_sigma"] = self.model.yolact.detect.nms_sigma
                self.model.yolact.detect.nms_sigma = 0.0  # hard NMS

        if config.get("disable_cbam"):
            if hasattr(self.model.yolact.fpn, "cbam_modules"):
                original["cbam_modules"] = []
                for cbam in self.model.yolact.fpn.cbam_modules:
                    original["cbam_modules"].append(cbam.training)
                    # Set CBAM to identity by zeroing weights
                    self.model._cbam_disabled = True

        return original

    def _restore_state(self, original: Dict[str, Any]) -> None:
        """Restore model to original state after ablation."""
        if "gate" in original:
            self.model.spatial_attention.gate.data = original["gate"]

        if "recalibrator_state" in original:
            self.model._recalibrator_disabled = False

        if "gmm_fitted" in original:
            self.model._gmm_disabled = False

        if "kde_disabled" in original:
            self.model._kde_disabled = False

        if "nms_sigma" in original:
            self.model.yolact.detect.nms_sigma = original["nms_sigma"]

        if "cbam_modules" in original:
            self.model._cbam_disabled = False

    def _compute_metrics(
        self,
        predictions: List[Dict],
        targets: List[Dict],
        iou_threshold: float,
    ) -> Dict[str, float]:
        """Compute mAP, AR, precision, recall, F1."""
        all_tp = 0
        all_fp = 0
        all_fn = 0
        all_scores = []
        all_matches = []

        for pred, target in zip(predictions, targets):
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            gt_boxes = target["boxes"]

            if len(pred_boxes) == 0:
                all_fn += len(gt_boxes)
                continue

            if len(gt_boxes) == 0:
                all_fp += len(pred_boxes)
                all_scores.extend(pred_scores.tolist())
                all_matches.extend([0] * len(pred_scores))
                continue

            # Compute IoU matrix
            iou_matrix = self._compute_iou(pred_boxes, gt_boxes)

            # Greedy matching
            gt_matched = set()
            sorted_indices = np.argsort(-pred_scores)

            for idx in sorted_indices:
                best_iou = 0
                best_gt = -1
                for g in range(len(gt_boxes)):
                    if g not in gt_matched and iou_matrix[idx, g] > best_iou:
                        best_iou = iou_matrix[idx, g]
                        best_gt = g

                if best_iou >= iou_threshold and best_gt >= 0:
                    gt_matched.add(best_gt)
                    all_tp += 1
                    all_scores.append(pred_scores[idx])
                    all_matches.append(1)
                else:
                    all_fp += 1
                    all_scores.append(pred_scores[idx])
                    all_matches.append(0)

            all_fn += len(gt_boxes) - len(gt_matched)

        # Compute metrics
        precision = all_tp / max(all_tp + all_fp, 1)
        recall = all_tp / max(all_tp + all_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        # Compute AP (area under PR curve)
        ap = self._compute_ap(all_scores, all_matches, all_tp + all_fn)

        # AR at different detection limits
        ar_100 = recall  # Simplified: using all detections

        return {
            "mAP_50": ap,
            "AR_100": ar_100,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "TP": all_tp,
            "FP": all_fp,
            "FN": all_fn,
        }

    def _per_density_analysis(
        self,
        predictions: List[Dict],
        targets: List[Dict],
        densities: List[int],
        iou_threshold: float,
    ) -> Dict[str, Dict]:
        """Compute metrics per density bucket."""
        bucket_preds = defaultdict(list)
        bucket_targets = defaultdict(list)

        for pred, target, density in zip(predictions, targets, densities):
            for bucket_name, (lo, hi) in DENSITY_BUCKETS.items():
                if lo <= density < hi:
                    bucket_preds[bucket_name].append(pred)
                    bucket_targets[bucket_name].append(target)
                    break

        density_metrics = {}
        for bucket_name in DENSITY_BUCKETS:
            if bucket_preds[bucket_name]:
                metrics = self._compute_metrics(
                    bucket_preds[bucket_name],
                    bucket_targets[bucket_name],
                    iou_threshold,
                )
                metrics["n_images"] = len(bucket_preds[bucket_name])
                density_metrics[bucket_name] = metrics
            else:
                density_metrics[bucket_name] = {"mAP_50": 0.0, "AR_100": 0.0, "F1": 0.0, "n_images": 0}

        return density_metrics

    def _compute_ap(self, scores: List[float], matches: List[int], total_gt: int) -> float:
        """Compute average precision from score-match pairs."""
        if not scores or total_gt == 0:
            return 0.0

        # Sort by descending score
        sorted_pairs = sorted(zip(scores, matches), key=lambda x: -x[0])

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for score, match in sorted_pairs:
            if match:
                tp_cumsum += 1
            else:
                fp_cumsum += 1

            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / total_gt

            precisions.append(precision)
            recalls.append(recall)

        # 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
            if prec_at_recall:
                ap += max(prec_at_recall) / 11.0

        return ap

    @staticmethod
    def _compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes."""
        N = len(boxes1)
        M = len(boxes2)
        iou = np.zeros((N, M))

        for i in range(N):
            for j in range(M):
                x1 = max(boxes1[i, 0], boxes2[j, 0])
                y1 = max(boxes1[i, 1], boxes2[j, 1])
                x2 = min(boxes1[i, 2], boxes2[j, 2])
                y2 = min(boxes1[i, 3], boxes2[j, 3])

                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (boxes1[i, 2] - boxes1[i, 0]) * (boxes1[i, 3] - boxes1[i, 1])
                area2 = (boxes2[j, 2] - boxes2[j, 0]) * (boxes2[j, 3] - boxes2[j, 1])
                union = area1 + area2 - inter

                iou[i, j] = inter / max(union, 1e-6)

        return iou

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save ablation results to JSON and generate summary."""
        # Save full results
        json_path = self.output_dir / "ablation_results.json"

        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = json.loads(json.dumps(results, default=convert))

        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Ablation results saved to %s", json_path)

        # Generate summary table
        summary_lines = [
            "=" * 90,
            "ABLATION STUDY RESULTS",
            "=" * 90,
            f"{'Variant':<25} {'mAP@0.5':>10} {'AR@100':>10} {'Precision':>10} "
            f"{'Recall':>10} {'F1':>10} {'Delta mAP':>10}",
            "-" * 90,
        ]

        for name in self.VARIANTS:
            if name not in results:
                continue
            m = results[name]
            delta = m.get("delta_mAP_50", 0.0)
            delta_str = f"{delta:+.4f}" if name != "full_hybrid" else "baseline"
            summary_lines.append(
                f"{name:<25} {m['mAP_50']:>10.4f} {m['AR_100']:>10.4f} "
                f"{m['precision']:>10.4f} {m['recall']:>10.4f} {m['F1']:>10.4f} "
                f"{delta_str:>10}"
            )

        summary_lines.extend(["=" * 90, ""])

        # Per-density summary
        summary_lines.append("PER-DENSITY ANALYSIS (Full Hybrid)")
        summary_lines.append("-" * 60)
        if "full_hybrid" in results and "density_analysis" in results["full_hybrid"]:
            for bucket, metrics in results["full_hybrid"]["density_analysis"].items():
                summary_lines.append(
                    f"  {bucket:>8}: mAP={metrics['mAP_50']:.4f} "
                    f"F1={metrics['F1']:.4f} "
                    f"({metrics['n_images']} images)"
                )

        summary = "\n".join(summary_lines)
        logger.info("\n%s", summary)

        # Save summary
        summary_path = self.output_dir / "ablation_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)
