#!/usr/bin/env python3
"""Run automated ablation study for the Hybrid Detector.

Evaluates all ablation variants on the validation set and generates
comprehensive metrics tables, per-density analysis, and visualizations.

Usage:
    python scripts/run_ablation.py --config configs/hybrid.yaml
    python scripts/run_ablation.py --checkpoint results/hybrid/checkpoints/hybrid_best.pth
"""

import argparse
import logging
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.models.hybrid import HybridDetector
from src.data.dataset import get_dataloaders
from src.evaluation.ablation import AblationFramework
from src.utils.helpers import get_device, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ablation Study")
    parser.add_argument("--config", type=str, default="configs/hybrid.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/hybrid/checkpoints/hybrid_best.pth")
    parser.add_argument("--output-dir", type=str, default="results/ablation")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def generate_ablation_charts(results, output_dir):
    """Generate ablation visualization charts."""
    output_dir = os.path.join(output_dir, "")
    os.makedirs(output_dir, exist_ok=True)

    variants = []
    map_values = []
    deltas = []

    for name in [
        "full_hybrid",
        "dl_only",
        "no_recalibrator",
        "no_spatial_attention",
        "no_row_model",
        "no_density_field",
        "hard_nms",
        "no_cbam",
    ]:
        if name in results:
            variants.append(name.replace("_", "\n"))
            map_values.append(results[name]["mAP_50"])
            deltas.append(results[name].get("delta_mAP_50", 0.0))

    # Chart 1: mAP comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]
    colors[0] = "#3498db"  # Highlight full hybrid
    bars = ax.bar(range(len(variants)), map_values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, fontsize=9)
    ax.set_ylabel("mAP@0.5", fontsize=12)
    ax.set_title("Ablation Study: Component Contribution to mAP@0.5", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, map_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_chart.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Chart 2: Delta mAP (component contribution)
    fig, ax = plt.subplots(figsize=(10, 5))
    variant_names = [v for v, d in zip(variants[1:], deltas[1:]) if True]
    delta_vals = deltas[1:]
    colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in delta_vals]

    ax.barh(range(len(variant_names)), delta_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(variant_names)))
    ax.set_yticklabels(variant_names, fontsize=9)
    ax.set_xlabel("Delta mAP@0.5 (vs Full Hybrid)", fontsize=12)
    ax.set_title("Component Contribution: Impact of Removal", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    for i, val in enumerate(delta_vals):
        ax.text(
            val + (0.001 if val >= 0 else -0.001),
            i,
            f"{val:+.4f}",
            ha="left" if val >= 0 else "right",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "component_contribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Chart 3: Per-density analysis
    if "full_hybrid" in results and "density_analysis" in results["full_hybrid"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        density_data = results["full_hybrid"]["density_analysis"]

        for ax_idx, metric in enumerate(["mAP_50", "recall", "F1"]):
            buckets = ["low", "medium", "high"]
            values = [density_data.get(b, {}).get(metric, 0) for b in buckets]

            axes[ax_idx].bar(buckets, values, color=["#3498db", "#f39c12", "#e74c3c"], edgecolor="black", linewidth=0.5)
            axes[ax_idx].set_title(metric.replace("_", "@"), fontsize=12, fontweight="bold")
            axes[ax_idx].set_ylabel("Score")
            axes[ax_idx].grid(axis="y", alpha=0.3)

            for i, val in enumerate(values):
                axes[ax_idx].text(i, val + 0.001, f"{val:.4f}", ha="center", fontsize=9)

        plt.suptitle("Performance by Scene Density", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "density_analysis.png"), dpi=300, bbox_inches="tight")
        plt.close()

    logging.getLogger(__name__).info("Ablation charts saved to %s", output_dir)


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    device = get_device()

    # Build model
    yolact_config = {
        "input_size": config.get("dataset", {}).get("input_size", 550),
        "num_classes": config.get("dataset", {}).get("num_classes", 1) + 1,
        "pretrained_backbone": config.get("backbone", {}).get("pretrained", True),
        "fpn_out_channels": config.get("fpn", {}).get("out_channels", 256),
        "num_prototypes": config.get("yolact", {}).get("num_prototypes", 32),
        "conf_threshold": config.get("yolact", {}).get("conf_threshold", 0.05),
        "nms_sigma": config.get("softnms", {}).get("sigma", 0.5),
        "max_detections": config.get("yolact", {}).get("max_detections", 300),
    }

    model = HybridDetector(
        yolact_config=yolact_config,
        hybrid_config=config.get("hybrid", {}),
    )

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        logger.info("Loaded checkpoint: %s", args.checkpoint)
    else:
        logger.warning("Checkpoint not found: %s. Using random weights.", args.checkpoint)

    # Load spatial engine
    spatial_dir = config.get("hybrid", {}).get("spatial_reasoning", {}).get("save_dir", "results/hybrid/spatial_models")
    if os.path.exists(os.path.join(spatial_dir, "spatial_engine.pkl")):
        model.load_spatial_engine(spatial_dir)

    model = model.to(device)

    # Get data loader
    _, val_loader = get_dataloaders(config)

    # Run ablation
    framework = AblationFramework(
        model=model,
        data_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
    )

    results = framework.run_all(iou_threshold=args.iou_threshold)

    # Generate charts
    generate_ablation_charts(results, args.output_dir)

    logger.info("Ablation study complete. Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
