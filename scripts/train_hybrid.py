#!/usr/bin/env python3
"""Training script for the Hybrid Neuro-Symbolic Detector.

Three-stage training pipeline:
    Stage fit-spatial: Fit ML spatial reasoning engine on GT boxes
    Stage train-hybrid: Train fusion components (3a frozen + 3b fine-tune)
    Stage all: Run both stages sequentially

Usage:
    # Full pipeline:
    python scripts/train_hybrid.py --config configs/hybrid.yaml

    # Individual stages:
    python scripts/train_hybrid.py --config configs/hybrid.yaml --stage fit-spatial
    python scripts/train_hybrid.py --config configs/hybrid.yaml --stage train-hybrid

    # Resume from checkpoint:
    python scripts/train_hybrid.py --config configs/hybrid.yaml --stage train-hybrid \\
        --resume results/hybrid/checkpoints/hybrid_epoch5.pth
"""

import argparse
import logging
import os
import sys
import time

# Ensure the project root is on the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

from src.models.hybrid import HybridDetector
from src.data.dataset import get_dataloaders
from src.training.hybrid_trainer import HybridTrainer
from src.utils.helpers import get_device, load_config, set_seed, format_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Hybrid Neuro-Symbolic Detector on SKU-110K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hybrid.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["fit-spatial", "train-hybrid", "all"],
        help="Training stage to run",
    )
    parser.add_argument(
        "--yolact-checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained YOLACT checkpoint (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to hybrid checkpoint to resume training from",
    )
    parser.add_argument(
        "--spatial-models",
        type=str,
        default=None,
        help="Path to fitted spatial models directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config(args.config)
    logger.info("Configuration loaded from %s", args.config)

    # Set random seed for reproducibility
    set_seed(42)

    # Get device
    device = get_device()
    logger.info("Using device: %s", device)

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = get_dataloaders(config)
    logger.info("  Train: %d batches", len(train_loader))
    logger.info("  Val: %d batches", len(val_loader))

    # Build hybrid model
    logger.info("Building Hybrid Detector...")
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

    hybrid_config = config.get("hybrid", {})
    model = HybridDetector(
        yolact_config=yolact_config,
        hybrid_config=hybrid_config,
    )

    # Load YOLACT pre-trained checkpoint
    yolact_ckpt = args.yolact_checkpoint or config.get("training", {}).get("yolact_resume")
    if yolact_ckpt and os.path.exists(yolact_ckpt):
        model.load_yolact_checkpoint(yolact_ckpt)
    else:
        logger.warning("No YOLACT checkpoint provided. Starting from ImageNet pretrained backbone.")

    # Resume from hybrid checkpoint if provided
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info("Resumed from hybrid checkpoint: %s (epoch %d)", args.resume, checkpoint.get("epoch", 0))

    model = model.to(device)

    # Print model info
    params = model.count_parameters()
    logger.info("Model Parameters:")
    logger.info("  YOLACT total: %s", format_params(params["yolact"]["total"]))
    logger.info("  Spatial Attention: %s", format_params(params["spatial_attention"]["total"]))
    logger.info("  Recalibrator: %s", format_params(params["recalibrator"]["total"]))
    logger.info("  Visual Projector: %s", format_params(params["visual_projector"]["total"]))
    logger.info("  Total: %s (trainable: %s)", format_params(params["total"]), format_params(params["trainable"]))

    # Create trainer
    trainer = HybridTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Run training stages
    total_start = time.time()

    if args.stage in ("fit-spatial", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: Fitting Spatial Reasoning Engine")
        logger.info("=" * 60)
        trainer.fit_spatial_engine()

    if args.stage in ("train-hybrid", "all"):
        # Load spatial models if not just fitted
        if args.stage == "train-hybrid":
            spatial_dir = args.spatial_models or config.get("hybrid", {}).get("spatial_reasoning", {}).get(
                "save_dir", "results/hybrid/spatial_models"
            )
            if os.path.exists(os.path.join(spatial_dir, "spatial_engine.pkl")):
                model.load_spatial_engine(spatial_dir)
            else:
                logger.warning("No fitted spatial models found. Running fit-spatial first...")
                trainer.fit_spatial_engine()

        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: Training Fusion Components")
        logger.info("=" * 60)
        results = trainer.train_fusion()

    total_time = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("Training complete! Total time: %.1f minutes", total_time / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
