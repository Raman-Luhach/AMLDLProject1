"""Hybrid Detector Training Loop.

Three-stage training pipeline:
    Stage 1: YOLACT pre-training (use existing train.py with --resume)
    Stage 2: Fit Spatial Reasoning Engine on GT boxes (sklearn, no GPU)
    Stage 3a: Train fusion components with frozen YOLACT backbone
    Stage 3b: Joint end-to-end fine-tuning with reduced learning rate

Device compatibility:
    - MPS (Apple Silicon): Full FP32 training, no AMP/GradScaler
    - CUDA: Optional AMP with GradScaler
    - CPU: Full FP32 training
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.models.hybrid import HybridDetector
from src.training.losses import YOLACTLoss
from src.training.hybrid_losses import HybridLoss
from src.data.augmentations import mixup_batch

logger = logging.getLogger(__name__)


class HybridTrainer:
    """Training loop for the Hybrid Neuro-Symbolic Detector.

    Handles the complete hybrid training workflow:
        - Loading YOLACT pre-trained checkpoint
        - Fitting spatial reasoning engine
        - Training fusion components (frozen/unfrozen)
        - Checkpointing, logging, and validation

    Args:
        model: HybridDetector instance (already on device).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Configuration dictionary.
        device: PyTorch device.
    """

    def __init__(
        self,
        model: HybridDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Training config
        train_cfg = config.get("training", {})
        hybrid_cfg = config.get("hybrid", {})
        loss_cfg = config.get("loss", {})

        # Loss function
        yolact_loss = YOLACTLoss(
            num_classes=config.get("dataset", {}).get("num_classes", 1) + 1,
            cls_weight=loss_cfg.get("cls_weight", 1.0),
            box_weight=loss_cfg.get("box_weight", 1.5),
            mask_weight=loss_cfg.get("mask_weight", 6.125),
            focal_alpha=loss_cfg.get("focal_alpha", 0.25),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            neg_pos_ratio=loss_cfg.get("neg_pos_ratio", 3),
            label_smoothing=loss_cfg.get("label_smoothing", 0.1),
        )

        hybrid_losses_cfg = hybrid_cfg.get("losses", {})
        self.criterion = HybridLoss(
            yolact_loss=yolact_loss,
            yolact_weight=hybrid_losses_cfg.get("yolact_weight", 1.0),
            recalibration_weight=hybrid_losses_cfg.get("recalibration_weight", 0.5),
            spatial_consistency_weight=hybrid_losses_cfg.get("spatial_consistency_weight", 0.3),
        )

        # Checkpoint directory
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "results/hybrid/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save interval
        self.save_interval = train_cfg.get("save_interval", 5)
        self.val_interval = train_cfg.get("val_interval", 2)

        # MixUp config
        aug_cfg = config.get("augmentation", {})
        self.use_mixup = aug_cfg.get("mixup", True)
        self.mixup_alpha = aug_cfg.get("mixup_alpha", 0.2)

        # Training state
        self.best_val_loss = float("inf")
        self.history = []

    def fit_spatial_engine(self) -> None:
        """Stage 2: Fit the spatial reasoning engine on GT boxes.

        Iterates through training data, collects all GT boxes, and fits
        the GMM + KDE models.
        """
        logger.info("=" * 60)
        logger.info("Stage 2: Fitting Spatial Reasoning Engine")
        logger.info("=" * 60)

        all_gt_boxes = []
        input_size = self.config.get("dataset", {}).get("input_size", 550)

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            for t in targets:
                boxes = t["boxes"].numpy()
                if len(boxes) > 0:
                    all_gt_boxes.append(boxes)

            if (batch_idx + 1) % 100 == 0:
                logger.info("  Collected GT from %d batches...", batch_idx + 1)

        logger.info("Collected GT boxes from %d images", len(all_gt_boxes))

        # Fit the spatial engine
        self.model.spatial_engine.fit(all_gt_boxes, image_size=input_size)

        # Save fitted models
        save_dir = (
            self.config.get("hybrid", {}).get("spatial_reasoning", {}).get("save_dir", "results/hybrid/spatial_models")
        )
        self.model.spatial_engine.save(Path(save_dir))
        logger.info("Spatial reasoning engine fitted and saved.")

    def train_fusion(self) -> Dict[str, Any]:
        """Stage 3: Train fusion components (attention + recalibrator).

        Stage 3a: Frozen YOLACT backbone, train only fusion components.
        Stage 3b: Unfreeze all, joint fine-tuning with lower LR.

        Returns:
            Training history dict.
        """
        train_cfg = self.config.get("training", {})
        frozen_epochs = train_cfg.get("frozen_epochs", 10)
        finetune_epochs = train_cfg.get("finetune_epochs", 5)
        total_epochs = frozen_epochs + finetune_epochs

        logger.info("=" * 60)
        logger.info("Stage 3: Training Fusion Components")
        logger.info("  3a: %d epochs (frozen backbone)", frozen_epochs)
        logger.info("  3b: %d epochs (joint fine-tuning)", finetune_epochs)
        logger.info("=" * 60)

        # ====== Stage 3a: Frozen backbone ======
        logger.info("\n--- Stage 3a: Frozen Backbone Training ---")
        self.model.freeze_yolact()

        optimizer_3a = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=train_cfg.get("frozen_lr", 0.001),
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=train_cfg.get("weight_decay", 0.0005),
        )

        scheduler_3a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_3a, T_max=frozen_epochs, eta_min=1e-6)

        for epoch in range(1, frozen_epochs + 1):
            train_losses = self._train_epoch(optimizer_3a, epoch, frozen_epochs)
            scheduler_3a.step()

            # Log
            entry = {"epoch": epoch, "stage": "3a", "train": train_losses, "lr": optimizer_3a.param_groups[0]["lr"]}

            # Validation
            if epoch % self.val_interval == 0 or epoch == frozen_epochs:
                val_losses = self._validate_epoch(epoch)
                entry["val"] = val_losses

                if val_losses["total"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total"]
                    self._save_checkpoint(epoch, optimizer_3a, is_best=True)

            self.history.append(entry)

            # Periodic checkpoint
            if epoch % self.save_interval == 0:
                self._save_checkpoint(epoch, optimizer_3a)

        # ====== Stage 3b: Joint fine-tuning ======
        logger.info("\n--- Stage 3b: Joint Fine-Tuning ---")
        self.model.unfreeze_yolact()

        optimizer_3b = torch.optim.SGD(
            self.model.parameters(),
            lr=train_cfg.get("finetune_lr", 0.0001),
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=train_cfg.get("weight_decay", 0.0005),
        )

        scheduler_3b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_3b, T_max=finetune_epochs, eta_min=1e-7)

        for epoch in range(frozen_epochs + 1, total_epochs + 1):
            train_losses = self._train_epoch(optimizer_3b, epoch, total_epochs)
            scheduler_3b.step()

            entry = {"epoch": epoch, "stage": "3b", "train": train_losses, "lr": optimizer_3b.param_groups[0]["lr"]}

            if epoch % self.val_interval == 0 or epoch == total_epochs:
                val_losses = self._validate_epoch(epoch)
                entry["val"] = val_losses

                if val_losses["total"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total"]
                    self._save_checkpoint(epoch, optimizer_3b, is_best=True)

            self.history.append(entry)

            if epoch % self.save_interval == 0 or epoch == total_epochs:
                self._save_checkpoint(epoch, optimizer_3b)

        # Save final model
        self._save_checkpoint(total_epochs, optimizer_3b, filename="hybrid_final.pth")

        # Save training log
        log_path = self.checkpoint_dir / "hybrid_training_log.json"
        with open(log_path, "w") as f:
            json.dump(
                {
                    "config": {
                        "frozen_epochs": frozen_epochs,
                        "finetune_epochs": finetune_epochs,
                        "device": str(self.device),
                    },
                    "best_val_loss": self.best_val_loss,
                    "history": self.history,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info("Training complete. Best val loss: %.4f", self.best_val_loss)
        logger.info("Training log saved to %s", log_path)

        return {"best_val_loss": self.best_val_loss, "history": self.history}

    def _train_epoch(self, optimizer: torch.optim.Optimizer, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_start = time.time()

        running_losses = {
            "total": 0.0,
            "yolact": 0.0,
            "spatial_consistency": 0.0,
            "cls": 0.0,
            "box": 0.0,
            "mask": 0.0,
        }
        num_batches = 0

        grad_clip = self.config.get("training", {}).get("gradient_clip", 10.0)

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # MixUp augmentation
            if self.use_mixup and len(images) > 1:
                images, targets, _, _ = mixup_batch(images, targets, alpha=self.mixup_alpha)

            # Forward pass
            optimizer.zero_grad()
            predictions = self.model(images, targets)

            # Compute loss
            losses = self.criterion(predictions, targets)
            loss = losses["total"]

            # Backward pass
            loss.backward()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            optimizer.step()

            # Accumulate losses
            for key in running_losses:
                if key in losses:
                    val = losses[key]
                    if isinstance(val, torch.Tensor):
                        running_losses[key] += val.item()
                    else:
                        running_losses[key] += val
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = running_losses["total"] / num_batches
                gate_val = losses.get("gate_value", 0.0)
                logger.info(
                    "  Epoch [%d/%d] Batch [%d/%d] Loss: %.4f Gate: %.4f",
                    epoch,
                    total_epochs,
                    batch_idx + 1,
                    len(self.train_loader),
                    avg_loss,
                    gate_val,
                )

        epoch_time = time.time() - epoch_start

        # Average losses
        avg_losses = {k: v / max(num_batches, 1) for k, v in running_losses.items()}
        avg_losses["epoch_time"] = epoch_time
        avg_losses["gate_value"] = losses.get("gate_value", 0.0)

        logger.info(
            "Epoch [%d/%d] Train Loss: %.4f (yolact: %.4f, spatial: %.4f) " "Gate: %.4f Time: %.0fs",
            epoch,
            total_epochs,
            avg_losses["total"],
            avg_losses["yolact"],
            avg_losses["spatial_consistency"],
            avg_losses["gate_value"],
            epoch_time,
        )

        return avg_losses

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Run validation epoch."""
        self.model.eval()
        val_start = time.time()

        running_losses = {
            "total": 0.0,
            "yolact": 0.0,
            "spatial_consistency": 0.0,
            "cls": 0.0,
            "box": 0.0,
            "mask": 0.0,
        }
        num_batches = 0

        for images, targets in self.val_loader:
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass (training mode for loss computation)
            self.model.train()
            predictions = self.model(images, targets)
            losses = self.criterion(predictions, targets)
            self.model.eval()

            for key in running_losses:
                if key in losses:
                    val = losses[key]
                    if isinstance(val, torch.Tensor):
                        running_losses[key] += val.item()
                    else:
                        running_losses[key] += val
            num_batches += 1

        val_time = time.time() - val_start
        avg_losses = {k: v / max(num_batches, 1) for k, v in running_losses.items()}
        avg_losses["val_time"] = val_time

        logger.info(
            "  Val Loss: %.4f (yolact: %.4f, spatial: %.4f) Time: %.0fs",
            avg_losses["total"],
            avg_losses["yolact"],
            avg_losses["spatial_consistency"],
            val_time,
        )

        return avg_losses

    def _save_checkpoint(
        self, epoch: int, optimizer: torch.optim.Optimizer, is_best: bool = False, filename: str = None
    ) -> None:
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if filename:
            path = self.checkpoint_dir / filename
        else:
            path = self.checkpoint_dir / f"hybrid_epoch{epoch}.pth"

        torch.save(state, path)
        logger.info("  Checkpoint saved: %s", path)

        if is_best:
            best_path = self.checkpoint_dir / "hybrid_best.pth"
            torch.save(state, best_path)
            logger.info("  Best model updated: %s", best_path)
