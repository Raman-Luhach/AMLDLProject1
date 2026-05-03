"""Loss functions for Hybrid Neuro-Symbolic Detector training.

Extends the base YOLACT loss with two additional terms:
    1. Recalibration Loss: BCE between recalibrated scores and ground-truth
       correctness (whether a detection actually overlaps with a GT box).
    2. Spatial Consistency Loss: Penalizes detections that violate learned
       spatial patterns (e.g., detections in very low-density regions).

Total Loss:
    L_total = L_yolact + lambda_r * L_recalibration + lambda_s * L_spatial
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecalibrationLoss(nn.Module):
    """BCE loss for confidence recalibration.

    Trains the recalibrator to output high scores for true positives
    (detections with IoU > threshold with GT) and low scores for
    false positives.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        super().__init__()
        self.iou_threshold = iou_threshold
        self.bce = nn.BCELoss(reduction="mean")

    def forward(
        self,
        recalibrated_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute recalibration loss.

        Args:
            recalibrated_scores: Predicted scores from recalibrator (N, 1).
            pred_boxes: Predicted boxes (N, 4) in [x1, y1, x2, y2].
            gt_boxes: Ground-truth boxes (M, 4) in [x1, y1, x2, y2].

        Returns:
            Scalar loss.
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return torch.tensor(0.0, device=recalibrated_scores.device, requires_grad=True)

        # Compute IoU between predictions and GT
        iou_matrix = self._compute_iou(pred_boxes, gt_boxes)

        # For each prediction, find max IoU with any GT box
        max_iou, _ = iou_matrix.max(dim=1)  # (N,)

        # Target: 1 if IoU > threshold (true positive), 0 otherwise
        targets = (max_iou >= self.iou_threshold).float().unsqueeze(1)

        # BCE loss
        loss = self.bce(recalibrated_scores, targets)

        return loss

    @staticmethod
    def _compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two sets of boxes.

        Args:
            boxes1: (N, 4) in [x1, y1, x2, y2].
            boxes2: (M, 4) in [x1, y1, x2, y2].

        Returns:
            IoU matrix (N, M).
        """
        N = boxes1.size(0)
        M = boxes2.size(0)

        # Expand for broadcasting
        b1 = boxes1.unsqueeze(1).expand(N, M, 4)  # (N, M, 4)
        b2 = boxes2.unsqueeze(0).expand(N, M, 4)  # (N, M, 4)

        # Intersection
        inter_x1 = torch.max(b1[..., 0], b2[..., 0])
        inter_y1 = torch.max(b1[..., 1], b2[..., 1])
        inter_x2 = torch.min(b1[..., 2], b2[..., 2])
        inter_y2 = torch.min(b1[..., 3], b2[..., 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union
        area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
        area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
        union_area = area1 + area2 - inter_area

        iou = inter_area / (union_area + 1e-6)
        return iou


class SpatialConsistencyLoss(nn.Module):
    """Penalizes detections that violate spatial consistency.

    Uses the density field to compute a penalty: detections in
    regions where the density model predicts few objects should
    have lower confidence. This encourages the model to be
    spatially coherent.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        class_preds: torch.Tensor,
        density_maps: torch.Tensor,
        anchors: torch.Tensor,
        input_size: int = 550,
    ) -> torch.Tensor:
        """Compute spatial consistency loss.

        High-confidence predictions in low-density regions are penalized.

        Args:
            class_preds: Class predictions (B, total_anchors, num_classes).
            density_maps: Density fields (B, 1, H, W) in [0, 1].
            anchors: Anchor boxes (total_anchors, 4) as [cx, cy, w, h] in [0, 1].
            input_size: Input image size.

        Returns:
            Scalar loss.
        """
        batch_size = class_preds.size(0)

        if density_maps is None or batch_size == 0:
            return torch.tensor(0.0, device=class_preds.device, requires_grad=True)

        # Get foreground confidence (class 1, ignoring background class 0)
        if class_preds.size(-1) > 1:
            fg_scores = torch.sigmoid(class_preds[:, :, 1])  # (B, total_anchors)
        else:
            fg_scores = torch.sigmoid(class_preds[:, :, 0])

        # Sample density at anchor center locations
        # anchors: (total_anchors, 4) as [cx, cy, w, h] in [0, 1]
        cx = anchors[:, 0]  # (total_anchors,)
        cy = anchors[:, 1]

        # Normalize to [-1, 1] for grid_sample
        grid_x = cx * 2 - 1
        grid_y = cy * 2 - 1

        # Create sampling grid (B, total_anchors, 1, 2)
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (total_anchors, 2)
        grid = grid.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, 1, 2)
        grid = grid.to(density_maps.device)

        # Sample density at anchor locations
        density_at_anchors = F.grid_sample(
            density_maps, grid, mode="bilinear", align_corners=False
        )  # (B, 1, total_anchors, 1)
        density_at_anchors = density_at_anchors.squeeze(1).squeeze(-1)  # (B, total_anchors)

        # Penalty: high confidence in low-density regions
        # Invert density: low_density → high penalty weight
        penalty_weight = 1.0 - density_at_anchors.clamp(0, 1)

        # Loss: weighted mean of foreground scores in low-density areas
        loss = (fg_scores * penalty_weight).mean()

        return loss


class HybridLoss(nn.Module):
    """Combined loss for hybrid model training.

    L_total = yolact_weight * L_yolact
            + recalibration_weight * L_recalibration
            + spatial_consistency_weight * L_spatial

    Args:
        yolact_loss: Pre-existing YOLACT loss instance.
        recalibration_weight: Weight for recalibration loss.
        spatial_consistency_weight: Weight for spatial consistency loss.
    """

    def __init__(
        self,
        yolact_loss: nn.Module,
        yolact_weight: float = 1.0,
        recalibration_weight: float = 0.5,
        spatial_consistency_weight: float = 0.3,
    ) -> None:
        super().__init__()

        self.yolact_loss = yolact_loss
        self.recalibration_loss = RecalibrationLoss()
        self.spatial_consistency_loss = SpatialConsistencyLoss()

        self.yolact_weight = yolact_weight
        self.recalibration_weight = recalibration_weight
        self.spatial_consistency_weight = spatial_consistency_weight

    def forward(
        self,
        predictions: dict,
        targets: list,
    ) -> dict:
        """Compute combined hybrid loss.

        Args:
            predictions: Dict from HybridDetector training forward pass.
            targets: List of target dicts.

        Returns:
            Dict with individual and total loss values.
        """
        # YOLACT base loss
        yolact_out = (
            predictions["class_preds"],
            predictions["box_preds"],
            predictions["mask_coeffs"],
            predictions["prototypes"],
            predictions["anchors"],
        )
        yolact_losses = self.yolact_loss(yolact_out, targets)
        yolact_loss_val = yolact_losses["total"]

        # Spatial consistency loss
        spatial_loss = self.spatial_consistency_loss(
            predictions["class_preds"],
            predictions.get("density_maps"),
            predictions["anchors"],
        )

        # Total loss
        total_loss = self.yolact_weight * yolact_loss_val + self.spatial_consistency_weight * spatial_loss

        return {
            "total": total_loss,
            "yolact": yolact_loss_val.detach(),
            "spatial_consistency": spatial_loss.detach(),
            "cls": yolact_losses.get("cls", torch.tensor(0.0)),
            "box": yolact_losses.get("box", torch.tensor(0.0)),
            "mask": yolact_losses.get("mask", torch.tensor(0.0)),
            "gate_value": predictions.get("gate_value", 0.0),
        }
