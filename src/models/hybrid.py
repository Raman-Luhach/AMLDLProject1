"""Hybrid Neuro-Symbolic Detector.

Combines YOLACT (DL) with Spatial Reasoning Engine (ML) in a synergistic
architecture with differentiable feedback loop.

Pipeline:
    1. YOLACT backbone + FPN → multi-scale features
    2. Initial prediction head → raw detections
    3. Spatial Reasoning Engine → spatial features + density field (ML)
    4. Spatial Prior Attention → modulate P3 with density prior (FEEDBACK)
    5. Re-run ProtoNet + Prediction Head on attended features
    6. Confidence Recalibrator → fuse score + spatial + visual (FUSION)
    7. Final refined detections

The architecture creates genuine symbiosis:
    - DL: Visual feature extraction (what objects look like)
    - ML: Spatial reasoning (where objects should be, shelf grammar)
    - Feedback: ML spatial priors improve DL feature extraction
    - Fusion: Combined confidence is better than either alone
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.models.yolact import YOLACT, DEFAULT_CONFIG
from src.models.spatial_reasoning import SpatialReasoningEngine
from src.models.spatial_attention import SpatialPriorAttention
from src.models.confidence_recalibrator import ConfidenceRecalibrator

logger = logging.getLogger(__name__)


class HybridDetector(nn.Module):
    """Neuro-Symbolic Hybrid Detection System.

    Wraps YOLACT and augments it with ML-based spatial reasoning,
    a differentiable feedback attention mechanism, and confidence
    recalibration.

    Args:
        yolact_config: Configuration dict for YOLACT base model.
        hybrid_config: Configuration dict for hybrid components.
    """

    def __init__(
        self,
        yolact_config: Optional[Dict[str, Any]] = None,
        hybrid_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        # Default hybrid configuration
        self.hybrid_config = {
            "spatial_reasoning": {
                "num_row_components": 8,
                "kde_bandwidth": 0.05,
                "min_detections": 5,
                "save_dir": "results/hybrid/spatial_models",
            },
            "recalibrator": {
                "spatial_dim": 8,
                "visual_dim": 64,
                "hidden_dims": [128, 64],
                "dropout": 0.1,
            },
            "spatial_attention": {
                "gate_init": 0.1,
                "conv_channels": 16,
            },
        }
        if hybrid_config:
            for key in hybrid_config:
                if key in self.hybrid_config and isinstance(self.hybrid_config[key], dict):
                    self.hybrid_config[key].update(hybrid_config[key])
                else:
                    self.hybrid_config[key] = hybrid_config[key]

        # ====== DL Component: YOLACT (existing, no modifications) ======
        self.yolact = YOLACT(config=yolact_config)

        # ====== ML Component: Spatial Reasoning Engine (sklearn) ======
        self.spatial_engine = SpatialReasoningEngine(self.hybrid_config["spatial_reasoning"])

        # ====== Fusion Components (differentiable) ======
        attn_cfg = self.hybrid_config["spatial_attention"]
        self.spatial_attention = SpatialPriorAttention(
            conv_channels=attn_cfg["conv_channels"],
            gate_init=attn_cfg["gate_init"],
        )

        recal_cfg = self.hybrid_config["recalibrator"]
        self.recalibrator = ConfidenceRecalibrator(
            spatial_dim=recal_cfg["spatial_dim"],
            visual_dim=recal_cfg["visual_dim"],
            hidden_dims=recal_cfg["hidden_dims"],
            dropout=recal_cfg["dropout"],
        )

        # Visual feature projector for recalibrator
        fpn_channels = 256
        self.roi_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_projector = nn.Sequential(
            nn.Linear(fpn_channels, recal_cfg["visual_dim"]),
            nn.ReLU(inplace=True),
        )

        self._input_size = DEFAULT_CONFIG.get("input_size", 550)
        if yolact_config:
            self._input_size = yolact_config.get("input_size", self._input_size)

    def load_yolact_checkpoint(self, checkpoint_path: str) -> None:
        """Load pre-trained YOLACT weights.

        Args:
            checkpoint_path: Path to YOLACT checkpoint file.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.yolact.load_state_dict(state_dict, strict=False)
        logger.info("Loaded YOLACT checkpoint from %s", checkpoint_path)

    def load_spatial_engine(self, path: Optional[str] = None) -> None:
        """Load fitted spatial reasoning models.

        Args:
            path: Directory containing spatial engine pickle files.
        """
        load_path = Path(path) if path else None
        self.spatial_engine.load(load_path)

    def freeze_yolact(self) -> None:
        """Freeze all YOLACT parameters (for Stage 3a training)."""
        for param in self.yolact.parameters():
            param.requires_grad = False
        logger.info("YOLACT parameters frozen.")

    def unfreeze_yolact(self) -> None:
        """Unfreeze all YOLACT parameters (for Stage 3b fine-tuning)."""
        for param in self.yolact.parameters():
            param.requires_grad = True
        logger.info("YOLACT parameters unfrozen.")

    def _extract_fpn_features(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Run YOLACT backbone + FPN and return features.

        Args:
            images: Input images (B, 3, H, W).

        Returns:
            Tuple of (fpn_features, prototypes):
                fpn_features: List of [P3, P4, P5, P6, P7] feature maps.
                prototypes: (B, num_prototypes, proto_H, proto_W).
        """
        backbone_features = self.yolact.backbone(images)
        fpn_features = self.yolact.fpn(backbone_features)
        return fpn_features

    def _get_spatial_features_from_boxes(self, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute spatial features and density field from boxes.

        Args:
            boxes: Detection boxes (N, 4) in [x1, y1, x2, y2] format.

        Returns:
            Tuple of (spatial_features, density_field):
                spatial_features: (8,) feature vector.
                density_field: (H, W) normalized density map.
        """
        spatial_features = self.spatial_engine.compute_spatial_features(boxes, image_size=self._input_size)

        # Generate density at a reduced resolution (P3 size)
        p3_size = self._input_size // 16  # stride 16
        density_field = self.spatial_engine.generate_density_field(
            boxes, shape=(p3_size, p3_size), image_size=self._input_size
        )

        return spatial_features, density_field

    def _extract_visual_features(self, fpn_feature: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Extract ROI-pooled visual features for each detection.

        Uses simple ROI alignment: crop feature map at box locations,
        pool to 1x1, then project to visual_dim.

        Args:
            fpn_feature: P3 feature map (B, C, H, W). Assumes B=1 for per-image.
            boxes: Detection boxes (N, 4) in [x1, y1, x2, y2] normalized to [0, 1].

        Returns:
            Visual features (N, visual_dim).
        """
        if len(boxes) == 0:
            return torch.zeros(0, self.hybrid_config["recalibrator"]["visual_dim"], device=fpn_feature.device)

        C, H, W = fpn_feature.shape[1], fpn_feature.shape[2], fpn_feature.shape[3]
        feat = fpn_feature[0]  # (C, H, W) for single image

        visual_feats = []
        for box in boxes:
            # Convert normalized [x1,y1,x2,y2] to feature map coordinates
            x1 = max(0, int(box[0].item() * W))
            y1 = max(0, int(box[1].item() * H))
            x2 = min(W, max(x1 + 1, int(box[2].item() * W)))
            y2 = min(H, max(y1 + 1, int(box[3].item() * H)))

            # Crop and pool
            roi = feat[:, y1:y2, x1:x2].unsqueeze(0)  # (1, C, h, w)
            pooled = self.roi_pool(roi).squeeze(-1).squeeze(-1)  # (1, C)
            visual_feats.append(pooled)

        visual_feats = torch.cat(visual_feats, dim=0)  # (N, C)
        visual_feats = self.visual_projector(visual_feats)  # (N, visual_dim)

        return visual_feats

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Union[
        Dict[str, torch.Tensor],
        List[Dict[str, torch.Tensor]],
    ]:
        """Forward pass through the hybrid detector.

        Training mode:
            Uses GT boxes for spatial features (teacher forcing).
            Returns dict with all predictions + losses info.

        Inference mode:
            Uses initial YOLACT detections for spatial features.
            Returns list of refined detection dicts.

        Args:
            images: Input images (B, 3, H, W).
            targets: Optional list of target dicts with 'boxes', 'labels', 'masks'.

        Returns:
            Training: Dict with class_preds, box_preds, mask_coeffs, prototypes,
                      anchors, spatial_features, density_maps, recalibrator inputs.
            Inference: List of detection dicts with refined boxes/scores.
        """
        if self.training:
            return self._training_forward(images, targets)
        else:
            return self._inference_forward(images)

    def _training_forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with teacher forcing.

        Uses GT boxes to compute spatial features for stable training.
        """
        batch_size = images.size(0)
        device = images.device

        # Step 1: Extract FPN features
        fpn_features = self._extract_fpn_features(images)

        # Step 2: Compute spatial features using GT boxes (teacher forcing)
        density_maps = []
        spatial_features_batch = []

        for i in range(batch_size):
            if targets is not None and i < len(targets):
                gt_boxes = targets[i]["boxes"].cpu().numpy()
            else:
                gt_boxes = np.zeros((0, 4))

            spatial_feats, density = self._get_spatial_features_from_boxes(gt_boxes)
            spatial_features_batch.append(spatial_feats)
            density_maps.append(density)

        # Convert density maps to tensor (B, 1, H, W)
        density_tensor = torch.tensor(np.stack(density_maps), dtype=torch.float32, device=device).unsqueeze(1)

        # Convert spatial features to tensor (B, 8)
        spatial_tensor = torch.tensor(np.stack(spatial_features_batch), dtype=torch.float32, device=device)

        # Step 3: Apply spatial prior attention to P3 features (FEEDBACK LOOP)
        attended_p3 = self.spatial_attention(fpn_features[0], density_tensor)

        # Replace P3 with attended version
        attended_fpn = [attended_p3] + fpn_features[1:]

        # Step 4: Generate prototypes from attended P3
        prototypes = self.yolact.protonet(attended_fpn[0])

        # Step 5: Run prediction head on attended features
        class_preds, box_preds, mask_coeffs = self.yolact.prediction_head(attended_fpn)

        # Step 6: Generate anchors
        fpn_shapes = [(f.size(2), f.size(3)) for f in attended_fpn]
        input_size = images.size(2)
        if self.yolact._anchors is None or self.yolact._anchor_input_size != input_size:
            self.yolact._anchors = self.yolact._generate_anchors(fpn_shapes, input_size)
            self.yolact._anchor_input_size = input_size
        anchors = self.yolact._anchors.to(device)

        return {
            "class_preds": class_preds,
            "box_preds": box_preds,
            "mask_coeffs": mask_coeffs,
            "prototypes": prototypes,
            "anchors": anchors,
            "spatial_features": spatial_tensor,
            "density_maps": density_tensor,
            "gate_value": self.spatial_attention.get_gate_value(),
        }

    def _inference_forward(self, images: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Inference forward pass with full hybrid pipeline."""
        batch_size = images.size(0)
        device = images.device

        # Step 1: Get initial YOLACT detections for spatial reasoning
        fpn_features = self._extract_fpn_features(images)
        prototypes_initial = self.yolact.protonet(fpn_features[0])
        class_preds_init, box_preds_init, mask_coeffs_init = self.yolact.prediction_head(fpn_features)

        fpn_shapes = [(f.size(2), f.size(3)) for f in fpn_features]
        input_size = images.size(2)
        if self.yolact._anchors is None or self.yolact._anchor_input_size != input_size:
            self.yolact._anchors = self.yolact._generate_anchors(fpn_shapes, input_size)
            self.yolact._anchor_input_size = input_size
        anchors = self.yolact._anchors.to(device)

        # Get initial detections
        with torch.no_grad():
            initial_dets = self.yolact.detect(
                class_preds_init, box_preds_init, mask_coeffs_init, prototypes_initial, anchors
            )

        # Step 2: For each image, compute spatial features and refine
        all_refined = []

        for i in range(batch_size):
            det = initial_dets[i]
            boxes = det["boxes"]  # (K, 4)
            scores = det["scores"]  # (K,)

            if len(boxes) == 0:
                all_refined.append(det)
                continue

            # Convert boxes to numpy for spatial reasoning
            boxes_np = boxes.cpu().numpy()

            # Compute spatial features and density field (ML reasoning)
            spatial_feats, density = self._get_spatial_features_from_boxes(boxes_np)

            # Apply spatial attention to P3 (FEEDBACK LOOP)
            density_tensor = torch.tensor(density, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            attended_p3 = self.spatial_attention(fpn_features[0][i : i + 1], density_tensor)

            # Re-generate prototypes and re-predict on attended features
            attended_fpn_single = [attended_p3] + [f[i : i + 1] for f in fpn_features[1:]]
            prototypes_refined = self.yolact.protonet(attended_fpn_single[0])
            class_ref, box_ref, mask_ref = self.yolact.prediction_head(attended_fpn_single)

            # Get refined detections
            refined_dets = self.yolact.detect(class_ref, box_ref, mask_ref, prototypes_refined, anchors)
            refined_det = refined_dets[0]

            if len(refined_det["boxes"]) == 0:
                all_refined.append(det)
                continue

            # Step 3: Recalibrate confidence scores (FUSION)
            refined_boxes = refined_det["boxes"]
            refined_scores = refined_det["scores"].unsqueeze(1)  # (K, 1)

            # Extract visual features for each detection
            visual_feats = self._extract_visual_features(attended_p3, refined_boxes / input_size)  # normalize to [0, 1]

            # Expand spatial features to match number of detections
            spatial_tensor = (
                torch.tensor(spatial_feats, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .expand(len(refined_boxes), -1)
            )

            # Recalibrate
            recalibrated_scores = self.recalibrator(refined_scores, spatial_tensor, visual_feats).squeeze(1)

            # Build final detection dict
            refined_det["scores"] = recalibrated_scores
            all_refined.append(refined_det)

        return all_refined

    def count_parameters(self) -> Dict[str, Any]:
        """Count parameters for all components."""

        def _count(module: nn.Module) -> Dict[str, int]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return {"total": total, "trainable": trainable}

        yolact_params = self.yolact.count_parameters()

        return {
            "yolact": yolact_params,
            "spatial_attention": _count(self.spatial_attention),
            "recalibrator": _count(self.recalibrator),
            "visual_projector": _count(self.visual_projector),
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
