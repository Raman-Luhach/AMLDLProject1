"""Confidence Recalibrator for Hybrid Detection.

Small MLP that recalibrates detection confidence scores by fusing:
    1. Original detection score (1-dim) from YOLACT
    2. Spatial context features (8-dim) from Spatial Reasoning Engine
    3. ROI-pooled visual features (64-dim) from FPN

This makes the ML-DL interaction DIFFERENTIABLE — gradients flow from
the recalibrated scores back through both the visual features (improving
the backbone) and influence the spatial feature weighting.

Architecture:
    Input: [score(1) + spatial(8) + visual(64)] = 73-dim
    → Linear(73, 128) + ReLU + Dropout(0.1)
    → Linear(128, 64) + ReLU + Dropout(0.1)
    → Linear(64, 1) + Sigmoid
    → recalibrated_score
"""

import torch
import torch.nn as nn


class ConfidenceRecalibrator(nn.Module):
    """MLP-based confidence recalibration using multi-modal features.

    Learns to improve detection confidence by combining the original
    YOLACT score with spatial context and visual features.

    Args:
        spatial_dim: Dimension of spatial feature vector (default: 8).
        visual_dim: Dimension of ROI-pooled visual embedding (default: 64).
        hidden_dims: Hidden layer dimensions (default: [128, 64]).
        dropout: Dropout rate for regularization (default: 0.1).
    """

    def __init__(
        self,
        spatial_dim: int = 8,
        visual_dim: int = 64,
        hidden_dims: list = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.spatial_dim = spatial_dim
        self.visual_dim = visual_dim
        input_dim = 1 + spatial_dim + visual_dim  # score + spatial + visual = 73

        # Build MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values for stable start."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize last layer bias to 0 so initial output ≈ 0.5
        # (sigmoid(0) = 0.5, preserving original scores initially)
        last_linear = None
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        scores: torch.Tensor,
        spatial_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """Recalibrate detection confidences.

        Args:
            scores: Original detection scores, shape (N, 1).
            spatial_features: Spatial context features, shape (N, 8).
            visual_features: ROI-pooled visual embeddings, shape (N, 64).

        Returns:
            Recalibrated scores, shape (N, 1) in [0, 1].
        """
        # Concatenate all features
        x = torch.cat([scores, spatial_features, visual_features], dim=-1)

        # Forward through MLP
        logits = self.mlp(x)

        # Blend: use recalibrator output as an adjustment to original score
        # This residual design helps stability during early training
        adjustment = torch.sigmoid(logits)

        # Final score = weighted blend of original and recalibrated
        # Start with mostly original (adjustment ≈ 0.5 initially)
        recalibrated = 0.5 * scores + 0.5 * adjustment

        return recalibrated
