"""Spatial Prior Attention Module for Hybrid Detection.

Converts spatial density priors from the ML Spatial Reasoning Engine
into a learnable attention map that modulates FPN features. This creates
the FEEDBACK LOOP from ML → DL: spatial reasoning improves feature extraction.

The module includes a learnable 'gate' parameter that controls how much
the model trusts the spatial prior. During training, if the spatial prior
is helpful, the gate opens (gate → 1); if not, it stays closed (gate → 0).

Architecture:
    density_map (1, H, W)
        → Conv2d(1, 16, 3) + ReLU
        → Conv2d(16, 1, 3) + Sigmoid
        → attention_map (1, H, W)

    output = fpn_feature * (1 + sigmoid(gate) * attention_map)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPriorAttention(nn.Module):
    """Converts density field into learnable spatial attention for FPN features.

    This module is the key innovation enabling the ML→DL feedback loop.
    The density field from SpatialReasoningEngine encodes where objects
    are expected based on shelf structure. This module learns to convert
    that prior into a feature-space attention map.

    Args:
        conv_channels: Intermediate channels in density encoder.
        gate_init: Initial value for the trust gate (0-1 range before sigmoid).
    """

    def __init__(self, conv_channels: int = 16, gate_init: float = 0.1) -> None:
        super().__init__()

        # Density field encoder: converts raw density → attention map
        self.density_encoder = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Learnable gate parameter — controls trust in spatial prior
        # Initialized small (0.1) so model starts mostly ignoring prior
        # and gradually learns to use it if beneficial
        self.gate = nn.Parameter(torch.tensor(gate_init))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolution weights."""
        for m in self.density_encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, fpn_feature: torch.Tensor, density_map: torch.Tensor) -> torch.Tensor:
        """Apply spatial prior attention to FPN features.

        The output is a residual modulation:
            output = fpn_feature * (1 + gate * attention_map)

        This ensures that with gate=0, the module is an identity function
        (no effect), allowing the model to gracefully learn whether the
        spatial prior is useful.

        Args:
            fpn_feature: FPN feature map of shape (B, C, H, W).
            density_map: Spatial density prior of shape (B, 1, H_d, W_d).
                         Will be resized to match fpn_feature spatial dims.

        Returns:
            Modulated feature map of shape (B, C, H, W).
        """
        B, C, H, W = fpn_feature.shape

        # Resize density map to match FPN spatial dimensions
        if density_map.shape[-2:] != (H, W):
            density_map = F.interpolate(density_map, size=(H, W), mode="bilinear", align_corners=False)

        # Encode density into attention map
        attention = self.density_encoder(density_map)  # (B, 1, H, W)

        # Apply gated residual attention
        gate = torch.sigmoid(self.gate)
        output = fpn_feature * (1.0 + gate * attention)

        return output

    def get_gate_value(self) -> float:
        """Return current gate value (after sigmoid) for monitoring."""
        return torch.sigmoid(self.gate).item()
