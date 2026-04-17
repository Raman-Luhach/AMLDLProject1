"""Convolutional Block Attention Module (CBAM) for feature refinement.

Implements the CBAM attention mechanism from Woo et al. (2018) which
sequentially applies channel attention and spatial attention to refine
feature maps. Used after FPN smoothing convolutions to enhance feature
selection for dense object detection.

Channel attention captures inter-channel relationships ("what" to attend to).
Spatial attention captures inter-spatial relationships ("where" to attend).

Reference:
    Woo, S., Park, J., Lee, J.Y. and Kweon, I.S., 2018.
    CBAM: Convolutional Block Attention Module. ECCV 2018.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention sub-module.

    Aggregates spatial information using both average-pooling and max-pooling,
    then processes through a shared MLP to produce channel attention weights.

    M_c(F) = sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))

    Args:
        channels: Number of input/output channels.
        reduction: Channel reduction ratio for the MLP bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.

        Args:
            x: Input feature map (B, C, H, W).

        Returns:
            Channel-attended feature map (B, C, H, W).
        """
        b, c, _, _ = x.size()
        # Global average and max pooling across spatial dims
        avg_out = x.mean(dim=(2, 3))  # (B, C)
        max_out = x.amax(dim=(2, 3))  # (B, C)
        # Shared MLP
        attn = torch.sigmoid(self.mlp(avg_out) + self.mlp(max_out))  # (B, C)
        return x * attn.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention sub-module.

    Aggregates channel information using average-pooling and max-pooling
    along the channel axis, then applies a convolution to produce spatial
    attention weights.

    M_s(F) = sigmoid(conv7x7([AvgPool(F); MaxPool(F)]))

    Args:
        kernel_size: Convolution kernel size (default 7).
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention.

        Args:
            x: Input feature map (B, C, H, W).

        Returns:
            Spatially-attended feature map (B, C, H, W).
        """
        avg_out = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        max_out = x.amax(dim=1, keepdim=True)  # (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attn = torch.sigmoid(self.conv(combined))  # (B, 1, H, W)
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Sequentially applies channel attention then spatial attention:
        F' = M_s(M_c(F) * F) * (M_c(F) * F)

    Args:
        channels: Number of input/output channels.
        reduction: Channel reduction ratio for the channel attention MLP.
        kernel_size: Kernel size for the spatial attention convolution.
    """

    def __init__(
        self, channels: int, reduction: int = 16, kernel_size: int = 7
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CBAM (channel then spatial attention).

        Args:
            x: Input feature map (B, C, H, W).

        Returns:
            Refined feature map (B, C, H, W).
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
