"""
Advanced Attention Mechanisms for Traffic Analysis
Implements CBAM-style Channel and Spatial Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM) - focuses on 'what' is important"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling branch
        avg_out = self.mlp(self.avg_pool(x))
        # Max pooling branch
        max_out = self.mlp(self.max_pool(x))
        # Combine
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM) - focuses on 'where' is important"""

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate along channel dimension
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # Generate spatial attention map
        attention = self.sigmoid(self.conv(x_cat))

        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines Channel and Spatial Attention

    Reference: https://arxiv.org/abs/1807.06521
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()

        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention first
        x = self.channel_attention(x)
        # Then apply spatial attention
        x = self.spatial_attention(x)
        return x


class EnhancedSpatialAttention(nn.Module):
    """
    Enhanced Spatial Attention for Traffic Analysis
    Multi-scale spatial attention with road region focus
    """

    def __init__(self, channels):
        super().__init__()

        # Multi-scale convolutions for different receptive fields
        self.conv1x1 = nn.Conv2d(2, 1, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.conv7x7 = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        # Combine multi-scale features
        self.combine = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)

        # Multi-scale attention maps
        attention_1x1 = self.conv1x1(pooled)
        attention_3x3 = self.conv3x3(pooled)
        attention_7x7 = self.conv7x7(pooled)

        # Combine multi-scale features
        multi_scale = torch.cat([attention_1x1, attention_3x3, attention_7x7], dim=1)
        attention = self.combine(multi_scale)

        return x * attention


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Lightweight channel attention

    Reference: https://arxiv.org/abs/1709.01507
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze: global spatial information
        y = self.squeeze(x).view(b, c)

        # Excitation: channel-wise scaling
        y = self.excitation(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


# Backward compatibility - keep old name
class SimpleSpatialAttention(nn.Module):
    """Original simple spatial attention from current model"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        attention = self.conv(avg_pool)
        return x * attention


if __name__ == "__main__":
    # Test attention modules
    print("Testing Attention Modules...")

    # Create dummy input
    batch_size = 4
    channels = 2048
    height, width = 28, 28
    x = torch.randn(batch_size, channels, height, width)

    print(f"Input shape: {x.shape}")

    # Test CBAM
    cbam = CBAM(channels, reduction=16)
    out = cbam(x)
    print(f"CBAM output shape: {out.shape}")
    assert out.shape == x.shape, "CBAM output shape mismatch!"

    # Test Enhanced Spatial Attention
    enhanced_spatial = EnhancedSpatialAttention(channels)
    out = enhanced_spatial(x)
    print(f"Enhanced Spatial output shape: {out.shape}")
    assert out.shape == x.shape, "Enhanced Spatial output shape mismatch!"

    # Test SE Block
    se = SEBlock(channels, reduction=16)
    out = se(x)
    print(f"SE Block output shape: {out.shape}")
    assert out.shape == x.shape, "SE Block output shape mismatch!"

    print("\n✅ All attention modules working correctly!")
