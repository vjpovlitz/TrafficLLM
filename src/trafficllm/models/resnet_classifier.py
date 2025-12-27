"""
Enhanced Traffic Classification Model
Improved version with better attention and classifier
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .attention_modules import CBAM, EnhancedSpatialAttention, SEBlock


class EnhancedTrafficNet(nn.Module):
    """
    Enhanced Traffic Classification Model

    Enhancements:
    1. CBAM attention (channel + spatial) instead of simple spatial
    2. Deeper classifier using full ResNet features
    3. Better multi-task head design with BatchNorm
    4. Optional feature fusion between tasks
    """

    def __init__(
        self,
        num_classes=5,
        attention_type='cbam',  # 'cbam', 'enhanced_spatial', 'se', or 'simple'
        feature_fusion=True,
        dropout=0.4,
        pretrained=True
    ):
        super().__init__()

        self.feature_fusion = feature_fusion

        # Load pretrained ResNet-50 backbone (remove final FC layers)
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Output: 2048 x 7 x 7

        # Attention mechanism selection
        if attention_type == 'cbam':
            self.attention = CBAM(channels=2048, reduction=16)
        elif attention_type == 'enhanced_spatial':
            self.attention = EnhancedSpatialAttention(channels=2048)
        elif attention_type == 'se':
            self.attention = SEBlock(channels=2048, reduction=16)
        else:  # simple (original)
            self.attention = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=7, padding=3),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

        # Improved regression heads with BatchNorm
        self.density_head = self._make_regression_head(2048, dropout)
        self.congestion_head = self._make_regression_head(2048, dropout)
        self.flow_rate_head = self._make_regression_head(2048, dropout)

        # Improved classification head
        if feature_fusion:
            # Use both backbone features AND regression outputs
            self.classifier = self._make_classifier_head(
                in_features=2048 + 3,  # ResNet features + 3 regression values
                num_classes=num_classes,
                dropout=dropout
            )
        else:
            # Use only regression outputs (original approach)
            self.classifier = self._make_classifier_head(
                in_features=3,
                num_classes=num_classes,
                dropout=dropout
            )

        # Initialize weights
        self._initialize_weights()

    def _make_regression_head(self, in_features, dropout=0.4):
        """Create improved regression head with BatchNorm"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def _make_classifier_head(self, in_features, num_classes, dropout=0.4):
        """Create deeper classification head"""
        if in_features == 3:
            # Simpler head for regression-only input
            return nn.Sequential(
                nn.Linear(3, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )
        else:
            # Rich head for fused features
            return nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.75),  # Slightly less dropout in later layers

                nn.Linear(256, num_classes)
            )

    def _initialize_weights(self):
        """Initialize weights for new layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features from ResNet backbone
        features = self.backbone(x)  # (B, 2048, H, W)

        # Apply attention
        attended_features = self.attention(features)  # (B, 2048, H, W)

        # Get traffic parameters from regression heads
        density = self.density_head(attended_features)
        congestion = self.congestion_head(attended_features)
        flow_rate = self.flow_rate_head(attended_features)

        # Combine regression outputs
        regression_params = torch.cat([density, congestion, flow_rate], dim=1)  # (B, 3)

        # Classification
        if self.feature_fusion:
            # Global average pooling of attended features
            global_features = F.adaptive_avg_pool2d(attended_features, 1).flatten(1)  # (B, 2048)

            # Concatenate features and regression outputs
            combined = torch.cat([global_features, regression_params], dim=1)  # (B, 2051)

            # Final classification
            traffic_level = self.classifier(combined)
        else:
            # Use only regression outputs
            traffic_level = self.classifier(regression_params)

        return {
            'traffic_level': traffic_level,
            'density': density,
            'congestion': congestion,
            'flow_rate': flow_rate
        }

    def get_feature_maps(self, x):
        """Get intermediate feature maps for visualization"""
        features = self.backbone(x)
        attended = self.attention(features)
        return {
            'backbone_features': features,
            'attended_features': attended
        }


class EnsembleTrafficNet(nn.Module):
    """
    Ensemble of multiple TrafficNet models for improved accuracy
    """

    def __init__(self, model_configs):
        """
        Args:
            model_configs: List of dicts with model configuration
                Example: [{'attention_type': 'cbam', 'feature_fusion': True},
                         {'attention_type': 'se', 'feature_fusion': True}]
        """
        super().__init__()

        self.models = nn.ModuleList([
            EnhancedTrafficNet(**config)
            for config in model_configs
        ])

    def forward(self, x):
        # Get predictions from all models
        all_predictions = [model(x) for model in self.models]

        # Average predictions
        averaged = {}
        for key in all_predictions[0].keys():
            stacked = torch.stack([pred[key] for pred in all_predictions])
            averaged[key] = torch.mean(stacked, dim=0)

        return averaged

    def load_ensemble_weights(self, checkpoint_paths):
        """Load weights for each model in ensemble"""
        assert len(checkpoint_paths) == len(self.models), \
            "Number of checkpoints must match number of models"

        for model, path in zip(self.models, checkpoint_paths):
            state_dict = torch.load(path, map_location='cpu')
            model.load_state_dict(state_dict)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("="*80)
    print("Testing Enhanced Traffic Model")
    print("="*80)

    # Create dummy input (fullscreen size)
    batch_size = 4
    x = torch.randn(batch_size, 3, 640, 640)
    print(f"\nInput shape: {x.shape}")

    # Test different attention types
    attention_types = ['cbam', 'enhanced_spatial', 'se', 'simple']

    for att_type in attention_types:
        print(f"\n{'='*80}")
        print(f"Testing with {att_type.upper()} attention")
        print(f"{'='*80}")

        # Create model
        model = EnhancedTrafficNet(
            num_classes=5,
            attention_type=att_type,
            feature_fusion=True,
            dropout=0.4
        )

        # Count parameters
        params = count_parameters(model)
        print(f"Trainable parameters: {params:,}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(x)

        # Check outputs
        print(f"\nOutputs:")
        for key, value in outputs.items():
            print(f"  {key}: {value.shape}")

        # Validate output shapes
        assert outputs['traffic_level'].shape == (batch_size, 5), "Traffic level shape mismatch!"
        assert outputs['density'].shape == (batch_size, 1), "Density shape mismatch!"
        assert outputs['congestion'].shape == (batch_size, 1), "Congestion shape mismatch!"
        assert outputs['flow_rate'].shape == (batch_size, 1), "Flow rate shape mismatch!"

    print(f"\n{'='*80}")
    print("✅ All model tests passed!")
    print(f"{'='*80}")
