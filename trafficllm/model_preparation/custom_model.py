import torch
import torch.nn as nn
import torchvision.models as models

class SpatialAttention(nn.Module):
    """Spatial attention module to focus on important areas in images"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Generate attention map
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        attention = self.conv(avg_pool)
        return x * attention

class CustomTrafficNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # Load pretrained ResNet backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Traffic analysis modules
        self.spatial_attention = SpatialAttention()
        
        # Multiple heads for different traffic parameters
        self.traffic_heads = nn.ModuleDict({
            'density': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(2048, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1)
            ),
            'congestion': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(2048, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1)
            ),
            'flow_rate': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(2048, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1)
            )
        })
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(3, 128),  # 3 features from traffic heads
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply spatial attention
        attended = self.spatial_attention(features)
        
        # Get traffic parameters
        traffic_params = {}
        for name, head in self.traffic_heads.items():
            traffic_params[name] = head(attended)
        
        # Combine parameters for final classification
        combined_params = torch.cat([
            traffic_params['density'],
            traffic_params['congestion'],
            traffic_params['flow_rate']
        ], dim=1)
        
        # Final traffic level classification
        traffic_level = self.classifier(combined_params)
        
        return {
            'traffic_level': traffic_level,
            'density': traffic_params['density'],
            'congestion': traffic_params['congestion'],
            'flow_rate': traffic_params['flow_rate']
        } 