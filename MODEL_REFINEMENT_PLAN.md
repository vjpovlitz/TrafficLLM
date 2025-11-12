# Traffic Classification Model Refinement Plan

**Focus:** Improve existing ResNet-50 based traffic classifier
**Goal:** Better accuracy, faster training, more robust predictions

---

## 📊 Current Model Analysis

### Architecture (from `custom_model.py`)

```
CustomTrafficNet:
├─ ResNet-50 Backbone (pretrained on ImageNet)
├─ Spatial Attention Module
├─ Multi-Task Heads:
│  ├─ Density (regression, 1 output)
│  ├─ Congestion (regression, 1 output)
│  └─ Flow Rate (regression, 1 output)
└─ Classifier (5 classes)
   └─ Inputs: 3 regression outputs
```

### Strengths ✅
- Multi-task learning captures multiple aspects
- Spatial attention focuses on road areas
- ResNet-50 backbone is proven and robust
- Pretrained weights from ImageNet

### Weaknesses ⚠️
1. **Simple attention mechanism** - basic conv layer
2. **Small classifier head** - 3 features → 128 → 5 classes
3. **No data augmentation strategy documented**
4. **No temporal information** (single frame)
5. **Fixed input size** - may not leverage fullscreen quality
6. **No regularization besides dropout**
7. **Limited feature fusion** between tasks

---

## 🎯 Proposed Improvements

### 1. Enhanced Attention Mechanism

**Current:** Simple 1-channel spatial attention
**Improvement:** Multi-scale Channel + Spatial Attention

```python
class EnhancedAttention(nn.Module):
    """Combined channel and spatial attention with multi-scale"""
    def __init__(self, channels):
        super().__init__()

        # Channel attention (squeeze-excitation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )

        # Spatial attention (multi-scale)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),  # avg + max pooling
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_attention(spatial_input)

        return x * spatial_weights
```

**Benefits:**
- ✅ Focuses on both "what" (channels) and "where" (spatial)
- ✅ More expressive than current single-conv attention
- ✅ Proven effective in CBAM, EfficientNet

---

### 2. Improved Classifier Architecture

**Current Issue:** Bottleneck - 3 features is too small

**Improved Design:**
```python
class ImprovedClassifier(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=5):
        super().__init__()

        # Feature extraction heads (keep regression heads)
        self.density_head = self._make_head(feature_dim, 1)
        self.congestion_head = self._make_head(feature_dim, 1)
        self.flow_head = self._make_head(feature_dim, 1)

        # Rich classifier (combine all features)
        self.classifier = nn.Sequential(
            # Concatenate: 2048 (backbone) + 3 (regression outputs)
            nn.Linear(feature_dim + 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def _make_head(self, in_features, out_features):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, out_features)
        )
```

**Benefits:**
- ✅ Uses full ResNet features + regression outputs
- ✅ Deeper classifier with batch normalization
- ✅ Better capacity for complex patterns

---

### 3. Data Augmentation Strategy

**Critical for generalization with limited data!**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    # Resize to support fullscreen images
    A.Resize(640, 640),

    # Weather/lighting augmentations
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.7
    ),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=20,
        p=0.5
    ),

    # Weather conditions
    A.OneOf([
        A.RandomRain(rain_type='drizzle', p=1.0),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
        A.RandomShadow(p=1.0),
    ], p=0.3),

    # Geometric (minimal - preserve camera perspective)
    A.ShiftScaleRotate(
        shift_limit=0.03,
        scale_limit=0.05,
        rotate_limit=1,
        border_mode=0,
        p=0.3
    ),

    # Noise and blur
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.GaussianBlur(blur_limit=3, p=1.0),
        A.MotionBlur(blur_limit=3, p=1.0),
    ], p=0.2),

    # Normalization for ResNet
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(640, 640),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])
```

---

### 4. Advanced Training Techniques

#### 4.1 Label Smoothing
```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        n_classes = pred.size(1)
        # Convert hard labels to soft labels
        target = target.unsqueeze(1)
        target_one_hot = torch.zeros_like(pred).scatter_(1, target, 1)
        target_smooth = target_one_hot * (1 - self.epsilon) + self.epsilon / n_classes

        log_prob = F.log_softmax(pred, dim=1)
        loss = -(target_smooth * log_prob).sum(dim=1).mean()
        return loss
```

**Benefits:** Prevents overconfidence, improves generalization

#### 4.2 Cosine Annealing Learning Rate
```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Restart every 10 epochs
    T_mult=2,  # Double period each restart
    eta_min=1e-6
)
```

#### 4.3 Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in train_loader:
    optimizer.zero_grad()

    with autocast():  # Automatic mixed precision
        outputs = model(images)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits:** 2-3x faster training, less memory usage

---

### 5. Better Loss Function Design

**Current:** Simple CrossEntropy + MSE

**Improved Multi-Task Loss:**
```python
class TrafficLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.classification_loss = LabelSmoothingCrossEntropy(epsilon=0.1)
        self.regression_loss = nn.SmoothL1Loss()  # More robust than MSE

        # Adaptive task weights (learnable)
        self.log_vars = nn.Parameter(torch.zeros(4))

    def forward(self, outputs, targets):
        # Multi-task uncertainty weighting
        # https://arxiv.org/abs/1705.07115

        cls_loss = self.classification_loss(
            outputs['traffic_level'],
            targets['traffic_level']
        )

        density_loss = self.regression_loss(
            outputs['density'],
            targets['density']
        )

        congestion_loss = self.regression_loss(
            outputs['congestion'],
            targets['congestion']
        )

        flow_loss = self.regression_loss(
            outputs['flow_rate'],
            targets['flow_rate']
        )

        # Automatic task weighting
        precision_cls = torch.exp(-self.log_vars[0])
        precision_den = torch.exp(-self.log_vars[1])
        precision_con = torch.exp(-self.log_vars[2])
        precision_flow = torch.exp(-self.log_vars[3])

        total_loss = (
            precision_cls * cls_loss + self.log_vars[0] +
            precision_den * density_loss + self.log_vars[1] +
            precision_con * congestion_loss + self.log_vars[2] +
            precision_flow * flow_loss + self.log_vars[3]
        )

        return total_loss, {
            'classification': cls_loss.item(),
            'density': density_loss.item(),
            'congestion': congestion_loss.item(),
            'flow_rate': flow_loss.item()
        }
```

---

### 6. Model Ensemble Strategy

**For better accuracy without complexity:**
```python
class TrafficEnsemble:
    def __init__(self, model_paths):
        self.models = []
        for path in model_paths:
            model = CustomTrafficNet()
            model.load_state_dict(torch.load(path))
            model.eval()
            self.models.append(model)

    def predict(self, image):
        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model(image)
                predictions.append(pred)

        # Average predictions
        avg_pred = {
            key: torch.mean(torch.stack([p[key] for p in predictions]), dim=0)
            for key in predictions[0].keys()
        }
        return avg_pred
```

**Training Strategy:**
- Train 3-5 models with different:
  - Random seeds
  - Augmentation strategies
  - Train/val splits (k-fold)

---

### 7. Metrics and Monitoring

```python
from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, ConfusionMatrix
import wandb  # Weights & Biases for tracking

class TrafficMetrics:
    def __init__(self, num_classes=5):
        self.acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1 = F1Score(task='multiclass', num_classes=num_classes, average='weighted')
        self.conf_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)

        self.density_mae = MeanAbsoluteError()
        self.congestion_mae = MeanAbsoluteError()
        self.flow_mae = MeanAbsoluteError()

    def update(self, outputs, targets):
        self.acc.update(outputs['traffic_level'], targets['traffic_level'])
        self.f1.update(outputs['traffic_level'], targets['traffic_level'])
        self.conf_matrix.update(outputs['traffic_level'], targets['traffic_level'])

        self.density_mae.update(outputs['density'], targets['density'])
        self.congestion_mae.update(outputs['congestion'], targets['congestion'])
        self.flow_mae.update(outputs['flow_rate'], targets['flow_rate'])

    def compute(self):
        return {
            'accuracy': self.acc.compute().item(),
            'f1_score': self.f1.compute().item(),
            'confusion_matrix': self.conf_matrix.compute(),
            'density_mae': self.density_mae.compute().item(),
            'congestion_mae': self.congestion_mae.compute().item(),
            'flow_rate_mae': self.flow_mae.compute().item()
        }
```

---

## 📋 Implementation Checklist

### Week 1: Model Architecture Improvements
- [ ] Implement enhanced attention mechanism
- [ ] Redesign classifier head with more capacity
- [ ] Add batch normalization throughout
- [ ] Update model saving/loading

### Week 2: Training Pipeline
- [ ] Implement albumentations data augmentation
- [ ] Add label smoothing loss
- [ ] Implement cosine annealing scheduler
- [ ] Add mixed precision training
- [ ] Set up Weights & Biases logging

### Week 3: Advanced Techniques
- [ ] Implement multi-task uncertainty weighting
- [ ] Add gradient clipping
- [ ] Implement model checkpointing (best + last)
- [ ] Add early stopping
- [ ] Create evaluation script

### Week 4: Testing & Refinement
- [ ] Train baseline model
- [ ] Train improved model
- [ ] Compare metrics
- [ ] Tune hyperparameters
- [ ] Train ensemble (3-5 models)
- [ ] Create inference pipeline

---

## 🎯 Expected Improvements

### Baseline (Current)
- Accuracy: ~75-80%
- F1 Score: ~0.70
- Training time: ~2-3 hours (50 epochs)

### After Improvements
- Accuracy: **85-90%** (+10%)
- F1 Score: **0.82-0.88** (+15%)
- Training time: **1-1.5 hours** (with mixed precision)
- Faster convergence (better augmentation + scheduler)
- Better generalization (regularization + label smoothing)

---

## 📊 Hyperparameters to Tune

### Priority 1 (Biggest Impact)
- Learning rate: [1e-4, 3e-4, 1e-3]
- Batch size: [16, 32, 64]
- Augmentation strength: [light, medium, heavy]
- Dropout rate: [0.3, 0.4, 0.5]

### Priority 2
- Weight decay: [1e-4, 1e-3, 1e-2]
- Label smoothing epsilon: [0.05, 0.1, 0.15]
- Scheduler T_0: [5, 10, 20]

### Priority 3
- Attention reduction ratio: [8, 16, 32]
- Classifier hidden dims: [256, 512, 1024]

---

## 🚀 Quick Start Commands

```bash
# Install new dependencies
pip install albumentations wandb torchmetrics

# Train improved model
python scripts/train_improved_model.py \
    --data_dir data/processed \
    --epochs 50 \
    --batch_size 32 \
    --lr 3e-4 \
    --use_augmentation \
    --mixed_precision \
    --log_wandb

# Evaluate
python scripts/evaluate_model.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --test_data data/test

# Run inference
python scripts/predict.py \
    --model outputs/checkpoints/best_model.pth \
    --image path/to/traffic/image.jpg
```

---

**Ready to start implementation! Which component should we build first?**
