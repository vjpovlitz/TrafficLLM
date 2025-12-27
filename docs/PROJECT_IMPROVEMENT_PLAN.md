# TrafficLLM Project Improvement Plan

**Date:** November 11, 2025
**Branch:** Nov25Work

## 📊 Current Project Analysis

### Existing Components

#### 1. **Data Collection System** (`TrafficLLM/scripts/data_collection/`)
- ✅ Enhanced pipeline with metadata collection
- ✅ Image validation and quality assessment
- ✅ Weather API integration
- 🆕 **NEW:** Fullscreen video-only capture (verified working)

#### 2. **ML Model** (`trafficllm/model_preparation/`)
**Current Architecture:**
- **Base:** ResNet-50 pretrained backbone
- **Attention:** Spatial attention module
- **Multi-task heads:**
  - Density prediction (regression)
  - Congestion prediction (regression)
  - Flow rate prediction (regression)
  - Traffic level classification (5 classes)

**Limitations:**
- ❌ No object detection (vehicles, pedestrians)
- ❌ No vehicle counting
- ❌ Limited temporal analysis (single frame)
- ❌ No lane-specific analysis
- ⚠️ Simple attention mechanism

#### 3. **Traditional CV Analysis** (`trafficllm/data_processing/traffic_analysis.py`)
- Edge detection-based density estimation
- Grayscale standard deviation for congestion
- Threshold-based traffic level classification

**Limitations:**
- ❌ Not learning-based
- ❌ Fixed thresholds
- ❌ No vehicle differentiation

---

## 🎯 Improvement Roadmap

### Phase 1: Infrastructure & Organization (Week 1)

#### 1.1 Integrate Fullscreen Capture into Pipeline

**Action Items:**
- [ ] Copy working fullscreen script to `scripts/data_collection/`
- [ ] Integrate with `enhanced_pipeline.py`
- [ ] Add fullscreen mode flag to camera config
- [ ] Update `CameraManager` class with fullscreen support
- [ ] Add screenshot size comparison logging

**New Config Structure:**
```json
{
  "cameras": {
    "us_50_sandy_point": {
      "name": "US_50_AT_SANDY_POINT",
      "url": "https://mdta.maryland.gov/...",
      "fullscreen_mode": true,
      "capture_quality": "high"
    }
  }
}
```

#### 1.2 Reorganize Project Structure

**Current Issues:**
- Duplicate folder structure (outer TrafficCamLLM / inner TrafficLLM)
- Scripts scattered between `/scripts/testing/` and `/TrafficLLM/scripts/`
- Test files mixed with production code

**Proposed New Structure:**
```
TrafficCamLLM/
├── CLAUDE.md
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
│
├── config/                          # ALL configuration files
│   ├── camera_config.json
│   ├── model_config.yaml
│   └── training_config.yaml
│
├── data/                            # Data storage (gitignored)
│   ├── raw/                         # Raw screenshots
│   ├── processed/                   # Preprocessed images
│   ├── annotated/                   # Labeled data for YOLO
│   ├── datasets/                    # Train/val/test splits
│   └── metadata/                    # JSON metadata files
│
├── trafficllm/                      # Main Python package
│   ├── __init__.py
│   │
│   ├── data_collection/             # Data capture (MOVED from scripts/)
│   │   ├── __init__.py
│   │   ├── camera_manager.py
│   │   ├── fullscreen_capture.py   # NEW
│   │   ├── image_validator.py
│   │   ├── metadata_collector.py
│   │   └── pipeline.py
│   │
│   ├── data_processing/             # Preprocessing & augmentation
│   │   ├── __init__.py
│   │   ├── preprocessor.py
│   │   ├── augmentations.py
│   │   └── dataset_builder.py
│   │
│   ├── models/                      # All model definitions
│   │   ├── __init__.py
│   │   ├── resnet_classifier.py    # Current ResNet model
│   │   ├── yolo_detector.py        # NEW: YOLO for object detection
│   │   ├── hybrid_model.py         # NEW: Combined detection + classification
│   │   └── attention_modules.py
│   │
│   ├── training/                    # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   │
│   ├── inference/                   # NEW: Inference & deployment
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── video_analyzer.py
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── logging.py
│       ├── visualization.py
│       └── config_loader.py
│
├── scripts/                         # Standalone scripts
│   ├── collect_data.py              # CLI for data collection
│   ├── train_model.py               # CLI for training
│   ├── annotate_data.py             # Annotation helper
│   └── evaluate_model.py
│
├── notebooks/                       # Jupyter notebooks for experiments
│   ├── exploratory_analysis.ipynb
│   ├── model_evaluation.ipynb
│   └── visualization.ipynb
│
├── tests/                           # Unit tests
│   ├── test_data_collection.py
│   ├── test_models.py
│   └── test_preprocessing.py
│
├── docs/                            # Documentation
│   ├── data_collection.md
│   ├── model_architecture.md
│   └── training_guide.md
│
└── outputs/                         # Training outputs (gitignored)
    ├── checkpoints/
    ├── logs/
    └── predictions/
```

---

### Phase 2: YOLO Integration for Object Detection (Week 2-3)

#### 2.1 Research: YOLO Model Selection

**Options:**

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| **YOLOv8n** (Nano) | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Real-time, edge devices |
| **YOLOv8s** (Small) | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | **RECOMMENDED** - Best balance |
| **YOLOv8m** (Medium) | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | High accuracy needed |
| **YOLOv9** | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Latest, best accuracy |
| **YOLOv11** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | **CUTTING EDGE** |

**Recommendation: YOLOv8s or YOLOv11 Small**
- Fast enough for real-time (60+ FPS on GPU)
- Excellent accuracy for vehicle detection
- Pre-trained on COCO (includes car, truck, bus, motorcycle)
- Easy fine-tuning on custom traffic data

#### 2.2 YOLO Implementation Plan

**Step 1: Install Ultralytics**
```bash
pip install ultralytics
```

**Step 2: Create YOLO Wrapper**
```python
# trafficllm/models/yolo_detector.py
from ultralytics import YOLO

class TrafficYOLODetector:
    def __init__(self, model_size='s', pretrained=True):
        if pretrained:
            self.model = YOLO(f'yolov8{model_size}.pt')
        else:
            self.model = YOLO(f'yolov8{model_size}.yaml')

        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

    def detect_vehicles(self, image):
        results = self.model(image)
        vehicles = []

        for r in results:
            for box in r.boxes:
                if r.names[int(box.cls)] in self.vehicle_classes:
                    vehicles.append({
                        'class': r.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].tolist(),
                        'center': box.xywh[0][:2].tolist()
                    })

        return vehicles

    def count_vehicles(self, image):
        vehicles = self.detect_vehicles(image)
        return {
            'total': len(vehicles),
            'by_type': {
                vtype: sum(1 for v in vehicles if v['class'] == vtype)
                for vtype in self.vehicle_classes
            }
        }
```

**Step 3: Data Annotation**
- Use Roboflow or Label Studio for annotation
- Export in YOLO format
- Store in `data/annotated/`

**Step 4: Fine-tune on Traffic Data**
```python
# Fine-tune YOLO on traffic camera data
model = YOLO('yolov8s.pt')
model.train(
    data='traffic_dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='traffic_yolo'
)
```

---

### Phase 3: Hybrid Model Architecture (Week 4-5)

#### 3.1 Combine YOLO + ResNet

**New Architecture:**
```
Input Image (1728x879)
    │
    ├─→ YOLO Detector ──→ Vehicle Detections
    │                      ├─ Count per class
    │                      ├─ Bounding boxes
    │                      ├─ Lane distribution
    │                      └─ Speed estimation (optional)
    │
    └─→ ResNet-50 Backbone ──→ Global Features
            │
            └─→ Spatial Attention
                    │
                    ├─→ Density Head
                    ├─→ Congestion Head
                    ├─→ Flow Rate Head
                    └─→ Classification Head
                            │
                            ├─ Vehicle counts (from YOLO)
                            ├─ Global features (from ResNet)
                            └─→ Final Traffic Level (0-4)
```

#### 3.2 Enhanced Features

**Add to Model:**
1. **Temporal Module** - Analyze sequences of frames
2. **Lane Detector** - Identify and analyze individual lanes
3. **Speed Estimator** - Track vehicles across frames
4. **Weather Classifier** - Detect rain, snow, fog from video

**Example Hybrid Model:**
```python
class HybridTrafficModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Object detection
        self.yolo = TrafficYOLODetector('s')

        # Scene understanding
        self.resnet_backbone = ResNet50Backbone()
        self.attention = SpatialAttention()

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 64, 512),  # ResNet features + YOLO stats
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Multi-task heads
        self.heads = MultiTaskHeads(512)

    def forward(self, x):
        # 1. Detect vehicles with YOLO
        detections = self.yolo.detect_vehicles(x)
        vehicle_features = self.extract_vehicle_features(detections)

        # 2. Extract global features with ResNet
        global_features = self.resnet_backbone(x)
        global_features = self.attention(global_features)

        # 3. Fuse features
        combined = torch.cat([global_features, vehicle_features], dim=1)
        fused = self.fusion(combined)

        # 4. Multi-task prediction
        return self.heads(fused)
```

---

### Phase 4: Advanced Improvements (Week 6+)

#### 4.1 Temporal Analysis

**Implement LSTM/Transformer for video:**
```python
class TemporalTrafficAnalyzer(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2):
        super().__init__()

        self.frame_encoder = HybridTrafficModel()  # Process each frame
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.predictor = nn.Linear(hidden_size, num_classes)

    def forward(self, video_frames):  # (B, T, C, H, W)
        batch_size, num_frames = video_frames.shape[:2]

        # Encode each frame
        frame_features = []
        for t in range(num_frames):
            feat = self.frame_encoder(video_frames[:, t])
            frame_features.append(feat)

        frame_features = torch.stack(frame_features, dim=1)  # (B, T, 512)

        # Temporal modeling
        lstm_out, _ = self.lstm(frame_features)

        # Predict traffic trend
        return self.predictor(lstm_out[:, -1])  # Last timestep
```

#### 4.2 Data Augmentation Strategies

```python
# trafficllm/data_processing/augmentations.py
import albumentations as A

traffic_augmentations = A.Compose([
    # Weather conditions
    A.RandomRain(p=0.3),
    A.RandomFog(p=0.2),
    A.RandomShadow(p=0.3),

    # Lighting
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.3),

    # Time of day simulation
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.5
    ),

    # Geometric (careful - preserve vehicle aspect ratios)
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.05,
        rotate_limit=2,  # Small rotation only
        p=0.3
    ),

    # Noise
    A.GaussNoise(p=0.2),
    A.ISONoise(p=0.2),
])
```

#### 4.3 Evaluation Metrics

**Define comprehensive metrics:**
```python
# trafficllm/training/metrics.py

class TrafficMetrics:
    def __init__(self):
        self.metrics = {
            # Detection metrics (YOLO)
            'vehicle_detection_mAP': MeanAveragePrecision(),
            'vehicle_recall': Recall(),
            'vehicle_precision': Precision(),

            # Classification metrics
            'traffic_level_accuracy': Accuracy(),
            'traffic_level_f1': F1Score(num_classes=5),

            # Regression metrics
            'density_mae': MeanAbsoluteError(),
            'congestion_mae': MeanAbsoluteError(),
            'flow_rate_mae': MeanAbsoluteError(),

            # Custom metrics
            'vehicle_count_error': VehicleCountError(),
        }
```

---

## 🎯 Immediate Next Steps (This Week)

### Priority 1: Integrate Fullscreen Capture ✅
- [x] Working fullscreen capture verified
- [ ] Move script to production location
- [ ] Update pipeline integration
- [ ] Test with multiple cameras

### Priority 2: File Reorganization
- [ ] Create new directory structure
- [ ] Move files to new locations
- [ ] Update all import statements
- [ ] Update documentation

### Priority 3: YOLO Prototype
- [ ] Install ultralytics
- [ ] Test YOLOv8s on sample traffic images
- [ ] Evaluate detection quality
- [ ] Plan annotation strategy

---

## 📝 Research Tasks

### YOLO Fine-tuning Research
- [ ] Review YOLOv8 documentation
- [ ] Find traffic camera datasets (UA-DETRAC, KITTI, BDD100K)
- [ ] Research vehicle tracking algorithms (DeepSORT, ByteTrack)
- [ ] Explore temporal models for video (TimeSformer, VideoMAE)

### Model Architecture Research
- [ ] Vision Transformers (ViT) vs CNN for traffic
- [ ] Efficient attention mechanisms (EfficientNet, CoAtNet)
- [ ] Multi-scale feature fusion (FPN, PANet)
- [ ] Self-supervised pretraining for traffic domain

---

## 📊 Success Metrics

### Data Quality
- ✅ Fullscreen capture: 1.2-2.2 MB images (vs 682 KB)
- ✅ Video-only content (no webpage elements)
- 🎯 Target: 10,000+ annotated traffic images

### Model Performance
- 🎯 Vehicle Detection mAP > 0.85
- 🎯 Traffic Classification Accuracy > 90%
- 🎯 Inference Speed > 30 FPS (YOLOv8s)
- 🎯 Real-time capable (<50ms per frame)

### Code Quality
- 🎯 100% type hints
- 🎯 80%+ test coverage
- 🎯 Organized modular structure
- 🎯 Comprehensive documentation

---

## 🚀 Long-term Vision

### v1.0 - Traffic Classification
- ✅ Screenshot capture working
- ✅ ResNet-based classifier
- 🔄 YOLO vehicle detection
- 🔄 Hybrid model

### v2.0 - Video Analysis
- Video stream processing
- Temporal traffic modeling
- Vehicle tracking and counting
- Speed estimation

### v3.0 - Smart Traffic System
- Real-time traffic prediction
- Incident detection
- Traffic flow optimization
- Dashboard and API

---

**Ready to begin implementation! Which phase should we start with?**
