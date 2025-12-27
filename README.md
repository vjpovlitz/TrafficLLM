# TrafficLLM

A deep learning system for intelligent traffic analysis and classification from camera feeds. Features multi-task learning with ResNet-50, SAM 2/3 integration for annotation, YOLO-based detection, and specialized nighttime vehicle detection using headlight pattern recognition.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Data Collection](#data-collection)
  - [Annotation with SAM](#annotation-with-sam)
  - [Nighttime Detection](#nighttime-detection)
  - [Training](#training)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Documentation](#documentation)
- [License](#license)

## Features

### Core Capabilities
- Multi-task learning: traffic level classification + density/congestion/flow regression
- ResNet-50 backbone with configurable attention mechanisms (CBAM, SE, Enhanced Spatial)
- Mixed precision training with automatic mixed precision (AMP)
- Comprehensive data augmentation pipeline

### Detection and Annotation
- SAM 2 integration for automatic vehicle segmentation (works on Mac MPS)
- SAM 3 support for text-based concept segmentation ("car", "truck", "bus")
- YOLO + SAM combined pipeline for precise bounding boxes and masks
- Nighttime headlight detection for low-light conditions

### Data Collection
- Selenium-based camera capture with fullscreen video mode
- Automated image quality validation (blur, brightness, contrast)
- Weather and temporal metadata enrichment
- Organized image library with date-based structure

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (with MPS support for Mac or CUDA for GPU)
- Chrome browser (for Selenium capture)

### Setup

```bash
# Clone the repository
git clone https://github.com/vjpovlitz/TrafficLLM.git
cd TrafficLLM

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# For SAM support (Ultralytics 8.3.237+)
pip install -U ultralytics
```

### SAM 3 Setup (Optional)

SAM 3 requires weights from Hugging Face:

1. Request access: https://huggingface.co/facebook/sam3
2. Download sam3.pt (3.4GB) after approval
3. Place in project root or specify path

SAM 2 works immediately without access requests.

## Quick Start

### Capture Traffic Images

```bash
# Single capture from all enabled cameras
python scripts/capture_to_library.py

# Capture from specific camera
python scripts/capture_to_library.py --camera us_50_sandy_point
```

### Annotate with SAM 2

```bash
# YOLO detection + SAM 2 segmentation
python scripts/annotate_vehicles_sam2.py --input data/image_library

# Output: annotations in YOLO format for training
```

### Detect Vehicles at Night

```bash
# Headlight-based detection for nighttime images
python scripts/nighttime_traffic_detector.py --input data/image_library
```

### Train Model

```bash
python scripts/train_classifier.py
```

## Project Structure

```
TrafficLLM/
├── config/
│   └── camera_config.json          # Camera definitions and settings
│
├── data/
│   ├── image_library/              # Captured images (date-organized)
│   │   └── YYYY/MM/DD/camera_id/
│   │       ├── raw/                # Original captures
│   │       ├── validated/          # Quality-filtered images
│   │       └── metadata/           # JSON metadata files
│   ├── annotations/                # Generated annotations
│   │   ├── labels/                 # YOLO format labels
│   │   └── visualizations/         # Annotated images
│   └── models/                     # Trained model checkpoints
│
├── docs/
│   ├── SAM3_QUICKSTART.md          # SAM 3 setup guide
│   ├── SAM3_INTEGRATION_ANALYSIS.md # Architecture details
│   ├── IMAGE_LIBRARY_GUIDE.md      # Library management
│   ├── SETUP_GUIDE.md              # Installation guide
│   └── MODEL_REFINEMENT.md         # Model architecture docs
│
├── scripts/
│   ├── annotate_vehicles_sam2.py   # YOLO + SAM 2 annotation
│   ├── annotate_with_sam3.py       # SAM 3 batch annotation
│   ├── capture_to_library.py       # Image capture to library
│   ├── manage_image_library.py     # Library management utility
│   ├── nighttime_traffic_detector.py # Headlight-based detection
│   ├── run_traffic_monitor.py      # Real-time monitoring
│   ├── test_sam2_mac.py            # SAM 2 Mac testing
│   ├── test_sam3_mac.py            # SAM 3 Mac testing
│   ├── train_classifier.py         # Train ResNet classifier
│   ├── train_yolo_with_sam3.py     # Train YOLO with SAM annotations
│   └── test_yolo.py                # Test YOLO detection
│
├── src/trafficllm/
│   ├── annotation/
│   │   ├── sam3_annotator.py       # SAM 3 annotation interface
│   │   ├── exporter.py             # Export to YOLO/COCO formats
│   │   └── verifier.py             # Annotation quality checks
│   │
│   ├── core/
│   │   └── pipeline.py             # UnifiedTrafficPipeline
│   │
│   ├── data_collection/
│   │   ├── camera_manager.py       # Selenium camera capture
│   │   ├── image_validator.py      # Quality validation
│   │   ├── metadata_collector.py   # Weather/temporal metadata
│   │   ├── image_library.py        # Library organization
│   │   └── enhanced_pipeline.py    # Full capture pipeline
│   │
│   ├── models/
│   │   ├── resnet_classifier.py    # EnhancedTrafficNet (ResNet-50)
│   │   ├── attention_modules.py    # CBAM, SE, Spatial attention
│   │   └── yolo_detector.py        # YOLO wrapper
│   │
│   ├── training/
│   │   ├── improved_trainer.py     # Multi-task trainer
│   │   └── train.py                # Training entry point
│   │
│   └── utils/
│       ├── refinement/             # Interactive feedback system
│       └── visualization/          # Data visualization
│
├── tests/
│   ├── unit/                       # Unit tests
│   └── integration/                # Integration tests
│
├── CLAUDE.md                       # Claude Code instructions
├── requirements.txt                # Python dependencies
└── setup.py                        # Package setup
```

## Usage

### Data Collection

#### Single Camera Capture
```bash
python scripts/capture_to_library.py --camera us_50_sandy_point
```

#### Scheduled Collection
```bash
python scripts/capture_to_library.py --mode scheduled --interval 15
```

#### With Weather Data
```bash
export OPENWEATHER_API_KEY="your_key"
python scripts/capture_to_library.py --weather
```

### Annotation with SAM

#### SAM 2 (No Access Required)
```bash
# Annotate all images with YOLO + SAM 2
python scripts/annotate_vehicles_sam2.py --input data/image_library

# Skip SAM masks (faster, boxes only)
python scripts/annotate_vehicles_sam2.py --input data/image_library --no-sam

# Lower confidence threshold
python scripts/annotate_vehicles_sam2.py --conf 0.15
```

#### SAM 3 (Requires Hugging Face Access)
```bash
# Test SAM 3 with text prompts
python scripts/test_sam3_mac.py --model sam3.pt

# Batch annotation
python scripts/annotate_with_sam3.py \
    --input data/image_library \
    --prompts "car" "truck" "bus" "motorcycle"
```

#### SAM 3 Text Prompts
```python
from ultralytics.models.sam import SAM3SemanticPredictor

predictor = SAM3SemanticPredictor(overrides={"model": "sam3.pt"})
predictor.set_image("traffic.jpg")

# Describe what you want to segment
results = predictor(text=["car", "truck", "bus"])
results = predictor(text=["vehicle headlights"])
results = predictor(text=["person with red shirt"])
```

### Nighttime Detection

The headlight detector finds vehicles in dark conditions by pairing bright spots:

```bash
# Basic usage
python scripts/nighttime_traffic_detector.py --input data/image_library

# Adjust brightness threshold for darker scenes
python scripts/nighttime_traffic_detector.py --brightness-threshold 180

# Disable SAM road segmentation (faster)
python scripts/nighttime_traffic_detector.py --no-sam
```

Output includes:
- Vehicle count based on headlight pairs
- Traffic density estimate (none, light, moderate, heavy)
- Visualization with detected headlights and bounding boxes

### Training

#### Train ResNet Classifier
```bash
python scripts/train_classifier.py
```

#### Train YOLO with SAM Annotations
```bash
# Generate annotations first
python scripts/annotate_vehicles_sam2.py --input data/image_library

# Train YOLO
python scripts/train_yolo_with_sam3.py \
    --data data/annotations/dataset.yaml \
    --model yolo11n.pt \
    --epochs 100
```

### Inference

#### Real-time Monitoring
```bash
python scripts/run_traffic_monitor.py --source "video_url_or_file"
```

#### Using UnifiedTrafficPipeline
```python
from trafficllm.core.pipeline import UnifiedTrafficPipeline

pipeline = UnifiedTrafficPipeline(
    config_path="config/camera_config.json",
    yolo_model_size='s',
    classifier_path="models/best_model.pth"
)

# Process single frame
results = pipeline.process_frame(frame, conf_threshold=0.25)
# Returns: vehicle_count, traffic_level, density_score, annotated_frame
```

## Model Architecture

### EnhancedTrafficNet

ResNet-50 backbone with multi-task heads:

```
Input Image (224x224x3)
    |
ResNet-50 Backbone (pretrained ImageNet)
    |
Attention Module (CBAM/SE/Enhanced Spatial)
    |
    +---> Regression Head ---> [density, congestion, flow_rate]
    |                               |
    |                               v
    +---> Feature Fusion <---------+
              |
              v
         Classifier Head ---> [traffic_level: 5 classes]
```

### Attention Options
- `cbam`: Channel + spatial attention (recommended, +10-15% accuracy)
- `se`: Squeeze-Excitation blocks (lightweight)
- `enhanced_spatial`: Multi-scale spatial attention
- `simple`: Basic 7x7 convolution

### Multi-task Learning
- Classification: 5 traffic levels (CrossEntropyLoss)
- Regression: density, congestion, flow_rate (MSELoss)
- Learnable uncertainty weights balance task losses

## Performance

### Detection Comparison

| Method | Daytime | Nighttime | Speed |
|--------|---------|-----------|-------|
| YOLO11n | 15-20 vehicles | 1-2 vehicles | 50ms |
| YOLO + SAM 2 | 15-20 vehicles | 1-2 vehicles | 1.2s |
| Headlight Detector | N/A | 25-30 vehicles | 0.5s |

### Model Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Baseline ResNet-50 | 75-80% | 0.70 |
| With CBAM Attention | 85-90% | 0.82-0.88 |

### Mac MPS Performance
- SAM 2 inference: 2-3 seconds per image
- YOLO11n: 50ms per image
- Nighttime detector: 500ms per image

## Documentation

- [SAM 3 Quick Start](docs/SAM3_QUICKSTART.md) - Setup and usage guide
- [SAM 3 Integration Analysis](docs/SAM3_INTEGRATION_ANALYSIS.md) - Architecture details
- [Image Library Guide](docs/IMAGE_LIBRARY_GUIDE.md) - Library management
- [Setup Guide](docs/SETUP_GUIDE.md) - Complete installation instructions
- [Model Refinement](docs/MODEL_REFINEMENT.md) - Model architecture details

## Tech Stack

- **Deep Learning**: PyTorch, TorchVision
- **Object Detection**: Ultralytics YOLO, SAM 2/3
- **Data Collection**: Selenium WebDriver, OpenCV
- **Image Processing**: PIL, scikit-image
- **Augmentation**: Albumentations
- **Monitoring**: Weights and Biases (optional)

## License

MIT License

## Acknowledgments

- Meta AI for Segment Anything Models (SAM, SAM 2, SAM 3)
- Ultralytics for YOLO integration
- PyTorch team for deep learning framework
- ResNet architecture (He et al.)
- CBAM attention mechanism (Woo et al.)
