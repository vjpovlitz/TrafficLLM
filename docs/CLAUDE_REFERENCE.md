# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TrafficLLM is a deep learning project for traffic analysis using computer vision and LLMs. The project captures traffic camera footage, validates image quality, enriches data with metadata, and trains models to classify traffic conditions.

## Architecture

### Two-Tier Structure

The repository has a nested structure: `TrafficCamLLM/` (outer) contains development/testing scripts, while `TrafficCamLLM/TrafficLLM/` (inner) is the main package containing production code.

**Key directories:**
- `TrafficCamLLM/TrafficLLM/trafficllm/` - Main Python package with core modules
- `TrafficCamLLM/TrafficLLM/scripts/data_collection/` - Production data collection pipeline
- `TrafficCamLLM/scripts/` - Development testing scripts (outer layer)

### Core Components

**1. Data Collection System** (`scripts/data_collection/`)
- **CameraManager** (`camera_manager.py`) - Unified camera configuration and concurrent screenshot capture using Selenium WebDriver with Chrome headless
- **ImageValidator** (`image_validator.py`) - Quality assessment: blur detection (Laplacian variance), brightness/contrast analysis, edge density, noise assessment. Includes ROIDetector for automatic region-of-interest detection
- **MetadataCollector** (`metadata_collector.py`) - Enriches captures with weather data (OpenWeatherMap API), timestamps, traffic context (rush hour, weekend, daytime detection)
- **EnhancedPipeline** (`enhanced_pipeline.py`) - Orchestrates the complete workflow: capture → validation → metadata → reporting

**2. Model Training** (`trafficllm/`)
- `model_preparation/` - Dataset loading (lazy loading with `load_from_disk`), TrafficDataset class, TrafficClassifier (ResNet-50 based)
- `training/` - Multi-task training with traffic_level classification, density/congestion/flow_rate regression
- `data_processing/` - Data preparation utilities
- `refinement/` - Interactive feedback system
- `data_visualization/` - Visualization tools

### Data Flow

```
Camera URLs (JSON config)
  → Selenium Screenshot Capture (concurrent, 4 workers)
  → Image Validation (quality scoring 0.0-1.0)
  → Metadata Collection (weather + context)
  → File Organization (raw/validated/metadata/reports)
  → Model Training (ResNet-50, multi-task)
```

### Configuration System

Cameras are defined in JSON config files (`camera_config.json`, `video_camera_config.json`) with structure:
```json
{
  "cameras": {
    "camera_id": {
      "name": "DISPLAY_NAME",
      "url": "https://...",
      "location": "Location",
      "region": "State",
      "enabled": true,
      "capture_interval": 15,
      "timeout": 30,
      "retry_attempts": 3,
      "roi_coordinates": null
    }
  }
}
```

## Development Commands

### Setup
```bash
cd TrafficCamLLM/TrafficLLM
pip install -r requirements.txt  # Core: torch, numpy, opencv-python, pillow

cd scripts/data_collection
pip install -r requirements.txt  # Selenium, schedule, requests, pytz
```

### Data Collection

**Single capture (all cameras):**
```bash
cd TrafficCamLLM/TrafficLLM/scripts/data_collection
python enhanced_pipeline.py --mode single
```

**Single capture (specific camera):**
```bash
python enhanced_pipeline.py --mode single --camera us_50_sandy_point
```

**Scheduled capture (every 10 minutes):**
```bash
python enhanced_pipeline.py --mode scheduled --interval 10
```

**With weather data:**
```bash
export OPENWEATHER_API_KEY="your_key"
python enhanced_pipeline.py --weather-key "your_key"
```

### Testing

**Test fullscreen capture:**
```bash
cd TrafficCamLLM/TrafficLLM/scripts/data_collection
python test_fullscreen_capture.py
```

**Test video capture:**
```bash
python test_video_capture.py
```

**Test data loader:**
```bash
cd TrafficCamLLM/TrafficLLM
python trafficllm/model_preparation/test_dataloader.py
```

### Training

**Run model training:**
```bash
cd TrafficCamLLM/TrafficLLM/trafficllm/training
python train.py
```

The training script automatically adds the project root to sys.path for imports.

### Linting/Formatting
```bash
black trafficllm/
isort trafficllm/
flake8 trafficllm/
```

### Running Tests
```bash
pytest tests/
```

## Key Implementation Details

### Image Quality Validation
- **Minimum requirements:** 640x480 resolution, 10KB file size, quality score ≥ 0.4
- **Quality scoring:** Weighted combination of brightness, contrast, blur (Laplacian), edge density, noise
- **Batch validation:** `ImageValidator.batch_validate()` processes directories with parallel workers

### Selenium WebDriver Setup
- Uses Chrome headless mode with optimized options
- Automatic driver cleanup after each capture to prevent memory leaks
- Concurrent capture with ThreadPoolExecutor (default 4 workers, configurable with `max_workers`)
- Retry logic with exponential backoff (2^attempt seconds)

### Error Handling Pattern
Throughout the codebase, operations use:
- Configurable retry attempts (typically 3)
- Exponential backoff for rate limiting
- Graceful degradation (individual camera failures don't stop pipeline)
- Comprehensive logging at INFO/WARNING/ERROR levels

### Output Directory Structure
```
enhanced_screenshots/
├── raw/          # Original screenshots
├── validated/    # Quality-validated images (score ≥ 0.4)
├── metadata/     # JSON files with weather + context
├── reports/      # Session reports (JSON)
└── logs/         # Pipeline execution logs
```

### Multi-Task Training
The model trains on multiple tasks simultaneously:
- Traffic level classification (CrossEntropyLoss)
- Density/congestion/flow_rate regression (MSELoss)
- Loss weights can be adjusted per task

### Legacy Scripts
The following scripts in `scripts/data_collection/` are legacy and superseded by the enhanced pipeline:
- `myAPTCamPipeline1.py` → Use CameraManager
- `spheadlessPipeline1.py` → Use EnhancedPipeline
- `screenshotCroppedPipeline2.py` → Use ROIDetector
- `SP.py` → Use JSON camera configs

## Working with This Codebase

### Adding New Cameras
1. Edit `scripts/data_collection/camera_config.json`
2. Add camera entry with required fields (name, url, location, region)
3. Run `python enhanced_pipeline.py --mode single --camera new_camera_id` to test

### Modifying Image Validation Criteria
- Edit `ImageValidator` class in `scripts/data_collection/image_validator.py`
- Adjust quality score weights in `calculate_quality_score()` method
- Update minimum thresholds in `validate_image()` method

### Extending Metadata Collection
- Add new data sources in `MetadataCollector` class (`metadata_collector.py`)
- Weather data uses OpenWeatherMap API (free tier: 1000 calls/month)
- Implement caching for multiple cameras in same location to avoid rate limits

### Model Architecture Changes
- Base model defined in `trafficllm/model_preparation/custom_model.py`
- Training loop in `trafficllm/training/train.py`
- Modify `TrafficClassifier` class for different pretrained models or architectures
- Adjust loss functions/weights in `TrafficTrainer.criteria` dict
