# TrafficLLM - Complete Setup and Usage Guide

## 🎯 What We've Built

### 1. **Dataset Adapter** ✅
- Bridges your existing HuggingFace datasets with the new training pipeline
- Automatically generates regression values (density, congestion, flow_rate) from traffic labels
- Supports augmentation transforms

### 2. **Fullscreen Video Capture** ✅
- Integrated into `enhanced_pipeline.py` and `camera_manager.py`
- Captures video-only screenshots (no webpage elements)
- 2x better quality (1.2-2.2 MB vs 682 KB)
- Use with `--fullscreen` flag

### 3. **Complete Training Pipeline** ✅
- Multi-task loss with automatic task weighting
- Label smoothing, cosine annealing, mixed precision
- Comprehensive metrics and checkpointing
- Ready-to-use training script

---

## 📦 Quick Start

### Step 1: Collect Data with Fullscreen Mode

```bash
cd TrafficLLM/scripts/data_collection

# Single capture with fullscreen video
python enhanced_pipeline.py --mode single --fullscreen

# Capture specific camera with fullscreen
python enhanced_pipeline.py --mode single --camera us_50_sandy_point --fullscreen

# Scheduled captures with fullscreen (every 10 minutes)
python enhanced_pipeline.py --mode scheduled --interval 10 --fullscreen

# With weather data
export OPENWEATHER_API_KEY="your_key"
python enhanced_pipeline.py --mode single --fullscreen --weather-key "$OPENWEATHER_API_KEY"
```

**What's Different with `--fullscreen`:**
- ✅ Video fills entire screenshot (no sidebars, no navigation)
- ✅ 2x larger files with more detail
- ✅ Clean video-only content
- ✅ Better for training models

---

### Step 2: Train the Improved Model

```bash
cd TrafficLLM/trafficllm/training

# Run training with your existing datasets
python3 run_training.py
```

**What Happens:**
1. Loads your existing Singapore & Bay Bridge datasets
2. Applies augmentation (weather, lighting, noise)
3. Trains improved model with CBAM attention
4. Saves checkpoints to `outputs/checkpoints/`
5. Generates training history to `outputs/logs/`

**Expected Training Time:**
- With mixed precision: ~1-2 hours for 50 epochs (on GPU)
- Without GPU: ~4-6 hours

---

## 🔧 Configuration Options

### Training Configuration

Edit `TrafficLLM/trafficllm/training/run_training.py` to adjust:

```python
config = TrainingConfig(
    # Model Architecture
    attention_type='cbam',      # Options: 'cbam', 'se', 'enhanced_spatial', 'simple'
    feature_fusion=True,         # Use full ResNet features + regression
    dropout=0.4,

    # Training
    epochs=50,
    batch_size=32,
    learning_rate=3e-4,

    # Augmentation
    augmentation_level='medium', # Options: 'light', 'medium', 'heavy'

    # Optimization
    use_mixed_precision=True,    # 2-3x faster training
    early_stopping=True,
    patience=10,

    # Logging
    use_wandb=False,             # Set True for Weights & Biases tracking
)
```

### Dataset Adaptation

If you want to change how regression values are generated, edit:

`TrafficLLM/trafficllm/data_processing/adapted_dataset.py`

```python
# Option 1: Label-based mapping (current default)
dataset = AdaptedTrafficDataset(
    path, label=0,
    generate_regression_values='auto'
)

# Option 2: Fixed values
dataset = AdaptedTrafficDataset(
    path, label=0,
    generate_regression_values={'density': 0.5, 'congestion': 0.3, 'flow_rate': 0.7'}
)

# Option 3: Random (for testing)
dataset = AdaptedTrafficDataset(
    path, label=0,
    generate_regression_values='random'
)
```

---

## 📊 Understanding Your Datasets

### Current Setup

You have two datasets:
1. **Singapore**: `processed_singapore_tensors_final` → Label 0 (Light traffic)
2. **Bay Bridge**: `processed_bay_bridge_dataset` → Label 1 (Moderate traffic)

### How Regression Values Are Generated

Since your datasets only have binary labels (0/1), the adapter automatically generates reasonable regression values:

| Traffic Level | Density | Congestion | Flow Rate | Description |
|--------------|---------|------------|-----------|-------------|
| 0 (Very Light) | 0.1 | 0.0 | 0.9 | Free flowing |
| 1 (Light) | 0.3 | 0.2 | 0.7 | Light traffic |
| 2 (Moderate) | 0.5 | 0.5 | 0.5 | Balanced |
| 3 (Heavy) | 0.7 | 0.7 | 0.3 | Congested |
| 4 (Very Heavy) | 0.9 | 0.9 | 0.1 | Gridlock |

**Note:** These are estimates with added noise. For better results, you should:
- Manually label traffic density/congestion for your images
- Or use the model's predictions to refine these values over time

---

## 🧪 Testing Components

### Test Dataset Adapter

```bash
cd TrafficLLM/trafficllm/data_processing
python3 adapted_dataset.py
```

Expected output:
```
✅ Successfully loaded a batch
   Image shape: torch.Size([4, 3, 640, 640])
   Traffic level: [0, 1, 0, 1]
   Density range: [0.052, 0.347]
```

### Test Augmentations

```bash
cd TrafficLLM/trafficllm/data_processing
python3 augmentations.py
```

Expected output:
```
✅ All augmentation tests passed!
```

### Test Improved Model

```bash
cd TrafficLLM/trafficllm/model_preparation
python3 improved_model.py
```

Expected output:
```
✅ All model tests passed!
   CBAM: 26,234,053 parameters
   SE: 26,152,325 parameters
```

### Test Training Pipeline

```bash
cd TrafficLLM/trafficllm/training
python3 test_trainer.py
```

Expected output:
```
✅ ALL TESTS PASSED!
```

---

## 📁 Output Directory Structure

After training, you'll have:

```
outputs/
├── checkpoints/
│   ├── best_model.pth              # Best validation loss
│   ├── last_model.pth              # Most recent epoch
│   ├── checkpoint_epoch_5.pth      # Periodic saves
│   ├── checkpoint_epoch_10.pth
│   └── ...
└── logs/
    ├── training_history.json       # Full training metrics
    └── pipeline_YYYYMMDD_HHMMSS.log
```

---

## 🚀 Next Steps

### 1. Improve Data Quality

**Option A: Collect More Data with Fullscreen**
```bash
# Run scheduled captures overnight
python enhanced_pipeline.py --mode scheduled --interval 15 --fullscreen
```

**Option B: Manually Label Regression Values**
- Create a script to annotate density/congestion/flow_rate
- Update `adapted_dataset.py` to load these annotations

### 2. Experiment with Model Variants

Train multiple models and compare:

```bash
# Model 1: CBAM attention
# (Edit run_training.py: attention_type='cbam')
python3 run_training.py

# Model 2: SE attention
# (Edit run_training.py: attention_type='se')
python3 run_training.py

# Model 3: Heavy augmentation
# (Edit run_training.py: augmentation_level='heavy')
python3 run_training.py
```

### 3. Create Model Ensemble

After training multiple models, combine them for better accuracy:

```python
from trafficllm.model_preparation.improved_model import EnsembleTrafficNet

ensemble = EnsembleTrafficNet([
    {'attention_type': 'cbam', 'feature_fusion': True},
    {'attention_type': 'se', 'feature_fusion': True},
    {'attention_type': 'enhanced_spatial', 'feature_fusion': True}
])

ensemble.load_ensemble_weights([
    'outputs/checkpoints/best_cbam.pth',
    'outputs/checkpoints/best_se.pth',
    'outputs/checkpoints/best_enhanced.pth'
])
```

### 4. Deploy for Inference

Create an inference script to classify new traffic images:

```python
import torch
from trafficllm.model_preparation.improved_model import ImprovedTrafficNet
from trafficllm.data_processing.augmentations import get_validation_transform

# Load model
model = ImprovedTrafficNet(attention_type='cbam', pretrained=False)
checkpoint = torch.load('outputs/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess image
transform = get_validation_transform(640)
image = ... # Load your image
transformed = transform(image=image)['image'].unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(transformed)
    traffic_level = outputs['traffic_level'].argmax(dim=1).item()
    density = outputs['density'].item()
    congestion = outputs['congestion'].item()
    flow_rate = outputs['flow_rate'].item()

print(f"Traffic Level: {traffic_level}")
print(f"Density: {density:.2f}")
print(f"Congestion: {congestion:.2f}")
print(f"Flow Rate: {flow_rate:.2f}")
```

---

## 🐛 Troubleshooting

### Issue: "Dataset not found"

**Solution:**
1. Verify dataset paths in `adapted_dataset.py`
2. Check that you've run data collection
3. Datasets should be at:
   - `TrafficLLM/trafficllm/data_processing/processed_singapore_tensors_final`
   - `TrafficLLM/trafficllm/data_processing/processed_bay_bridge_dataset`

### Issue: "CUDA out of memory"

**Solutions:**
1. Reduce batch size in `run_training.py` (try 16 or 8)
2. Reduce image size to 512 instead of 640
3. Disable mixed precision (slower but uses less memory)
4. Use fewer data augmentations

### Issue: Training is very slow

**Solutions:**
1. Enable mixed precision: `use_mixed_precision=True`
2. Reduce number of workers: `num_workers=2`
3. Use lighter augmentation: `augmentation_level='light'`
4. Train on GPU instead of CPU

### Issue: Model not improving

**Solutions:**
1. Check your data labels are correct
2. Try different learning rates (1e-4, 3e-4, 1e-3)
3. Increase augmentation strength
4. Train for more epochs
5. Check for class imbalance in your dataset

---

## 📚 Files Reference

### Core Training Files
- `trafficllm/training/run_training.py` - Main training script (START HERE)
- `trafficllm/training/improved_trainer.py` - Training pipeline implementation
- `trafficllm/model_preparation/improved_model.py` - Enhanced model architecture
- `trafficllm/data_processing/adapted_dataset.py` - Dataset adapter
- `trafficllm/data_processing/augmentations.py` - Data augmentation

### Data Collection Files
- `scripts/data_collection/enhanced_pipeline.py` - Production pipeline (with fullscreen)
- `scripts/data_collection/camera_manager.py` - Camera management (with fullscreen)
- `scripts/data_collection/image_validator.py` - Quality validation
- `scripts/data_collection/metadata_collector.py` - Metadata enrichment

### Configuration Files
- `scripts/data_collection/camera_config.json` - Camera definitions
- `IMPLEMENTATION_STATUS.md` - What's been built
- `MODEL_REFINEMENT_PLAN.md` - Original improvement plan

---

## 💡 Tips for Best Results

1. **Data Quality Matters Most**
   - Use fullscreen capture for better quality
   - Validate images before training
   - Collect diverse conditions (day/night, weather)

2. **Start Simple, Then Scale**
   - Begin with 'light' augmentation
   - Train for fewer epochs first (10-20)
   - Gradually increase complexity

3. **Monitor Training Carefully**
   - Watch for overfitting (val loss increases while train loss decreases)
   - Use early stopping to prevent wasted training time
   - Save checkpoints frequently

4. **Experiment Systematically**
   - Change one thing at a time
   - Keep notes on what works
   - Use W&B for experiment tracking

---

**Questions? Check:**
- `IMPLEMENTATION_STATUS.md` - Feature documentation
- `MODEL_REFINEMENT_PLAN.md` - Architecture details
- `CLAUDE.md` - General project guide

**Ready to train?**
```bash
cd TrafficLLM/trafficllm/training
python3 run_training.py
```

Good luck! 🚀
