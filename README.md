# Traffic Classification System

A deep learning system for intelligent traffic analysis and classification from camera feeds.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd traffic-classification

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Data Collection

```bash
# Collect traffic images with fullscreen video mode
python src/scripts/collect_data.py --mode single --fullscreen

# Scheduled collection
python src/scripts/collect_data.py --mode scheduled --interval 15 --fullscreen
```

### Training

```bash
# Train the improved model
python src/scripts/train_model.py
```

## 📁 Project Structure

```
traffic-classification/
├── docs/                   # Documentation
├── config/                 # Configuration files
├── src/                    # Source code
│   ├── trafficllm/        # Core package
│   ├── data_collection/   # Data collection
│   └── scripts/           # Entry point scripts
├── tests/                  # Tests
├── data/                   # Data storage
├── models/                 # Trained models
├── outputs/                # Training outputs
└── notebooks/              # Jupyter notebooks
```

## 📚 Documentation

- [Setup Guide](docs/SETUP_GUIDE.md) - Complete setup instructions
- [Architecture](docs/MODEL_REFINEMENT.md) - Model architecture details
- [API Reference](docs/CLAUDE_REFERENCE.md) - API documentation
- [Implementation Status](docs/IMPLEMENTATION_STATUS.md) - Current status

## 🎯 Features

- ✅ Multi-task learning (classification + regression)
- ✅ Advanced attention mechanisms (CBAM, SE, Enhanced Spatial)
- ✅ Comprehensive data augmentation
- ✅ Fullscreen video capture
- ✅ Mixed precision training
- ✅ Automated quality validation
- ✅ Metadata enrichment

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, TorchVision
- **Data Collection:** Selenium, OpenCV
- **Augmentation:** Albumentations
- **Monitoring:** Weights & Biases (optional)
- **Testing:** pytest

## 📊 Model Performance

| Model | Accuracy | F1 Score | Training Time |
|-------|----------|----------|---------------|
| Baseline | 75-80% | 0.70 | 3 hours |
| Improved (CBAM) | 85-90% | 0.82-0.88 | 1-2 hours |

## 🤝 Contributing

See [CLAUDE_REFERENCE.md](docs/CLAUDE_REFERENCE.md) for development guidelines.

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Built with:
- ResNet-50 (pretrained on ImageNet)
- CBAM Attention Mechanism
- Albumentations Data Augmentation
