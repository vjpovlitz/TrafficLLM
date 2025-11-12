# Project Restructuring Plan

## 🎯 Goal
Transform the current confusing nested structure into a clean, professional project layout.

---

## ❌ Current Structure (CONFUSING!)

```
TrafficCamLLM/                          # Outer directory
├── .git/                               # Git repo 1
├── TrafficLLM/                         # NESTED subdirectory (confusing!)
│   ├── .git/                          # Git repo 2 (duplicate!)
│   ├── trafficllm/                    # Core package
│   │   ├── data_processing/
│   │   ├── model_preparation/
│   │   ├── training/
│   │   └── ...
│   └── scripts/
│       └── data_collection/
├── scripts/                            # Duplicate scripts folder
│   ├── capture/
│   └── testing/
├── CLAUDE.md                           # Docs scattered everywhere
├── IMPLEMENTATION_STATUS.md
├── MODEL_REFINEMENT_PLAN.md
└── ...
```

**Problems:**
- ❌ Two `TrafficLLM` levels (TrafficCamLLM and TrafficLLM)
- ❌ Two `.git` repositories
- ❌ Two `scripts/` directories in different places
- ❌ Documentation scattered at multiple levels
- ❌ Unclear where to put new files
- ❌ Import paths are confusing

---

## ✅ New Structure (CLEAN!)

```
traffic-classification/                 # Clear, professional name
├── .git/                              # Single git repo
├── .gitignore                         # Single gitignore
│
├── docs/                              # 📚 ALL documentation
│   ├── README.md                      # Main readme
│   ├── SETUP_GUIDE.md                 # Setup instructions
│   ├── API_REFERENCE.md               # API docs
│   ├── ARCHITECTURE.md                # System architecture
│   └── CHANGELOG.md                   # Version history
│
├── config/                            # ⚙️  ALL configuration
│   ├── camera_config.json             # Camera definitions
│   ├── training_config.yaml           # Training hyperparameters
│   └── deployment_config.yaml         # Deployment settings
│
├── src/                               # 📦 ALL source code
│   ├── trafficllm/                    # Core package
│   │   ├── __init__.py
│   │   ├── models/                    # Model definitions
│   │   │   ├── __init__.py
│   │   │   ├── improved_model.py
│   │   │   ├── attention_modules.py
│   │   │   └── custom_model.py
│   │   │
│   │   ├── data/                      # Data handling
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py
│   │   │   ├── augmentations.py
│   │   │   └── preprocessing.py
│   │   │
│   │   ├── training/                  # Training logic
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py
│   │   │   ├── losses.py
│   │   │   └── metrics.py
│   │   │
│   │   └── utils/                     # Utilities
│   │       ├── __init__.py
│   │       ├── logging.py
│   │       └── visualization.py
│   │
│   ├── data_collection/               # Data collection scripts
│   │   ├── __init__.py
│   │   ├── camera_manager.py
│   │   ├── image_validator.py
│   │   ├── metadata_collector.py
│   │   └── enhanced_pipeline.py
│   │
│   └── scripts/                       # Standalone scripts
│       ├── train_model.py             # Training entry point
│       ├── evaluate_model.py          # Evaluation script
│       ├── collect_data.py            # Data collection entry point
│       └── deploy_model.py            # Deployment script
│
├── tests/                             # 🧪 ALL tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_training.py
│   └── test_data_collection.py
│
├── data/                              # 💾 Data storage (gitignored)
│   ├── raw/                           # Raw captures
│   ├── processed/                     # Processed datasets
│   ├── validated/                     # Validated images
│   └── metadata/                      # Metadata files
│
├── models/                            # 🤖 Trained models (gitignored)
│   ├── checkpoints/                   # Training checkpoints
│   ├── best_model.pth                 # Best model
│   └── production/                    # Production models
│
├── outputs/                           # 📊 Outputs (gitignored)
│   ├── logs/                          # Training logs
│   ├── metrics/                       # Metrics and reports
│   └── visualizations/                # Plots and visualizations
│
├── notebooks/                         # 📓 Jupyter notebooks
│   ├── exploration.ipynb
│   └── analysis.ipynb
│
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
├── pyproject.toml                     # Modern Python config
└── README.md                          # Main project readme
```

**Benefits:**
- ✅ Clear, single-level structure
- ✅ Everything in its logical place
- ✅ Easy to find and add new files
- ✅ Professional project layout
- ✅ Clean import paths
- ✅ Follows Python best practices

---

## 📋 File Mapping

### Documentation Files
```
OLD → NEW
TrafficCamLLM/CLAUDE.md                         → docs/CLAUDE_REFERENCE.md
TrafficCamLLM/SETUP_AND_USAGE_GUIDE.md         → docs/SETUP_GUIDE.md
TrafficCamLLM/IMPLEMENTATION_STATUS.md          → docs/IMPLEMENTATION_STATUS.md
TrafficCamLLM/MODEL_REFINEMENT_PLAN.md         → docs/MODEL_REFINEMENT.md
TrafficCamLLM/PROJECT_IMPROVEMENT_PLAN.md      → docs/PROJECT_PLAN.md
TrafficCamLLM/FULLSCREEN_FIX_SUMMARY.md        → docs/FULLSCREEN_FIX.md
TrafficCamLLM/TrafficLLM/README.md             → README.md (root)
```

### Configuration Files
```
OLD → NEW
TrafficCamLLM/camera_config.json                           → config/camera_config.json
TrafficCamLLM/TrafficLLM/scripts/data_collection/*.json   → config/
```

### Core Package
```
OLD → NEW
TrafficCamLLM/TrafficLLM/trafficllm/model_preparation/     → src/trafficllm/models/
TrafficCamLLM/TrafficLLM/trafficllm/data_processing/       → src/trafficllm/data/
TrafficCamLLM/TrafficLLM/trafficllm/training/              → src/trafficllm/training/
TrafficCamLLM/TrafficLLM/trafficllm/data_visualization/    → src/trafficllm/utils/visualization/
TrafficCamLLM/TrafficLLM/trafficllm/refinement/            → src/trafficllm/utils/refinement/
```

### Data Collection
```
OLD → NEW
TrafficCamLLM/TrafficLLM/scripts/data_collection/         → src/data_collection/
```

### Scripts
```
OLD → NEW
TrafficCamLLM/TrafficLLM/trafficllm/training/run_training.py    → src/scripts/train_model.py
TrafficCamLLM/scripts/capture/                                   → (merge into src/data_collection/)
TrafficCamLLM/scripts/testing/                                   → tests/integration/
```

### Tests
```
OLD → NEW
TrafficCamLLM/TrafficLLM/tests/                           → tests/unit/
TrafficCamLLM/scripts/testing/                            → tests/integration/
TrafficCamLLM/TrafficLLM/trafficllm/training/test_trainer.py → tests/unit/test_trainer.py
```

### Other
```
OLD → NEW
TrafficCamLLM/utils/                                      → src/trafficllm/utils/
TrafficCamLLM/screenshots/                                → data/raw/screenshots/
TrafficCamLLM/logs/                                       → outputs/logs/
TrafficCamLLM/test_results/                               → outputs/test_results/
```

---

## 🔧 Updated Import Paths

### Before (CONFUSING):
```python
# From inside TrafficCamLLM/TrafficLLM/trafficllm/training/
from model_preparation.improved_model import ImprovedTrafficNet
from data_processing.augmentations import get_training_augmentation

# From scripts
sys.path.append(str(Path(__file__).parent.parent.parent))
from trafficllm.model_preparation.improved_model import ImprovedTrafficNet
```

### After (CLEAN):
```python
# From anywhere in the project
from trafficllm.models import ImprovedTrafficNet
from trafficllm.data import get_training_augmentation

# Simple, predictable paths!
```

---

## 📦 Updated Package Structure

### setup.py
```python
from setuptools import setup, find_packages

setup(
    name="trafficllm",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "albumentations>=1.3.0",
        "selenium>=4.0.0",
        "pillow>=9.0.0",
        "numpy>=1.24.0",
        # ... etc
    ],
)
```

### pyproject.toml (Modern Python)
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "trafficllm"
version = "1.0.0"
description = "Deep learning system for traffic classification"
readme = "README.md"
requires-python = ">=3.8"

[project.scripts]
traffic-train = "scripts.train_model:main"
traffic-collect = "scripts.collect_data:main"
traffic-evaluate = "scripts.evaluate_model:main"
```

---

## 🚀 Migration Steps

### Phase 1: Create New Structure (Safe - No Deletion)
1. Create new directories
2. Copy files to new locations
3. Update imports in copied files
4. Test copied files work

### Phase 2: Update Configuration
5. Update all config files with new paths
6. Update documentation with new structure
7. Create new README.md

### Phase 3: Clean Up (After Testing)
8. Archive old structure
9. Remove duplicates
10. Clean up old directories

---

## ✅ Validation Checklist

After restructuring, verify:
- [ ] All imports work
- [ ] Training script runs
- [ ] Data collection works
- [ ] Tests pass
- [ ] Documentation is accurate
- [ ] No broken links
- [ ] Git history preserved
- [ ] All files accounted for

---

## 🎯 Next Steps

1. **Review this plan** - Make sure it makes sense
2. **Backup current state** - Create git tag or zip backup
3. **Execute restructuring** - Run migration script
4. **Test everything** - Make sure nothing broke
5. **Update documentation** - Reflect new structure
6. **Commit changes** - Clean git history

---

**Ready to proceed?** This will make your project:
- ✅ Professional
- ✅ Easy to navigate
- ✅ Easy to maintain
- ✅ Easy to share
- ✅ Python best practices compliant
