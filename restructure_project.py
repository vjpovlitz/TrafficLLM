#!/usr/bin/env python3
"""
Safe Project Restructuring Script
Migrates from confusing nested structure to clean professional layout
"""
import os
import shutil
from pathlib import Path
import json

class ProjectRestructurer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.backup_created = False
        self.changes = []

    def backup_project(self):
        """Create a backup before making changes"""
        print("\n" + "="*80)
        print("STEP 1: Creating Backup")
        print("="*80)

        backup_dir = self.base_dir.parent / f"{self.base_dir.name}_backup"

        if backup_dir.exists():
            print(f"⚠️  Backup already exists at: {backup_dir}")
            response = input("Overwrite? (yes/no): ")
            if response.lower() != 'yes':
                print("❌ Backup cancelled. Exiting.")
                return False

        print(f"Creating backup at: {backup_dir}")
        try:
            # Copy everything except .git and large data folders
            shutil.copytree(
                self.base_dir,
                backup_dir,
                ignore=shutil.ignore_patterns(
                    '.git', '__pycache__', '*.pyc', '*.pyo',
                    'data', 'models', 'outputs', 'screenshots'
                )
            )
            self.backup_created = True
            print("✅ Backup created successfully")
            return True
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False

    def create_new_structure(self):
        """Create the new directory structure"""
        print("\n" + "="*80)
        print("STEP 2: Creating New Directory Structure")
        print("="*80)

        new_dirs = [
            "docs",
            "config",
            "src/trafficllm/models",
            "src/trafficllm/data",
            "src/trafficllm/training",
            "src/trafficllm/utils/visualization",
            "src/trafficllm/utils/refinement",
            "src/data_collection",
            "src/scripts",
            "tests/unit",
            "tests/integration",
            "data/raw",
            "data/processed",
            "data/validated",
            "data/metadata",
            "models/checkpoints",
            "models/production",
            "outputs/logs",
            "outputs/metrics",
            "outputs/visualizations",
            "notebooks"
        ]

        for dir_path in new_dirs:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created: {dir_path}")

        # Create __init__.py files
        init_dirs = [
            "src/trafficllm",
            "src/trafficllm/models",
            "src/trafficllm/data",
            "src/trafficllm/training",
            "src/trafficllm/utils",
            "src/trafficllm/utils/visualization",
            "src/trafficllm/utils/refinement",
            "src/data_collection",
            "tests",
            "tests/unit",
            "tests/integration"
        ]

        for dir_path in init_dirs:
            init_file = self.base_dir / dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text("")
                print(f"  ✅ Created: {dir_path}/__init__.py")

        print("\n✅ New structure created")

    def move_documentation(self):
        """Move documentation files"""
        print("\n" + "="*80)
        print("STEP 3: Moving Documentation")
        print("="*80)

        doc_mappings = {
            "CLAUDE.md": "docs/CLAUDE_REFERENCE.md",
            "SETUP_AND_USAGE_GUIDE.md": "docs/SETUP_GUIDE.md",
            "IMPLEMENTATION_STATUS.md": "docs/IMPLEMENTATION_STATUS.md",
            "MODEL_REFINEMENT_PLAN.md": "docs/MODEL_REFINEMENT.md",
            "PROJECT_IMPROVEMENT_PLAN.md": "docs/PROJECT_PLAN.md",
            "FULLSCREEN_FIX_SUMMARY.md": "docs/FULLSCREEN_FIX.md",
            "RESTRUCTURING_PLAN.md": "docs/RESTRUCTURING_PLAN.md",
            "TrafficLLM/README.md": "docs/OLD_README.md"
        }

        for old, new in doc_mappings.items():
            old_path = self.base_dir / old
            new_path = self.base_dir / new

            if old_path.exists():
                shutil.copy2(old_path, new_path)
                print(f"  ✅ {old} → {new}")
                self.changes.append(f"Moved: {old} → {new}")
            else:
                print(f"  ⚠️  Not found: {old}")

    def move_config(self):
        """Move configuration files"""
        print("\n" + "="*80)
        print("STEP 4: Moving Configuration")
        print("="*80)

        config_files = [
            "camera_config.json",
            "TrafficLLM/scripts/data_collection/video_camera_config.json"
        ]

        for config_file in config_files:
            old_path = self.base_dir / config_file
            if old_path.exists():
                new_path = self.base_dir / "config" / old_path.name
                shutil.copy2(old_path, new_path)
                print(f"  ✅ {config_file} → config/{old_path.name}")
                self.changes.append(f"Moved: {config_file} → config/{old_path.name}")

    def move_source_code(self):
        """Move source code files"""
        print("\n" + "="*80)
        print("STEP 5: Moving Source Code")
        print("="*80)

        # Move core package modules
        module_mappings = {
            "TrafficLLM/trafficllm/model_preparation": "src/trafficllm/models",
            "TrafficLLM/trafficllm/data_processing": "src/trafficllm/data",
            "TrafficLLM/trafficllm/training": "src/trafficllm/training",
            "TrafficLLM/trafficllm/data_visualization": "src/trafficllm/utils/visualization",
            "TrafficLLM/trafficllm/refinement": "src/trafficllm/utils/refinement",
        }

        for old, new in module_mappings.items():
            old_path = self.base_dir / old
            new_path = self.base_dir / new

            if old_path.exists():
                # Copy all Python files
                for py_file in old_path.glob("*.py"):
                    shutil.copy2(py_file, new_path / py_file.name)
                    print(f"  ✅ {old}/{py_file.name} → {new}/{py_file.name}")
                    self.changes.append(f"Moved: {old}/{py_file.name} → {new}/{py_file.name}")
            else:
                print(f"  ⚠️  Not found: {old}")

        # Move data collection
        data_collection_src = self.base_dir / "TrafficLLM/scripts/data_collection"
        data_collection_dst = self.base_dir / "src/data_collection"

        if data_collection_src.exists():
            for py_file in data_collection_src.glob("*.py"):
                shutil.copy2(py_file, data_collection_dst / py_file.name)
                print(f"  ✅ data_collection/{py_file.name} → src/data_collection/{py_file.name}")
                self.changes.append(f"Moved: data_collection/{py_file.name}")

    def move_tests(self):
        """Move test files"""
        print("\n" + "="*80)
        print("STEP 6: Moving Tests")
        print("="*80)

        # Move unit tests
        unit_test_src = self.base_dir / "TrafficLLM/tests"
        if unit_test_src.exists():
            for test_file in unit_test_src.glob("*.py"):
                shutil.copy2(test_file, self.base_dir / "tests/unit" / test_file.name)
                print(f"  ✅ {test_file.name} → tests/unit/")
                self.changes.append(f"Moved test: {test_file.name}")

        # Move integration tests from scripts/testing
        integration_test_src = self.base_dir / "scripts/testing"
        if integration_test_src.exists():
            for test_file in integration_test_src.glob("*.py"):
                shutil.copy2(test_file, self.base_dir / "tests/integration" / test_file.name)
                print(f"  ✅ {test_file.name} → tests/integration/")
                self.changes.append(f"Moved test: {test_file.name}")

    def create_main_readme(self):
        """Create new main README"""
        print("\n" + "="*80)
        print("STEP 7: Creating Main README")
        print("="*80)

        readme_content = """# Traffic Classification System

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
"""

        readme_path = self.base_dir / "README.md"
        readme_path.write_text(readme_content)
        print("  ✅ Created new README.md")
        self.changes.append("Created: README.md")

    def create_requirements(self):
        """Create comprehensive requirements.txt"""
        print("\n" + "="*80)
        print("STEP 8: Creating requirements.txt")
        print("="*80)

        requirements = """# Core Deep Learning
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0

# Data Processing
albumentations>=1.3.0
opencv-python>=4.7.0
pillow>=9.0.0

# Data Collection
selenium>=4.0.0
requests>=2.28.0
beautifulsoup4>=4.11.0

# Training & Monitoring
wandb>=0.15.0
torchmetrics>=1.0.0
tqdm>=4.65.0

# Utilities
pytz>=2023.3
schedule>=1.1.0
python-dateutil>=2.8.2

# Testing
pytest>=7.3.0
pytest-cov>=4.1.0

# Development
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0
"""

        req_path = self.base_dir / "requirements.txt"
        req_path.write_text(requirements)
        print("  ✅ Created requirements.txt")
        self.changes.append("Created: requirements.txt")

    def create_setup_py(self):
        """Create setup.py"""
        print("\n" + "="*80)
        print("STEP 9: Creating setup.py")
        print("="*80)

        setup_content = """from setuptools import setup, find_packages

setup(
    name="trafficllm",
    version="1.0.0",
    description="Deep learning system for traffic classification",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "albumentations>=1.3.0",
        "selenium>=4.0.0",
        "pillow>=9.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.7.0",
        "torchmetrics>=1.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "traffic-train=scripts.train_model:main",
            "traffic-collect=scripts.collect_data:main",
            "traffic-evaluate=scripts.evaluate_model:main",
        ],
    },
)
"""

        setup_path = self.base_dir / "setup.py"
        setup_path.write_text(setup_content)
        print("  ✅ Created setup.py")
        self.changes.append("Created: setup.py")

    def generate_report(self):
        """Generate restructuring report"""
        print("\n" + "="*80)
        print("STEP 10: Generating Report")
        print("="*80)

        report_path = self.base_dir / "RESTRUCTURING_REPORT.txt"
        with open(report_path, 'w') as f:
            f.write("Project Restructuring Report\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total changes: {len(self.changes)}\n\n")
            f.write("Changes made:\n")
            for change in self.changes:
                f.write(f"  - {change}\n")

        print(f"  ✅ Report saved to: {report_path}")

    def run(self, skip_backup=False):
        """Run the complete restructuring"""
        print("\n" + "="*80)
        print("PROJECT RESTRUCTURING")
        print("="*80)
        print(f"Project: {self.base_dir}")
        print("\nThis will:")
        print("  1. Create a backup")
        print("  2. Create new directory structure")
        print("  3. Copy files to new locations")
        print("  4. Update configurations")
        print("  5. Generate new documentation")
        print("\n⚠️  IMPORTANT: The old structure will remain untouched")
        print("   You can manually delete it after verifying the new structure works")

        if not skip_backup:
            response = input("\nProceed? (yes/no): ")
            if response.lower() != 'yes':
                print("❌ Cancelled")
                return False

            if not self.backup_project():
                print("❌ Backup failed. Aborting.")
                return False

        # Run all steps
        self.create_new_structure()
        self.move_documentation()
        self.move_config()
        self.move_source_code()
        self.move_tests()
        self.create_main_readme()
        self.create_requirements()
        self.create_setup_py()
        self.generate_report()

        print("\n" + "="*80)
        print("✅ RESTRUCTURING COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Review the new structure")
        print("  2. Test that imports work: cd src && python -c 'import trafficllm'")
        print("  3. Run tests: pytest tests/")
        print("  4. If everything works, you can delete:")
        print("     - TrafficLLM/ (old nested directory)")
        print("     - scripts/ (old outer scripts)")
        print("     - Old doc files at root")
        print(f"\n  Backup saved at: {self.base_dir.parent / f'{self.base_dir.name}_backup'}")
        print(f"  Report saved at: {self.base_dir / 'RESTRUCTURING_REPORT.txt'}")

        return True


if __name__ == "__main__":
    import sys

    # Get the project directory
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]
    else:
        project_dir = Path(__file__).parent

    restructurer = ProjectRestructurer(project_dir)
    restructurer.run()
