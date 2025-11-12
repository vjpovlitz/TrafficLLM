from setuptools import setup, find_packages

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
