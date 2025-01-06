from setuptools import setup, find_packages

setup(
    name="trafficllm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "pandas",
        "pillow",
        "opencv-python",
    ],
    author="vjpovlitz",
    description="Traffic analysis and prediction using LLMs",
    python_requires=">=3.8",
) 