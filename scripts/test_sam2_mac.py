#!/usr/bin/env python3
"""
Test SAM 2 on MacBook with MPS (Metal) backend.

This script tests the Segment Anything Model 2 for traffic image annotation
on Apple Silicon Macs using the Metal Performance Shaders backend.

Usage:
    python scripts/test_sam2_mac.py
    python scripts/test_sam2_mac.py --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
import cv2


def check_environment():
    """Check and print environment info."""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        device = "mps"
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = "cpu"
        print(f"Using device: CPU (MPS not available)")

    return device


def load_sam_model(device: str):
    """Load SAM 2 model using Ultralytics."""
    print("\n" + "=" * 60)
    print("Loading SAM 2 Model")
    print("=" * 60)

    try:
        from ultralytics import SAM

        # SAM 2 models available:
        # - sam2_t.pt (tiny) - fastest, lowest memory
        # - sam2_s.pt (small)
        # - sam2_b.pt (base)
        # - sam2_l.pt (large) - most accurate
        # For Mac testing, start with tiny or small
        model_name = "sam2_t.pt"  # Tiny model for testing

        print(f"Loading {model_name}... (will download on first run)")
        start = time.time()

        model = SAM(model_name)

        print(f"Model loaded in {time.time() - start:.2f}s")
        print(f"Model type: SAM2 = {model.is_sam2}")

        return model

    except ImportError as e:
        print(f"Error: Ultralytics not installed properly: {e}")
        print("Run: pip install -U ultralytics")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def annotate_image(model, image_path: Path, device: str):
    """
    Annotate a traffic image using SAM 2.

    Uses automatic mask generation (no prompts) to detect all objects.
    """
    print("\n" + "=" * 60)
    print(f"Annotating: {image_path.name}")
    print("=" * 60)

    # Load and display image info
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")

    # Run SAM 2 prediction
    # For automatic mask generation (no prompts):
    print("\nRunning SAM 2 inference...")
    start = time.time()

    results = model(
        str(image_path),
        device=device,
        verbose=False
    )

    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.2f}s")

    # Parse results
    if results and len(results) > 0:
        result = results[0]

        # Get masks
        if result.masks is not None:
            num_masks = len(result.masks)
            print(f"Detected {num_masks} segments")

            # Get mask data
            masks = result.masks.data.cpu().numpy()  # (N, H, W)

            # Create visualization
            visualize_masks(image_path, masks)

            return masks
        else:
            print("No masks generated")
            return None
    else:
        print("No results returned")
        return None


def annotate_with_prompts(model, image_path: Path, device: str, prompts: list):
    """
    Annotate using bounding box or point prompts.

    SAM 2 supports:
    - bboxes: [[x1, y1, x2, y2], ...]
    - points: [[x, y], ...]
    - labels: [1, 0, ...] (1=foreground, 0=background)
    """
    print("\n" + "=" * 60)
    print(f"Annotating with prompts: {prompts}")
    print("=" * 60)

    image = Image.open(image_path)
    width, height = image.size

    # Example: detect center region (where vehicles likely are)
    # Create a bounding box for the center/lower portion of the image
    center_bbox = [
        width * 0.1,   # x1: 10% from left
        height * 0.3,  # y1: 30% from top
        width * 0.9,   # x2: 90% from left
        height * 0.9   # y2: 90% from top
    ]

    print(f"Using bbox prompt: {center_bbox}")

    start = time.time()
    results = model(
        str(image_path),
        bboxes=[center_bbox],
        device=device,
        verbose=False
    )

    print(f"Inference time: {time.time() - start:.2f}s")

    if results and len(results) > 0 and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        print(f"Generated {len(masks)} masks from prompt")
        visualize_masks(image_path, masks, suffix="_prompted")
        return masks

    return None


def visualize_masks(image_path: Path, masks: np.ndarray, suffix: str = ""):
    """Create visualization of segmentation masks."""
    # Load original image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create colored overlay
    overlay = image.copy()

    # Generate random colors for each mask
    np.random.seed(42)
    colors = np.random.randint(0, 255, (len(masks), 3))

    for i, mask in enumerate(masks):
        # Resize mask to image size if needed
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.float32),
                (image.shape[1], image.shape[0])
            ) > 0.5

        # Apply color to mask region
        color = colors[i].tolist()
        overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5

    # Save visualization
    output_path = image_path.parent / f"{image_path.stem}_sam2{suffix}.png"
    overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), overlay_bgr)

    print(f"Saved visualization: {output_path}")


def find_test_images(project_root: Path) -> list:
    """Find available test images in the project."""
    patterns = [
        "data/image_library/**/*.png",
        "data/image_library/**/*.jpg",
        "data/screenshots/**/*.png",
        "data/screenshots/**/*.jpg",
    ]

    images = []
    for pattern in patterns:
        images.extend(project_root.glob(pattern))

    return sorted(images)[:5]  # Return up to 5 images


def main():
    parser = argparse.ArgumentParser(description="Test SAM 2 on MacBook")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file (uses project images if not specified)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sam2_t.pt",
        choices=["sam2_t.pt", "sam2_s.pt", "sam2_b.pt", "sam2_l.pt"],
        help="SAM 2 model size (t=tiny, s=small, b=base, l=large)"
    )
    args = parser.parse_args()

    # Check environment
    device = check_environment()

    # Load model
    model = load_sam_model(device)

    # Find test images
    if args.image:
        images = [Path(args.image)]
    else:
        images = find_test_images(project_root)
        if not images:
            print("\nNo test images found! Capture some traffic images first:")
            print("  python scripts/run_traffic_monitor.py")
            sys.exit(1)
        print(f"\nFound {len(images)} test images")

    # Test annotation on each image
    for image_path in images:
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue

        # Automatic segmentation
        masks = annotate_image(model, image_path, device)

        # Also test with prompts
        annotate_with_prompts(model, image_path, device, ["vehicle region"])

    print("\n" + "=" * 60)
    print("SAM 2 Test Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check the *_sam2.png files for visualizations")
    print("2. To annotate all images: python scripts/annotate_with_sam3.py")
    print("3. The annotator needs updating for Mac - let me know if you want that")


if __name__ == "__main__":
    main()
