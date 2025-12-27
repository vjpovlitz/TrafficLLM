#!/usr/bin/env python3
"""
Test SAM 3 on MacBook with MPS (Metal) backend.

SAM 3 (Segment Anything Model 3) was released by Meta on November 20, 2025.
It supports text-based "concept segmentation" - describe what you want to segment!

PREREQUISITES:
1. Request access at: https://huggingface.co/facebook/sam3
2. Once approved, download sam3.pt (3.4GB):
   https://huggingface.co/facebook/sam3/resolve/main/sam3.pt?download=true
3. Place sam3.pt in this directory or specify full path

Usage:
    python scripts/test_sam3_mac.py
    python scripts/test_sam3_mac.py --model /path/to/sam3.pt --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
import cv2


def check_environment():
    """Check environment and SAM 3 availability."""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # Auto-select device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Check Ultralytics version
    try:
        import ultralytics
        print(f"Ultralytics: {ultralytics.__version__}")
        if ultralytics.__version__ < "8.3.237":
            print("  WARNING: Need 8.3.237+ for SAM 3. Run: pip install -U ultralytics")
    except ImportError:
        print("ERROR: Ultralytics not installed. Run: pip install -U ultralytics")

    return device


def check_sam3_weights(model_path: str) -> bool:
    """Check if SAM 3 weights are available."""
    path = Path(model_path)
    if path.exists():
        size_gb = path.stat().st_size / (1024**3)
        print(f"\nSAM 3 weights found: {path}")
        print(f"  Size: {size_gb:.2f} GB")
        return True
    else:
        print(f"\nSAM 3 weights NOT found at: {path}")
        print("\nTo get SAM 3 weights:")
        print("1. Request access: https://huggingface.co/facebook/sam3")
        print("2. Download sam3.pt (3.4GB):")
        print("   https://huggingface.co/facebook/sam3/resolve/main/sam3.pt?download=true")
        print(f"3. Save to: {path.absolute()}")
        return False


def test_sam3_semantic(model_path: str, image_path: str, device: str):
    """
    Test SAM 3 semantic (text-based) segmentation.

    SAM 3's killer feature: describe what you want to segment in plain text!
    """
    print("\n" + "=" * 60)
    print("Testing SAM 3 Semantic Segmentation")
    print("=" * 60)

    from ultralytics.models.sam import SAM3SemanticPredictor

    # Configure predictor
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=model_path,
        half=True,  # FP16 for speed
        device=device,
        save=False,
        verbose=False,
    )

    print("Loading SAM 3 model...")
    start = time.time()
    predictor = SAM3SemanticPredictor(overrides=overrides)
    print(f"Model loaded in {time.time() - start:.2f}s")

    # Load and set image
    print(f"\nProcessing: {image_path}")
    predictor.set_image(image_path)

    # Test with traffic-related text prompts
    text_prompts = [
        ["car", "truck", "bus"],
        ["vehicle headlights"],
        ["road", "traffic"],
        ["person", "pedestrian"],
    ]

    img = cv2.imread(image_path)

    for prompts in text_prompts:
        print(f"\n  Text prompts: {prompts}")
        start = time.time()

        results = predictor(text=prompts)

        print(f"  Inference time: {time.time() - start:.2f}s")

        if results and len(results) > 0:
            result = results[0]
            if result.masks is not None:
                num_masks = len(result.masks)
                print(f"  Detected: {num_masks} segments")

                # Save visualization
                vis = visualize_results(img.copy(), result, prompts)
                prompt_str = "_".join(prompts).replace(" ", "-")[:30]
                output_path = Path(image_path).parent / f"{Path(image_path).stem}_sam3_{prompt_str}.jpg"
                cv2.imwrite(str(output_path), vis)
                print(f"  Saved: {output_path}")
            else:
                print("  No masks generated")
        else:
            print("  No results")

    return results


def visualize_results(image: np.ndarray, result, prompts: list) -> np.ndarray:
    """Visualize SAM 3 segmentation results."""
    if result.masks is None:
        return image

    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
    classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
    confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

    # Generate colors
    np.random.seed(42)
    colors = np.random.randint(0, 255, (len(prompts), 3)).tolist()

    # Apply masks
    for i, mask in enumerate(masks):
        # Resize mask if needed
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask.astype(np.float32), (image.shape[1], image.shape[0])) > 0.5

        # Get class index for color
        cls_idx = int(classes[i]) if i < len(classes) else 0
        color = colors[cls_idx % len(colors)]

        # Apply colored mask
        image[mask] = image[mask] * 0.5 + np.array(color) * 0.5

    # Draw boxes and labels
    for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
        x1, y1, x2, y2 = map(int, box)
        cls_idx = int(cls)
        color = colors[cls_idx % len(colors)]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{prompts[cls_idx] if cls_idx < len(prompts) else 'obj'} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


def find_test_images(project_root: Path) -> list:
    """Find available test images."""
    patterns = [
        "data/image_library/**/*.png",
        "data/image_library/**/*.jpg",
    ]
    images = []
    for pattern in patterns:
        images.extend(project_root.glob(pattern))
    # Filter out processed images
    images = [img for img in images if not any(x in img.stem for x in ['_sam', '_detected', '_annotated'])]
    return sorted(images)[:3]


def main():
    parser = argparse.ArgumentParser(description="Test SAM 3 on MacBook")
    parser.add_argument(
        "--model",
        type=str,
        default="sam3.pt",
        help="Path to sam3.pt weights file"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to test image (uses project images if not specified)"
    )
    args = parser.parse_args()

    # Check environment
    device = check_environment()

    # Check for SAM 3 weights
    if not check_sam3_weights(args.model):
        print("\n" + "=" * 60)
        print("SAM 3 SETUP REQUIRED")
        print("=" * 60)
        print("\nSAM 3 is not yet installed. Follow these steps:")
        print("\n1. Request access (takes ~24 hours):")
        print("   https://huggingface.co/facebook/sam3")
        print("\n2. Download sam3.pt (3.4GB)")
        print("\n3. Run this script again")
        print("\n" + "=" * 60)
        print("\nIn the meantime, you can use SAM 2 which works without access:")
        print("  python scripts/test_sam2_mac.py")
        print("  python scripts/annotate_vehicles_sam2.py")
        return

    # Find test images
    if args.image:
        images = [Path(args.image)]
    else:
        images = find_test_images(project_root)
        if not images:
            print("No test images found!")
            return

    print(f"\nFound {len(images)} test images")

    # Test SAM 3
    for image_path in images:
        test_sam3_semantic(args.model, str(image_path), device)

    print("\n" + "=" * 60)
    print("SAM 3 Test Complete!")
    print("=" * 60)
    print("\nSAM 3 features you can use:")
    print("- Text prompts: segment('car', 'person with red shirt')")
    print("- Image exemplars: show an example, find similar")
    print("- Video tracking: track objects across frames")


if __name__ == "__main__":
    main()
