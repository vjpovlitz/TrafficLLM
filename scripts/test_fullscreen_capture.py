"""
Test Fullscreen Screenshot Capture

Tests the fullscreen video capture mode and validates image quality.
Compares standard vs fullscreen mode side-by-side.

Usage:
    # Test single camera
    python test_fullscreen_capture.py --camera us_50_sandy_point

    # Test all cameras
    python test_fullscreen_capture.py --all

    # Save to library
    python test_fullscreen_capture.py --camera us_50_sandy_point --use-library
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from trafficllm.data_collection.camera_manager import CameraManager
from trafficllm.data_collection.image_validator import ImageValidator
from trafficllm.data_collection.image_library import ImageLibrary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compare_modes(camera_manager, camera_id, output_dir):
    """
    Capture and compare standard vs fullscreen mode.

    Args:
        camera_manager: CameraManager instance
        camera_id: Camera to test
        output_dir: Directory for test outputs

    Returns:
        Dictionary with comparison results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = camera_manager.cameras.get(camera_id)
    if not camera:
        logger.error(f"Camera {camera_id} not found")
        return None

    results = {
        'camera_id': camera_id,
        'camera_name': camera.name,
        'timestamp': datetime.now().isoformat(),
        'standard': {},
        'fullscreen': {}
    }

    logger.info(f"\n{'='*80}")
    logger.info(f"Testing Camera: {camera.name} ({camera_id})")
    logger.info(f"URL: {camera.url}")
    logger.info(f"{'='*80}\n")

    # Test 1: Standard capture
    logger.info("Test 1: Standard capture (webpage + video)...")
    success_std, path_std, metadata_std = camera_manager.capture_screenshot(
        camera_id,
        camera,
        fullscreen_video=False
    )

    if success_std:
        logger.info(f"✅ Standard capture successful")
        logger.info(f"   Path: {path_std}")
        logger.info(f"   Size: {metadata_std['file_size'] / 1024:.1f} KB")
        logger.info(f"   Duration: {metadata_std['capture_duration']:.2f}s")

        # Get image dimensions
        with Image.open(path_std) as img:
            width_std, height_std = img.size
            logger.info(f"   Dimensions: {width_std}x{height_std}")

        results['standard'] = {
            'success': True,
            'path': path_std,
            'size_kb': metadata_std['file_size'] / 1024,
            'width': width_std,
            'height': height_std,
            'duration': metadata_std['capture_duration']
        }

        # Copy to comparison folder
        import shutil
        shutil.copy(path_std, output_dir / f"{camera_id}_standard.png")
    else:
        logger.error(f"❌ Standard capture failed: {metadata_std.get('error')}")
        results['standard']['success'] = False

    # Wait between captures
    import time
    time.sleep(2)

    # Test 2: Fullscreen capture
    logger.info("\nTest 2: Fullscreen capture (video only, maximized)...")
    success_fs, path_fs, metadata_fs = camera_manager.capture_screenshot(
        camera_id,
        camera,
        fullscreen_video=True
    )

    if success_fs:
        logger.info(f"✅ Fullscreen capture successful")
        logger.info(f"   Path: {path_fs}")
        logger.info(f"   Size: {metadata_fs['file_size'] / 1024:.1f} KB")
        logger.info(f"   Duration: {metadata_fs['capture_duration']:.2f}s")

        # Get image dimensions
        with Image.open(path_fs) as img:
            width_fs, height_fs = img.size
            logger.info(f"   Dimensions: {width_fs}x{height_fs}")

        results['fullscreen'] = {
            'success': True,
            'path': path_fs,
            'size_kb': metadata_fs['file_size'] / 1024,
            'width': width_fs,
            'height': height_fs,
            'duration': metadata_fs['capture_duration']
        }

        # Copy to comparison folder
        import shutil
        shutil.copy(path_fs, output_dir / f"{camera_id}_fullscreen.png")
    else:
        logger.error(f"❌ Fullscreen capture failed: {metadata_fs.get('error')}")
        results['fullscreen']['success'] = False

    # Comparison
    if success_std and success_fs:
        logger.info(f"\n{'='*80}")
        logger.info("Comparison Results:")
        logger.info(f"{'='*80}")

        size_increase = (results['fullscreen']['size_kb'] / results['standard']['size_kb'] - 1) * 100
        logger.info(f"File size increase: {size_increase:+.1f}%")
        logger.info(f"  Standard: {results['standard']['size_kb']:.1f} KB")
        logger.info(f"  Fullscreen: {results['fullscreen']['size_kb']:.1f} KB")

        pixel_increase = (
            (results['fullscreen']['width'] * results['fullscreen']['height']) /
            (results['standard']['width'] * results['standard']['height']) - 1
        ) * 100
        logger.info(f"\nPixel count increase: {pixel_increase:+.1f}%")
        logger.info(f"  Standard: {results['standard']['width']}x{results['standard']['height']}")
        logger.info(f"  Fullscreen: {results['fullscreen']['width']}x{results['fullscreen']['height']}")

        logger.info(f"\nComparison images saved to: {output_dir}/")
        logger.info(f"  - {camera_id}_standard.png")
        logger.info(f"  - {camera_id}_fullscreen.png")

        results['improvement'] = {
            'size_increase_pct': size_increase,
            'pixel_increase_pct': pixel_increase
        }

    return results


def validate_quality(image_path, validator):
    """
    Validate image quality.

    Args:
        image_path: Path to image
        validator: ImageValidator instance

    Returns:
        Quality metrics dict
    """
    logger.info(f"\nValidating quality: {Path(image_path).name}")

    is_valid, quality_score, metrics = validator.validate_image(image_path)

    logger.info(f"  Valid: {'✅ Yes' if is_valid else '❌ No'}")
    logger.info(f"  Quality Score: {quality_score:.3f}")
    logger.info(f"  Metrics:")
    logger.info(f"    - Brightness: {metrics['brightness']:.3f}")
    logger.info(f"    - Contrast: {metrics['contrast']:.3f}")
    logger.info(f"    - Blur (Laplacian): {metrics['blur']:.3f}")
    logger.info(f"    - Edge Density: {metrics['edge_density']:.3f}")

    return {
        'valid': is_valid,
        'score': quality_score,
        'metrics': metrics
    }


def test_with_library(camera_manager, camera_id, library_root):
    """
    Test capture and store in ImageLibrary.

    Args:
        camera_manager: CameraManager instance
        camera_id: Camera to test
        library_root: Path to image library

    Returns:
        ImageRecord
    """
    library = ImageLibrary(library_root)

    camera = camera_manager.cameras[camera_id]

    logger.info(f"\n{'='*80}")
    logger.info(f"Testing with ImageLibrary: {camera.name}")
    logger.info(f"Library: {library_root}")
    logger.info(f"{'='*80}\n")

    # Capture fullscreen
    success, image_path, metadata = camera_manager.capture_screenshot(
        camera_id,
        camera,
        fullscreen_video=True
    )

    if not success:
        logger.error(f"Capture failed: {metadata.get('error')}")
        return None

    # Store in library
    logger.info("Storing in library...")
    record = library.store_image(
        camera_id=camera_id,
        camera_name=camera.name,
        image_path=Path(image_path),
        category="raw",
        metadata={
            'location': camera.location,
            'region': camera.region,
            'fullscreen_mode': True
        }
    )

    logger.info(f"✅ Stored in library:")
    logger.info(f"   Path: {record.file_path}")
    logger.info(f"   Size: {record.file_size / 1024:.1f} KB")

    # Validate quality
    validator = ImageValidator()
    is_valid, quality_score, metrics = validator.validate_image(record.file_path)

    if is_valid:
        # Store validated copy
        validated_record = library.store_image(
            camera_id=camera_id,
            camera_name=camera.name,
            image_path=Path(record.file_path),
            category="validated",
            metadata={
                'location': camera.location,
                'region': camera.region,
                'quality_score': quality_score,
                'fullscreen_mode': True
            }
        )
        logger.info(f"✅ Validated (score: {quality_score:.3f})")
        logger.info(f"   Validated path: {validated_record.file_path}")

    # Show library stats
    stats = library.get_camera_stats(camera_id=camera_id, days=1)
    logger.info(f"\nLibrary stats for {camera_id}:")
    logger.info(f"  Total images: {stats['total_images']}")
    logger.info(f"  Validated: {stats['validated']}")
    logger.info(f"  Fullscreen: {stats['fullscreen']}")
    logger.info(f"  Total size: {stats['total_size_mb']:.1f} MB")

    return record


def main():
    parser = argparse.ArgumentParser(
        description="Test fullscreen screenshot capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single camera comparison
  python test_fullscreen_capture.py --camera us_50_sandy_point

  # Test all cameras
  python test_fullscreen_capture.py --all

  # Test with ImageLibrary integration
  python test_fullscreen_capture.py --camera us_50_sandy_point --use-library
        """
    )

    parser.add_argument(
        '--camera', '-c',
        help='Camera ID to test'
    )

    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Test all enabled cameras'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('test_captures'),
        help='Output directory for test images (default: test_captures)'
    )

    parser.add_argument(
        '--use-library',
        action='store_true',
        help='Store results in ImageLibrary'
    )

    parser.add_argument(
        '--library-path',
        type=Path,
        default=Path('data/image_library'),
        help='Path to image library (default: data/image_library)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/camera_config.json'),
        help='Path to camera config'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.camera and not args.all:
        logger.error("Must specify --camera or --all")
        return 1

    # Initialize camera manager
    logger.info("Initializing camera manager...")
    config_path = project_root / args.config
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    camera_manager = CameraManager(
        config_path=str(config_path),
        output_dir=str(args.output)
    )

    # Get cameras to test
    if args.all:
        camera_ids = [cid for cid, cam in camera_manager.cameras.items() if cam.enabled]
        logger.info(f"Testing all {len(camera_ids)} enabled cameras")
    else:
        if args.camera not in camera_manager.cameras:
            logger.error(f"Camera not found: {args.camera}")
            logger.error(f"Available cameras: {list(camera_manager.cameras.keys())}")
            return 1
        camera_ids = [args.camera]

    # Run tests
    all_results = []

    for camera_id in camera_ids:
        try:
            if args.use_library:
                # Test with library
                record = test_with_library(
                    camera_manager,
                    camera_id,
                    args.library_path
                )
                all_results.append({'camera_id': camera_id, 'record': record})
            else:
                # Test comparison
                results = compare_modes(
                    camera_manager,
                    camera_id,
                    args.output
                )
                all_results.append(results)

        except Exception as e:
            logger.error(f"Error testing {camera_id}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("Test Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Cameras tested: {len(camera_ids)}")
    logger.info(f"Successful: {sum(1 for r in all_results if r is not None)}")

    if not args.use_library and all_results:
        # Calculate averages
        successful = [r for r in all_results if r and r.get('improvement')]
        if successful:
            avg_size_increase = sum(r['improvement']['size_increase_pct'] for r in successful) / len(successful)
            avg_pixel_increase = sum(r['improvement']['pixel_increase_pct'] for r in successful) / len(successful)

            logger.info(f"\nAverage improvements:")
            logger.info(f"  File size: {avg_size_increase:+.1f}%")
            logger.info(f"  Pixel count: {avg_pixel_increase:+.1f}%")

    logger.info(f"\nOutput directory: {args.output}")

    if args.use_library:
        logger.info(f"Image library: {args.library_path}")
        logger.info(f"\nNext steps:")
        logger.info(f"1. Review images in library")
        logger.info(f"2. Run SAM 3 annotation:")
        logger.info(f"   python scripts/annotate_with_sam3.py --input {args.library_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
