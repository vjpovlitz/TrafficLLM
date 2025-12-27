"""
Capture Traffic Images to Organized Library

Integrates CameraManager with ImageLibrary for organized storage.
Automatically validates quality and stores in date/camera hierarchy.

Usage:
    # Single capture to library
    python capture_to_library.py --mode single --fullscreen

    # Scheduled captures (every 15 minutes)
    python capture_to_library.py --mode scheduled --interval 15 --fullscreen

    # Specific camera only
    python capture_to_library.py --mode single --camera us_50_sandy_point --fullscreen
"""

import argparse
import sys
import time
import schedule
from pathlib import Path
from datetime import datetime
import logging

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


class LibraryCapture:
    """
    Integrated capture system with automatic library storage.
    """

    def __init__(
        self,
        camera_config: Path,
        library_path: Path,
        fullscreen: bool = True,
        validate: bool = True
    ):
        """
        Initialize capture system.

        Args:
            camera_config: Path to camera configuration
            library_path: Path to image library
            fullscreen: Use fullscreen video mode
            validate: Validate and store validated copies
        """
        self.fullscreen = fullscreen
        self.validate_images = validate

        # Initialize components
        logger.info("Initializing capture system...")

        self.camera_manager = CameraManager(
            config_path=str(camera_config),
            output_dir="temp_captures"  # Temporary, will move to library
        )

        self.library = ImageLibrary(library_path)

        if self.validate_images:
            self.validator = ImageValidator()

        logger.info(f"Library: {library_path}")
        logger.info(f"Fullscreen mode: {fullscreen}")
        logger.info(f"Validation: {validate}")

    def capture_single(self, camera_id: str = None):
        """
        Capture from single camera or all cameras.

        Args:
            camera_id: Specific camera (all if None)

        Returns:
            Dictionary with capture results
        """
        if camera_id:
            cameras = {camera_id: self.camera_manager.cameras[camera_id]}
        else:
            cameras = {
                cid: cam for cid, cam in self.camera_manager.cameras.items()
                if cam.enabled
            }

        logger.info(f"\n{'='*80}")
        logger.info(f"Starting capture: {len(cameras)} camera(s)")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}\n")

        results = {
            'timestamp': datetime.now().isoformat(),
            'cameras': {},
            'summary': {
                'total': len(cameras),
                'successful': 0,
                'validated': 0,
                'failed': 0
            }
        }

        for cid, camera in cameras.items():
            logger.info(f"Capturing {camera.name} ({cid})...")

            try:
                # Capture screenshot
                success, image_path, metadata = self.camera_manager.capture_screenshot(
                    cid,
                    camera,
                    fullscreen_video=self.fullscreen
                )

                if not success:
                    logger.error(f"  ❌ Failed: {metadata.get('error')}")
                    results['cameras'][cid] = {'success': False, 'error': metadata.get('error')}
                    results['summary']['failed'] += 1
                    continue

                logger.info(f"  ✅ Captured: {metadata['file_size'] / 1024:.1f} KB")

                # Store in library (raw)
                capture_time = datetime.fromisoformat(metadata['timestamp'])

                record = self.library.store_image(
                    camera_id=cid,
                    camera_name=camera.name,
                    image_path=Path(image_path),
                    capture_time=capture_time,
                    category="raw",
                    metadata={
                        'location': camera.location,
                        'region': camera.region,
                        'fullscreen_mode': self.fullscreen
                    },
                    move=True  # Move from temp to library
                )

                logger.info(f"  📁 Stored: {Path(record.file_path).parent.name}/")

                # Validate if enabled
                validated = False
                quality_score = None

                if self.validate_images:
                    validation_result = self.validator.validate_image(record.file_path)
                    is_valid = validation_result["is_valid"]
                    quality_score = validation_result["quality_score"]
                    metrics = {
                        'brightness': validation_result.get('brightness_score', 0),
                        'contrast': validation_result.get('contrast_score', 0),
                        'blur': validation_result.get('blur_score', 0)
                    }

                    if is_valid:
                        # Store validated copy
                        validated_record = self.library.store_image(
                            camera_id=cid,
                            camera_name=camera.name,
                            image_path=Path(record.file_path),
                            capture_time=capture_time,
                            category="validated",
                            metadata={
                                'location': camera.location,
                                'region': camera.region,
                                'quality_score': quality_score,
                                'fullscreen_mode': self.fullscreen,
                                'validation_metrics': metrics
                            }
                        )

                        logger.info(f"  ✓ Validated: score {quality_score:.3f}")
                        validated = True
                        results['summary']['validated'] += 1
                    else:
                        logger.info(f"  ⚠️  Quality too low: {quality_score:.3f}")

                results['cameras'][cid] = {
                    'success': True,
                    'file_path': record.file_path,
                    'size_kb': record.file_size / 1024,
                    'validated': validated,
                    'quality_score': quality_score
                }

                results['summary']['successful'] += 1

            except Exception as e:
                logger.error(f"  ❌ Error: {e}")
                results['cameras'][cid] = {'success': False, 'error': str(e)}
                results['summary']['failed'] += 1

        # Summary
        logger.info(f"\n{'='*80}")
        logger.info("Capture Summary")
        logger.info(f"{'='*80}")
        logger.info(f"Total: {results['summary']['total']}")
        logger.info(f"Successful: {results['summary']['successful']}")
        logger.info(f"Validated: {results['summary']['validated']}")
        logger.info(f"Failed: {results['summary']['failed']}")

        # Library stats
        stats = self.library.get_camera_stats(days=1)
        logger.info(f"\nLibrary (last 24h):")
        logger.info(f"  Total images: {stats['total_images']}")
        logger.info(f"  Total size: {stats['total_size_mb']:.1f} MB")
        logger.info(f"  Validated: {stats['validated']}")

        return results

    def run_scheduled(self, interval_minutes: int):
        """
        Run scheduled captures.

        Args:
            interval_minutes: Capture interval in minutes
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Scheduled Capture Mode")
        logger.info(f"{'='*80}")
        logger.info(f"Interval: Every {interval_minutes} minutes")
        logger.info(f"Press Ctrl+C to stop")
        logger.info(f"{'='*80}\n")

        # Schedule job
        schedule.every(interval_minutes).minutes.do(self.capture_single)

        # Run immediately on start
        self.capture_single()

        # Run scheduler
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n\nScheduled capture stopped by user")


def main():
    parser = argparse.ArgumentParser(
        description="Capture traffic images to organized library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single capture, all cameras, fullscreen
  python capture_to_library.py --mode single --fullscreen

  # Scheduled captures every 15 minutes
  python capture_to_library.py --mode scheduled --interval 15 --fullscreen

  # Single camera only
  python capture_to_library.py --mode single --camera us_50_sandy_point

  # Without validation (faster)
  python capture_to_library.py --mode single --no-validate
        """
    )

    parser.add_argument(
        '--mode', '-m',
        choices=['single', 'scheduled'],
        required=True,
        help='Capture mode'
    )

    parser.add_argument(
        '--camera', '-c',
        help='Specific camera ID (all cameras if not specified)'
    )

    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=15,
        help='Capture interval in minutes for scheduled mode (default: 15)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/camera_config.json'),
        help='Path to camera config (default: config/camera_config.json)'
    )

    parser.add_argument(
        '--library',
        type=Path,
        default=Path('data/image_library'),
        help='Path to image library (default: data/image_library)'
    )

    parser.add_argument(
        '--fullscreen',
        action='store_true',
        default=True,
        help='Use fullscreen video mode (default: True)'
    )

    parser.add_argument(
        '--no-fullscreen',
        dest='fullscreen',
        action='store_false',
        help='Disable fullscreen video mode'
    )

    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip quality validation'
    )

    args = parser.parse_args()

    # Validate config
    config_path = project_root / args.config
    if not config_path.exists():
        logger.error(f"Camera config not found: {config_path}")
        return 1

    # Initialize capture system
    try:
        capture_system = LibraryCapture(
            camera_config=config_path,
            library_path=project_root / args.library,
            fullscreen=args.fullscreen,
            validate=not args.no_validate
        )
    except Exception as e:
        logger.error(f"Failed to initialize capture system: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run capture
    try:
        if args.mode == 'single':
            capture_system.capture_single(camera_id=args.camera)
        elif args.mode == 'scheduled':
            capture_system.run_scheduled(interval_minutes=args.interval)

        return 0

    except Exception as e:
        logger.error(f"Capture failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
