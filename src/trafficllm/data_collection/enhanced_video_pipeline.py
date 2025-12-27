"""
Enhanced Video Pipeline Integration
Integrates fullscreen video capture with existing traffic camera data collection pipeline
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import schedule
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fullscreen_video_capture import FullscreenVideoCapture, capture_video_screenshot
from camera_manager import CameraManager, TrafficCamera
from image_validator import ImageValidator
from metadata_collector import MetadataCollector


class VideoTrafficCamera(TrafficCamera):
    """Extended TrafficCamera class for video-based cameras"""

    def __init__(self, camera_id: str, name: str, url: str, location: str = "",
                 region: str = "", enabled: bool = True, capture_type: str = "video",
                 fullscreen_enabled: bool = True, wait_time: int = 5):
        super().__init__(camera_id, name, url, location, region, enabled)
        self.capture_type = capture_type  # "video" or "static"
        self.fullscreen_enabled = fullscreen_enabled
        self.wait_time = wait_time

    def to_dict(self) -> Dict:
        """Convert to dictionary with video-specific fields"""
        base_dict = super().to_dict()
        base_dict.update({
            'capture_type': self.capture_type,
            'fullscreen_enabled': self.fullscreen_enabled,
            'wait_time': self.wait_time
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict) -> 'VideoTrafficCamera':
        """Create VideoTrafficCamera from dictionary"""
        return cls(
            camera_id=data['id'],
            name=data['name'],
            url=data['url'],
            location=data.get('location', ''),
            region=data.get('region', ''),
            enabled=data.get('enabled', True),
            capture_type=data.get('capture_type', 'video'),
            fullscreen_enabled=data.get('fullscreen_enabled', True),
            wait_time=data.get('wait_time', 5)
        )


class EnhancedVideoCameraManager(CameraManager):
    """Enhanced camera manager that handles both static and video cameras"""

    def __init__(self, config_path: str = "video_camera_config.json", output_dir: str = "video_screenshots"):
        super().__init__(config_path, output_dir)
        self.fullscreen_capturer = FullscreenVideoCapture(
            headless=True,
            screenshot_dir=output_dir
        )

    def load_cameras(self):
        """Load cameras from config file with video support"""
        if not os.path.exists(self.config_path):
            # Create a sample config for video cameras
            self._create_sample_video_config()

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            self.cameras = {}
            for camera_data in config.get('cameras', []):
                if camera_data.get('capture_type') == 'video':
                    camera = VideoTrafficCamera.from_dict(camera_data)
                else:
                    camera = TrafficCamera.from_dict(camera_data)
                self.cameras[camera.id] = camera

            self.logger.info(f"Loaded {len(self.cameras)} cameras from config")
            video_cameras = sum(1 for c in self.cameras.values()
                              if isinstance(c, VideoTrafficCamera))
            self.logger.info(f"Video cameras: {video_cameras}")

        except Exception as e:
            self.logger.error(f"Failed to load camera config: {e}")
            self.cameras = {}

    def _create_sample_video_config(self):
        """Create a sample configuration file for video cameras"""
        sample_config = {
            "cameras": [
                {
                    "id": "youtube_traffic_1",
                    "name": "YouTube Live Traffic Camera 1",
                    "url": "https://www.youtube.com/watch?v=1EiC9bvVGnk",
                    "location": "Online Stream",
                    "region": "Sample",
                    "enabled": True,
                    "capture_type": "video",
                    "fullscreen_enabled": True,
                    "wait_time": 5
                },
                {
                    "id": "sample_traffic_video",
                    "name": "Sample Traffic Video",
                    "url": "https://vimeo.com/34741214",
                    "location": "Sample Location",
                    "region": "Demo",
                    "enabled": False,
                    "capture_type": "video",
                    "fullscreen_enabled": True,
                    "wait_time": 3
                }
            ]
        }

        with open(self.config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)

        self.logger.info(f"Created sample video camera config: {self.config_path}")

    def capture_camera(self, camera_id: str) -> Dict:
        """Capture from a single camera (handles both static and video)"""
        if camera_id not in self.cameras:
            return {"success": False, "error": f"Camera {camera_id} not found"}

        camera = self.cameras[camera_id]

        if not camera.enabled:
            return {"success": False, "error": f"Camera {camera_id} is disabled"}

        try:
            if isinstance(camera, VideoTrafficCamera) and camera.capture_type == "video":
                # Use fullscreen video capture
                result = self._capture_video_camera(camera)
            else:
                # Use standard static capture
                result = super().capture_camera(camera_id)

            return result

        except Exception as e:
            self.logger.error(f"Failed to capture {camera_id}: {e}")
            return {"success": False, "error": str(e)}

    def _capture_video_camera(self, camera: VideoTrafficCamera) -> Dict:
        """Capture from a video camera using fullscreen method"""
        self.logger.info(f"Capturing video camera: {camera.name}")

        try:
            # Use fullscreen capture if enabled
            if camera.fullscreen_enabled:
                result = capture_video_screenshot(
                    camera.url,
                    headless=True,
                    wait_time=camera.wait_time,
                    screenshot_dir=self.output_dir
                )
            else:
                # Use basic video capture (non-fullscreen)
                result = self.fullscreen_capturer.capture_fullscreen_video(camera.url, camera.wait_time)

            # Add camera metadata
            if result['success']:
                result['camera_id'] = camera.id
                result['camera_name'] = camera.name
                result['camera_location'] = camera.location
                result['camera_region'] = camera.region

            return result

        except Exception as e:
            self.logger.error(f"Video capture failed for {camera.name}: {e}")
            return {"success": False, "error": str(e)}


class EnhancedVideoDataPipeline:
    """Complete data pipeline for video-based traffic cameras"""

    def __init__(self, config_path: str = "video_camera_config.json",
                 output_dir: str = "enhanced_video_screenshots",
                 weather_api_key: str = None):

        # Setup directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / "raw").mkdir(exist_ok=True)
        (self.output_dir / "validated").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize components
        self.camera_manager = EnhancedVideoCameraManager(
            config_path,
            str(self.output_dir / "raw")
        )
        self.image_validator = ImageValidator()
        self.metadata_collector = MetadataCollector(weather_api_key)

        # Statistics tracking
        self.session_stats = {
            "session_start": datetime.now().isoformat(),
            "total_captures": 0,
            "successful_captures": 0,
            "valid_images": 0,
            "fullscreen_successes": 0,
            "video_cameras": 0,
            "static_cameras": 0,
            "average_quality": 0.0,
            "cameras_status": {}
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger("enhanced_video_pipeline")
        logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(
            log_dir / f"video_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def capture_single_camera(self, camera_id: str) -> Dict:
        """Capture and process a single camera"""
        self.logger.info(f"Starting capture for camera: {camera_id}")

        # Capture
        result = self.camera_manager.capture_camera(camera_id)

        # Update statistics
        self.session_stats["total_captures"] += 1

        if result.get("success", False):
            self.session_stats["successful_captures"] += 1

            # Track fullscreen success for video cameras
            if result.get("fullscreen_success", False):
                self.session_stats["fullscreen_successes"] += 1

            # Validate image
            if result.get("screenshot_path"):
                validation_result = self.image_validator.validate_image(result["screenshot_path"])
                result["validation"] = validation_result

                if validation_result.get("is_valid", False):
                    self.session_stats["valid_images"] += 1

                    # Move valid image to validated directory
                    self._move_to_validated(result["screenshot_path"], result)

                # Collect metadata
                self._collect_and_save_metadata(camera_id, result, validation_result)

        # Update camera statistics
        self._update_camera_stats(camera_id, result)

        return result

    def capture_all_cameras(self) -> Dict:
        """Capture from all enabled cameras"""
        self.logger.info("Starting capture cycle for all cameras")

        results = {}
        for camera_id, camera in self.camera_manager.cameras.items():
            if camera.enabled:
                results[camera_id] = self.capture_single_camera(camera_id)

                # Track camera types
                if isinstance(camera, VideoTrafficCamera):
                    self.session_stats["video_cameras"] += 1
                else:
                    self.session_stats["static_cameras"] += 1

        return results

    def _move_to_validated(self, source_path: str, result: Dict):
        """Move valid image to validated directory"""
        try:
            import shutil
            source_path_obj = Path(source_path)
            validated_path = self.output_dir / "validated" / source_path_obj.name
            shutil.copy2(source_path, validated_path)
            result["validated_path"] = str(validated_path)
        except Exception as e:
            self.logger.warning(f"Failed to move validated image: {e}")

    def _collect_and_save_metadata(self, camera_id: str, result: Dict, validation_result: Dict):
        """Collect and save comprehensive metadata"""
        try:
            camera = self.camera_manager.cameras[camera_id]

            camera_config = {
                "camera_id": camera_id,
                "name": camera.name,
                "url": camera.url,
                "location": camera.location,
                "region": camera.region,
                "capture_type": getattr(camera, 'capture_type', 'static')
            }

            quality_metrics = {
                "quality_score": validation_result.get("quality_score", 0.0),
                "blur_score": validation_result.get("blur_score", 0.0),
                "brightness_score": validation_result.get("brightness_score", 0.0),
                "contrast_score": validation_result.get("contrast_score", 0.0)
            }

            # Add video-specific metadata
            if isinstance(camera, VideoTrafficCamera):
                quality_metrics.update({
                    "fullscreen_success": result.get("fullscreen_success", False),
                    "player_detected": result.get("player_info") is not None,
                    "video_player_info": result.get("player_info", {})
                })

            complete_metadata = self.metadata_collector.collect_complete_metadata(
                camera_config,
                result["screenshot_path"],
                validation_result,
                quality_metrics
            )

            # Save metadata
            metadata_path = self.metadata_collector.create_metadata_filename(result["screenshot_path"])
            self.metadata_collector.save_metadata(complete_metadata, metadata_path)

            result["metadata_path"] = metadata_path

        except Exception as e:
            self.logger.warning(f"Failed to collect metadata for {camera_id}: {e}")

    def _update_camera_stats(self, camera_id: str, result: Dict):
        """Update camera-specific statistics"""
        camera = self.camera_manager.cameras[camera_id]
        camera_name = camera.name

        if camera_name not in self.session_stats["cameras_status"]:
            self.session_stats["cameras_status"][camera_name] = {
                "attempts": 0,
                "successes": 0,
                "valid_images": 0,
                "fullscreen_successes": 0,
                "total_quality": 0.0,
                "camera_type": getattr(camera, 'capture_type', 'static')
            }

        stats = self.session_stats["cameras_status"][camera_name]
        stats["attempts"] += 1

        if result.get("success", False):
            stats["successes"] += 1

            if result.get("fullscreen_success", False):
                stats["fullscreen_successes"] += 1

            if result.get("validation", {}).get("is_valid", False):
                stats["valid_images"] += 1
                stats["total_quality"] += result["validation"].get("quality_score", 0.0)

    def generate_session_report(self) -> Dict:
        """Generate comprehensive session report"""
        # Calculate averages
        if self.session_stats["valid_images"] > 0:
            total_quality = sum(
                stats["total_quality"] for stats in self.session_stats["cameras_status"].values()
            )
            self.session_stats["average_quality"] = total_quality / self.session_stats["valid_images"]

        # Add success rates
        for camera_name, stats in self.session_stats["cameras_status"].items():
            if stats["attempts"] > 0:
                stats["success_rate"] = stats["successes"] / stats["attempts"]
                stats["validation_rate"] = (
                    stats["valid_images"] / stats["successes"] if stats["successes"] > 0 else 0
                )
                stats["fullscreen_rate"] = (
                    stats["fullscreen_successes"] / stats["attempts"] if stats["attempts"] > 0 else 0
                )
                stats["average_quality"] = (
                    stats["total_quality"] / stats["valid_images"] if stats["valid_images"] > 0 else 0
                )

        # Session summary
        self.session_stats["session_end"] = datetime.now().isoformat()

        if self.session_stats["total_captures"] > 0:
            self.session_stats["overall_success_rate"] = (
                self.session_stats["successful_captures"] / self.session_stats["total_captures"]
            )
            self.session_stats["overall_validation_rate"] = (
                self.session_stats["valid_images"] / self.session_stats["successful_captures"]
                if self.session_stats["successful_captures"] > 0 else 0
            )
            self.session_stats["fullscreen_success_rate"] = (
                self.session_stats["fullscreen_successes"] / self.session_stats["total_captures"]
            )

        return self.session_stats

    def save_session_report(self, report: Dict = None) -> str:
        """Save session report to file"""
        if report is None:
            report = self.generate_session_report()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / "reports" / f"video_session_report_{timestamp}.json"

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Session report saved to: {report_path}")
        return str(report_path)

    def run_scheduled_collection(self, interval_minutes: int = 15):
        """Run scheduled data collection"""
        self.logger.info(f"Starting scheduled video collection every {interval_minutes} minutes")

        schedule.every(interval_minutes).minutes.do(self._scheduled_capture)

        # Run immediately
        self._scheduled_capture()

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            self.logger.info("Stopping scheduled collection")
            self._final_report()

    def _scheduled_capture(self):
        """Execute scheduled capture"""
        self.logger.info("Executing scheduled video capture")
        results = self.capture_all_cameras()

        # Log results
        successful = sum(1 for r in results.values() if r.get("success", False))
        valid = sum(1 for r in results.values() if r.get("validation", {}).get("is_valid", False))
        fullscreen = sum(1 for r in results.values() if r.get("fullscreen_success", False))

        self.logger.info(
            f"Capture cycle complete: {successful}/{len(results)} successful, "
            f"{valid} valid images, {fullscreen} fullscreen successes"
        )

    def _final_report(self):
        """Generate and save final session report"""
        report = self.generate_session_report()
        report_path = self.save_session_report(report)

        self.logger.info("Final Video Pipeline Session Report:")
        self.logger.info(f"  Total captures: {report['total_captures']}")
        self.logger.info(f"  Successful captures: {report['successful_captures']}")
        self.logger.info(f"  Valid images: {report['valid_images']}")
        self.logger.info(f"  Fullscreen successes: {report['fullscreen_successes']}")
        self.logger.info(f"  Video cameras: {report['video_cameras']}")
        self.logger.info(f"  Success rate: {report.get('overall_success_rate', 0):.2%}")
        self.logger.info(f"  Fullscreen rate: {report.get('fullscreen_success_rate', 0):.2%}")
        self.logger.info(f"  Report saved to: {report_path}")


def main():
    """Main function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Video Traffic Camera Pipeline")
    parser.add_argument("--config", default="video_camera_config.json",
                       help="Video camera configuration file")
    parser.add_argument("--output", default="enhanced_video_screenshots",
                       help="Output directory")
    parser.add_argument("--weather-key", help="OpenWeatherMap API key")
    parser.add_argument("--interval", type=int, default=15,
                       help="Capture interval in minutes for scheduled mode")
    parser.add_argument("--mode", choices=["single", "scheduled"], default="single",
                       help="Capture mode")
    parser.add_argument("--camera", help="Specific camera ID to capture")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = EnhancedVideoDataPipeline(
        config_path=args.config,
        output_dir=args.output,
        weather_api_key=args.weather_key
    )

    if args.mode == "single":
        if args.camera:
            result = pipeline.capture_single_camera(args.camera)
            print(f"Capture result: {json.dumps(result, indent=2, default=str)}")
        else:
            results = pipeline.capture_all_cameras()
            print(f"Captured {len(results)} cameras")

        pipeline._final_report()

    elif args.mode == "scheduled":
        pipeline.run_scheduled_collection(args.interval)


if __name__ == "__main__":
    main()