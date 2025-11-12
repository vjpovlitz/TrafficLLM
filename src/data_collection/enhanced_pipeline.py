"""
Enhanced Traffic Camera Data Collection Pipeline
Integrates camera management, image validation, and metadata collection
"""
import os
import sys
import argparse
import logging
import schedule
import time
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from camera_manager import CameraManager
from image_validator import ImageValidator, ROIDetector
from metadata_collector import MetadataCollector, EnhancedCameraCapture

class EnhancedDataCollectionPipeline:
    """Complete data collection pipeline with validation and metadata"""
    
    def __init__(self, config_path: str = "camera_config.json",
                 output_dir: str = "enhanced_screenshots",
                 weather_api_key: str = None,
                 fullscreen_video: bool = False):

        # Setup directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Store fullscreen mode setting
        self.fullscreen_video = fullscreen_video
        
        # Create subdirectories
        (self.output_dir / "raw").mkdir(exist_ok=True)
        (self.output_dir / "validated").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.camera_manager = CameraManager(config_path, str(self.output_dir / "raw"))
        self.image_validator = ImageValidator()
        self.metadata_collector = MetadataCollector(weather_api_key)
        self.roi_detector = ROIDetector()
        
        # Enhanced capture system
        self.enhanced_capture = EnhancedCameraCapture(
            self.camera_manager, 
            self.metadata_collector, 
            self.image_validator
        )
        
        # Statistics tracking
        self.session_stats = {
            "session_start": datetime.now().isoformat(),
            "total_captures": 0,
            "successful_captures": 0,
            "valid_images": 0,
            "average_quality": 0.0,
            "cameras_status": {}
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create logger
        logger = logging.getLogger("enhanced_pipeline")
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(
            log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        """Capture and process a single camera with full pipeline"""
        self.logger.info(f"Starting enhanced capture for camera: {camera_id}")
        
        # Capture with metadata
        result = self.enhanced_capture.capture_with_metadata(camera_id)
        
        # Update statistics
        self.session_stats["total_captures"] += 1
        
        if result.get("success", False):
            self.session_stats["successful_captures"] += 1
            
            if result.get("is_valid", False):
                self.session_stats["valid_images"] += 1
                
                # Move valid image to validated directory
                source_path = Path(result["image_path"])
                validated_path = self.output_dir / "validated" / source_path.name
                
                try:
                    import shutil
                    shutil.copy2(source_path, validated_path)
                    result["validated_path"] = str(validated_path)
                    
                    # Detect ROI for valid images
                    optimal_roi = self.roi_detector.suggest_optimal_roi(str(validated_path))
                    if optimal_roi:
                        result["suggested_roi"] = optimal_roi
                        self.logger.info(f"Suggested ROI for {camera_id}: {optimal_roi}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to copy validated image: {e}")
        
        # Update camera statistics
        camera_name = self.camera_manager.cameras.get(camera_id, type('obj', (object,), {'name': camera_id})).name
        if camera_name not in self.session_stats["cameras_status"]:
            self.session_stats["cameras_status"][camera_name] = {
                "attempts": 0,
                "successes": 0,
                "valid_images": 0,
                "total_quality": 0.0
            }
        
        stats = self.session_stats["cameras_status"][camera_name]
        stats["attempts"] += 1
        
        if result.get("success", False):
            stats["successes"] += 1
            
            if result.get("is_valid", False):
                stats["valid_images"] += 1
                stats["total_quality"] += result.get("quality_score", 0.0)
        
        return result
    
    def capture_all_cameras(self, concurrent: bool = True) -> Dict:
        """Capture from all enabled cameras"""
        self.logger.info("Starting full camera capture cycle")
        
        if concurrent:
            # Use the camera manager's concurrent capture
            capture_results = self.camera_manager.capture_all_cameras(
                max_workers=4,
                fullscreen_video=self.fullscreen_video
            )
            
            # Process each result through validation and metadata
            processed_results = {}
            for camera_id, capture_metadata in capture_results.items():
                if capture_metadata.get("success", False):
                    # Process through validation pipeline
                    image_path = capture_metadata["file_path"]
                    validation_result = self.image_validator.validate_image(image_path)
                    
                    # Collect metadata
                    camera = self.camera_manager.cameras[camera_id]
                    camera_config = {
                        "camera_id": camera_id,
                        "name": camera.name,
                        "url": camera.url,
                        "location": camera.location,
                        "region": camera.region
                    }
                    
                    quality_metrics = {
                        "quality_score": validation_result.get("quality_score", 0.0),
                        "blur_score": validation_result.get("blur_score", 0.0),
                        "brightness_score": validation_result.get("brightness_score", 0.0),
                        "contrast_score": validation_result.get("contrast_score", 0.0)
                    }
                    
                    complete_metadata = self.metadata_collector.collect_complete_metadata(
                        camera_config, image_path, validation_result, quality_metrics
                    )
                    
                    # Save metadata
                    metadata_path = self.metadata_collector.create_metadata_filename(image_path)
                    self.metadata_collector.save_metadata(complete_metadata, metadata_path)
                    
                    processed_results[camera_id] = {
                        "success": True,
                        "image_path": image_path,
                        "metadata_path": metadata_path,
                        "is_valid": validation_result["is_valid"],
                        "quality_score": validation_result["quality_score"]
                    }
                    
                    # Update statistics
                    self.session_stats["total_captures"] += 1
                    self.session_stats["successful_captures"] += 1
                    
                    if validation_result["is_valid"]:
                        self.session_stats["valid_images"] += 1
                else:
                    processed_results[camera_id] = {
                        "success": False,
                        "error": capture_metadata.get("error", "Unknown error")
                    }
                    self.session_stats["total_captures"] += 1
            
            return processed_results
        else:
            # Sequential capture
            results = {}
            for camera_id in self.camera_manager.cameras:
                if self.camera_manager.cameras[camera_id].enabled:
                    results[camera_id] = self.capture_single_camera(camera_id)
            return results
    
    def generate_session_report(self) -> Dict:
        """Generate comprehensive session report"""
        # Calculate average quality
        total_quality = 0.0
        quality_count = 0
        
        for camera_stats in self.session_stats["cameras_status"].values():
            if camera_stats["valid_images"] > 0:
                avg_quality = camera_stats["total_quality"] / camera_stats["valid_images"]
                total_quality += avg_quality * camera_stats["valid_images"]
                quality_count += camera_stats["valid_images"]
        
        if quality_count > 0:
            self.session_stats["average_quality"] = total_quality / quality_count
        
        # Add success rates
        for camera_name, stats in self.session_stats["cameras_status"].items():
            if stats["attempts"] > 0:
                stats["success_rate"] = stats["successes"] / stats["attempts"]
                stats["validation_rate"] = stats["valid_images"] / stats["successes"] if stats["successes"] > 0 else 0
                stats["average_quality"] = stats["total_quality"] / stats["valid_images"] if stats["valid_images"] > 0 else 0
        
        # Add session summary
        self.session_stats["session_end"] = datetime.now().isoformat()
        self.session_stats["success_rate"] = (
            self.session_stats["successful_captures"] / self.session_stats["total_captures"]
            if self.session_stats["total_captures"] > 0 else 0
        )
        self.session_stats["validation_rate"] = (
            self.session_stats["valid_images"] / self.session_stats["successful_captures"]
            if self.session_stats["successful_captures"] > 0 else 0
        )
        
        return self.session_stats
    
    def save_session_report(self, report: Dict = None):
        """Save session report to file"""
        if report is None:
            report = self.generate_session_report()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / "reports" / f"session_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Session report saved to: {report_path}")
        return report_path
    
    def run_scheduled_collection(self, interval_minutes: int = 15):
        """Run scheduled data collection"""
        self.logger.info(f"Starting scheduled collection every {interval_minutes} minutes")
        
        # Schedule the job
        schedule.every(interval_minutes).minutes.do(self._scheduled_capture)
        
        # Run immediately
        self._scheduled_capture()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            self.logger.info("Stopping scheduled collection")
            self._final_report()
    
    def _scheduled_capture(self):
        """Execute scheduled capture"""
        self.logger.info("Executing scheduled capture")
        results = self.capture_all_cameras(concurrent=True)
        
        # Log results
        successful = sum(1 for r in results.values() if r.get("success", False))
        valid = sum(1 for r in results.values() if r.get("is_valid", False))
        
        self.logger.info(f"Capture cycle complete: {successful}/{len(results)} successful, {valid} valid images")
    
    def _final_report(self):
        """Generate and save final session report"""
        report = self.generate_session_report()
        report_path = self.save_session_report(report)
        
        self.logger.info("Final Session Report:")
        self.logger.info(f"  Total captures: {report['total_captures']}")
        self.logger.info(f"  Successful captures: {report['successful_captures']}")
        self.logger.info(f"  Valid images: {report['valid_images']}")
        self.logger.info(f"  Success rate: {report['success_rate']:.2%}")
        self.logger.info(f"  Validation rate: {report['validation_rate']:.2%}")
        self.logger.info(f"  Average quality: {report['average_quality']:.2f}")
        self.logger.info(f"  Report saved to: {report_path}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Enhanced Traffic Camera Data Collection Pipeline")
    
    parser.add_argument("--config", default="camera_config.json", 
                       help="Camera configuration file")
    parser.add_argument("--output", default="enhanced_screenshots",
                       help="Output directory")
    parser.add_argument("--weather-key", help="OpenWeatherMap API key")
    parser.add_argument("--interval", type=int, default=15,
                       help="Capture interval in minutes for scheduled mode")
    parser.add_argument("--mode", choices=["single", "scheduled"], default="single",
                       help="Capture mode: single run or scheduled")
    parser.add_argument("--camera", help="Specific camera ID to capture (single mode only)")
    parser.add_argument("--fullscreen", action="store_true",
                       help="Enable fullscreen video capture mode (video-only, no webpage elements)")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = EnhancedDataCollectionPipeline(
        config_path=args.config,
        output_dir=args.output,
        weather_api_key=args.weather_key,
        fullscreen_video=args.fullscreen
    )
    
    if args.mode == "single":
        if args.camera:
            # Single camera capture
            result = pipeline.capture_single_camera(args.camera)
            print(f"Capture result: {result}")
        else:
            # All cameras capture
            results = pipeline.capture_all_cameras()
            print(f"Captured {len(results)} cameras")
            
        # Generate and save report
        pipeline._final_report()
        
    elif args.mode == "scheduled":
        # Scheduled capture
        pipeline.run_scheduled_collection(args.interval)

if __name__ == "__main__":
    main()