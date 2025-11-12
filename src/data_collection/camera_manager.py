"""
Unified Camera Management System for Traffic Data Collection
"""
import os
import json
import datetime
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException

@dataclass
class CameraConfig:
    """Camera configuration data structure"""
    name: str
    url: str
    location: str
    region: str
    enabled: bool = True
    capture_interval: int = 15  # minutes
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    roi_coordinates: Optional[Dict] = None

class TrafficCamera:
    """Base traffic camera class"""

    def __init__(self, camera_id: str, name: str, url: str, location: str = "",
                 region: str = "", enabled: bool = True):
        self.id = camera_id
        self.name = name
        self.url = url
        self.location = location
        self.region = region
        self.enabled = enabled

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'url': self.url,
            'location': self.location,
            'region': self.region,
            'enabled': self.enabled
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TrafficCamera':
        """Create TrafficCamera from dictionary"""
        return cls(
            camera_id=data['id'],
            name=data['name'],
            url=data['url'],
            location=data.get('location', ''),
            region=data.get('region', ''),
            enabled=data.get('enabled', True)
        )

class CameraManager:
    """Unified camera management and data collection system"""
    
    def __init__(self, config_path: str = "camera_config.json", output_dir: str = "screenshots"):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load camera configurations
        self.cameras = self._load_camera_config()
        
        # Chrome options for headless operation
        self.chrome_options = self._setup_chrome_options()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup centralized logging system"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"camera_manager_{datetime.datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _setup_chrome_options(self) -> Options:
        """Configure Chrome browser options"""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        return options
    
    def _load_camera_config(self) -> Dict[str, CameraConfig]:
        """Load camera configurations from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            cameras = {}
            for camera_id, camera_data in config_data.get('cameras', {}).items():
                cameras[camera_id] = CameraConfig(**camera_data)
            
            self.logger.info(f"Loaded {len(cameras)} camera configurations")
            return cameras
            
        except FileNotFoundError:
            self.logger.warning(f"Config file {self.config_path} not found. Creating default config.")
            return self._create_default_config()
        except Exception as e:
            self.logger.error(f"Error loading camera config: {e}")
            return {}
    
    def _create_default_config(self) -> Dict[str, CameraConfig]:
        """Create default camera configuration file"""
        default_cameras = {
            "us_50_sandy_point": CameraConfig(
                name="US_50_AT_SANDY_POINT",
                url="https://mdta.maryland.gov/index.php/traffic-cameras-by-facility#camera-28",
                location="Sandy Point",
                region="Maryland"
            ),
            "us_50_ex23_md2": CameraConfig(
                name="US_50_AT_EX23_MD2",
                url="https://mdta.maryland.gov/index.php/traffic-cameras-by-facility#camera-48",
                location="Exit 23 MD2",
                region="Maryland"
            )
        }
        
        # Save default config
        self.save_camera_config(default_cameras)
        return default_cameras
    
    def save_camera_config(self, cameras: Dict[str, CameraConfig] = None):
        """Save camera configurations to JSON file"""
        if cameras is None:
            cameras = self.cameras
            
        config_data = {
            "version": "1.0",
            "last_updated": datetime.datetime.now().isoformat(),
            "cameras": {}
        }
        
        for camera_id, camera in cameras.items():
            config_data["cameras"][camera_id] = {
                "name": camera.name,
                "url": camera.url,
                "location": camera.location,
                "region": camera.region,
                "enabled": camera.enabled,
                "capture_interval": camera.capture_interval,
                "timeout": camera.timeout,
                "retry_attempts": camera.retry_attempts,
                "roi_coordinates": camera.roi_coordinates
            }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"Saved camera configuration to {self.config_path}")

    def _expand_video_fullscreen(self, driver) -> Dict:
        """
        Expand video element to fill viewport and remove all other page elements
        Returns: Dict with success status and any error messages
        """
        try:
            result = driver.execute_script("""
                var video = document.querySelector('video');
                if (!video) return {success: false, error: 'No video element found'};

                // Wait for video to be ready
                if (video.readyState < 2) {
                    return {success: false, error: 'Video not ready'};
                }

                // Clear page and show ONLY the video
                video.remove();
                document.body.innerHTML = '';
                document.body.appendChild(video);

                // Reset body and html styles
                document.body.style.margin = '0';
                document.body.style.padding = '0';
                document.body.style.overflow = 'hidden';
                document.body.style.backgroundColor = 'black';
                document.documentElement.style.margin = '0';
                document.documentElement.style.padding = '0';
                document.documentElement.style.overflow = 'hidden';

                // Make video fill entire viewport
                video.style.position = 'fixed';
                video.style.top = '0';
                video.style.left = '0';
                video.style.width = '100vw';
                video.style.height = '100vh';
                video.style.margin = '0';
                video.style.padding = '0';
                video.style.zIndex = '9999999';
                video.style.objectFit = 'contain';
                video.style.backgroundColor = 'black';

                // Remove video controls
                video.removeAttribute('controls');
                video.controls = false;

                // Ensure video is playing
                if (video.paused) video.play();

                return {
                    success: true,
                    videoSize: {
                        width: video.offsetWidth,
                        height: video.offsetHeight
                    }
                };
            """)
            return result if result else {'success': False, 'error': 'Script returned null'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def capture_screenshot(self, camera_id: str, camera: CameraConfig, fullscreen_video: bool = False) -> Tuple[bool, str, Dict]:
        """
        Capture screenshot from a single camera with retry logic

        Args:
            camera_id: Unique camera identifier
            camera: Camera configuration
            fullscreen_video: If True, expands video to fullscreen (video-only, no webpage elements)
        """
        metadata = {
            "camera_id": camera_id,
            "camera_name": camera.name,
            "timestamp": datetime.datetime.now().isoformat(),
            "location": camera.location,
            "region": camera.region,
            "success": False,
            "error": None,
            "file_path": None,
            "file_size": None,
            "capture_duration": None,
            "fullscreen_mode": fullscreen_video
        }
        
        start_time = time.time()
        
        # Generate filename
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{camera.name}_{timestamp_str}.png"
        file_path = self.output_dir / filename
        metadata["file_path"] = str(file_path)
        
        for attempt in range(camera.retry_attempts):
            driver = None
            try:
                self.logger.info(f"Capturing {camera.name} (attempt {attempt + 1}/{camera.retry_attempts})")
                
                # Initialize WebDriver
                service = Service()
                driver = webdriver.Chrome(service=service, options=self.chrome_options)
                driver.set_page_load_timeout(camera.timeout)
                
                # Navigate and capture
                driver.get(camera.url)
                time.sleep(5)  # Wait for page load

                # If fullscreen video mode is enabled, expand video to fill viewport
                if fullscreen_video:
                    self.logger.info(f"Applying fullscreen video mode for {camera.name}")
                    fullscreen_result = self._expand_video_fullscreen(driver)
                    if not fullscreen_result.get('success'):
                        self.logger.warning(f"Fullscreen expansion failed: {fullscreen_result.get('error')}")
                        # Continue anyway - we'll still get some screenshot
                    else:
                        time.sleep(2)  # Let video stabilize after expansion

                driver.save_screenshot(str(file_path))
                
                # Verify file was created and has content
                if file_path.exists() and file_path.stat().st_size > 0:
                    metadata["success"] = True
                    metadata["file_size"] = file_path.stat().st_size
                    metadata["capture_duration"] = time.time() - start_time
                    
                    self.logger.info(f"Successfully captured {camera.name}: {file_path}")
                    return True, str(file_path), metadata
                else:
                    raise Exception("Screenshot file was not created or is empty")
                    
            except Exception as e:
                error_msg = f"Attempt {attempt + 1} failed for {camera.name}: {str(e)}"
                self.logger.warning(error_msg)
                metadata["error"] = str(e)
                
                if attempt < camera.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
            finally:
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass
        
        metadata["capture_duration"] = time.time() - start_time
        self.logger.error(f"Failed to capture {camera.name} after {camera.retry_attempts} attempts")
        return False, "", metadata
    
    def capture_all_cameras(self, max_workers: int = 4, fullscreen_video: bool = False) -> Dict[str, Dict]:
        """
        Capture screenshots from all enabled cameras concurrently

        Args:
            max_workers: Number of concurrent capture threads
            fullscreen_video: If True, expands video to fullscreen for all cameras
        """
        enabled_cameras = {k: v for k, v in self.cameras.items() if v.enabled}

        if not enabled_cameras:
            self.logger.warning("No enabled cameras found")
            return {}

        mode_str = "fullscreen video" if fullscreen_video else "standard"
        self.logger.info(f"Starting {mode_str} capture for {len(enabled_cameras)} cameras")
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all capture tasks
            future_to_camera = {
                executor.submit(self.capture_screenshot, camera_id, camera, fullscreen_video): camera_id
                for camera_id, camera in enabled_cameras.items()
            }
            
            # Collect results
            for future in as_completed(future_to_camera):
                camera_id = future_to_camera[future]
                try:
                    success, file_path, metadata = future.result()
                    results[camera_id] = metadata
                except Exception as e:
                    self.logger.error(f"Unexpected error for camera {camera_id}: {e}")
                    results[camera_id] = {
                        "camera_id": camera_id,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
        
        # Log summary
        successful = sum(1 for r in results.values() if r.get("success", False))
        self.logger.info(f"Capture complete: {successful}/{len(enabled_cameras)} successful")
        
        return results
    
    def add_camera(self, camera_id: str, name: str, url: str, location: str, region: str, **kwargs):
        """Add a new camera to the configuration"""
        self.cameras[camera_id] = CameraConfig(
            name=name,
            url=url,
            location=location,
            region=region,
            **kwargs
        )
        self.save_camera_config()
        self.logger.info(f"Added new camera: {camera_id}")
    
    def remove_camera(self, camera_id: str):
        """Remove a camera from the configuration"""
        if camera_id in self.cameras:
            del self.cameras[camera_id]
            self.save_camera_config()
            self.logger.info(f"Removed camera: {camera_id}")
        else:
            self.logger.warning(f"Camera {camera_id} not found")
    
    def enable_camera(self, camera_id: str, enabled: bool = True):
        """Enable or disable a camera"""
        if camera_id in self.cameras:
            self.cameras[camera_id].enabled = enabled
            self.save_camera_config()
            status = "enabled" if enabled else "disabled"
            self.logger.info(f"Camera {camera_id} {status}")
        else:
            self.logger.warning(f"Camera {camera_id} not found")
    
    def get_camera_status(self) -> Dict:
        """Get status of all cameras"""
        return {
            camera_id: {
                "name": camera.name,
                "enabled": camera.enabled,
                "location": camera.location,
                "url": camera.url
            }
            for camera_id, camera in self.cameras.items()
        }

def main():
    """Main function for testing the camera manager"""
    manager = CameraManager()
    
    # Display current cameras
    print("Current camera configuration:")
    for camera_id, status in manager.get_camera_status().items():
        print(f"  {camera_id}: {status['name']} ({'enabled' if status['enabled'] else 'disabled'})")
    
    # Capture from all cameras
    results = manager.capture_all_cameras(max_workers=2)
    
    # Display results
    print("\nCapture Results:")
    for camera_id, result in results.items():
        status = "SUCCESS" if result.get("success", False) else "FAILED"
        print(f"  {camera_id}: {status}")
        if result.get("file_path"):
            print(f"    File: {result['file_path']}")
        if result.get("error"):
            print(f"    Error: {result['error']}")

if __name__ == "__main__":
    main()