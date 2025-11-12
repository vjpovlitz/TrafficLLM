"""
Final Enhanced Camera Manager
Incorporates all video interaction improvements for accurate traffic camera capture
"""
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import tempfile
import uuid

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException

class FinalEnhancedCameraManager:
    """Production-ready camera manager with video interaction"""

    def __init__(self, config_path: str = "camera_config.json", output_dir: str = "final_enhanced_screenshots"):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Load cameras
        self.cameras = self._load_cameras()

        # Camera-specific configurations
        self.camera_configs = {
            "us_50_sandy_point": {
                "anchor": "#camera-28",
                "search_terms": ["Sandy Point", "camera-28"],
                "wait_time": 10
            },
            "us_50_ex23_md2": {
                "anchor": "#camera-48",
                "search_terms": ["Exit 23", "camera-48"],
                "wait_time": 10
            }
        }

    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _load_cameras(self) -> Dict:
        """Load camera configurations"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config.get('cameras', {})
        except:
            return {}

    def _setup_chrome(self) -> webdriver.Chrome:
        """Setup Chrome with optimized options for video capture"""
        options = Options()

        # Core options
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")

        # Video optimization
        options.add_argument("--autoplay-policy=no-user-gesture-required")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")

        # Unique session to avoid conflicts
        temp_dir = tempfile.mkdtemp(prefix=f"chrome_{uuid.uuid4().hex[:8]}_")
        options.add_argument(f"--user-data-dir={temp_dir}")

        # User agent
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        driver = webdriver.Chrome(service=Service(), options=options)
        driver.set_page_load_timeout(30)
        return driver

    def _activate_camera_video(self, driver: webdriver.Chrome, camera_id: str) -> bool:
        """Activate specific camera video content"""
        if camera_id not in self.camera_configs:
            return False

        config = self.camera_configs[camera_id]

        try:
            # Method 1: Direct anchor click
            try:
                anchor_element = driver.find_element(By.CSS_SELECTOR, f"a[href*='{config['anchor']}']")
                if anchor_element.is_displayed():
                    self.logger.info(f"Clicking anchor {config['anchor']} for {camera_id}")
                    driver.execute_script("arguments[0].click();", anchor_element)
                    time.sleep(5)
                    return True
            except NoSuchElementException:
                pass

            # Method 2: Search by text content
            for term in config['search_terms']:
                try:
                    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{term}')]")
                    for element in elements:
                        if element.is_displayed() and element.tag_name == 'a':
                            self.logger.info(f"Found {term} link for {camera_id}")
                            driver.execute_script("arguments[0].click();", element)
                            time.sleep(5)
                            return True
                except:
                    continue

        except Exception as e:
            self.logger.warning(f"Camera activation failed for {camera_id}: {e}")

        return False

    def _wait_and_activate_video(self, driver: webdriver.Chrome, wait_time: int = 10):
        """Wait for and activate video content"""
        time.sleep(3)  # Initial wait

        # Find and try to play video elements
        try:
            video_elements = driver.find_elements(By.TAG_NAME, "video")
            self.logger.info(f"Found {len(video_elements)} video elements")

            for i, video in enumerate(video_elements):
                try:
                    if video.is_displayed():
                        self.logger.info(f"Activating video element {i}")
                        driver.execute_script("arguments[0].play();", video)
                        time.sleep(2)
                except Exception as e:
                    self.logger.debug(f"Video {i} activation failed: {e}")

        except Exception as e:
            self.logger.debug(f"Video search failed: {e}")

        # Wait for content to stabilize
        time.sleep(wait_time)

    def capture_camera(self, camera_id: str, camera_config: Dict) -> Dict:
        """Capture from a single camera with full video interaction"""
        result = {
            "camera_id": camera_id,
            "camera_name": camera_config.get('name', camera_id),
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "file_path": None,
            "file_size": None,
            "error": None,
            "camera_activated": False,
            "video_detected": False
        }

        driver = None
        try:
            self.logger.info(f"Starting capture for {camera_config.get('name', camera_id)}")

            # Setup driver
            driver = self._setup_chrome()

            # Navigate to URL
            url = camera_config.get('url')
            self.logger.info(f"Loading {url}")
            driver.get(url)

            # Wait for page load
            time.sleep(8)

            # Activate specific camera
            camera_activated = self._activate_camera_video(driver, camera_id)
            result["camera_activated"] = camera_activated

            # Check for video elements
            video_elements = driver.find_elements(By.TAG_NAME, "video")
            result["video_detected"] = len(video_elements) > 0

            # Wait and activate video content
            wait_time = self.camera_configs.get(camera_id, {}).get('wait_time', 10)
            self._wait_and_activate_video(driver, wait_time)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{camera_config.get('name', camera_id)}_{timestamp}_final.png"
            file_path = self.output_dir / filename

            # Take screenshot
            self.logger.info(f"Taking screenshot: {filename}")
            driver.save_screenshot(str(file_path))

            # Verify file
            if file_path.exists() and file_path.stat().st_size > 0:
                result["success"] = True
                result["file_path"] = str(file_path)
                result["file_size"] = file_path.stat().st_size
                self.logger.info(f"✅ Captured {camera_id}: {filename} ({result['file_size']:,} bytes)")
            else:
                result["error"] = "Screenshot file not created"

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"❌ Capture failed for {camera_id}: {e}")

        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

        return result

    def capture_all_cameras(self) -> Dict[str, Dict]:
        """Capture from all enabled cameras"""
        if not self.cameras:
            self.logger.warning("No cameras configured")
            return {}

        self.logger.info(f"Starting capture for {len(self.cameras)} cameras")
        results = {}

        # Sequential processing for better reliability
        for camera_id, camera_config in self.cameras.items():
            if camera_config.get('enabled', True):
                results[camera_id] = self.capture_camera(camera_id, camera_config)
                time.sleep(3)  # Brief pause between captures

        # Summary
        successful = sum(1 for r in results.values() if r.get("success", False))
        activated = sum(1 for r in results.values() if r.get("camera_activated", False))
        with_video = sum(1 for r in results.values() if r.get("video_detected", False))

        self.logger.info(f"Capture complete: {successful}/{len(results)} successful, {activated} activated, {with_video} with video")

        return results

def main():
    """Run the final enhanced camera manager"""
    print("=== FINAL ENHANCED TRAFFIC CAMERA CAPTURE ===\n")

    manager = FinalEnhancedCameraManager()

    print(f"📹 Configured cameras: {len(manager.cameras)}")
    for camera_id in manager.cameras:
        print(f"   - {camera_id}: {manager.cameras[camera_id].get('name', 'Unknown')}")

    print("\n🚀 Starting capture process...\n")

    results = manager.capture_all_cameras()

    print("\n=== FINAL RESULTS ===")
    for camera_id, result in results.items():
        status_icon = "✅" if result.get("success") else "❌"
        camera_icon = "🎯" if result.get("camera_activated") else "📷"
        video_icon = "🎥" if result.get("video_detected") else "📺"

        print(f"{status_icon} {camera_id}: {result.get('camera_name', 'Unknown')}")
        print(f"   {camera_icon} Camera activation: {'SUCCESS' if result.get('camera_activated') else 'FAILED'}")
        print(f"   {video_icon} Video detection: {'YES' if result.get('video_detected') else 'NO'}")

        if result.get("success"):
            print(f"   📁 File: {result['file_path']}")
            print(f"   📊 Size: {result['file_size']:,} bytes")

        if result.get("error"):
            print(f"   ⚠️ Error: {result['error']}")
        print()

    print("🎉 Enhanced capture process complete!")
    print("📂 Check the 'final_enhanced_screenshots' directory for results.")

if __name__ == "__main__":
    main()