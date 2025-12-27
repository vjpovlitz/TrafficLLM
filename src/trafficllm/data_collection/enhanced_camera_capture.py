"""
Enhanced Camera Capture with Video Interaction
Specifically designed for traffic camera video feeds that require interaction
"""
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, WebDriverException,
    ElementClickInterceptedException, JavascriptException
)

class EnhancedTrafficCameraCapture:
    """Enhanced traffic camera capture with video interaction capabilities"""

    def __init__(self, config_path: str = "camera_config.json", output_dir: str = "enhanced_traffic_screenshots"):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Load camera configurations
        self.cameras = self._load_camera_config()

        # Video interaction selectors
        self.video_selectors = [
            "video",  # HTML5 video tag
            ".video-player",
            ".player",
            ".media-player",
            "iframe[src*='video']",
            "[data-video]",
            ".jwplayer",
            ".flowplayer",
            ".video-js"
        ]

        self.play_button_selectors = [
            ".play-button",
            ".btn-play",
            ".video-play-button",
            "[aria-label*='play']",
            "[title*='play']",
            ".play",
            ".fa-play",
            ".glyphicon-play",
            "button[title*='Play']",
            ".ytp-play-button"
        ]

    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"enhanced_capture_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _load_camera_config(self) -> Dict:
        """Load camera configurations"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            cameras = config.get('cameras', {})
            self.logger.info(f"Loaded {len(cameras)} camera configurations")
            return cameras
        except FileNotFoundError:
            self.logger.warning(f"Config file {self.config_path} not found")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading camera config: {e}")
            return {}

    def _setup_chrome_options(self) -> Options:
        """Configure Chrome options for video capture"""
        options = Options()

        # Essential options
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")

        # Fix session conflict with unique user data directory
        import tempfile
        temp_dir = tempfile.mkdtemp()
        options.add_argument(f"--user-data-dir={temp_dir}")

        # Video-specific options
        options.add_argument("--autoplay-policy=no-user-gesture-required")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--enable-automation")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")

        # Media permissions
        options.add_experimental_option("prefs", {
            "profile.default_content_setting_values.media_stream_camera": 1,
            "profile.default_content_setting_values.media_stream_mic": 1,
            "profile.default_content_settings.popups": 0
        })

        # User agent to avoid detection
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        return options

    def _wait_for_video_load(self, driver: webdriver.Chrome, timeout: int = 30) -> bool:
        """Wait for video content to load"""
        wait = WebDriverWait(driver, timeout)

        # Strategy 1: Wait for video element to be present and have data
        try:
            self.logger.info("Waiting for video elements to load...")

            # Look for video elements
            video_elements = driver.find_elements(By.TAG_NAME, "video")
            if video_elements:
                self.logger.info(f"Found {len(video_elements)} video elements")

                # Try to play the video
                for i, video in enumerate(video_elements):
                    try:
                        if video.is_displayed():
                            self.logger.info(f"Attempting to play video element {i}")
                            driver.execute_script("arguments[0].play();", video)
                            time.sleep(2)

                            # Check if video is actually playing
                            is_playing = driver.execute_script(
                                "return !arguments[0].paused && !arguments[0].ended;", video
                            )
                            if is_playing:
                                self.logger.info(f"Video {i} is now playing")
                                return True
                    except Exception as e:
                        self.logger.debug(f"Could not play video {i}: {e}")

            # Strategy 2: Look for and click play buttons
            self.logger.info("Looking for play buttons...")
            for selector in self.play_button_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            self.logger.info(f"Clicking play button: {selector}")
                            element.click()
                            time.sleep(3)
                            return True
                except Exception as e:
                    self.logger.debug(f"Play button selector {selector} failed: {e}")

            # Strategy 3: Try JavaScript video activation
            self.logger.info("Attempting JavaScript video activation...")
            js_commands = [
                "document.querySelectorAll('video').forEach(v => v.play());",
                "if(window.player) window.player.play();",
                "if(window.jwplayer) jwplayer().play();",
                "Array.from(document.querySelectorAll('[class*=\"play\"]')).forEach(el => el.click());"
            ]

            for cmd in js_commands:
                try:
                    driver.execute_script(cmd)
                    time.sleep(2)
                except Exception as e:
                    self.logger.debug(f"JS command failed: {e}")

            return True

        except Exception as e:
            self.logger.warning(f"Video load wait failed: {e}")
            return False

    def _click_specific_camera(self, driver: webdriver.Chrome, camera_id: str) -> bool:
        """Click on a specific camera from the camera list"""
        camera_links = {
            "us_50_sandy_point": ["#camera-28", "camera-28", "Sandy Point"],
            "us_50_ex23_md2": ["#camera-48", "camera-48", "Exit 23"]
        }

        if camera_id not in camera_links:
            return False

        selectors = camera_links[camera_id]

        for selector in selectors:
            try:
                # Try direct ID/anchor link
                if selector.startswith('#'):
                    element = driver.find_element(By.CSS_SELECTOR, selector)
                else:
                    # Try finding by text content
                    element = driver.find_element(By.XPATH, f"//*[contains(text(), '{selector}')]")

                if element.is_displayed():
                    self.logger.info(f"Clicking camera selector: {selector}")
                    driver.execute_script("arguments[0].click();", element)
                    time.sleep(3)
                    return True

            except Exception as e:
                self.logger.debug(f"Camera selector {selector} failed: {e}")

        return False

    def capture_traffic_camera(self, camera_id: str, camera_config: Dict) -> Dict:
        """Capture from a specific traffic camera with video interaction"""
        result = {
            "camera_id": camera_id,
            "camera_name": camera_config.get('name', camera_id),
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "file_path": None,
            "file_size": None,
            "error": None,
            "video_loaded": False,
            "interaction_attempted": False
        }

        driver = None
        try:
            self.logger.info(f"Starting enhanced capture for {camera_config.get('name', camera_id)}")

            # Setup Chrome driver
            chrome_options = self._setup_chrome_options()
            service = Service()
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(30)

            # Navigate to camera URL
            url = camera_config.get('url')
            self.logger.info(f"Loading URL: {url}")
            driver.get(url)

            # Wait for initial page load
            time.sleep(5)

            # Try to click on specific camera if it's a camera list page
            if "camera-" in url or "#camera" in url:
                self.logger.info("Detected camera list page, attempting to select specific camera")
                camera_selected = self._click_specific_camera(driver, camera_id)
                if camera_selected:
                    time.sleep(5)  # Wait for camera to load
                    result["interaction_attempted"] = True

            # Wait for video content to load
            video_loaded = self._wait_for_video_load(driver, timeout=15)
            result["video_loaded"] = video_loaded

            if video_loaded:
                self.logger.info("Video content detected, waiting for stream to stabilize...")
                time.sleep(8)  # Allow video to load and stabilize
            else:
                self.logger.warning("No video content detected, capturing page as-is")
                time.sleep(3)

            # Generate filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{camera_config.get('name', camera_id)}_{timestamp_str}_enhanced.png"
            file_path = self.output_dir / filename

            # Take screenshot
            self.logger.info(f"Taking screenshot: {file_path}")
            driver.save_screenshot(str(file_path))

            # Verify screenshot was created
            if file_path.exists() and file_path.stat().st_size > 0:
                result["success"] = True
                result["file_path"] = str(file_path)
                result["file_size"] = file_path.stat().st_size
                self.logger.info(f"Successfully captured {camera_id}: {file_path} ({result['file_size']} bytes)")
            else:
                result["error"] = "Screenshot file was not created or is empty"

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Enhanced capture failed for {camera_id}: {e}")

        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

        return result

    def capture_all_cameras(self, max_workers: int = 2) -> Dict[str, Dict]:
        """Capture from all configured cameras"""
        if not self.cameras:
            self.logger.warning("No cameras configured")
            return {}

        self.logger.info(f"Starting enhanced capture for {len(self.cameras)} cameras")
        results = {}

        # Use sequential processing for better video handling
        for camera_id, camera_config in self.cameras.items():
            if camera_config.get('enabled', True):
                results[camera_id] = self.capture_traffic_camera(camera_id, camera_config)
                time.sleep(2)  # Brief pause between captures

        # Log summary
        successful = sum(1 for r in results.values() if r.get("success", False))
        video_loaded = sum(1 for r in results.values() if r.get("video_loaded", False))

        self.logger.info(f"Enhanced capture complete: {successful}/{len(results)} successful, {video_loaded} with video")

        return results

def main():
    """Test the enhanced camera capture"""
    capture = EnhancedTrafficCameraCapture()

    print("=== Enhanced Traffic Camera Capture Test ===")
    print(f"Configured cameras: {len(capture.cameras)}")

    # Capture from all cameras
    results = capture.capture_all_cameras()

    print(f"\n=== Results ===")
    for camera_id, result in results.items():
        status = "✅ SUCCESS" if result.get("success") else "❌ FAILED"
        video_status = "🎥 Video Loaded" if result.get("video_loaded") else "📷 Static Only"
        interaction = "🖱️ Interaction" if result.get("interaction_attempted") else "🔄 Direct"

        print(f"{camera_id}: {status} {video_status} {interaction}")
        if result.get("file_path"):
            print(f"   File: {result['file_path']} ({result.get('file_size', 0):,} bytes)")
        if result.get("error"):
            print(f"   Error: {result['error']}")

if __name__ == "__main__":
    main()