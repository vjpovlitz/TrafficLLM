"""
Fullscreen Enhanced Traffic Camera Capture
Adds fullscreen capability for maximum image quality
"""
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import uuid

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException

class FullscreenTrafficCameraCapture:
    """Enhanced camera capture with fullscreen video capability"""

    def __init__(self, config_path: str = "camera_config.json", output_dir: str = "fullscreen_screenshots"):
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

        # Fullscreen button selectors (in order of preference)
        self.fullscreen_selectors = [
            # Common fullscreen button classes and attributes
            "[title*='fullscreen' i]",
            "[title*='Fullscreen' i]",
            "[aria-label*='fullscreen' i]",
            "[aria-label*='Fullscreen' i]",
            ".fullscreen-button",
            ".btn-fullscreen",
            ".fs-button",
            ".fullscreen",
            ".expand-button",
            ".maximize-button",

            # Icon-based selectors
            ".fa-expand",
            ".glyphicon-fullscreen",
            ".icon-fullscreen",
            ".icon-expand",

            # Video player specific
            ".video-fullscreen",
            ".player-fullscreen",
            ".vjs-fullscreen-control",

            # Generic button patterns
            "button[class*='fullscreen' i]",
            "button[class*='expand' i]",
            "button[class*='maximize' i]",

            # SVG icons (common in modern players)
            "svg[class*='fullscreen' i]",
            "svg[class*='expand' i]"
        ]

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
        """Setup Chrome with fullscreen-optimized options"""
        options = Options()

        # Core options
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--start-maximized")

        # Video and fullscreen optimization
        options.add_argument("--autoplay-policy=no-user-gesture-required")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--enable-features=VaapiVideoDecoder")

        # Allow fullscreen
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-infobars")

        # Unique session
        temp_dir = tempfile.mkdtemp(prefix=f"chrome_fs_{uuid.uuid4().hex[:8]}_")
        options.add_argument(f"--user-data-dir={temp_dir}")

        # User agent
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        driver = webdriver.Chrome(service=Service(), options=options)
        driver.set_page_load_timeout(30)

        # Maximize window for better fullscreen support
        driver.maximize_window()

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

    def _enter_fullscreen(self, driver: webdriver.Chrome) -> bool:
        """Attempt to enter fullscreen mode using multiple strategies"""
        self.logger.info("🔍 Attempting to enter fullscreen mode...")

        # Strategy 1: Look for fullscreen buttons
        for selector in self.fullscreen_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if element.is_displayed() and element.is_enabled():
                        try:
                            self.logger.info(f"🎯 Found fullscreen button: {selector}")
                            # Try different click methods
                            try:
                                element.click()
                            except:
                                driver.execute_script("arguments[0].click();", element)

                            time.sleep(3)

                            # Check if fullscreen was successful
                            if self._is_fullscreen(driver):
                                self.logger.info("✅ Fullscreen activated via button click")
                                return True

                        except Exception as e:
                            self.logger.debug(f"Button click failed: {e}")

            except Exception as e:
                self.logger.debug(f"Selector {selector} failed: {e}")

        # Strategy 2: JavaScript fullscreen API
        self.logger.info("🔧 Trying JavaScript fullscreen API...")
        js_fullscreen_methods = [
            # Try on video elements
            """
            var videos = document.querySelectorAll('video');
            for (var i = 0; i < videos.length; i++) {
                if (videos[i].requestFullscreen) {
                    videos[i].requestFullscreen();
                    break;
                } else if (videos[i].webkitRequestFullscreen) {
                    videos[i].webkitRequestFullscreen();
                    break;
                } else if (videos[i].mozRequestFullScreen) {
                    videos[i].mozRequestFullScreen();
                    break;
                }
            }
            """,

            # Try on video containers
            """
            var containers = document.querySelectorAll('[class*="video"], [class*="player"]');
            for (var i = 0; i < containers.length; i++) {
                if (containers[i].requestFullscreen) {
                    containers[i].requestFullscreen();
                    break;
                }
            }
            """,

            # Try on document
            """
            if (document.documentElement.requestFullscreen) {
                document.documentElement.requestFullscreen();
            } else if (document.documentElement.webkitRequestFullscreen) {
                document.documentElement.webkitRequestFullscreen();
            } else if (document.documentElement.mozRequestFullScreen) {
                document.documentElement.mozRequestFullScreen();
            }
            """
        ]

        for i, js_method in enumerate(js_fullscreen_methods):
            try:
                self.logger.info(f"🚀 Executing JS fullscreen method {i+1}")
                driver.execute_script(js_method)
                time.sleep(3)

                if self._is_fullscreen(driver):
                    self.logger.info(f"✅ Fullscreen activated via JS method {i+1}")
                    return True

            except Exception as e:
                self.logger.debug(f"JS method {i+1} failed: {e}")

        # Strategy 3: Keyboard shortcut (F11 or F key)
        self.logger.info("⌨️ Trying keyboard shortcuts...")
        try:
            # First click on video to ensure focus
            video_elements = driver.find_elements(By.TAG_NAME, "video")
            if video_elements:
                video_elements[0].click()
                time.sleep(1)

            # Try F key (common video player fullscreen)
            actions = ActionChains(driver)
            actions.send_keys("f").perform()
            time.sleep(3)

            if self._is_fullscreen(driver):
                self.logger.info("✅ Fullscreen activated via 'F' key")
                return True

            # Try F11 (browser fullscreen)
            actions.send_keys(Keys.F11).perform()
            time.sleep(3)

            if self._is_fullscreen(driver):
                self.logger.info("✅ Fullscreen activated via F11 key")
                return True

        except Exception as e:
            self.logger.debug(f"Keyboard shortcuts failed: {e}")

        # Strategy 4: Double-click on video
        self.logger.info("🖱️ Trying double-click on video...")
        try:
            video_elements = driver.find_elements(By.TAG_NAME, "video")
            for video in video_elements:
                if video.is_displayed():
                    actions = ActionChains(driver)
                    actions.double_click(video).perform()
                    time.sleep(3)

                    if self._is_fullscreen(driver):
                        self.logger.info("✅ Fullscreen activated via double-click")
                        return True

        except Exception as e:
            self.logger.debug(f"Double-click failed: {e}")

        self.logger.warning("⚠️ Could not activate fullscreen mode")
        return False

    def _is_fullscreen(self, driver: webdriver.Chrome) -> bool:
        """Check if browser/video is in fullscreen mode"""
        try:
            # Check various fullscreen indicators
            fullscreen_checks = [
                "return document.fullscreenElement !== null;",
                "return document.webkitFullscreenElement !== null;",
                "return document.mozFullScreenElement !== null;",
                "return document.msFullscreenElement !== null;",
                "return window.innerHeight === screen.height;",
                "return window.outerHeight === screen.height;"
            ]

            for check in fullscreen_checks:
                try:
                    result = driver.execute_script(check)
                    if result:
                        return True
                except:
                    continue

            return False

        except Exception:
            return False

    def capture_camera_fullscreen(self, camera_id: str, camera_config: Dict) -> Dict:
        """Capture from camera with fullscreen capability"""
        result = {
            "camera_id": camera_id,
            "camera_name": camera_config.get('name', camera_id),
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "file_path": None,
            "file_size": None,
            "error": None,
            "camera_activated": False,
            "video_detected": False,
            "fullscreen_activated": False,
            "fullscreen_method": None
        }

        driver = None
        try:
            self.logger.info(f"🚀 Starting fullscreen capture for {camera_config.get('name', camera_id)}")

            # Setup driver
            driver = self._setup_chrome()

            # Navigate to URL
            url = camera_config.get('url')
            self.logger.info(f"📡 Loading {url}")
            driver.get(url)

            # Wait for page load
            time.sleep(8)

            # Activate specific camera
            camera_activated = self._activate_camera_video(driver, camera_id)
            result["camera_activated"] = camera_activated

            # Check for video elements
            video_elements = driver.find_elements(By.TAG_NAME, "video")
            result["video_detected"] = len(video_elements) > 0
            self.logger.info(f"🎥 Found {len(video_elements)} video elements")

            # Wait and activate video content
            wait_time = self.camera_configs.get(camera_id, {}).get('wait_time', 10)
            self._wait_and_activate_video(driver, wait_time)

            # Attempt to enter fullscreen
            fullscreen_success = self._enter_fullscreen(driver)
            result["fullscreen_activated"] = fullscreen_success

            if fullscreen_success:
                self.logger.info("🎯 Fullscreen activated! Waiting for stabilization...")
                time.sleep(5)  # Extra wait for fullscreen to stabilize
            else:
                self.logger.warning("⚠️ Fullscreen not activated, capturing standard view")
                time.sleep(2)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = "fullscreen" if fullscreen_success else "standard"
            filename = f"{camera_config.get('name', camera_id)}_{timestamp}_{mode}.png"
            file_path = self.output_dir / filename

            # Take screenshot
            self.logger.info(f"📸 Taking {mode} screenshot: {filename}")
            driver.save_screenshot(str(file_path))

            # Verify file
            if file_path.exists() and file_path.stat().st_size > 0:
                result["success"] = True
                result["file_path"] = str(file_path)
                result["file_size"] = file_path.stat().st_size

                mode_icon = "🎯" if fullscreen_success else "📷"
                self.logger.info(f"✅ {mode_icon} Captured {camera_id}: {filename} ({result['file_size']:,} bytes)")
            else:
                result["error"] = "Screenshot file not created"

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"❌ Fullscreen capture failed for {camera_id}: {e}")

        finally:
            if driver:
                try:
                    # Exit fullscreen before closing
                    if result.get("fullscreen_activated"):
                        try:
                            driver.execute_script("if (document.exitFullscreen) document.exitFullscreen();")
                        except:
                            pass
                    driver.quit()
                except:
                    pass

        return result

    def capture_all_cameras_fullscreen(self) -> Dict[str, Dict]:
        """Capture from all cameras with fullscreen capability"""
        if not self.cameras:
            self.logger.warning("No cameras configured")
            return {}

        self.logger.info(f"🎬 Starting fullscreen capture for {len(self.cameras)} cameras")
        results = {}

        # Sequential processing for better reliability
        for camera_id, camera_config in self.cameras.items():
            if camera_config.get('enabled', True):
                results[camera_id] = self.capture_camera_fullscreen(camera_id, camera_config)
                time.sleep(3)  # Brief pause between captures

        # Summary
        successful = sum(1 for r in results.values() if r.get("success", False))
        activated = sum(1 for r in results.values() if r.get("camera_activated", False))
        with_video = sum(1 for r in results.values() if r.get("video_detected", False))
        fullscreen = sum(1 for r in results.values() if r.get("fullscreen_activated", False))

        self.logger.info(f"🎉 Fullscreen capture complete: {successful}/{len(results)} successful, {fullscreen} fullscreen")

        return results

def main():
    """Run the fullscreen camera capture"""
    print("=== 🎬 FULLSCREEN TRAFFIC CAMERA CAPTURE ===\n")

    capture = FullscreenTrafficCameraCapture()

    print(f"📹 Configured cameras: {len(capture.cameras)}")
    for camera_id in capture.cameras:
        print(f"   - {camera_id}: {capture.cameras[camera_id].get('name', 'Unknown')}")

    print("\n🚀 Starting fullscreen capture process...\n")

    results = capture.capture_all_cameras_fullscreen()

    print("\n=== 🎯 FULLSCREEN RESULTS ===")
    for camera_id, result in results.items():
        status_icon = "✅" if result.get("success") else "❌"
        camera_icon = "🎯" if result.get("camera_activated") else "📷"
        video_icon = "🎥" if result.get("video_detected") else "📺"
        fullscreen_icon = "🎬" if result.get("fullscreen_activated") else "📐"

        print(f"{status_icon} {camera_id}: {result.get('camera_name', 'Unknown')}")
        print(f"   {camera_icon} Camera activation: {'SUCCESS' if result.get('camera_activated') else 'FAILED'}")
        print(f"   {video_icon} Video detection: {'YES' if result.get('video_detected') else 'NO'}")
        print(f"   {fullscreen_icon} Fullscreen mode: {'SUCCESS' if result.get('fullscreen_activated') else 'FAILED'}")

        if result.get("success"):
            print(f"   📁 File: {result['file_path']}")
            print(f"   📊 Size: {result['file_size']:,} bytes")

        if result.get("error"):
            print(f"   ⚠️ Error: {result['error']}")
        print()

    print("🎉 Fullscreen capture process complete!")
    print("📂 Check the 'fullscreen_screenshots' directory for results.")

if __name__ == "__main__":
    main()