"""
Fullscreen Video Capture Module
Automates finding video players on webpages, entering fullscreen mode, and capturing high-quality screenshots
"""

import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
    JavascriptException
)


class VideoPlayerDetector:
    """Detects video players on web pages using multiple strategies"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Common video player selectors in order of preference
        self.video_selectors = [
            # HTML5 video tags
            "video",

            # Common video player containers
            ".video-player",
            ".player",
            ".video-container",
            ".media-player",

            # YouTube specific
            "#movie_player",
            ".html5-video-player",
            ".ytp-player",

            # Vimeo specific
            ".vp-player",
            ".player-container",

            # Twitch specific
            ".video-player__container",
            ".player-overlay-background",

            # Generic iframe for embedded videos
            "iframe[src*='youtube']",
            "iframe[src*='vimeo']",
            "iframe[src*='twitch']",
            "iframe[src*='video']",
            "iframe[src*='player']",

            # Other common patterns
            "[data-video-id]",
            "[data-player]",
            ".jwplayer",
            ".flowplayer",
            ".video-js",
        ]

        # JavaScript snippets to detect video elements
        self.js_detection_scripts = [
            # Find all video elements
            "return document.querySelectorAll('video');",

            # Find elements with video-related attributes
            """
            return document.querySelectorAll(
                '[data-video], [data-player], [data-video-id], [data-stream]'
            );
            """,

            # Find elements containing 'player' or 'video' in className
            """
            return Array.from(document.querySelectorAll('*')).filter(el =>
                el.className && (
                    el.className.includes('player') ||
                    el.className.includes('video')
                )
            );
            """,
        ]

    def detect_video_players(self, driver: webdriver.Chrome) -> List[Dict]:
        """
        Detect all potential video players on the current page

        Returns:
            List of dictionaries containing player information:
            {
                'element': WebElement,
                'selector': str,
                'method': str,
                'dimensions': tuple,
                'is_visible': bool,
                'has_video_tag': bool
            }
        """
        players = []

        self.logger.info("Starting video player detection...")

        # Method 1: CSS Selector detection
        for selector in self.video_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if element.is_displayed():
                        player_info = self._analyze_element(element, selector, "css_selector")
                        if player_info:
                            players.append(player_info)
                            self.logger.info(f"Found player via CSS selector '{selector}': {player_info['dimensions']}")
            except Exception as e:
                self.logger.debug(f"CSS selector '{selector}' failed: {e}")

        # Method 2: JavaScript detection
        for i, script in enumerate(self.js_detection_scripts):
            try:
                elements = driver.execute_script(script)
                for element in elements:
                    if element.is_displayed():
                        player_info = self._analyze_element(element, f"js_script_{i}", "javascript")
                        if player_info:
                            players.append(player_info)
                            self.logger.info(f"Found player via JavaScript {i}: {player_info['dimensions']}")
            except Exception as e:
                self.logger.debug(f"JavaScript detection {i} failed: {e}")

        # Remove duplicates (same element found by different methods)
        unique_players = []
        seen_elements = set()

        for player in players:
            element_id = id(player['element'])
            if element_id not in seen_elements:
                seen_elements.add(element_id)
                unique_players.append(player)

        self.logger.info(f"Detected {len(unique_players)} unique video players")
        return unique_players

    def _analyze_element(self, element, selector: str, method: str) -> Optional[Dict]:
        """Analyze a potential video element and return player information"""
        try:
            # Get element dimensions
            size = element.size
            location = element.location

            # Skip tiny elements (likely not main video players)
            if size['width'] < 100 or size['height'] < 100:
                return None

            # Check if element contains or is a video tag
            has_video_tag = False
            try:
                # Check if element itself is a video tag
                if element.tag_name.lower() == 'video':
                    has_video_tag = True
                else:
                    # Check if element contains video tags
                    video_children = element.find_elements(By.TAG_NAME, "video")
                    has_video_tag = len(video_children) > 0
            except:
                pass

            return {
                'element': element,
                'selector': selector,
                'method': method,
                'dimensions': (size['width'], size['height']),
                'location': (location['x'], location['y']),
                'is_visible': element.is_displayed(),
                'has_video_tag': has_video_tag,
                'tag_name': element.tag_name,
                'area': size['width'] * size['height']
            }
        except Exception as e:
            self.logger.debug(f"Failed to analyze element: {e}")
            return None

    def select_best_player(self, players: List[Dict]) -> Optional[Dict]:
        """
        Select the best video player from detected players

        Priority:
        1. Largest area
        2. Has video tag
        3. Visible
        """
        if not players:
            return None

        # Filter visible players
        visible_players = [p for p in players if p['is_visible']]
        if not visible_players:
            self.logger.warning("No visible players found")
            return None

        # Sort by priority: has_video_tag (descending), area (descending)
        best_player = max(visible_players, key=lambda p: (p['has_video_tag'], p['area']))

        self.logger.info(f"Selected best player: {best_player['selector']} "
                        f"({best_player['dimensions']}, video_tag: {best_player['has_video_tag']})")

        return best_player


class FullscreenVideoCapture:
    """Main class for capturing fullscreen video screenshots"""

    def __init__(self, headless: bool = True, screenshot_dir: str = "fullscreen_screenshots"):
        self.headless = headless
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # Initialize detector
        self.detector = VideoPlayerDetector()

        # Driver will be initialized when needed
        self.driver = None

    def _setup_logging(self):
        """Setup logging configuration"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _initialize_driver(self) -> webdriver.Chrome:
        """Initialize Chrome WebDriver with optimal settings"""
        chrome_options = Options()

        if self.headless:
            chrome_options.add_argument("--headless")

        # Optimize for video handling
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")

        # Enable autoplay (important for videos)
        chrome_options.add_argument("--autoplay-policy=no-user-gesture-required")

        # Disable notifications and popups
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")

        service = Service()
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Set page load timeout
        driver.set_page_load_timeout(30)

        return driver

    def capture_fullscreen_video(self, url: str, wait_time: int = 5) -> Dict:
        """
        Capture fullscreen screenshot of video player

        Args:
            url: Website URL containing video
            wait_time: Time to wait after entering fullscreen (seconds)

        Returns:
            Dictionary with capture results
        """
        result = {
            'success': False,
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'screenshot_path': None,
            'error': None,
            'player_info': None,
            'fullscreen_success': False
        }

        try:
            # Initialize driver
            self.driver = self._initialize_driver()
            self.logger.info(f"Loading URL: {url}")

            # Load the page
            self.driver.get(url)

            # Wait for page to load
            time.sleep(3)

            # Detect video players
            players = self.detector.detect_video_players(self.driver)

            if not players:
                result['error'] = "No video players detected on the page"
                self.logger.error(result['error'])
                return result

            # Select best player
            best_player = self.detector.select_best_player(players)
            result['player_info'] = {
                'selector': best_player['selector'],
                'method': best_player['method'],
                'dimensions': best_player['dimensions'],
                'has_video_tag': best_player['has_video_tag']
            }

            # Attempt to enter fullscreen
            fullscreen_success = self._enter_fullscreen(best_player['element'])
            result['fullscreen_success'] = fullscreen_success

            if not fullscreen_success:
                self.logger.warning("Failed to enter fullscreen, capturing standard size")

            # Wait for fullscreen transition
            time.sleep(wait_time)

            # Take screenshot
            screenshot_path = self._take_screenshot(url, fullscreen_success)
            result['screenshot_path'] = screenshot_path
            result['success'] = True

            self.logger.info(f"Successfully captured screenshot: {screenshot_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Capture failed: {e}")

        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None

        return result

    def _enter_fullscreen(self, video_element) -> bool:
        """
        Attempt to enter fullscreen mode for the video element

        Returns:
            True if fullscreen was successfully triggered
        """
        fullscreen_methods = [
            # Method 1: JavaScript requestFullscreen on video element
            lambda: self.driver.execute_script("arguments[0].requestFullscreen();", video_element),

            # Method 2: Click fullscreen button if available
            self._click_fullscreen_button,

            # Method 3: Double-click on video element
            lambda: ActionChains(self.driver).double_click(video_element).perform(),

            # Method 4: Press F key (common fullscreen shortcut)
            lambda: self._send_fullscreen_key(video_element),

            # Method 5: Try alternative fullscreen JS methods
            lambda: self._try_alternative_fullscreen_js(video_element),
        ]

        for i, method in enumerate(fullscreen_methods):
            try:
                self.logger.info(f"Trying fullscreen method {i + 1}")
                method()

                # Wait briefly and check if fullscreen was successful
                time.sleep(1)

                # Check if we're in fullscreen mode
                is_fullscreen = self.driver.execute_script(
                    "return document.fullscreenElement !== null || "
                    "document.webkitFullscreenElement !== null || "
                    "document.mozFullScreenElement !== null || "
                    "document.msFullscreenElement !== null;"
                )

                if is_fullscreen:
                    self.logger.info(f"Fullscreen method {i + 1} successful")
                    return True

            except Exception as e:
                self.logger.debug(f"Fullscreen method {i + 1} failed: {e}")

        self.logger.warning("All fullscreen methods failed")
        return False

    def _click_fullscreen_button(self):
        """Try to find and click a fullscreen button"""
        fullscreen_selectors = [
            # Common fullscreen button selectors
            ".ytp-fullscreen-button",  # YouTube
            ".vp-fullscreen",          # Vimeo
            ".fullscreen-button",
            ".fs-button",
            "[title*='fullscreen']",
            "[title*='Fullscreen']",
            "[aria-label*='fullscreen']",
            "[aria-label*='Fullscreen']",
            ".icon-fullscreen",
            ".fa-expand",
            ".glyphicon-fullscreen",
        ]

        for selector in fullscreen_selectors:
            try:
                button = self.driver.find_element(By.CSS_SELECTOR, selector)
                if button.is_displayed():
                    button.click()
                    return
            except NoSuchElementException:
                continue

        raise Exception("No fullscreen button found")

    def _send_fullscreen_key(self, video_element):
        """Send fullscreen keyboard shortcut"""
        # Click on video element first to ensure focus
        video_element.click()
        time.sleep(0.5)

        # Try different fullscreen keys
        actions = ActionChains(self.driver)
        actions.send_keys("f").perform()  # Common fullscreen key

    def _try_alternative_fullscreen_js(self, video_element):
        """Try alternative JavaScript fullscreen methods"""
        js_methods = [
            # Webkit fullscreen
            "if (arguments[0].webkitRequestFullscreen) arguments[0].webkitRequestFullscreen();",

            # Mozilla fullscreen
            "if (arguments[0].mozRequestFullScreen) arguments[0].mozRequestFullScreen();",

            # MS fullscreen
            "if (arguments[0].msRequestFullscreen) arguments[0].msRequestFullscreen();",

            # Try on parent container
            "if (arguments[0].parentElement.requestFullscreen) arguments[0].parentElement.requestFullscreen();",

            # Try document fullscreen
            "document.documentElement.requestFullscreen();"
        ]

        for js_method in js_methods:
            try:
                self.driver.execute_script(js_method, video_element)
                time.sleep(0.5)
                return
            except JavascriptException:
                continue

        raise Exception("All JavaScript fullscreen methods failed")

    def _take_screenshot(self, url: str, is_fullscreen: bool) -> str:
        """Take and save screenshot"""
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        url_part = url.replace("://", "_").replace("/", "_").replace("?", "_")[:50]
        mode = "fullscreen" if is_fullscreen else "standard"

        filename = f"{timestamp}_{url_part}_{mode}.png"
        filepath = self.screenshot_dir / filename

        # Take screenshot
        self.driver.save_screenshot(str(filepath))

        return str(filepath)


# Convenience function for integration
def capture_video_screenshot(url: str, headless: bool = True, wait_time: int = 5,
                           screenshot_dir: str = "fullscreen_screenshots") -> Dict:
    """
    Convenience function to capture a fullscreen video screenshot

    Args:
        url: Website URL containing video
        headless: Run browser in headless mode
        wait_time: Time to wait after entering fullscreen
        screenshot_dir: Directory to save screenshots

    Returns:
        Dictionary with capture results
    """
    capturer = FullscreenVideoCapture(headless=headless, screenshot_dir=screenshot_dir)
    return capturer.capture_fullscreen_video(url, wait_time)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Capture fullscreen video screenshots")
    parser.add_argument("url", help="URL of webpage with video")
    parser.add_argument("--headless", action="store_true", default=True,
                       help="Run in headless mode")
    parser.add_argument("--wait", type=int, default=5,
                       help="Wait time after fullscreen (seconds)")
    parser.add_argument("--output", default="fullscreen_screenshots",
                       help="Output directory for screenshots")

    args = parser.parse_args()

    # Setup logging for standalone usage
    logging.basicConfig(level=logging.INFO)

    result = capture_video_screenshot(
        args.url,
        headless=args.headless,
        wait_time=args.wait,
        screenshot_dir=args.output
    )

    print(f"Capture result: {result}")