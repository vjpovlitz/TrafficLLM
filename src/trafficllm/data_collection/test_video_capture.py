"""
Simple test script to capture traffic camera video content
"""
import time
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def setup_chrome():
    """Setup Chrome with video-optimized options"""
    options = Options()

    # Basic headless options
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    # Video-specific options
    options.add_argument("--autoplay-policy=no-user-gesture-required")
    options.add_argument("--disable-web-security")
    options.add_argument("--disable-features=VizDisplayCompositor")

    # Unique session
    import tempfile
    import uuid
    temp_dir = tempfile.mkdtemp(prefix=f"chrome_session_{uuid.uuid4().hex[:8]}_")
    options.add_argument(f"--user-data-dir={temp_dir}")

    return webdriver.Chrome(service=Service(), options=options)

def test_sandy_point_camera():
    """Test capturing Sandy Point camera specifically"""
    driver = None
    try:
        print("🚀 Starting Sandy Point camera test...")

        driver = setup_chrome()
        driver.set_page_load_timeout(30)

        # Navigate to camera page
        url = "https://mdta.maryland.gov/index.php/traffic-cameras-by-facility#camera-28"
        print(f"📡 Loading: {url}")
        driver.get(url)

        # Wait for page load
        print("⏳ Waiting for page to load...")
        time.sleep(8)

        # Try to find and click on camera-28 specifically
        print("🎯 Looking for camera-28 link...")
        try:
            # Method 1: Direct anchor link
            camera_link = driver.find_element(By.CSS_SELECTOR, "a[href*='#camera-28']")
            if camera_link.is_displayed():
                print("✅ Found camera-28 link, clicking...")
                driver.execute_script("arguments[0].click();", camera_link)
                time.sleep(5)
        except:
            print("⚠️ Direct link not found, trying alternative methods...")

            # Method 2: Find by text content
            try:
                elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Sandy Point') or contains(text(), 'camera-28')]")
                for element in elements:
                    if element.is_displayed() and element.tag_name == 'a':
                        print("✅ Found Sandy Point link, clicking...")
                        driver.execute_script("arguments[0].click();", element)
                        time.sleep(5)
                        break
            except Exception as e:
                print(f"⚠️ Text-based search failed: {e}")

        # Look for video elements
        print("🎥 Searching for video elements...")
        video_elements = driver.find_elements(By.TAG_NAME, "video")
        print(f"Found {len(video_elements)} video elements")

        # Try to activate videos
        for i, video in enumerate(video_elements):
            try:
                if video.is_displayed():
                    print(f"🎬 Attempting to play video {i}...")
                    driver.execute_script("arguments[0].play();", video)
                    time.sleep(2)
            except Exception as e:
                print(f"⚠️ Video {i} play failed: {e}")

        # Look for iframe content
        print("🖼️ Checking for iframes...")
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        print(f"Found {len(iframes)} iframes")

        # Wait for content to load
        print("⏳ Waiting for video content to stabilize...")
        time.sleep(10)

        # Take screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sandy_point_test_{timestamp}.png"

        print(f"📸 Taking screenshot: {filename}")
        driver.save_screenshot(filename)

        # Get page source for analysis
        with open(f"sandy_point_page_source_{timestamp}.html", "w") as f:
            f.write(driver.page_source)

        print(f"✅ Test complete! Screenshot saved as {filename}")
        print(f"📄 Page source saved for analysis")

        return filename

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

    finally:
        if driver:
            driver.quit()

def test_exit23_camera():
    """Test capturing Exit 23 camera specifically"""
    driver = None
    try:
        print("🚀 Starting Exit 23 camera test...")

        driver = setup_chrome()
        driver.set_page_load_timeout(30)

        # Navigate to camera page
        url = "https://mdta.maryland.gov/index.php/traffic-cameras-by-facility#camera-48"
        print(f"📡 Loading: {url}")
        driver.get(url)

        # Wait for page load
        print("⏳ Waiting for page to load...")
        time.sleep(8)

        # Try to find and click on camera-48 specifically
        print("🎯 Looking for camera-48 link...")
        try:
            # Method 1: Direct anchor link
            camera_link = driver.find_element(By.CSS_SELECTOR, "a[href*='#camera-48']")
            if camera_link.is_displayed():
                print("✅ Found camera-48 link, clicking...")
                driver.execute_script("arguments[0].click();", camera_link)
                time.sleep(5)
        except:
            print("⚠️ Direct link not found, trying alternative methods...")

        # Look for video elements and activate
        print("🎥 Activating video content...")
        video_elements = driver.find_elements(By.TAG_NAME, "video")
        for video in video_elements:
            try:
                if video.is_displayed():
                    driver.execute_script("arguments[0].play();", video)
            except:
                pass

        # Wait for content
        print("⏳ Waiting for video content...")
        time.sleep(10)

        # Screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exit23_test_{timestamp}.png"

        print(f"📸 Taking screenshot: {filename}")
        driver.save_screenshot(filename)

        print(f"✅ Test complete! Screenshot saved as {filename}")
        return filename

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    print("=== TRAFFIC CAMERA VIDEO CAPTURE TEST ===\n")

    # Test both cameras
    sandy_result = test_sandy_point_camera()
    print("\n" + "="*50 + "\n")
    exit23_result = test_exit23_camera()

    print("\n=== RESULTS ===")
    print(f"Sandy Point: {'✅ SUCCESS' if sandy_result else '❌ FAILED'}")
    print(f"Exit 23: {'✅ SUCCESS' if exit23_result else '❌ FAILED'}")

    if sandy_result or exit23_result:
        print("\n📁 Check the generated screenshots to verify video content capture!")
    else:
        print("\n❌ Both tests failed. Check the log output above for details.")