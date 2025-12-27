"""
Test Fullscreen Traffic Camera Capture
Based on the working test_video_capture.py with fullscreen enhancements
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
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

def setup_chrome():
    """Setup Chrome with video-optimized options"""
    options = Options()

    # Basic headless options
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")

    # Video-specific options
    options.add_argument("--autoplay-policy=no-user-gesture-required")
    options.add_argument("--disable-web-security")
    options.add_argument("--disable-features=VizDisplayCompositor")

    # Fullscreen support
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-infobars")

    # Unique session
    import tempfile
    import uuid
    temp_dir = tempfile.mkdtemp(prefix=f"chrome_fullscreen_{uuid.uuid4().hex[:8]}_")
    options.add_argument(f"--user-data-dir={temp_dir}")

    driver = webdriver.Chrome(service=Service(), options=options)
    driver.maximize_window()
    return driver

def try_fullscreen(driver):
    """Attempt multiple fullscreen strategies"""
    print("🎯 Attempting fullscreen activation...")

    # Strategy 1: Look for fullscreen buttons
    fullscreen_selectors = [
        "[title*='fullscreen' i]",
        "[title*='Fullscreen' i]",
        "[aria-label*='fullscreen' i]",
        ".fullscreen-button",
        ".btn-fullscreen",
        ".fs-button",
        ".fullscreen",
        ".fa-expand",
        ".icon-fullscreen",
        "button[class*='fullscreen' i]",
        "svg[class*='fullscreen' i]"
    ]

    for selector in fullscreen_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                if element.is_displayed() and element.is_enabled():
                    print(f"   🔘 Found fullscreen button: {selector}")
                    try:
                        element.click()
                        time.sleep(3)
                        if is_fullscreen(driver):
                            print("   ✅ Fullscreen activated via button!")
                            return True
                    except:
                        try:
                            driver.execute_script("arguments[0].click();", element)
                            time.sleep(3)
                            if is_fullscreen(driver):
                                print("   ✅ Fullscreen activated via JS click!")
                                return True
                        except:
                            pass
        except:
            pass

    # Strategy 2: JavaScript fullscreen API
    print("   🔧 Trying JavaScript fullscreen...")
    js_methods = [
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

    for i, js_method in enumerate(js_methods):
        try:
            driver.execute_script(js_method)
            time.sleep(3)
            if is_fullscreen(driver):
                print(f"   ✅ Fullscreen activated via JS method {i+1}!")
                return True
        except:
            pass

    # Strategy 3: Keyboard shortcuts
    print("   ⌨️ Trying keyboard shortcuts...")
    try:
        # Click on video first for focus
        video_elements = driver.find_elements(By.TAG_NAME, "video")
        if video_elements:
            video_elements[0].click()
            time.sleep(1)

        # Try F key (video player fullscreen)
        actions = ActionChains(driver)
        actions.send_keys("f").perform()
        time.sleep(3)
        if is_fullscreen(driver):
            print("   ✅ Fullscreen activated via 'F' key!")
            return True

        # Try F11 (browser fullscreen)
        actions.send_keys(Keys.F11).perform()
        time.sleep(3)
        if is_fullscreen(driver):
            print("   ✅ Fullscreen activated via F11!")
            return True

    except:
        pass

    # Strategy 4: Double-click
    print("   🖱️ Trying double-click...")
    try:
        video_elements = driver.find_elements(By.TAG_NAME, "video")
        for video in video_elements:
            if video.is_displayed():
                actions = ActionChains(driver)
                actions.double_click(video).perform()
                time.sleep(3)
                if is_fullscreen(driver):
                    print("   ✅ Fullscreen activated via double-click!")
                    return True
    except:
        pass

    print("   ⚠️ Could not activate fullscreen")
    return False

def is_fullscreen(driver):
    """Check if in fullscreen mode"""
    try:
        checks = [
            "return document.fullscreenElement !== null;",
            "return document.webkitFullscreenElement !== null;",
            "return document.mozFullScreenElement !== null;",
            "return window.innerHeight === screen.height;",
            "return window.outerHeight === screen.height;"
        ]

        for check in checks:
            try:
                if driver.execute_script(check):
                    return True
            except:
                pass
        return False
    except:
        return False

def test_fullscreen_sandy_point():
    """Test Sandy Point with fullscreen"""
    driver = None
    try:
        print("🚀 Starting Sandy Point FULLSCREEN test...")

        driver = setup_chrome()
        driver.set_page_load_timeout(30)

        # Navigate
        url = "https://mdta.maryland.gov/index.php/traffic-cameras-by-facility#camera-28"
        print(f"📡 Loading: {url}")
        driver.get(url)

        # Wait for page load
        print("⏳ Waiting for page to load...")
        time.sleep(8)

        # Click camera-28
        print("🎯 Looking for camera-28...")
        try:
            camera_link = driver.find_element(By.CSS_SELECTOR, "a[href*='#camera-28']")
            if camera_link.is_displayed():
                print("✅ Found camera-28 link, clicking...")
                driver.execute_script("arguments[0].click();", camera_link)
                time.sleep(5)
        except:
            print("⚠️ Camera link not found")

        # Activate video
        print("🎥 Activating video...")
        video_elements = driver.find_elements(By.TAG_NAME, "video")
        print(f"Found {len(video_elements)} video elements")

        for i, video in enumerate(video_elements):
            try:
                if video.is_displayed():
                    print(f"🎬 Playing video {i}...")
                    driver.execute_script("arguments[0].play();", video)
                    time.sleep(2)
            except Exception as e:
                print(f"⚠️ Video {i} play failed: {e}")

        # Wait for video to stabilize
        print("⏳ Waiting for video to stabilize...")
        time.sleep(10)

        # Attempt fullscreen
        fullscreen_success = try_fullscreen(driver)

        if fullscreen_success:
            print("🎯 FULLSCREEN ACTIVATED! Waiting for stabilization...")
            time.sleep(8)  # Extra wait for fullscreen
        else:
            print("📷 Capturing in standard mode...")
            time.sleep(3)

        # Screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "FULLSCREEN" if fullscreen_success else "STANDARD"
        filename = f"sandy_point_{mode}_{timestamp}.png"

        print(f"📸 Taking {mode} screenshot: {filename}")
        driver.save_screenshot(filename)

        print(f"✅ {mode} test complete! Screenshot: {filename}")
        return filename, fullscreen_success

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None, False

    finally:
        if driver:
            try:
                # Exit fullscreen before closing
                driver.execute_script("if (document.exitFullscreen) document.exitFullscreen();")
            except:
                pass
            driver.quit()

def test_fullscreen_exit23():
    """Test Exit 23 with fullscreen"""
    driver = None
    try:
        print("🚀 Starting Exit 23 FULLSCREEN test...")

        driver = setup_chrome()
        driver.set_page_load_timeout(30)

        # Navigate
        url = "https://mdta.maryland.gov/index.php/traffic-cameras-by-facility#camera-48"
        print(f"📡 Loading: {url}")
        driver.get(url)

        # Wait for page load
        print("⏳ Waiting for page to load...")
        time.sleep(8)

        # Click camera-48
        print("🎯 Looking for camera-48...")
        try:
            camera_link = driver.find_element(By.CSS_SELECTOR, "a[href*='#camera-48']")
            if camera_link.is_displayed():
                print("✅ Found camera-48 link, clicking...")
                driver.execute_script("arguments[0].click();", camera_link)
                time.sleep(5)
        except:
            print("⚠️ Camera link not found")

        # Activate video
        print("🎥 Activating video...")
        video_elements = driver.find_elements(By.TAG_NAME, "video")
        for video in video_elements:
            try:
                if video.is_displayed():
                    driver.execute_script("arguments[0].play();", video)
            except:
                pass

        # Wait for video
        print("⏳ Waiting for video...")
        time.sleep(10)

        # Attempt fullscreen
        fullscreen_success = try_fullscreen(driver)

        if fullscreen_success:
            print("🎯 FULLSCREEN ACTIVATED! Waiting...")
            time.sleep(8)
        else:
            print("📷 Standard mode capture...")
            time.sleep(3)

        # Screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "FULLSCREEN" if fullscreen_success else "STANDARD"
        filename = f"exit23_{mode}_{timestamp}.png"

        print(f"📸 Taking {mode} screenshot: {filename}")
        driver.save_screenshot(filename)

        print(f"✅ {mode} test complete! Screenshot: {filename}")
        return filename, fullscreen_success

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None, False

    finally:
        if driver:
            try:
                driver.execute_script("if (document.exitFullscreen) document.exitFullscreen();")
            except:
                pass
            driver.quit()

if __name__ == "__main__":
    print("=== 🎬 FULLSCREEN TRAFFIC CAMERA TEST ===\n")

    # Test both cameras
    sandy_result, sandy_fullscreen = test_fullscreen_sandy_point()
    print("\n" + "="*60 + "\n")
    exit23_result, exit23_fullscreen = test_fullscreen_exit23()

    print("\n=== 🎯 FULLSCREEN TEST RESULTS ===")

    sandy_icon = "🎬" if sandy_fullscreen else "📷"
    exit23_icon = "🎬" if exit23_fullscreen else "📷"

    print(f"{sandy_icon} Sandy Point: {'✅ SUCCESS' if sandy_result else '❌ FAILED'}")
    if sandy_result:
        print(f"   📁 File: {sandy_result}")
        print(f"   🎯 Fullscreen: {'YES' if sandy_fullscreen else 'NO'}")

    print(f"{exit23_icon} Exit 23: {'✅ SUCCESS' if exit23_result else '❌ FAILED'}")
    if exit23_result:
        print(f"   📁 File: {exit23_result}")
        print(f"   🎯 Fullscreen: {'YES' if exit23_fullscreen else 'NO'}")

    if sandy_fullscreen or exit23_fullscreen:
        print("\n🎉 FULLSCREEN SUCCESS! Check the screenshots for improved quality!")
    else:
        print("\n⚠️ Fullscreen not achieved, but standard captures completed.")

    print("\n📊 Compare file sizes to see the improvement in detail capture!")