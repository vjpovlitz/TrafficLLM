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

    # REMOVED --headless - causes black screen issues with fullscreen video
    # options.add_argument("--headless")  # DISABLED
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

    # Don't use user-data-dir to avoid conflicts
    # Just use a fresh browser instance each time

    driver = webdriver.Chrome(service=Service(), options=options)
    driver.maximize_window()
    return driver

def wait_for_video_playing(driver, timeout=30):
    """Wait for video to actually be playing - CRITICAL to avoid black screens"""
    print("⏳ Waiting for video to play...")
    import time
    start = time.time()

    while (time.time() - start) < timeout:
        try:
            result = driver.execute_script("""
                var video = document.querySelector('video');
                if (video && video.offsetWidth > 0) {
                    // Check if video is playing and has rendered frames
                    if (!video.paused && video.currentTime > 0 && video.readyState >= 3) {
                        return {playing: true, time: video.currentTime, state: video.readyState};
                    }
                    // Try to play if paused
                    if (video.paused) { video.play(); }
                }
                return {playing: false};
            """)

            if result.get('playing'):
                print(f"   ✅ Video playing at {result.get('time', 0):.1f}s, readyState={result.get('state')}")
                time.sleep(2)  # Wait for stable playback
                return True

        except Exception as e:
            print(f"   ⚠️ Check failed: {e}")

        time.sleep(1)

    print("   ⚠️ Video not playing after timeout")
    return False

def try_fullscreen(driver):
    """Attempt multiple fullscreen strategies - TARGET VIDEO PLAYER CONTROLS"""
    print("🎯 Attempting video player fullscreen activation...")

    # PRIORITY: Strategy 1 - Hover over video to reveal controls, then click fullscreen button
    print("   🖱️ Strategy 1: Hover and click video player controls...")
    try:
        video_elements = driver.find_elements(By.TAG_NAME, "video")
        if video_elements:
            video = video_elements[0]

            # Hover over video to make controls visible
            actions = ActionChains(driver)
            actions.move_to_element(video).perform()
            time.sleep(2)  # Wait for controls to appear

            # Try clicking fullscreen button in video controls
            # These are common selectors for video player fullscreen buttons
            video_control_selectors = [
                # Look for buttons within or near the video
                "video ~ * button[title*='fullscreen' i]",
                "video ~ * button[aria-label*='fullscreen' i]",
                "video + * button[title*='fullscreen' i]",
                ".video-js .vjs-fullscreen-control",  # Video.js
                ".plyr__controls button[data-plyr='fullscreen']",  # Plyr
                "button.ytp-fullscreen-button",  # YouTube player
                ".html5-video-player .ytp-fullscreen-button",
                "button[class*='fullscreen']",
                "button[aria-label='Fullscreen']",
                "button[title='Fullscreen']",
                # Try SVG icons in buttons
                "button svg[class*='fullscreen']",
                "button svg[class*='expand']",
            ]

            for selector in video_control_selectors:
                try:
                    buttons = driver.find_elements(By.CSS_SELECTOR, selector)
                    for btn in buttons:
                        if btn.is_displayed():
                            print(f"   🎯 Found video control button: {selector}")
                            try:
                                # Hover to ensure controls are visible
                                actions.move_to_element(btn).perform()
                                time.sleep(0.5)
                                btn.click()
                                print("   ⏳ Waiting for video fullscreen...")
                                time.sleep(4)

                                if is_fullscreen(driver):
                                    print("   ✅ VIDEO PLAYER fullscreen activated!")
                                    return True
                            except Exception as e:
                                print(f"   ⚠️ Click failed: {e}")
                                try:
                                    driver.execute_script("arguments[0].click();", btn)
                                    time.sleep(4)
                                    if is_fullscreen(driver):
                                        print("   ✅ VIDEO PLAYER fullscreen activated via JS!")
                                        return True
                                except:
                                    pass
                except Exception as e:
                    continue
    except Exception as e:
        print(f"   ⚠️ Hover strategy failed: {e}")

    # Strategy 2: Try finding any button near the video with fullscreen-related attributes
    print("   🔍 Strategy 2: Searching for fullscreen buttons near video...")
    try:
        # Use JavaScript to find buttons with fullscreen in their attributes
        fullscreen_button = driver.execute_script("""
            // Find video element
            var video = document.querySelector('video');
            if (!video) return null;

            // Search for fullscreen button in parent containers
            var container = video.closest('div');
            if (!container) container = video.parentElement;

            // Look for button with fullscreen keywords
            var buttons = container.querySelectorAll('button, a, div[role="button"]');
            for (var i = 0; i < buttons.length; i++) {
                var btn = buttons[i];
                var attrs = btn.getAttribute('title') + ' ' +
                           btn.getAttribute('aria-label') + ' ' +
                           btn.className + ' ' +
                           btn.getAttribute('data-title');

                if (attrs && attrs.toLowerCase().includes('fullscreen')) {
                    return btn;
                }
            }
            return null;
        """)

        if fullscreen_button:
            print("   🎯 Found fullscreen button via JavaScript search!")
            try:
                driver.execute_script("arguments[0].click();", fullscreen_button)
                time.sleep(4)
                if is_fullscreen(driver):
                    print("   ✅ VIDEO PLAYER fullscreen activated!")
                    return True
            except Exception as e:
                print(f"   ⚠️ JS click failed: {e}")
    except Exception as e:
        print(f"   ⚠️ JS search failed: {e}")

    # Strategy 3: Keyboard shortcut 'f' (works for many video players)
    print("   ⌨️ Strategy 3: Keyboard 'F' key...")
    try:
        video_elements = driver.find_elements(By.TAG_NAME, "video")
        if video_elements:
            video = video_elements[0]
            video.click()  # Focus on video
            time.sleep(1)

            actions = ActionChains(driver)
            actions.send_keys("f").perform()
            time.sleep(4)

            if is_fullscreen(driver):
                print("   ✅ VIDEO PLAYER fullscreen via 'F' key!")
                return True
    except Exception as e:
        print(f"   ⚠️ Keyboard 'F' failed: {e}")

    # Strategy 4: Double-click on video (common trigger for fullscreen)
    print("   🖱️ Strategy 4: Double-click video...")
    try:
        video_elements = driver.find_elements(By.TAG_NAME, "video")
        if video_elements:
            video = video_elements[0]
            actions = ActionChains(driver)
            actions.double_click(video).perform()
            time.sleep(4)

            if is_fullscreen(driver):
                print("   ✅ VIDEO PLAYER fullscreen via double-click!")
                return True
    except Exception as e:
        print(f"   ⚠️ Double-click failed: {e}")

    # Strategy 5: JavaScript API on video element (fallback)
    print("   🔧 Strategy 5: JavaScript Fullscreen API...")
    try:
        driver.execute_script("""
            var video = document.querySelector('video');
            if (video) {
                if (video.requestFullscreen) {
                    video.requestFullscreen();
                } else if (video.webkitRequestFullscreen) {
                    video.webkitRequestFullscreen();
                } else if (video.mozRequestFullScreen) {
                    video.mozRequestFullScreen();
                }
            }
        """)
        time.sleep(4)

        if is_fullscreen(driver):
            print("   ✅ Fullscreen activated via JS API!")
            return True
    except Exception as e:
        print(f"   ⚠️ JS API failed: {e}")

    print("   ❌ Could not activate video player fullscreen")
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

        # CRITICAL: Wait for video to actually play
        print("⏳ Waiting for video to stabilize...")
        video_playing = wait_for_video_playing(driver, timeout=30)

        if not video_playing:
            print("⚠️ Warning: Video may not be playing properly")

        # Attempt fullscreen
        fullscreen_success = try_fullscreen(driver)

        if fullscreen_success:
            print("🎯 FULLSCREEN ACTIVATED! Verifying video still plays...")
            time.sleep(3)

            # CRITICAL: Verify video is still playing in fullscreen
            still_playing = driver.execute_script("""
                var v = document.querySelector('video');
                return v && !v.paused && v.currentTime > 0 && v.readyState >= 3;
            """)

            if still_playing:
                print("   ✅ Video confirmed playing in fullscreen")
                time.sleep(5)  # Wait for stable frames
            else:
                print("   ⚠️ Video paused in fullscreen, resuming...")
                driver.execute_script("var v = document.querySelector('video'); if(v) v.play();")
                time.sleep(5)
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

        # CRITICAL: Wait for video to actually play
        print("⏳ Waiting for video...")
        video_playing = wait_for_video_playing(driver, timeout=30)

        if not video_playing:
            print("⚠️ Warning: Video may not be playing properly")

        # Attempt fullscreen
        fullscreen_success = try_fullscreen(driver)

        if fullscreen_success:
            print("🎯 FULLSCREEN ACTIVATED! Verifying video still plays...")
            time.sleep(3)

            # Verify video still playing in fullscreen
            still_playing = driver.execute_script("""
                var v = document.querySelector('video');
                return v && !v.paused && v.currentTime > 0 && v.readyState >= 3;
            """)

            if still_playing:
                print("   ✅ Video confirmed playing in fullscreen")
                time.sleep(5)
            else:
                print("   ⚠️ Video paused in fullscreen, resuming...")
                driver.execute_script("var v = document.querySelector('video'); if(v) v.play();")
                time.sleep(5)
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