"""
Test expanding the video element itself to fill the viewport
Instead of using fullscreen API, we'll resize the video element
"""
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def setup_chrome():
    """Setup Chrome"""
    options = Options()
    # NOT headless to see what's happening
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")
    options.add_argument("--autoplay-policy=no-user-gesture-required")

    driver = webdriver.Chrome(service=Service(), options=options)
    driver.maximize_window()
    return driver

def wait_for_video_playing(driver, timeout=30):
    """Wait for video to be playing"""
    print("⏳ Waiting for video to play...")
    start = time.time()

    while (time.time() - start) < timeout:
        try:
            result = driver.execute_script("""
                var video = document.querySelector('video');
                if (video && !video.paused && video.currentTime > 0 && video.readyState >= 3) {
                    return {playing: true, time: video.currentTime};
                }
                if (video && video.paused) video.play();
                return {playing: false};
            """)

            if result.get('playing'):
                print(f"   ✅ Video playing at {result.get('time', 0):.1f}s")
                time.sleep(2)
                return True
        except:
            pass

        time.sleep(1)

    return False

def expand_video_to_fill_viewport(driver):
    """Expand video element to fill the entire browser viewport"""
    print("🎯 Expanding video to fill viewport...")

    try:
        # Use JavaScript to expand the video element and hide EVERYTHING else
        result = driver.execute_script("""
            var video = document.querySelector('video');
            if (!video) return {success: false, error: 'No video found'};

            // NUCLEAR OPTION: Hide absolutely everything and show only the video

            // 1. Hide ALL body children
            var allElements = document.body.querySelectorAll('*');
            for (var i = 0; i < allElements.length; i++) {
                var el = allElements[i];
                if (el !== video && !el.contains(video)) {
                    el.style.display = 'none';
                    el.style.visibility = 'hidden';
                    el.style.opacity = '0';
                }
            }

            // 2. Reset body to clean state
            document.body.style.margin = '0';
            document.body.style.padding = '0';
            document.body.style.overflow = 'hidden';
            document.body.style.backgroundColor = 'black';

            // 3. Reset html element
            document.documentElement.style.margin = '0';
            document.documentElement.style.padding = '0';
            document.documentElement.style.overflow = 'hidden';

            // 4. Detach video from its parent and append directly to body
            var parent = video.parentElement;
            video.remove();
            document.body.innerHTML = '';  // Clear everything
            document.body.appendChild(video);  // Add only the video

            // 5. Make video fill the entire viewport
            video.style.position = 'fixed';
            video.style.top = '0';
            video.style.left = '0';
            video.style.width = '100vw';
            video.style.height = '100vh';
            video.style.margin = '0';
            video.style.padding = '0';
            video.style.zIndex = '9999999';
            video.style.objectFit = 'contain';  // Maintain aspect ratio
            video.style.backgroundColor = 'black';

            // 6. REMOVE VIDEO CONTROLS - this is the key!
            video.removeAttribute('controls');
            video.controls = false;

            // 7. Ensure video is playing
            if (video.paused) {
                video.play();
            }

            return {
                success: true,
                videoSize: {
                    width: video.offsetWidth,
                    height: video.offsetHeight
                }
            };
        """)

        if result.get('success'):
            print(f"   ✅ Video expanded to {result['videoSize']['width']}x{result['videoSize']['height']}")
            return True
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_sandy_point():
    """Test Sandy Point with video expansion"""
    driver = None
    try:
        print("🚀 Starting Sandy Point VIDEO EXPANSION test...\n")

        driver = setup_chrome()
        driver.set_page_load_timeout(30)

        # Navigate
        url = "https://mdta.maryland.gov/index.php/traffic-cameras-by-facility#camera-28"
        print(f"📡 Loading: {url}")
        driver.get(url)
        time.sleep(8)

        # Click camera link
        print("🎯 Looking for camera-28...")
        try:
            camera_link = driver.find_element(By.CSS_SELECTOR, "a[href*='#camera-28']")
            driver.execute_script("arguments[0].click();", camera_link)
            print("   ✅ Clicked camera link")
            time.sleep(5)
        except:
            print("   ⚠️ Camera link not found")

        # Activate and wait for video
        print("🎥 Activating video...")
        videos = driver.find_elements(By.TAG_NAME, "video")
        for video in videos:
            try:
                driver.execute_script("arguments[0].play();", video)
            except:
                pass

        # Wait for video to play
        if not wait_for_video_playing(driver):
            print("   ⚠️ Video not playing, continuing anyway...")

        time.sleep(3)

        # Expand video to fill viewport
        expand_success = expand_video_to_fill_viewport(driver)

        if expand_success:
            print("⏳ Waiting for video rendering...")
            time.sleep(5)

        # Take screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "EXPANDED" if expand_success else "NORMAL"
        filename = f"sandy_point_{mode}_{timestamp}.png"

        print(f"📸 Taking screenshot: {filename}")
        driver.save_screenshot(filename)

        # Check file
        import os
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✅ Screenshot saved: {filename} ({size:,} bytes)")
            return filename, expand_success
        else:
            print("❌ Screenshot not saved")
            return None, False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None, False

    finally:
        if driver:
            time.sleep(2)  # Brief pause to see result
            driver.quit()

if __name__ == "__main__":
    print("=== 🎬 VIDEO EXPANSION TEST ===\n")

    filename, success = test_sandy_point()

    print("\n=== 📊 RESULTS ===")
    if filename:
        print(f"✅ Screenshot: {filename}")
        print(f"🎯 Video Expanded: {'YES' if success else 'NO'}")
        print("\n💡 The video should now fill the entire screenshot!")
    else:
        print("❌ Test failed")
