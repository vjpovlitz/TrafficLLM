"""
Final Fullscreen Traffic Camera Capture - Video Only
Captures ONLY the video content without any webpage elements
"""
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def setup_chrome():
    """Setup Chrome with optimized options"""
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")
    options.add_argument("--autoplay-policy=no-user-gesture-required")

    driver = webdriver.Chrome(service=Service(), options=options)
    driver.maximize_window()
    return driver

def wait_for_video_playing(driver, timeout=30):
    """Wait for video to be actively playing"""
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

def expand_video_fullscreen(driver):
    """Expand video to fill viewport and remove all other elements"""
    print("🎯 Expanding video to fullscreen (video only)...")

    try:
        result = driver.execute_script("""
            var video = document.querySelector('video');
            if (!video) return {success: false, error: 'No video found'};

            // Nuclear option: Clear page and show ONLY the video

            // 1. Detach video and clear everything
            video.remove();
            document.body.innerHTML = '';
            document.body.appendChild(video);

            // 2. Reset body and html
            document.body.style.margin = '0';
            document.body.style.padding = '0';
            document.body.style.overflow = 'hidden';
            document.body.style.backgroundColor = 'black';
            document.documentElement.style.margin = '0';
            document.documentElement.style.padding = '0';
            document.documentElement.style.overflow = 'hidden';

            // 3. Make video fill entire viewport
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

            // 4. Remove video controls
            video.removeAttribute('controls');
            video.controls = false;

            // 5. Ensure video is playing
            if (video.paused) video.play();

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

def capture_camera(camera_name, url, camera_anchor):
    """Capture a single camera"""
    driver = None
    try:
        print(f"\n{'='*80}")
        print(f"🚀 Starting {camera_name} capture...")
        print(f"{'='*80}\n")

        driver = setup_chrome()
        driver.set_page_load_timeout(30)

        # Navigate
        print(f"📡 Loading: {url}")
        driver.get(url)
        time.sleep(8)

        # Click camera link
        print(f"🎯 Activating camera: {camera_anchor}")
        try:
            camera_link = driver.find_element(By.CSS_SELECTOR, f"a[href*='{camera_anchor}']")
            driver.execute_script("arguments[0].click();", camera_link)
            print("   ✅ Camera activated")
            time.sleep(5)
        except:
            print("   ⚠️ Camera link not found, continuing anyway")

        # Activate video
        print("🎥 Starting video playback...")
        videos = driver.find_elements(By.TAG_NAME, "video")
        print(f"   Found {len(videos)} video element(s)")
        for video in videos:
            try:
                driver.execute_script("arguments[0].play();", video)
            except:
                pass

        # Wait for video to play
        video_playing = wait_for_video_playing(driver)
        if not video_playing:
            print("   ⚠️ Video not confirmed playing, continuing anyway")

        time.sleep(3)

        # Expand video to fullscreen
        expand_success = expand_video_fullscreen(driver)

        if expand_success:
            print("⏳ Waiting for video to stabilize in fullscreen...")
            time.sleep(5)
        else:
            print("   ⚠️ Expansion failed, capturing current state")
            time.sleep(2)

        # Take screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{camera_name.lower().replace(' ', '_')}_FULLSCREEN_{timestamp}.png"

        print(f"📸 Taking screenshot: {filename}")
        driver.save_screenshot(filename)

        # Verify file
        import os
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✅ Screenshot saved: {filename}")
            print(f"   📊 Size: {size:,} bytes ({size/1024/1024:.2f} MB)")
            return {
                'success': True,
                'filename': filename,
                'size': size,
                'expanded': expand_success
            }
        else:
            print("❌ Screenshot not saved")
            return {'success': False}

    except Exception as e:
        print(f"❌ Capture failed: {e}")
        return {'success': False, 'error': str(e)}

    finally:
        if driver:
            time.sleep(1)
            driver.quit()

def main():
    """Capture both cameras"""
    print("=== 🎬 FULLSCREEN TRAFFIC CAMERA CAPTURE (VIDEO ONLY) ===\n")

    cameras = [
        {
            'name': 'Sandy_Point',
            'url': 'https://mdta.maryland.gov/index.php/traffic-cameras-by-facility#camera-28',
            'anchor': '#camera-28'
        },
        {
            'name': 'Exit_23',
            'url': 'https://mdta.maryland.gov/index.php/traffic-cameras-by-facility#camera-48',
            'anchor': '#camera-48'
        }
    ]

    results = []

    for camera in cameras:
        result = capture_camera(camera['name'], camera['url'], camera['anchor'])
        results.append({**camera, **result})
        time.sleep(2)  # Brief pause between cameras

    # Summary
    print("\n" + "="*80)
    print("📊 CAPTURE SUMMARY")
    print("="*80)

    for result in results:
        status = "✅ SUCCESS" if result.get('success') else "❌ FAILED"
        expanded = "🎯 VIDEO ONLY" if result.get('expanded') else "📷 NORMAL"

        print(f"\n{status} - {result['name']}")
        if result.get('success'):
            print(f"   📁 File: {result['filename']}")
            print(f"   📊 Size: {result['size']:,} bytes ({result['size']/1024/1024:.2f} MB)")
            print(f"   {expanded}")
        elif 'error' in result:
            print(f"   ⚠️ Error: {result['error']}")

    successful = sum(1 for r in results if r.get('success'))
    expanded_count = sum(1 for r in results if r.get('expanded'))

    print(f"\n🎉 Completed: {successful}/{len(results)} successful, {expanded_count} fullscreen")
    print("\n💡 Screenshots contain ONLY the video content (no webpage elements)!")

if __name__ == "__main__":
    main()
