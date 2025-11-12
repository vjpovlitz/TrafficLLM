#!/usr/bin/env python3
"""Simple test to verify the fix works"""
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Setup Chrome with headless mode (server has no display)
import tempfile
import os

options = Options()
options.add_argument("--headless=new")  # Use new headless mode
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
options.add_argument("--start-maximized")
options.add_argument("--autoplay-policy=no-user-gesture-required")
options.add_argument("--disable-popup-blocking")
options.add_argument("--enable-features=NetworkService,NetworkServiceInProcess")

# Create unique temp directory
temp_dir = tempfile.mkdtemp(prefix="chrome_test_")
options.add_argument(f"--user-data-dir={temp_dir}")
print(f"Using temp dir: {temp_dir}")
print("ℹ️  Using headless mode - fullscreen may not work, will capture standard view")

driver = webdriver.Chrome(service=Service(), options=options)

try:
    print("🚀 Testing Sandy Point with video verification...")

    url = "https://mdta.maryland.gov/index.php/traffic-cameras-by-facility#camera-28"
    driver.get(url)
    time.sleep(8)

    # Click camera
    try:
        camera_link = driver.find_element(By.CSS_SELECTOR, "a[href*='#camera-28']")
        driver.execute_script("arguments[0].click();", camera_link)
        time.sleep(5)
    except:
        print("⚠️ Could not click camera link")

    # Wait for video to play
    print("⏳ Waiting for video...")
    for i in range(30):
        result = driver.execute_script("""
            var v = document.querySelector('video');
            if (v && v.offsetWidth > 0 && !v.paused && v.currentTime > 0 && v.readyState >= 3) {
                return {playing: true, time: v.currentTime};
            }
            if (v && v.paused) { v.play(); }
            return {playing: false};
        """)

        if result.get('playing'):
            print(f"✅ Video playing at {result.get('time', 0):.1f}s")
            break
        time.sleep(1)

    # Try fullscreen
    print("🎯 Trying fullscreen...")
    fs_result = driver.execute_script("""
        var v = document.querySelector('video');
        if (v && v.requestFullscreen) {
            v.requestFullscreen();
            return true;
        }
        return false;
    """)

    if fs_result:
        time.sleep(3)
        print("✅ Fullscreen requested")

        # Verify still playing
        still = driver.execute_script("""
            var v = document.querySelector('video');
            return v && !v.paused && v.currentTime > 0;
        """)

        if still:
            print("✅ Video still playing in fullscreen")
            time.sleep(5)
        else:
            print("⚠️ Video paused, resuming...")
            driver.execute_script("document.querySelector('video').play();")
            time.sleep(5)

    # Screenshot
    filename = f"test_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    driver.save_screenshot(filename)
    print(f"📸 Screenshot saved: {filename}")

finally:
    driver.quit()
