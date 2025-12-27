import os
import datetime
import time
import schedule

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Sandy Point Camera
CAMERA_NAME = "US_50_AT_SANDY_POINT"
CAMERA_URL = "https://mdta.maryland.gov/index.php/traffic-cameras-by-facility#camera-28"
OUTPUT_DIR = "screenshots"

def capture_screenshot():
    """
    Opens the specified MDTA camera URL in a Selenium-controlled browser
    (headless), takes a screenshot, and saves it in a standardized format:
    YYYYMMDD_HHMMSS_CAMERA_NAME.jpg
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Build a standardized file name like 20250101_130000_US_50_AT_SANDY_POINT.jpg
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp_str}_{CAMERA_NAME}.jpg"
    file_path = os.path.join(OUTPUT_DIR, filename)

    # Configure Chrome to run headless
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    # Initialize the WebDriver (assuming chromedriver is on your system PATH)
    service = Service()
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Navigate to the camera URL
        driver.get(CAMERA_URL)
        # Wait a few seconds for the page to load fully
        time.sleep(5)
        # Take a screenshot of the full page
        driver.save_screenshot(file_path)

        print(f"[{datetime.datetime.now()}] Screenshot saved to: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()

def main():
    """
    Takes a screenshot immediately, then schedules a screenshot
    every 15 minutes thereafter.
    """
    # 1. Take the initial screenshot
    capture_screenshot()

    # 2. Schedule the job to run every 15 minutes
    schedule.every(15).minutes.do(capture_screenshot)
    print("Script is now running headless. It will capture a screenshot every 15 minutes.")
    print("Press Ctrl+C to exit.")

    # 3. Keep the script running indefinitely
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting script.")

if __name__ == "__main__":
    main()
