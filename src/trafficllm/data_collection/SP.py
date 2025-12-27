import os
import datetime
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

def capture_mdta_camera_screenshot(
    camera_name="US_50_AT_SANDY_POINT",
    url="https://mdta.maryland.gov/index.php/traffic-cameras-by-facility#camera-28",
    output_dir="screenshots"
):
    """
    Opens the specified MDTA camera URL in a Selenium-controlled browser,
    takes a screenshot, and saves it in a standardized format.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build a standardized file name like US_50_AT_SANDY_POINT_20230101_130000.jpg
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{camera_name}_{timestamp_str}.jpg"
    file_path = os.path.join(output_dir, filename)
    
    # (Optional) Use headless mode for Chrome
    chrome_options = Options()
    # Uncomment to enable headless mode:
    # chrome_options.add_argument("--headless")

    # Initialize Chrome WebDriver
    service = Service()  # If chromedriver is on your PATH, just Service()
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Navigate to the camera URL
        driver.get(url)
        
        # Wait a few seconds for the page to load resources
        time.sleep(5)  # Adjust as necessary

        # Take full screenshot of the page
        driver.save_screenshot(file_path)
        print(f"Screenshot saved to: {file_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the browser
        driver.quit()

if __name__ == "__main__":
    capture_mdta_camera_screenshot()
