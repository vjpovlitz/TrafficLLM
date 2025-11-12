"""
Debug script to inspect the video player controls and find the fullscreen button
"""
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def setup_chrome():
    """Setup Chrome - NOT headless so we can see what's happening"""
    options = Options()
    # NO headless - we want to see the browser
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")
    options.add_argument("--autoplay-policy=no-user-gesture-required")

    driver = webdriver.Chrome(service=Service(), options=options)
    driver.maximize_window()
    return driver

def inspect_video_controls(url, camera_anchor):
    """Inspect the video player and its controls"""
    driver = setup_chrome()

    try:
        print(f"🔍 Loading: {url}")
        driver.get(url)
        time.sleep(8)

        # Click camera link
        print(f"🎯 Clicking camera link: {camera_anchor}")
        try:
            camera_link = driver.find_element(By.CSS_SELECTOR, f"a[href*='{camera_anchor}']")
            driver.execute_script("arguments[0].click();", camera_link)
            time.sleep(5)
        except:
            print("⚠️ Camera link not found")

        # Activate video
        print("🎥 Activating video...")
        videos = driver.find_elements(By.TAG_NAME, "video")
        for video in videos:
            try:
                driver.execute_script("arguments[0].play();", video)
            except:
                pass
        time.sleep(8)

        # Hover over video to reveal controls
        print("🖱️ Hovering over video to reveal controls...")
        if videos:
            from selenium.webdriver.common.action_chains import ActionChains
            actions = ActionChains(driver)
            actions.move_to_element(videos[0]).perform()
            time.sleep(3)

        # Inspect the DOM structure around the video
        print("\n" + "="*80)
        print("🔎 INSPECTING VIDEO PLAYER STRUCTURE")
        print("="*80)

        structure = driver.execute_script("""
            var video = document.querySelector('video');
            if (!video) return {error: 'No video found'};

            var result = {
                video_info: {
                    tagName: video.tagName,
                    id: video.id,
                    className: video.className,
                    width: video.offsetWidth,
                    height: video.offsetHeight
                },
                parent: null,
                siblings: [],
                all_buttons: []
            };

            // Get parent info
            var parent = video.parentElement;
            if (parent) {
                result.parent = {
                    tagName: parent.tagName,
                    id: parent.id,
                    className: parent.className
                };

                // Search for buttons in parent AND grandparent (broader search)
                var containers = [parent];
                if (parent.parentElement) containers.push(parent.parentElement);

                for (var c = 0; c < containers.length; c++) {
                    var container = containers[c];
                    var buttons = container.querySelectorAll('button, [role="button"], a, div[onclick]');

                    for (var i = 0; i < buttons.length; i++) {
                        var btn = buttons[i];
                        var isVisible = btn.offsetWidth > 0 && btn.offsetHeight > 0;

                        // Also check style.display
                        var style = window.getComputedStyle(btn);
                        isVisible = isVisible && style.display !== 'none' && style.visibility !== 'hidden';

                        result.all_buttons.push({
                            index: result.all_buttons.length,
                            tagName: btn.tagName,
                            id: btn.id || '',
                            className: btn.className || '',
                            title: btn.getAttribute('title') || '',
                            ariaLabel: btn.getAttribute('aria-label') || '',
                            innerText: (btn.innerText || '').substring(0, 50),
                            visible: isVisible,
                            container: c === 0 ? 'parent' : 'grandparent'
                        });
                    }
                }
            }

            return result;
        """)

        print("\n📺 VIDEO ELEMENT:")
        if 'error' in structure:
            print(f"   ❌ {structure['error']}")
            return

        print(f"   Tag: {structure['video_info']['tagName']}")
        print(f"   ID: {structure['video_info']['id']}")
        print(f"   Class: {structure['video_info']['className']}")
        print(f"   Size: {structure['video_info']['width']}x{structure['video_info']['height']}")

        if structure['parent']:
            print(f"\n📦 PARENT CONTAINER:")
            print(f"   Tag: {structure['parent']['tagName']}")
            print(f"   ID: {structure['parent']['id']}")
            print(f"   Class: {structure['parent']['className']}")

        print(f"\n🔘 BUTTONS IN PARENT CONTAINER ({len(structure['all_buttons'])} found):")
        for btn in structure['all_buttons']:
            if btn['visible']:
                print(f"\n   Button #{btn['index']}:")
                print(f"      Tag: {btn['tagName']}")
                print(f"      ID: '{btn['id']}'")
                print(f"      Class: '{btn['className']}'")
                print(f"      Title: '{btn['title']}'")
                print(f"      Aria-Label: '{btn['ariaLabel']}'")
                print(f"      Text: '{btn['innerText'][:50]}'")

                # Check if this looks like a fullscreen button
                fullscreen_keywords = ['fullscreen', 'full screen', 'expand', 'maximize', 'full-screen']
                attrs = f"{btn['className']} {btn['title']} {btn['ariaLabel']} {btn['id']}".lower()
                if any(keyword in attrs for keyword in fullscreen_keywords):
                    print(f"      ⭐ POTENTIAL FULLSCREEN BUTTON!")

        print("\n" + "="*80)
        print("✅ Inspection complete!")
        print("="*80)

        # Auto-close after 5 seconds
        print("\n⏸️  Browser will close in 5 seconds...")
        time.sleep(5)

    finally:
        driver.quit()

if __name__ == "__main__":
    print("=== 🔍 VIDEO PLAYER CONTROLS DEBUG ===\n")

    # Test Sandy Point
    inspect_video_controls(
        url="https://mdta.maryland.gov/index.php/traffic-cameras-by-facility#camera-28",
        camera_anchor="#camera-28"
    )
