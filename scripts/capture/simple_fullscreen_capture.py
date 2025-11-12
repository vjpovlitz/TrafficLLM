"""
Simple Traffic Camera Fullscreen Capture
Works with live traffic streams without overthinking video states
"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("❌ Playwright not installed. Run: pip install playwright && python -m playwright install")
    exit(1)

class SimpleFullscreenCapture:
    """Simple fullscreen capture that works with live traffic streams"""

    def __init__(self, config_path: str = "camera_config.json"):
        self.config_path = config_path
        self.output_dir = Path("FULLSCREEN_FINAL_RESULTS")
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.cameras = self._load_cameras()

    def _load_cameras(self) -> Dict:
        """Load camera configurations"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config.get('cameras', {})
        except Exception as e:
            self.logger.error(f"Failed to load camera config: {e}")
            return {}

    async def capture_traffic_camera(self, camera_id: str, camera_config: Dict) -> Dict:
        """Capture traffic camera with simple, reliable approach"""
        result = {
            "camera_id": camera_id,
            "camera_name": camera_config.get('name', camera_id),
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "file_path": None,
            "file_size": None,
            "fullscreen_activated": False,
            "error": None
        }

        browser = None
        try:
            self.logger.info(f"🚀 Starting {camera_config.get('name', camera_id)}")

            # Setup browser
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(
                headless=False,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--autoplay-policy=no-user-gesture-required",
                    "--start-maximized",
                    "--disable-popup-blocking"
                ]
            )

            page = await browser.new_page(viewport={"width": 1920, "height": 1080})

            # Navigate
            url = camera_config.get('url')
            self.logger.info(f"📡 Loading {url}")
            await page.goto(url, wait_until="networkidle")

            # Wait for page to load
            self.logger.info("⏳ Waiting for page content...")
            await page.wait_for_timeout(8000)

            # Look for video elements
            try:
                await page.wait_for_selector("video", timeout=10000)
                self.logger.info("📹 Video found")

                # Click on video to activate it
                videos = await page.locator("video").all()
                if videos:
                    await videos[0].click()
                    self.logger.info("🖱️ Clicked on video")

                # Give video time to load
                await page.wait_for_timeout(10000)

                # Try to activate fullscreen
                self.logger.info("🎯 Attempting fullscreen...")

                fullscreen_success = await page.evaluate("""
                    () => {
                        const videos = document.querySelectorAll('video');
for (const video of videos) {
                            if (video.offsetWidth > 0) {
                                try {
                                    if (video.requestFullscreen) {
                                        video.requestFullscreen();
                                        return true;
                                    } else if (video.webkitRequestFullscreen) {
                                        video.webkitRequestFullscreen();
                                        return true;
                                    }
                                } catch (e) {
                                    console.log('Fullscreen failed:', e);
                                }
                            }
                        }
                        return false;
                    }
                """)

                if fullscreen_success:
                    # Wait for fullscreen transition
                    await page.wait_for_timeout(3000)

                    # Check if we're actually in fullscreen
                    is_fullscreen = await page.evaluate("""
                        () => !!(document.fullscreenElement || document.webkitFullscreenElement)
                    """)

                    if is_fullscreen:
                        self.logger.info("✅ Fullscreen activated!")
                        result["fullscreen_activated"] = True

                        # Give more time for video to render in fullscreen
                        self.logger.info("⏳ Waiting for fullscreen content...")
                        await page.wait_for_timeout(15000)

                        # Try to ensure video is playing
                        await page.evaluate("""
                            () => {
                                const videos = document.querySelectorAll('video');
                                videos.forEach(video => {
                                    video.play().catch(e => console.log('Play error:', e));
                                });
                            }
                        """)

                        # Final wait
                        await page.wait_for_timeout(5000)

                    else:
                        self.logger.warning("⚠️ Fullscreen not confirmed")
                else:
                    self.logger.warning("⚠️ Fullscreen activation failed")

            except Exception as e:
                self.logger.warning(f"Video handling issue: {e}")

            # Take screenshot regardless
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = "FULLSCREEN" if result["fullscreen_activated"] else "STANDARD"
            filename = f"{camera_config.get('name', camera_id)}_{mode}_{timestamp}.png"
            file_path = self.output_dir / filename

            self.logger.info(f"📸 Taking screenshot: {filename}")
            await page.screenshot(path=str(file_path), type="png")

            # Verify file
            if file_path.exists() and file_path.stat().st_size > 0:
                result["success"] = True
                result["file_path"] = str(file_path)
                result["file_size"] = file_path.stat().st_size
                self.logger.info(f"✅ Captured: {filename} ({result['file_size']:,} bytes)")
            else:
                result["error"] = "Screenshot file not created"

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"❌ Capture failed: {e}")

        finally:
            if browser:
                try:
                    await browser.close()
                except:
                    pass

        return result

async def main():
    """Simple capture of all traffic cameras"""
    print("=== 🚗 SIMPLE TRAFFIC CAMERA FULLSCREEN CAPTURE ===\n")

    capture = SimpleFullscreenCapture()

    if not capture.cameras:
        print("❌ No cameras configured")
        return

    print(f"📹 Cameras: {len(capture.cameras)}")
    for camera_id, config in capture.cameras.items():
        print(f"   - {config.get('name', camera_id)}")

    print("\n🚀 Starting simple captures...\n")

    results = {}
    for camera_id, camera_config in capture.cameras.items():
        if camera_config.get('enabled', True):
            result = await capture.capture_traffic_camera(camera_id, camera_config)
            results[camera_id] = result
            await asyncio.sleep(2)

    # Results
    print("\n=== 🎯 RESULTS ===")
    successful = sum(1 for r in results.values() if r.get("success"))
    fullscreen = sum(1 for r in results.values() if r.get("fullscreen_activated"))

    for camera_id, result in results.items():
        status = "✅" if result.get("success") else "❌"
        mode = "🎯 FULLSCREEN" if result.get("fullscreen_activated") else "📷 STANDARD"

        print(f"{status} {result.get('camera_name', camera_id)}")
        print(f"   Mode: {mode}")

        if result.get("success"):
            print(f"   📁 {result['file_path']}")
            print(f"   📊 {result['file_size']:,} bytes")

        if result.get("error"):
            print(f"   ⚠️ {result['error']}")
        print()

    print(f"🎉 Complete: {successful}/{len(results)} successful, {fullscreen} fullscreen")
    print("📂 Results: FULLSCREEN_FINAL_RESULTS/")

if __name__ == "__main__":
    asyncio.run(main())