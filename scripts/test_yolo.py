"""
Test script for YOLO Detector
"""
import sys
import os
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from trafficllm.models.yolo_detector import TrafficYOLODetector

def main():
    print("🚀 Testing TrafficYOLODetector...")
    
    try:
        # Initialize detector
        print("Initializing detector...")
        detector = TrafficYOLODetector(model_size='s') # Use small for better accuracy
        print("✅ Detector initialized")
        
        # Find test images
        screenshots_dir = project_root / "screenshots"
        test_images = list(screenshots_dir.glob("**/*.png"))
        
        if not test_images:
            print("⚠️ No test images found in screenshots/")
            return

        print(f"Found {len(test_images)} images. Testing first 3...")
        
        for i, test_image in enumerate(test_images[:3]):
            print(f"\n📸 Testing image {i+1}: {test_image.name}")
            
            # Test detection
            vehicles = detector.detect_vehicles(str(test_image))
            print(f"   ✅ Detected {len(vehicles)} vehicles")
            
            if vehicles:
                print("   Top 3 detections:")
                for v in vehicles[:3]:
                    print(f"     - {v['class']}: {v['confidence']:.2f}")
            
            # Test counting
            counts = detector.count_vehicles(str(test_image))
            print(f"   📊 Counts: {counts['by_type']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
