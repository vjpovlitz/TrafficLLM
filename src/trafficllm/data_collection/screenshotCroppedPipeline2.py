import cv2
import os
import glob
from PIL import Image

def select_roi_and_crop(input_folder="screenshots", output_folder="processed"):
    """
    Opens first image for ROI selection, then automatically processes all images with that selection.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image files and sort them
    image_files = glob.glob(os.path.join(input_folder, "*.jpg")) + \
                 glob.glob(os.path.join(input_folder, "*.png"))
    image_files.sort()  # Sort files to process them in order
    
    if not image_files:
        print("No images found in input folder")
        return

    # Read first image and verify it's loaded correctly
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"Error: Could not read first image: {image_files[0]}")
        return

    print("Instructions:")
    print("1. Left click and drag to select area")
    print("2. Press ENTER to confirm selection")
    print("3. Press 'c' to process all images")
    print("4. Press 'q' to quit")

    window_name = "Select Region (Press ENTER to confirm, 'c' to process all, 'q' to quit)"
    cv2.namedWindow(window_name)
    roi = cv2.selectROI(window_name, first_img, False)
    
    # Verify ROI selection
    x, y, w, h = map(int, roi)
    if w == 0 or h == 0:
        print("Error: Invalid selection. Please select a region by clicking and dragging.")
        cv2.destroyAllWindows()
        return
    
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if key == ord('q'):
        return
    elif key == ord('c'):
        total = len(image_files)
        print(f"\nProcessing {total} images...")
        print(f"Using ROI coordinates: x={x}, y={y}, w={w}, h={h}")
        
        for idx, img_path in enumerate(image_files, 1):
            try:
                # Read image using PIL first
                with Image.open(img_path) as pil_img:
                    # Convert to RGB if needed
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    
                    # Verify ROI is within image bounds
                    img_w, img_h = pil_img.size
                    if x + w > img_w or y + h > img_h:
                        print(f"Warning: ROI dimensions exceed image size for {img_path} - skipping")
                        continue
                    
                    # Crop using PIL
                    cropped_pil = pil_img.crop((x, y, x+w, y+h))
                    
                    # Verify crop size
                    if cropped_pil.size[0] == 0 or cropped_pil.size[1] == 0:
                        print(f"Warning: Invalid crop dimensions for {img_path} - skipping")
                        continue
                    
                    # Save the cropped image
                    base_name = os.path.basename(img_path)
                    file_root, _ = os.path.splitext(base_name)
                    output_path = os.path.join(output_folder, f"{file_root}_cropped.jpg")
                    cropped_pil.save(output_path, quality=95)
                    
                    print(f"[{idx}/{total}] Successfully processed: {base_name}")

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

        print("\nProcessing complete!")

if __name__ == "__main__":
    select_roi_and_crop()
