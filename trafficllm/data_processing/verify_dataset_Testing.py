from datasets import load_from_disk
import torch
import matplotlib.pyplot as plt
import random
from torchvision.transforms import ToPILImage
import os
import numpy as np

def verify_processed_dataset():
    print("Checking for dataset directory...")
    if not os.path.exists("processed_singapore_test_dataset"):
        print("Dataset directory not found!")
        return False

    print("Loading processed test dataset...")
    try:
        dataset = load_from_disk("processed_singapore_test_dataset")
        print(f"\nDataset size: {len(dataset)} images")
        
        def show_tensor_image(pixel_values, title):
            try:
                # Debug the structure
                print(f"\nDebug for {title}:")
                print(f"Type: {type(pixel_values)}")
                if isinstance(pixel_values, list):
                    print(f"List structure: {len(pixel_values)} x {len(pixel_values[0]) if pixel_values else 'empty'}")
                    # Convert nested lists to numpy array first
                    array = np.array(pixel_values)
                    print(f"Array shape after conversion: {array.shape}")
                    
                    # Reshape to proper image dimensions
                    if len(array.shape) > 3:
                        array = array[0, 0]  # Take first image if batched
                    
                    # Convert to tensor
                    tensor = torch.from_numpy(array)
                    if tensor.shape[0] != 3:  # If channels are not first
                        tensor = tensor.permute(2, 0, 1)  # Rearrange to [C, H, W]
                    
                    print(f"Final tensor shape: {tensor.shape}")
                    
                    # Denormalize
                    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                    img_tensor = tensor * std + mean
                    
                    # Ensure proper range
                    img_tensor = img_tensor.clamp(0, 1)
                    
                    # Convert to PIL Image and display
                    img = ToPILImage()(img_tensor)
                    
                    plt.figure(figsize=(10, 8))
                    plt.imshow(img)
                    plt.title(title)
                    plt.axis('on')
                    plt.show()
                    plt.close()  # Explicitly close the figure
                    
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Display 5 random images
        print("\nDisplaying 5 random samples...")
        indices = random.sample(range(len(dataset)), 5)
        
        for idx in indices:
            sample = dataset[idx]
            pixel_values = sample['pixel_values']
            print(f"\nSample {idx}:")
            show_tensor_image(pixel_values, f"Processed Image {idx}")
            
        return True
        
    except Exception as e:
        print(f"Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_processed_dataset()
    if success:
        print("\nVerification complete! Dataset appears to be properly processed.")
    else:
        print("\nVerification failed. Please check the errors above.")