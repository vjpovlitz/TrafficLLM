from datasets import load_dataset
import torch
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def process_bay_bridge():
    print("Loading Bay Bridge dataset...")
    bay_bridge_dataset = load_dataset("imagefolder", data_dir="processed")
    
    print("\nInitializing processor...")
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    
    def process_batch(examples):
        try:
            if 'image' in examples:
                raw_images = examples['image']
            else:
                raise KeyError("No image key found")
            
            if not isinstance(raw_images, list):
                raw_images = [raw_images]
            
            processed_images = []
            successful = 0
            failed = 0
            
            for image in raw_images:
                try:
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    inputs = processor(image, return_tensors="pt")
                    tensor = inputs['pixel_values'][0].numpy()
                    processed_images.append(tensor)
                    successful += 1
                    
                except Exception as e:
                    failed += 1
                    continue
            
            if successful > 0:
                return {
                    "pixel_values": processed_images,
                    "label": [0] * len(processed_images)  # Placeholder labels
                }
            return None
            
        except Exception as e:
            print(f"\nBatch failed: {str(e)}")
            return None
    
    print("\nProcessing Bay Bridge images...")
    processed_dataset = bay_bridge_dataset['train'].map(
        process_batch,
        batched=True,
        batch_size=32,
        remove_columns=bay_bridge_dataset['train'].column_names
    )
    
    print("\nSaving processed dataset...")
    processed_dataset.save_to_disk("processed_bay_bridge_dataset")
    
    # Verify the saved dataset
    print("\nVerifying saved dataset...")
    from datasets import load_from_disk
    verify_dataset = load_from_disk("processed_bay_bridge_dataset")
    print(f"Verified dataset size: {len(verify_dataset)} images")
    
    return verify_dataset

if __name__ == "__main__":
    process_bay_bridge() 