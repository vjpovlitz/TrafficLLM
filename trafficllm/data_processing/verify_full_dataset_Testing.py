from datasets import load_from_disk
import torch
import random
import numpy as np
from PIL import Image

def verify_full_dataset():
    print("Loading full processed dataset...")
    dataset = load_from_disk("processed_singapore_full_dataset")
    
    print(f"\nDataset Statistics:")
    print(f"Total images: {len(dataset)}")
    print(f"Features available: {list(dataset.features.keys())}")
    
    # Check if we have raw images instead of processed tensors
    if 'image_url' in dataset.features:
        print("\nWARNING: Dataset contains raw images instead of processed tensors!")
        print("The processing step needs to be rerun with proper tensor saving.")
        
        # Check what we actually have
        sample = dataset[0]
        print("\nSample data structure:")
        for key, value in sample.items():
            print(f"{key}: {type(value)}")
            
        return False
    
    return True

if __name__ == "__main__":
    success = verify_full_dataset()
    if not success:
        print("\nRecommended action: Rerun the processing script with the following modification:")
        print("\nIn firstBayBridgeDataSet.py, modify the prepare_dataset function to ensure it returns:")
        print("return {")
        print("    'pixel_values': processed_images,")
        print("    'label': examples.get('label', [0] * len(processed_images))")
        print("}") 