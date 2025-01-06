from datasets import load_from_disk, Dataset
import torch
from tqdm import tqdm
import numpy as np
import os

def convert_dataset():
    print("Loading original processed dataset...")
    original_dataset = load_from_disk("processed_singapore_full_dataset")
    print(f"Found {len(original_dataset)} images")

    print("\nConverting to tensor format...")
    
    def process_batch(examples, idx_start, batch_size):
        processed_images = []
        
        for i in range(batch_size):
            if idx_start + i >= len(examples):
                break
                
            try:
                image = examples[idx_start + i]['image_url']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Process image to tensor
                inputs = processor(image, return_tensors="pt")
                tensor = inputs['pixel_values'][0]
                processed_images.append(tensor.numpy())
                
            except Exception as e:
                print(f"\nError processing image {idx_start + i}: {str(e)}")
                continue
        
        return processed_images

    # Initialize processor
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    # Process in smaller chunks
    CHUNK_SIZE = 5000  # Process 5000 images at a time
    total_chunks = (len(original_dataset) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    print(f"\nProcessing in {total_chunks} chunks of {CHUNK_SIZE} images each")
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, len(original_dataset))
        
        print(f"\nProcessing chunk {chunk_idx + 1}/{total_chunks}")
        print(f"Images {start_idx} to {end_idx}")
        
        # Process this chunk
        batch_size = 100
        chunk_images = []
        
        for idx in tqdm(range(start_idx, end_idx, batch_size)):
            batch_end = min(idx + batch_size, end_idx)
            batch_images = process_batch(original_dataset, idx, batch_end - idx)
            chunk_images.extend(batch_images)
        
        # Save this chunk
        chunk_dataset = Dataset.from_dict({
            "pixel_values": chunk_images,
            "label": [0] * len(chunk_images)
        })
        
        chunk_path = f"processed_singapore_tensors_chunk_{chunk_idx}"
        print(f"Saving chunk to {chunk_path}")
        chunk_dataset.save_to_disk(chunk_path)
        
        print(f"Chunk {chunk_idx + 1} saved successfully")
        print(f"Chunk size: {len(chunk_images)} images")
    
    print("\nAll chunks processed and saved!")
    print("\nTo combine chunks, use the combine_chunks.py script")

if __name__ == "__main__":
    convert_dataset() 