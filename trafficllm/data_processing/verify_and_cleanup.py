from datasets import load_from_disk
import torch
import os
import shutil
import numpy as np

def verify_and_cleanup():
    # Verify the final dataset
    print("Loading combined dataset...")
    try:
        dataset = load_from_disk("processed_singapore_tensors_final")
        print(f"\nDataset Statistics:")
        print(f"Total images: {len(dataset)}")
        print(f"Features: {list(dataset.features.keys())}")
        
        # Check first few samples
        print("\nChecking first sample:")
        first_sample = dataset[0]
        
        # Convert list to numpy array then to tensor
        if isinstance(first_sample['pixel_values'], list):
            pixel_values = np.array(first_sample['pixel_values'])
            tensor = torch.from_numpy(pixel_values)
            print(f"Tensor shape: {tensor.shape}")
            print(f"Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        
        # Clean up chunks
        print("\nCleaning up chunk directories...")
        chunk_dirs = [d for d in os.listdir() if d.startswith("processed_singapore_tensors_chunk_")]
        for chunk_dir in chunk_dirs:
            print(f"Removing {chunk_dir}...")
            shutil.rmtree(chunk_dir)
        
        print("\nVerification complete!")
        print("\nTo use this dataset in your training code:")
        print("""
from datasets import load_from_disk
import torch
import numpy as np

# Load dataset
dataset = load_from_disk('processed_singapore_tensors_final')

# Convert to tensors when loading
def preprocess_batch(batch):
    # Convert lists to tensors
    images = [torch.from_numpy(np.array(x)) for x in batch['pixel_values']]
    images = torch.stack(images)
    labels = torch.tensor(batch['label'])
    return {'pixel_values': images, 'label': labels}

# Use with DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=preprocess_batch
)
""")
        
    except Exception as e:
        print(f"Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_and_cleanup() 