from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
import numpy as np

def preprocess_batch(batch_items):
    # batch_items is a list of dictionaries
    # Convert each item's pixel_values to tensor
    images = [torch.from_numpy(np.array(item['pixel_values'])) for item in batch_items]
    labels = [item['label'] for item in batch_items]
    
    # Stack into batches
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    return {'pixel_values': images, 'label': labels}

def test_dataset():
    print("Loading dataset...")
    dataset = load_from_disk('processed_singapore_tensors_final')
    print(f"Dataset size: {len(dataset)}")
    
    # Look at raw data structure
    print("\nRaw data structure:")
    first_item = dataset[0]
    print(f"First item keys: {first_item.keys()}")
    print(f"Pixel values type: {type(first_item['pixel_values'])}")
    
    print("\nSetting up dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=preprocess_batch
    )

    # Test first batch
    print("\nTesting dataloader...")
    try:
        for batch in dataloader:
            images = batch['pixel_values']
            labels = batch['label']
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")
            break  # Just test first batch
        print("\nDataloader test successful!")
        
    except Exception as e:
        print(f"\nError in dataloader: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset() 