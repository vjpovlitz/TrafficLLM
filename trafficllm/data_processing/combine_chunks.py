from datasets import load_from_disk, concatenate_datasets
import glob
import os

def combine_chunks():
    print("Finding all chunk directories...")
    chunk_paths = sorted(glob.glob("processed_singapore_tensors_chunk_*"))
    
    if not chunk_paths:
        print("No chunks found!")
        return
    
    print(f"Found {len(chunk_paths)} chunks")
    
    # Load all chunks
    datasets = []
    for path in chunk_paths:
        print(f"Loading {path}...")
        dataset = load_from_disk(path)
        datasets.append(dataset)
    
    # Combine all chunks
    print("\nCombining chunks...")
    combined_dataset = concatenate_datasets(datasets)
    
    # Save combined dataset
    print("\nSaving combined dataset...")
    combined_dataset.save_to_disk("processed_singapore_tensors_final")
    
    print(f"\nFinal dataset saved with {len(combined_dataset)} images")
    
    # Cleanup chunk files
    if input("\nDelete chunk files? (y/n): ").lower() == 'y':
        for path in chunk_paths:
            os.system(f"rm -rf {path}")
        print("Chunks deleted")

if __name__ == "__main__":
    combine_chunks() 