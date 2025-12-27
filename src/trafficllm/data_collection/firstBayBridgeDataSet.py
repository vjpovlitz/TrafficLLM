from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import requests
from PIL import Image
from io import BytesIO
import multiprocessing
import time
import os

# Add freeze_support for Windows
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # Optimize for i7-9700K and RTX 2080 Super
    NUM_CORES = 6  # Leave 2 cores free for system tasks
    BATCH_SIZE = NUM_CORES * 12  # 12 images per core (72 total per batch)
    USE_GPU = torch.cuda.is_available()
    
    print(f"Hardware configuration:")
    print(f"- CPU: i7-9700K (8 cores)")
    print(f"- Using {NUM_CORES} cores for processing")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- GPU available: {USE_GPU}")
    if USE_GPU:
        print(f"- GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load the datasets
    print("Loading datasets...")
    traffic_dataset = load_dataset("Sayali9141/traffic_signal_images", trust_remote_code=True)
    bay_bridge_dataset = load_dataset("imagefolder", data_dir="processed")

    # Function to prepare dataset
    def prepare_dataset(examples):
        try:
            if 'image_url' in examples:
                raw_images = examples['image_url']
            elif 'image' in examples:
                raw_images = examples['image']
            else:
                raise KeyError("No image or image_url key found")
            
            if not isinstance(raw_images, list):
                raw_images = [raw_images]
            
            processed_images = []
            successful = 0
            failed = 0
            
            # Process in chunks
            chunk_size = 24
            for i in range(0, len(raw_images), chunk_size):
                chunk = raw_images[i:i + chunk_size]
                
                for image in chunk:
                    try:
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        inputs = processor(image, return_tensors="pt")
                        if USE_GPU:
                            inputs = {k: v.cuda() for k, v in inputs.items()}
                            tensor = inputs['pixel_values'][0].cpu()
                        else:
                            tensor = inputs['pixel_values'][0]
                        
                        # Convert tensor to numpy for storage
                        processed_images.append(tensor.numpy())
                        successful += 1
                        
                    except Exception as e:
                        failed += 1
                        continue
                
                if (successful + failed) % 200 == 0:
                    print(f"\rProcessed: {successful} successful, {failed} failed", end="")
            
            if successful > 0:
                # Return only processed tensors and labels
                return {
                    "pixel_values": processed_images,
                    "label": [0] * len(processed_images)  # Placeholder labels
                }
            return None
            
        except Exception as e:
            print(f"\nBatch failed: {str(e)}")
            return None
            
    # Initialize the model
    print("\nInitializing model...")
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    # Process datasets with error tracking
    def process_dataset(dataset, name):
        start_time = time.time()
        print(f"\nProcessing {name} dataset...")
        total_images = len(dataset['train'])
        
        try:
            processed = dataset['train'].map(
                prepare_dataset,
                batched=True,
                batch_size=BATCH_SIZE,
                remove_columns=dataset['train'].column_names,
                num_proc=NUM_CORES,
                load_from_cache_file=False
            )
            
            # Final statistics
            elapsed_time = time.time() - start_time
            success_rate = len(processed)/total_images * 100
            print(f"\n{name} processing complete:")
            print(f"- Total attempted: {total_images}")
            print(f"- Successfully processed: {len(processed)}")
            print(f"- Success rate: {success_rate:.1f}%")
            print(f"- Time taken: {elapsed_time/3600:.2f} hours")
            print(f"- Speed: {len(processed)/elapsed_time:.2f} images/second")
            return processed
            
        except Exception as e:
            print(f"\nError processing {name} dataset: {str(e)}")
            return None

    # Process full Singapore traffic dataset
    print("\nProcessing full Singapore traffic dataset...")
    processed_traffic = process_dataset(traffic_dataset, "Singapore traffic")

    # Save the processed dataset with progress tracking
    if processed_traffic:
        print("\nSaving processed dataset...")
        try:
            save_path = "processed_singapore_full_dataset"
            processed_traffic.save_to_disk(save_path)
            print(f"Full dataset successfully saved to {save_path}")
            print(f"Total images in saved dataset: {len(processed_traffic)}")
        except Exception as e:
            print(f"Error saving dataset: {str(e)}")
    else:
        print("\nProcessing failed")

