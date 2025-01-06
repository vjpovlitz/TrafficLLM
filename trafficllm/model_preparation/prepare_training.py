from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import numpy as np
from transformers import ResNetForImageClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
import psutil

class TrafficDataset(Dataset):
    def __init__(self, dataset_path, label):
        """
        Load images directly from disk instead of processing all at once
        """
        self.dataset = load_from_disk(dataset_path)
        self.label = label
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Load and process single image on demand
        item = self.dataset[idx]
        return {
            'pixel_values': torch.tensor(item['pixel_values']),
            'label': self.label
        }

class TrafficClassifier:
    def __init__(self):
        print("Initializing Traffic Classifier...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Get the project root directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load datasets
        self.load_and_prepare_datasets()
        
        # Initialize model
        print("\nInitializing ResNet model...")
        self.model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=2,
            ignore_mismatched_sizes=True
        ).to(self.device)
    
    def process_batch(self, batch):
        """Process a batch of images"""
        try:
            return [np.array(item['pixel_values'], dtype=np.float32) for item in batch]
        except Exception as e:
            print(f"Error processing batch: {e}")
            return []
    
    def load_and_prepare_datasets(self):
        # CUDA Diagnostic
        print("\nCUDA Diagnostic:")
        print(f"CUDA is available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
            
            # Test CUDA with a simple operation
            print("\nTesting CUDA with simple tensor operation...")
            test_tensor = torch.tensor([1., 2., 3.], device='cuda')
            print(f"Test tensor on CUDA: {test_tensor.device}")
            
            # GPU Memory Test
            print("\nGPU Memory Status:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Calculate optimal resources
        total_cores = os.cpu_count()
        num_workers = int(total_cores * 0.75)  # Use 75% of cores (6/8)
        BATCH_SIZE = 2000  # Increased batch size for better GPU utilization
        
        # GPU setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"\nGPU Info:")
            print(f"- Using: {torch.cuda.get_device_name(0)}")
            print(f"- Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
        
        # Define dataset paths
        singapore_path = os.path.join(
            self.project_root,
            'data_processing',
            'processed_singapore_tensors_final'
        )
        bay_bridge_path = os.path.join(
            self.project_root,
            'data_processing',
            'processed_bay_bridge_dataset'
        )
        
        print(f"\nLooking for datasets in:")
        print(f"Singapore: {singapore_path}")
        print(f"Bay Bridge: {bay_bridge_path}")
        
        print("\nCreating datasets...")
        # Create datasets without processing everything at once
        singapore_dataset = TrafficDataset(singapore_path, label=0)
        bay_bridge_dataset = TrafficDataset(bay_bridge_path, label=1)
        
        # Combine dataset lengths for splitting
        total_len = len(singapore_dataset) + len(bay_bridge_dataset)
        singapore_indices = list(range(len(singapore_dataset)))
        bay_bridge_indices = list(range(len(bay_bridge_dataset)))
        
        # Split indices
        train_sg, temp_sg = train_test_split(singapore_indices, test_size=0.3)
        val_sg, test_sg = train_test_split(temp_sg, test_size=0.5)
        
        train_bb, temp_bb = train_test_split(bay_bridge_indices, test_size=0.3)
        val_bb, test_bb = train_test_split(temp_bb, test_size=0.5)
        
        # Create ConcatDataset for each split
        self.train_dataset = ConcatDataset([
            Subset(singapore_dataset, train_sg),
            Subset(bay_bridge_dataset, train_bb)
        ])
        
        self.val_dataset = ConcatDataset([
            Subset(singapore_dataset, val_sg),
            Subset(bay_bridge_dataset, val_bb)
        ])
        
        self.test_dataset = ConcatDataset([
            Subset(singapore_dataset, test_sg),
            Subset(bay_bridge_dataset, test_bb)
        ])
        
        # Create dataloaders with proper num_workers
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=min(os.cpu_count() - 1, 6),  # Leave one core free
            pin_memory=True  # Faster data transfer to GPU
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=min(os.cpu_count() - 1, 6),
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=min(os.cpu_count() - 1, 6),
            pin_memory=True
        )
        
        print(f"\nDataset preparation complete:")
        print(f"Train: {len(self.train_dataset)} images")
        print(f"Validation: {len(self.val_dataset)} images")
        print(f"Test: {len(self.test_dataset)} images")

if __name__ == "__main__":
    classifier = TrafficClassifier()
    print("\nSetup complete! Ready for training.")