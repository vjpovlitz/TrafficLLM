import logging
import os
import torch
import numpy as np
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from trafficllm.data_processing.traffic_analysis import TrafficImageAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CustomTrafficDataset(Dataset):
    """
    Custom dataset for multi-parameter traffic analysis
    
    Features:
    - Handles multiple traffic parameters
    - Supports smaller dataset for testing
    - Includes data augmentation
    """
    def __init__(self, data_path, subset_size=None):
        logger.info(f"Initializing CustomTrafficDataset from {data_path}")
        
        # Load original dataset
        self.data = load_from_disk(data_path)
        logger.info(f"Full dataset size: {len(self.data)}")
        
        # Create subset if specified
        if subset_size:
            indices = np.random.choice(
                len(self.data), 
                size=min(subset_size, len(self.data)), 
                replace=False
            )
            self.data = self.data.select(indices)
            logger.info(f"Created subset with {len(self.data)} samples")
        
        self.analyzer = TrafficImageAnalyzer()
        self.estimate_traffic_parameters()
    
    def estimate_traffic_parameters(self):
        """Estimate traffic parameters with error handling"""
        logger.info("Estimating traffic parameters...")
        self.parameters = []
        
        for idx in tqdm(range(len(self.data)), desc="Processing images"):
            try:
                # Get image data
                image_data = self.data[idx]['pixel_values']
                
                # Analyze traffic
                analysis = self.analyzer.analyze_traffic(image_data)
                
                # Ensure all required keys exist with default values
                self.parameters.append({
                    'density': analysis.get('density', 0.0),
                    'congestion': analysis.get('congestion', 0.0),
                    'traffic_level': analysis.get('traffic_level', 0),
                    'flow_rate': analysis.get('edge_density', 0.0),  # Use edge_density as flow_rate
                    'valid': analysis.get('valid', False)
                })
                
            except Exception as e:
                logger.error(f"Error processing image {idx}: {str(e)}")
                # Provide default values if analysis fails
                self.parameters.append({
                    'density': 0.0,
                    'congestion': 0.0,
                    'traffic_level': 0,
                    'flow_rate': 0.0,
                    'valid': False
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample with all parameters"""
        try:
            # Get image data correctly
            image = np.array(self.data[idx]['pixel_values'])
            image_tensor = torch.tensor(image, dtype=torch.float32)
            
            # Get parameters
            params = self.parameters[idx]
            
            return {
                'image': image_tensor,
                'density': torch.tensor(params['density'], dtype=torch.float32),
                'congestion': torch.tensor(params['congestion'], dtype=torch.float32),
                'flow_rate': torch.tensor(params['flow_rate'], dtype=torch.float32),
                'traffic_level': torch.tensor(params['traffic_level'], dtype=torch.long)
            }
        except Exception as e:
            logger.error(f"Error accessing item at index {idx}")
            logger.error(f"Data structure: {self.data[idx]}")
            raise

def prepare_data(subset_size=1000):
    """
    Prepare a small dataset for testing the custom model
    """
    logger.info(f"Preparing data with subset size: {subset_size}")
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Define possible dataset locations (in order of preference)
    possible_locations = [
        os.path.join(project_root, 'data', 'processed', 'processed_singapore_tensors_final'),
        os.path.join(project_root, 'processed_singapore_tensors_final'),
        os.path.join(project_root, 'Traffic LLM Construction', 'data_processing', 'processed_singapore_tensors_final')
    ]
    
    # Try to find dataset
    data_path = None
    for loc in possible_locations:
        if os.path.exists(loc):
            data_path = loc
            logger.info(f"Found dataset at: {data_path}")
            break
    
    if data_path is None:
        logger.error("Dataset not found in any of the expected locations:")
        for loc in possible_locations:
            logger.error(f"- {loc}")
        logger.error("\nTo fix this:")
        logger.error("1. Create the directory: data/processed/")
        logger.error("2. Move your dataset to: data/processed/processed_singapore_tensors_final")
        logger.error("   OR")
        logger.error("3. Run the data processing scripts to generate the dataset")
        raise FileNotFoundError("Dataset not found. See logs for details and solutions.")
    
    # Create dataset with subset
    dataset = CustomTrafficDataset(data_path, subset_size=subset_size)
    
    # Split sizes
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    logger.info(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=min(os.cpu_count() - 1, 6)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=min(os.cpu_count() - 1, 6)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=min(os.cpu_count() - 1, 6)
    )
    
    logger.info("Data preparation complete")
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test data preparation with small subset
    train_loader, val_loader, test_loader = prepare_data(subset_size=1000)
    logger.info("Data preparation test successful") 