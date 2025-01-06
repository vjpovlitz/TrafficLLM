#!/usr/bin/env python3
"""
Utility script to set up data directories and move datasets to the correct locations.
"""

import os
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary data directories"""
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Directories to create
    directories = [
        project_root / 'data' / 'raw',
        project_root / 'data' / 'processed',
        project_root / 'trafficllm' / 'data_visualization' / 'outputs',
        project_root / 'training' / 'checkpoints'
    ]
    
    # Create directories
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def move_datasets():
    """Move datasets to correct locations"""
    project_root = Path(__file__).parent.parent
    
    # Define dataset locations
    dataset_moves = [
        {
            'from': project_root / 'processed_singapore_tensors_final',
            'to': project_root / 'data' / 'processed' / 'processed_singapore_tensors_final'
        },
        {
            'from': project_root / 'Traffic LLM Construction' / 'data_processing' / 'processed_singapore_tensors_final',
            'to': project_root / 'data' / 'processed' / 'processed_singapore_tensors_final'
        },
        {
            'from': project_root / 'processed_bay_bridge_dataset',
            'to': project_root / 'data' / 'processed' / 'processed_bay_bridge_dataset'
        }
    ]
    
    # Move datasets if they exist
    for move in dataset_moves:
        if move['from'].exists():
            if move['to'].exists():
                logger.warning(f"Destination already exists: {move['to']}")
                continue
                
            logger.info(f"Moving {move['from']} to {move['to']}")
            try:
                shutil.move(str(move['from']), str(move['to']))
                logger.info("Move successful")
            except Exception as e:
                logger.error(f"Error moving dataset: {e}")

def move_screenshots():
    """Move screenshots to raw data directory"""
    project_root = Path(__file__).parent.parent
    screenshots_dir = project_root / 'screenshots'
    raw_dir = project_root / 'data' / 'raw' / 'screenshots'
    
    if screenshots_dir.exists():
        if raw_dir.exists():
            logger.warning(f"Screenshots directory already exists in raw data: {raw_dir}")
            return
            
        logger.info(f"Moving screenshots to {raw_dir}")
        try:
            shutil.move(str(screenshots_dir), str(raw_dir))
            logger.info("Move successful")
        except Exception as e:
            logger.error(f"Error moving screenshots: {e}")

def main():
    """Main function to set up data structure"""
    logger.info("Setting up data directories...")
    setup_directories()
    
    logger.info("\nMoving datasets to correct locations...")
    move_datasets()
    
    logger.info("\nMoving screenshots to raw data...")
    move_screenshots()
    
    logger.info("\nSetup complete!")
    logger.info("\nNext steps:")
    logger.info("1. Check that your datasets are in data/processed/")
    logger.info("2. Check that your raw data is in data/raw/")
    logger.info("3. Run your processing scripts if needed to regenerate datasets")

if __name__ == "__main__":
    main() 