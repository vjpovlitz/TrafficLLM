import sys
import os
# Safe OpenMP configuration
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads

# Add GPU memory management
import torch
if torch.cuda.is_available():
    # Start with empty cache
    torch.cuda.empty_cache()
    # Set memory allocation limits
    torch.cuda.set_per_process_memory_fraction(0.7)  # Use up to 70% of GPU memory

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import matplotlib.pyplot as plt
import numpy as np
from trafficllm.data_processing.prepare_custom_data import prepare_data
import logging
from datetime import datetime
import json
import shutil
from trafficllm.refinement.interactive_refinement import InteractiveRefinement
import time
from colorama import init, Fore, Style
from trafficllm.refinement.feedback_learner import FeedbackLearner
from trafficllm.refinement.apply_feedback import apply_feedback_learning

# Add resource monitoring
import psutil
def monitor_resources():
    """Monitor system resources"""
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
        logger.info(f"Resource Usage - CPU: {cpu_percent}%, RAM: {memory_percent}%, GPU Memory: {gpu_memory:.1f}%")
    else:
        logger.info(f"Resource Usage - CPU: {cpu_percent}%, RAM: {memory_percent}%")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_dataset(train_loader=None):
    """Unified visualization function to avoid double processing"""
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(project_root, 'data_visualization', 'outputs', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Creating visualizations in: {output_dir}")
    
    # Get data only if not provided
    if train_loader is None:
        train_loader, _, _ = prepare_data(subset_size=1000)
    
    # Collect all parameters in one pass
    distributions = {
        'densities': [],
        'congestions': [],
        'flow_rates': [],
        'traffic_levels': []
    }
    
    # Single pass through data
    for batch in train_loader:
        distributions['densities'].extend(batch['density'].numpy().tolist())
        distributions['congestions'].extend(batch['congestion'].numpy().tolist())
        distributions['flow_rates'].extend(batch['flow_rate'].numpy().tolist())
        distributions['traffic_levels'].extend(batch['traffic_level'].numpy().tolist())
    
    # Convert lists to numpy arrays for processing
    for key in distributions:
        distributions[key] = np.array(distributions[key])
    
    # Create visualizations
    create_sample_visualization(train_loader, output_dir)
    create_distribution_plots(distributions, output_dir, timestamp)
    
    return distributions, output_dir

def create_sample_visualization(train_loader, output_dir):
    """Create sample image visualization"""
    batch = next(iter(train_loader))
    
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    fig.suptitle('Traffic Analysis Samples', fontsize=16)
    
    for i in range(min(12, len(batch['image']))):
        row, col = i // 4, i % 4
        
        # Normalize image properly
        image = batch['image'][i].numpy().transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())
        
        axes[row, col].imshow(image)
        axes[row, col].axis('off')
        
        title = f"Level: {batch['traffic_level'][i]}\n" \
                f"Density: {batch['density'][i]:.2f}\n" \
                f"Cong: {batch['congestion'][i]:.2f}"
        axes[row, col].set_title(title, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_images.png'))
    plt.close()

def create_distribution_plots(distributions, output_dir, timestamp):
    """Create distribution plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Parameter Distributions (Run: {timestamp})', fontsize=16)
    
    # Traffic Density
    axes[0, 0].hist(distributions['densities'], bins=30, color='blue', alpha=0.7)
    axes[0, 0].set_title('Traffic Density')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Congestion Levels
    axes[0, 1].hist(distributions['congestions'], bins=30, color='red', alpha=0.7)
    axes[0, 1].set_title('Congestion Levels')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Flow Rates
    axes[1, 0].hist(distributions['flow_rates'], bins=30, color='green', alpha=0.7)
    axes[1, 0].set_title('Flow Rates')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Traffic Levels
    traffic_counts = np.bincount(distributions['traffic_levels'].astype(int))
    axes[1, 1].bar(range(len(traffic_counts)), traffic_counts, color='purple', alpha=0.7)
    axes[1, 1].set_title('Traffic Levels')
    axes[1, 1].set_xticks(range(5))
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_distributions.png'))
    plt.close()

def visualize_dataset_with_feedback(train_loader=None, subset_size=500):
    """Visualize dataset and collect user feedback"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(project_root, 'data_visualization', 'outputs', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Creating visualizations in: {output_dir}")
    
    if train_loader is None:
        train_loader, _, _ = prepare_data(subset_size=subset_size)  # Reduced subset size
    
    # Get first batch
    batch = next(iter(train_loader))
    
    # Initialize refinement system
    refiner = InteractiveRefinement(save_dir=output_dir)
    feedback_data = {
        'timestamp': timestamp,
        'feedback': []
    }
    
    # Create and display figure
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    fig.suptitle('Traffic Analysis Samples', fontsize=16)
    
    # First, plot all images
    images_data = []
    for i in range(min(12, len(batch['image']))):
        row = i // 4
        col = i % 4
        
        # Get image and parameters
        image = batch['image'][i].numpy().transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())
        
        density = batch['density'][i].item()
        congestion = batch['congestion'][i].item()
        traffic_level = batch['traffic_level'][i].item()
        
        # Plot image
        axes[row, col].imshow(image)
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Initial Level: {traffic_level}\nDensity: {density:.2f}\nCong: {congestion:.2f}', 
                               fontsize=8)
        
        # Store data for feedback phase
        images_data.append({
            'image': image,
            'density': density,
            'congestion': congestion,
            'traffic_level': traffic_level
        })
    
    plt.tight_layout()
    
    # Show the plot
    plt.show(block=False)  # Don't block execution
    plt.pause(1)  # Give time for the window to appear
    
    session_start = time.time()
    i = 0
    while i < min(12, len(batch['image'])):
        # Get user feedback
        correct_level, is_correct, action = refiner.get_user_feedback(
            i, 
            images_data[i]['traffic_level'],
            total_images=min(12, len(batch['image']))
        )
        
        if action == 'quit':
            print(f"\n{Fore.YELLOW}Session ended by user{Style.RESET_ALL}")
            break
        elif action == 'back':
            if i > 0:
                i -= 1
                # Remove last feedback if it exists
                if feedback_data['feedback'] and i < len(feedback_data['feedback']):
                    feedback_data['feedback'].pop()
            continue
        
        # Update plot with feedback
        row = i // 4
        col = i % 4
        
        # Color code title based on agreement
        title_color = 'green' if is_correct else 'red'
        
        # Update title with feedback
        title = f'Model: {images_data[i]["traffic_level"]} | Actual: {correct_level}\n'
        title += f'Density: {images_data[i]["density"]:.2f}\nCong: {images_data[i]["congestion"]:.2f}'
        axes[row, col].set_title(title, fontsize=8, color=title_color)
        
        # Store feedback
        feedback_data['feedback'].append({
            'image_index': i,
            'model_prediction': float(images_data[i]['traffic_level']),
            'user_correction': float(correct_level),
            'was_correct': is_correct,
            'density': float(images_data[i]['density']),
            'congestion': float(images_data[i]['congestion'])
        })
        
        # Update the plot
        plt.draw()
        plt.pause(0.1)
        
        i += 1
    
    # Add session duration
    feedback_data['session_duration'] = time.time() - session_start
    
    # Save final visualization with feedback
    output_path = os.path.join(output_dir, 'feedback_visualization.png')
    plt.savefig(output_path)
    logger.info(f"Visualization saved as '{output_path}'")
    
    # Save feedback data
    refiner.save_feedback(feedback_data)
    
    # After collecting feedback and saving:
    print(f"\n{Fore.CYAN}Feedback collection complete. Applying learning...{Style.RESET_ALL}")
    
    try:
        # Calculate accuracy from this session before applying new learning
        correct_predictions = sum(1 for f in feedback_data['feedback'] if f['was_correct'])
        total_predictions = len(feedback_data['feedback'])
        current_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Add accuracy to metadata
        feedback_data['metadata'] = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'total_images': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy_before': current_accuracy,
            'session_duration': feedback_data.get('session_duration', 0)
        }
        
        # Apply feedback learning
        updated_analyzer = apply_feedback_learning()
        
        print(f"\n{Fore.GREEN}Successfully applied feedback learning!{Style.RESET_ALL}")
        print("\nUpdated parameters:")
        for param, value in updated_analyzer.get_parameters().items():
            print(f"{param}: {value}")
            
        # Show improvement metrics
        print("\nImprovement Metrics:")
        print(f"Session accuracy: {current_accuracy:.2%}")
        print(f"Total images reviewed: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        
    except Exception as e:
        print(f"\n{Fore.RED}Error applying feedback: {str(e)}{Style.RESET_ALL}")
        logger.error(f"Error in feedback application: {str(e)}", exc_info=True)
    
    return feedback_data, updated_analyzer

if __name__ == "__main__":
    init()  # Initialize colorama
    logger.info("Starting visualization with feedback collection...")
    feedback, analyzer = visualize_dataset_with_feedback(subset_size=500)