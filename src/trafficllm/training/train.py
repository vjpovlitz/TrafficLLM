"""
Traffic Classification Trainer
"""

import sys
import os
# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

print(f"Python path includes:")
print(f"- Project root: {project_root}")
print(f"- Current sys.path:")
for path in sys.path:
    print(f"  - {path}")

try:
    from trafficllm.model_preparation.prepare_training import TrafficClassifier
    print("Successfully imported TrafficClassifier")
except ImportError as e:
    print(f"Error importing TrafficClassifier: {e}")
    print("\nChecking directory structure...")
    
    expected_paths = {
        'model_preparation': os.path.join(project_root, 'trafficllm', 'model_preparation'),
        'prepare_training.py': os.path.join(project_root, 'trafficllm', 'model_preparation', 'prepare_training.py'),
        'training': os.path.join(project_root, 'trafficllm', 'training'),
        'data_processing': os.path.join(project_root, 'trafficllm', 'data_processing')
    }
    
    for name, path in expected_paths.items():
        exists = os.path.exists(path)
        print(f"- {name}: {'✓' if exists else '✗'} ({path})")
    
    sys.exit(1)

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime

class TrafficTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model and data
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Multiple loss functions for different tasks
        self.criteria = {
            'traffic_level': nn.CrossEntropyLoss(),
            'density': nn.MSELoss(),
            'congestion': nn.MSELoss(),
            'flow_rate': nn.MSELoss()
        }
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create checkpoint directory
        self.checkpoint_dir = os.path.join('training', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train_one_epoch(self, epoch, num_epochs):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            # Get data
            images = batch['pixel_values'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs.logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss/total,
                'acc': 100.*correct/total
            })
        
        return running_loss/len(self.train_loader), 100.*correct/total
    
    def validate(self):
        """Run validation"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                images = batch['pixel_values'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs.logits, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return val_loss/len(self.val_loader), 100.*correct/total
    
    def train(self, num_epochs=10):
        """Full training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch, num_epochs)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save if best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc)
    
    def save_checkpoint(self, epoch, val_acc):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'traffic_model_epoch_{epoch+1}_acc_{val_acc:.2f}.pth'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")

def main():
    """Main training function"""
    print("Loading classifier and data...")
    classifier = TrafficClassifier()
    
    # Create trainer
    trainer = TrafficTrainer(
        model=classifier.model,
        train_loader=classifier.train_loader,
        val_loader=classifier.val_loader,
        learning_rate=1e-4
    )
    
    # Start training
    trainer.train(num_epochs=10)

if __name__ == "__main__":
    main() 