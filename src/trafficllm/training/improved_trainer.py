"""
Improved Training Pipeline for Traffic Classification
Includes:
- Multi-task loss with automatic uncertainty weighting
- Label smoothing
- Cosine annealing scheduler
- Mixed precision training
- Model checkpointing & early stopping
- Comprehensive metrics & logging
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import numpy as np
from tqdm import tqdm

# Optional: Weights & Biases logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.")

# Optional: TorchMetrics for comprehensive metrics
try:
    from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, ConfusionMatrix
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not available. Install with 'pip install torchmetrics' for advanced metrics.")


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    data_dir: str = "data/processed"
    output_dir: str = "outputs"
    num_workers: int = 4

    # Model
    num_classes: int = 5
    attention_type: str = 'cbam'  # 'cbam', 'se', 'enhanced_spatial', 'simple'
    feature_fusion: bool = True
    dropout: float = 0.4
    pretrained: bool = True  # Use pretrained ResNet weights

    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    # Augmentation
    image_size: int = 640
    augmentation_level: str = 'medium'  # 'light', 'medium', 'heavy'

    # Optimization
    use_mixed_precision: bool = True
    gradient_clip: float = 1.0

    # Scheduler
    scheduler_T0: int = 10  # Restart every 10 epochs
    scheduler_T_mult: int = 2
    scheduler_eta_min: float = 1e-6

    # Loss
    label_smoothing: float = 0.1

    # Task weights (initial - will be learned)
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        'classification': 1.0,
        'density': 1.0,
        'congestion': 1.0,
        'flow_rate': 1.0
    })

    # Checkpointing
    save_best: bool = True
    save_last: bool = True
    checkpoint_every: int = 5  # Save every N epochs

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4

    # Logging
    log_interval: int = 10  # Log every N batches
    use_wandb: bool = False
    wandb_project: str = "traffic-classification"
    wandb_entity: Optional[str] = None

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing"""

    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(1)

        # Convert to one-hot
        target = target.unsqueeze(1)
        target_one_hot = torch.zeros_like(pred).scatter_(1, target, 1)

        # Apply label smoothing
        target_smooth = target_one_hot * (1 - self.epsilon) + self.epsilon / n_classes

        # Calculate loss
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(target_smooth * log_prob).sum(dim=1).mean()

        return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with automatic uncertainty weighting
    Reference: https://arxiv.org/abs/1705.07115
    """

    def __init__(self, num_tasks: int = 4, label_smoothing: float = 0.1):
        super().__init__()

        self.num_tasks = num_tasks

        # Learnable log variance for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

        # Loss functions
        self.classification_loss = LabelSmoothingCrossEntropy(epsilon=label_smoothing)
        self.regression_loss = nn.SmoothL1Loss()  # More robust than MSE

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: Dict with keys ['traffic_level', 'density', 'congestion', 'flow_rate']
            targets: Dict with same keys

        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Individual loss values for logging
        """
        # Calculate individual losses
        cls_loss = self.classification_loss(
            outputs['traffic_level'],
            targets['traffic_level']
        )

        density_loss = self.regression_loss(
            outputs['density'],
            targets['density']
        )

        congestion_loss = self.regression_loss(
            outputs['congestion'],
            targets['congestion']
        )

        flow_loss = self.regression_loss(
            outputs['flow_rate'],
            targets['flow_rate']
        )

        # Automatic task weighting using uncertainty
        # Higher uncertainty (larger log_var) -> lower weight for that task
        precision_cls = torch.exp(-self.log_vars[0])
        precision_den = torch.exp(-self.log_vars[1])
        precision_con = torch.exp(-self.log_vars[2])
        precision_flow = torch.exp(-self.log_vars[3])

        # Total loss with regularization term (log_vars)
        total_loss = (
            precision_cls * cls_loss + self.log_vars[0] +
            precision_den * density_loss + self.log_vars[1] +
            precision_con * congestion_loss + self.log_vars[2] +
            precision_flow * flow_loss + self.log_vars[3]
        )

        # For logging
        loss_dict = {
            'total': total_loss.item(),
            'classification': cls_loss.item(),
            'density': density_loss.item(),
            'congestion': congestion_loss.item(),
            'flow_rate': flow_loss.item(),
            'weight_cls': precision_cls.item(),
            'weight_den': precision_den.item(),
            'weight_con': precision_con.item(),
            'weight_flow': precision_flow.item()
        }

        return total_loss, loss_dict


class TrafficMetrics:
    """Comprehensive metrics tracking"""

    def __init__(self, num_classes: int = 5, device: str = 'cpu'):
        self.device = device
        self.num_classes = num_classes

        if TORCHMETRICS_AVAILABLE:
            # Classification metrics
            self.accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(device)
            self.f1 = F1Score(task='multiclass', num_classes=num_classes, average='weighted').to(device)
            self.conf_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)

            # Regression metrics
            self.density_mae = MeanAbsoluteError().to(device)
            self.congestion_mae = MeanAbsoluteError().to(device)
            self.flow_mae = MeanAbsoluteError().to(device)
        else:
            # Fallback to manual computation
            self.accuracy_sum = 0
            self.total_samples = 0
            self.density_error_sum = 0
            self.congestion_error_sum = 0
            self.flow_error_sum = 0

    def update(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """Update metrics with batch results"""
        if TORCHMETRICS_AVAILABLE:
            # Get predictions
            preds = torch.argmax(outputs['traffic_level'], dim=1)

            self.accuracy.update(preds, targets['traffic_level'])
            self.f1.update(preds, targets['traffic_level'])
            self.conf_matrix.update(preds, targets['traffic_level'])

            self.density_mae.update(outputs['density'].squeeze(), targets['density'].squeeze())
            self.congestion_mae.update(outputs['congestion'].squeeze(), targets['congestion'].squeeze())
            self.flow_mae.update(outputs['flow_rate'].squeeze(), targets['flow_rate'].squeeze())
        else:
            # Manual computation
            preds = torch.argmax(outputs['traffic_level'], dim=1)
            self.accuracy_sum += (preds == targets['traffic_level']).sum().item()
            self.total_samples += targets['traffic_level'].size(0)

            self.density_error_sum += torch.abs(outputs['density'] - targets['density']).sum().item()
            self.congestion_error_sum += torch.abs(outputs['congestion'] - targets['congestion']).sum().item()
            self.flow_error_sum += torch.abs(outputs['flow_rate'] - targets['flow_rate']).sum().item()

    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        if TORCHMETRICS_AVAILABLE:
            metrics = {
                'accuracy': self.accuracy.compute().item(),
                'f1_score': self.f1.compute().item(),
                'density_mae': self.density_mae.compute().item(),
                'congestion_mae': self.congestion_mae.compute().item(),
                'flow_rate_mae': self.flow_mae.compute().item()
            }
        else:
            metrics = {
                'accuracy': self.accuracy_sum / max(self.total_samples, 1),
                'f1_score': 0.0,  # Not computed without torchmetrics
                'density_mae': self.density_error_sum / max(self.total_samples, 1),
                'congestion_mae': self.congestion_error_sum / max(self.total_samples, 1),
                'flow_rate_mae': self.flow_error_sum / max(self.total_samples, 1)
            }

        return metrics

    def reset(self):
        """Reset metrics for new epoch"""
        if TORCHMETRICS_AVAILABLE:
            self.accuracy.reset()
            self.f1.reset()
            self.conf_matrix.reset()
            self.density_mae.reset()
            self.congestion_mae.reset()
            self.flow_mae.reset()
        else:
            self.accuracy_sum = 0
            self.total_samples = 0
            self.density_error_sum = 0
            self.congestion_error_sum = 0
            self.flow_error_sum = 0


class EarlyStopping:
    """Early stopping handler"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Returns True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class ImprovedTrainer:
    """Improved training pipeline"""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Create output directories
        self.checkpoint_dir = Path(config.output_dir) / "checkpoints"
        self.log_dir = Path(config.output_dir) / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb if requested
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.__dict__
            )

        # Model (imported here to avoid circular imports)
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from model_preparation.improved_model import ImprovedTrafficNet

        self.model = ImprovedTrafficNet(
            num_classes=config.num_classes,
            attention_type=config.attention_type,
            feature_fusion=config.feature_fusion,
            dropout=config.dropout,
            pretrained=config.pretrained
        ).to(config.device)

        # Loss
        self.criterion = MultiTaskLoss(
            num_tasks=4,
            label_smoothing=config.label_smoothing
        ).to(config.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.scheduler_T0,
            T_mult=config.scheduler_T_mult,
            eta_min=config.scheduler_eta_min
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None

        # Metrics
        self.train_metrics = TrafficMetrics(config.num_classes, config.device)
        self.val_metrics = TrafficMetrics(config.num_classes, config.device)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode='min'
        ) if config.early_stopping else None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()

        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.config.device)
            targets = {
                'traffic_level': batch['traffic_level'].to(self.config.device),
                'density': batch['density'].to(self.config.device),
                'congestion': batch['congestion'].to(self.config.device),
                'flow_rate': batch['flow_rate'].to(self.config.device)
            }

            # Forward pass with mixed precision
            self.optimizer.zero_grad()

            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss, loss_dict = self.criterion(outputs, targets)

                # Backward with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, targets)
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )

                self.optimizer.step()

            # Update metrics
            self.train_metrics.update(outputs, targets)
            epoch_losses.append(loss_dict['total'])

            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'cls': f"{loss_dict['classification']:.4f}",
                    'den': f"{loss_dict['density']:.4f}"
                })

        # Compute epoch metrics
        metrics = self.train_metrics.compute()
        metrics['loss'] = np.mean(epoch_losses)

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        self.val_metrics.reset()

        epoch_losses = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.config.device)
                targets = {
                    'traffic_level': batch['traffic_level'].to(self.config.device),
                    'density': batch['density'].to(self.config.device),
                    'congestion': batch['congestion'].to(self.config.device),
                    'flow_rate': batch['flow_rate'].to(self.config.device)
                }

                if self.config.use_mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss, loss_dict = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss, loss_dict = self.criterion(outputs, targets)

                self.val_metrics.update(outputs, targets)
                epoch_losses.append(loss_dict['total'])

        metrics = self.val_metrics.compute()
        metrics['loss'] = np.mean(epoch_losses)

        return metrics

    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        print(f"\n{'='*80}")
        print(f"Starting Training - {self.config.epochs} epochs")
        print(f"{'='*80}\n")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self.validate(val_loader)
            self.val_history.append(val_metrics)

            # Update scheduler
            self.scheduler.step()

            # Log
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics['accuracy'],
                    'train/f1': train_metrics['f1_score'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/f1': val_metrics['f1_score'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })

            # Save checkpoints
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            if self.config.save_best and is_best:
                self.save_checkpoint(
                    self.checkpoint_dir / f"best_epoch_{epoch}.pth",
                    is_best=True
                )

            if self.config.save_last:
                self.save_checkpoint(self.checkpoint_dir / "last_model.pth")

            if epoch % self.config.checkpoint_every == 0:
                self.save_checkpoint(self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_metrics['loss']):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

        # Save final training history
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history
            }, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # Example usage
    print("Improved Trainer Module - Ready for use!")
    print("\nExample usage:")
    print("""
from improved_trainer import ImprovedTrainer, TrainingConfig
from torch.utils.data import DataLoader

# Configure training
config = TrainingConfig(
    data_dir="data/processed",
    output_dir="outputs",
    epochs=50,
    batch_size=32,
    learning_rate=3e-4,
    attention_type='cbam',
    augmentation_level='medium',
    use_mixed_precision=True,
    use_wandb=True
)

# Create data loaders (you need to implement these)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# Train
trainer = ImprovedTrainer(config)
trainer.train(train_loader, val_loader)
""")
