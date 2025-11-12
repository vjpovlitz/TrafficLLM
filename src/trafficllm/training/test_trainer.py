"""
Test script for the improved training pipeline
Creates dummy data and runs a quick training test
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from improved_trainer import ImprovedTrainer, TrainingConfig


class DummyTrafficDataset(Dataset):
    """Dummy dataset for testing"""

    def __init__(self, num_samples=100, image_size=640):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create random image
        image = torch.randn(3, self.image_size, self.image_size)

        # Create random targets
        traffic_level = torch.randint(0, 5, (1,)).item()
        density = torch.rand(1)
        congestion = torch.rand(1)
        flow_rate = torch.rand(1)

        return {
            'image': image,
            'traffic_level': traffic_level,
            'density': density,
            'congestion': congestion,
            'flow_rate': flow_rate
        }


def test_trainer_initialization():
    """Test 1: Trainer initialization"""
    print("\n" + "="*80)
    print("TEST 1: Trainer Initialization")
    print("="*80)

    config = TrainingConfig(
        data_dir="data/processed",
        output_dir="outputs/test",
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        attention_type='cbam',
        pretrained=False,  # Don't download weights for testing
        use_mixed_precision=True,
        use_wandb=False,
        early_stopping=False
    )

    try:
        trainer = ImprovedTrainer(config)
        print("✅ Trainer initialized successfully")
        print(f"   Model device: {next(trainer.model.parameters()).device}")
        print(f"   Mixed precision: {config.use_mixed_precision}")
        return trainer, config
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        raise


def test_forward_pass(trainer):
    """Test 2: Forward pass"""
    print("\n" + "="*80)
    print("TEST 2: Forward Pass")
    print("="*80)

    try:
        # Create dummy batch
        batch_size = 4
        images = torch.randn(batch_size, 3, 640, 640).to(trainer.config.device)

        trainer.model.eval()
        with torch.no_grad():
            outputs = trainer.model(images)

        print("✅ Forward pass successful")
        print(f"   Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            print(f"   {key}: {value.shape}")

        # Verify output shapes
        assert outputs['traffic_level'].shape == (batch_size, 5)
        assert outputs['density'].shape == (batch_size, 1)
        assert outputs['congestion'].shape == (batch_size, 1)
        assert outputs['flow_rate'].shape == (batch_size, 1)
        print("✅ Output shapes verified")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise


def test_loss_calculation(trainer):
    """Test 3: Loss calculation"""
    print("\n" + "="*80)
    print("TEST 3: Loss Calculation")
    print("="*80)

    try:
        batch_size = 4
        outputs = {
            'traffic_level': torch.randn(batch_size, 5).to(trainer.config.device),
            'density': torch.randn(batch_size, 1).to(trainer.config.device),
            'congestion': torch.randn(batch_size, 1).to(trainer.config.device),
            'flow_rate': torch.randn(batch_size, 1).to(trainer.config.device)
        }

        targets = {
            'traffic_level': torch.randint(0, 5, (batch_size,)).to(trainer.config.device),
            'density': torch.randn(batch_size, 1).to(trainer.config.device),
            'congestion': torch.randn(batch_size, 1).to(trainer.config.device),
            'flow_rate': torch.randn(batch_size, 1).to(trainer.config.device)
        }

        loss, loss_dict = trainer.criterion(outputs, targets)

        print("✅ Loss calculation successful")
        print(f"   Loss dict keys: {list(loss_dict.keys())}")
        for key, value in loss_dict.items():
            print(f"   {key}: {value:.4f}")

        # Verify loss values are reasonable
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        print("✅ Loss values verified (no NaN/Inf)")

    except Exception as e:
        print(f"❌ Loss calculation failed: {e}")
        raise


def test_training_step(trainer, train_loader):
    """Test 4: Single training step"""
    print("\n" + "="*80)
    print("TEST 4: Single Training Step")
    print("="*80)

    try:
        trainer.model.train()
        batch = next(iter(train_loader))

        images = batch['image'].to(trainer.config.device)
        targets = {
            'traffic_level': batch['traffic_level'].to(trainer.config.device),
            'density': batch['density'].to(trainer.config.device),
            'congestion': batch['congestion'].to(trainer.config.device),
            'flow_rate': batch['flow_rate'].to(trainer.config.device)
        }

        trainer.optimizer.zero_grad()

        # Forward
        outputs = trainer.model(images)
        loss, loss_dict = trainer.criterion(outputs, targets)

        # Backward
        loss.backward()
        trainer.optimizer.step()

        print("✅ Training step successful")
        print(f"   Batch size: {images.shape[0]}")
        print(f"   Loss: {loss.item():.4f}")

        # Check gradients
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in trainer.model.parameters()
        )
        assert has_gradients, "No gradients computed!"
        print("✅ Gradients computed successfully")

    except Exception as e:
        print(f"❌ Training step failed: {e}")
        raise


def test_full_epoch(trainer, train_loader, val_loader):
    """Test 5: Full epoch"""
    print("\n" + "="*80)
    print("TEST 5: Full Training Epoch")
    print("="*80)

    try:
        trainer.current_epoch = 0

        # Train epoch
        train_metrics = trainer.train_epoch(train_loader)
        print("\n✅ Training epoch completed")
        print(f"   Train metrics: {train_metrics}")

        # Validation epoch
        val_metrics = trainer.validate(val_loader)
        print("✅ Validation epoch completed")
        print(f"   Val metrics: {val_metrics}")

        # Verify metrics have expected keys
        expected_keys = ['loss', 'accuracy', 'f1_score', 'density_mae',
                        'congestion_mae', 'flow_rate_mae']
        for key in expected_keys:
            assert key in train_metrics, f"Missing key: {key}"
            assert key in val_metrics, f"Missing key: {key}"

        print("✅ Metrics structure verified")

    except Exception as e:
        print(f"❌ Full epoch failed: {e}")
        raise


def test_checkpointing(trainer):
    """Test 6: Model checkpointing"""
    print("\n" + "="*80)
    print("TEST 6: Model Checkpointing")
    print("="*80)

    try:
        checkpoint_path = trainer.checkpoint_dir / "test_checkpoint.pth"
        trainer.save_checkpoint(str(checkpoint_path), is_best=True)

        assert checkpoint_path.exists(), "Checkpoint file not created"
        print(f"✅ Checkpoint saved: {checkpoint_path}")

        # Try loading
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        expected_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict',
                        'scheduler_state_dict', 'config']
        for key in expected_keys:
            assert key in checkpoint, f"Missing key in checkpoint: {key}"

        print("✅ Checkpoint structure verified")
        print(f"   Checkpoint keys: {list(checkpoint.keys())}")

        # Clean up
        checkpoint_path.unlink()
        print("✅ Test checkpoint cleaned up")

    except Exception as e:
        print(f"❌ Checkpointing failed: {e}")
        raise


def test_mixed_precision(config):
    """Test 7: Mixed precision training"""
    print("\n" + "="*80)
    print("TEST 7: Mixed Precision Training")
    print("="*80)

    try:
        # Test with mixed precision enabled
        config_mp = TrainingConfig(
            output_dir="outputs/test",
            epochs=1,
            batch_size=4,
            pretrained=False,
            use_mixed_precision=True,
            use_wandb=False
        )

        trainer_mp = ImprovedTrainer(config_mp)
        train_dataset = DummyTrafficDataset(num_samples=20)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        # Run one batch with mixed precision
        batch = next(iter(train_loader))
        images = batch['image'].to(trainer_mp.config.device)
        targets = {
            'traffic_level': batch['traffic_level'].to(trainer_mp.config.device),
            'density': batch['density'].to(trainer_mp.config.device),
            'congestion': batch['congestion'].to(trainer_mp.config.device),
            'flow_rate': batch['flow_rate'].to(trainer_mp.config.device)
        }

        trainer_mp.optimizer.zero_grad()

        from torch.cuda.amp import autocast
        with autocast():
            outputs = trainer_mp.model(images)
            loss, _ = trainer_mp.criterion(outputs, targets)

        trainer_mp.scaler.scale(loss).backward()
        trainer_mp.scaler.step(trainer_mp.optimizer)
        trainer_mp.scaler.update()

        print("✅ Mixed precision training successful")

    except Exception as e:
        print(f"❌ Mixed precision training failed: {e}")
        raise


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("TESTING IMPROVED TRAINING PIPELINE")
    print("="*80)

    # Create dummy datasets
    train_dataset = DummyTrafficDataset(num_samples=32, image_size=640)
    val_dataset = DummyTrafficDataset(num_samples=16, image_size=640)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Run tests
    try:
        trainer, config = test_trainer_initialization()
        test_forward_pass(trainer)
        test_loss_calculation(trainer)
        test_training_step(trainer, train_loader)
        test_full_epoch(trainer, train_loader, val_loader)
        test_checkpointing(trainer)
        test_mixed_precision(config)

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80 + "\n")

    except Exception as e:
        print("\n" + "="*80)
        print("❌ TESTS FAILED")
        print(f"Error: {e}")
        print("="*80 + "\n")
        raise


if __name__ == "__main__":
    run_all_tests()
