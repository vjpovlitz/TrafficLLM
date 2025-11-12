"""
Training Run Script - Using Your Existing Data with New Improved Model
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.improved_trainer import ImprovedTrainer, TrainingConfig
from data_processing.adapted_dataset import create_adapted_dataloaders
from data_processing.augmentations import get_training_augmentation, get_validation_transform

def main():
    print("="*80)
    print("Traffic Classification Training - Improved Model")
    print("="*80)

    # ============================================================================
    # CONFIGURATION - Adjust these settings as needed
    # ============================================================================

    config = TrainingConfig(
        # Data
        data_dir="data/processed",  # Not used with adapted dataset
        output_dir="outputs",
        num_workers=4,

        # Model Architecture
        num_classes=5,  # Adjust based on your traffic levels (currently uses 0-1 from dataset)
        attention_type='cbam',  # Options: 'cbam', 'se', 'enhanced_spatial', 'simple'
        feature_fusion=True,
        dropout=0.4,
        pretrained=True,  # Use ImageNet pretrained ResNet-50

        # Training Hyperparameters
        epochs=50,
        batch_size=32,
        learning_rate=3e-4,
        weight_decay=0.01,

        # Augmentation
        image_size=640,
        augmentation_level='medium',  # Options: 'light', 'medium', 'heavy'

        # Optimization
        use_mixed_precision=True,
        gradient_clip=1.0,

        # Scheduler
        scheduler_T0=10,  # Restart every 10 epochs
        scheduler_T_mult=2,
        scheduler_eta_min=1e-6,

        # Loss
        label_smoothing=0.1,

        # Checkpointing
        save_best=True,
        save_last=True,
        checkpoint_every=5,

        # Early Stopping
        early_stopping=True,
        patience=10,
        min_delta=1e-4,

        # Logging
        log_interval=10,
        use_wandb=False,  # Set to True if you want W&B logging
        wandb_project="traffic-classification",
        wandb_entity=None  # Your W&B username (optional)
    )

    print("\n" + "="*80)
    print("Training Configuration")
    print("="*80)
    print(f"  Model: ImprovedTrafficNet ({config.attention_type} attention)")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Augmentation: {config.augmentation_level}")
    print(f"  Mixed Precision: {config.use_mixed_precision}")
    print(f"  Early Stopping: {config.early_stopping} (patience={config.patience})")
    print(f"  Output Directory: {config.output_dir}")
    print("="*80)

    # ============================================================================
    # CREATE AUGMENTATION TRANSFORMS
    # ============================================================================

    print("\n📊 Creating augmentation transforms...")
    train_transform = get_training_augmentation(
        image_size=config.image_size,
        augmentation_level=config.augmentation_level
    )

    val_transform = get_validation_transform(
        image_size=config.image_size
    )

    print("  ✅ Training augmentation created")
    print("  ✅ Validation transform created")

    # ============================================================================
    # CREATE DATALOADERS
    # ============================================================================

    print("\n📊 Creating dataloaders from your existing datasets...")
    try:
        train_loader, val_loader, test_loader = create_adapted_dataloaders(
            project_root=project_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            train_transform=train_transform,
            val_transform=val_transform,
            test_split=0.15,
            val_split=0.15
        )

        print(f"  ✅ Dataloaders created successfully")
        print(f"     Train batches: {len(train_loader)}")
        print(f"     Val batches: {len(val_loader)}")
        print(f"     Test batches: {len(test_loader)}")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Solution:")
        print("   1. Make sure you have collected data using the data collection pipeline")
        print("   2. Check that the dataset paths in adapted_dataset.py are correct")
        print("   3. Your datasets should be at:")
        print(f"      - {project_root}/data_processing/processed_singapore_tensors_final")
        print(f"      - {project_root}/data_processing/processed_bay_bridge_dataset")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============================================================================
    # TEST BATCH LOADING
    # ============================================================================

    print("\n🧪 Testing batch loading...")
    try:
        test_batch = next(iter(train_loader))
        print(f"  ✅ Successfully loaded a batch")
        print(f"     Image shape: {test_batch['image'].shape}")
        print(f"     Traffic level shape: {test_batch['traffic_level'].shape}")
        print(f"     Density shape: {test_batch['density'].shape}")
        print(f"     Congestion shape: {test_batch['congestion'].shape}")
        print(f"     Flow rate shape: {test_batch['flow_rate'].shape}")
    except Exception as e:
        print(f"  ❌ Error loading batch: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============================================================================
    # INITIALIZE TRAINER
    # ============================================================================

    print("\n🚀 Initializing trainer...")
    try:
        trainer = ImprovedTrainer(config)
        print(f"  ✅ Trainer initialized")
        print(f"     Device: {config.device}")
        print(f"     Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    except Exception as e:
        print(f"  ❌ Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============================================================================
    # START TRAINING
    # ============================================================================

    print("\n" + "="*80)
    input_msg = "Ready to start training. Press Enter to continue or Ctrl+C to cancel..."
    try:
        input(input_msg)
    except KeyboardInterrupt:
        print("\n\n❌ Training cancelled by user")
        sys.exit(0)

    print("="*80)
    print("\n🏋️  Starting training...")
    print("="*80)

    try:
        trainer.train(train_loader, val_loader)

        print("\n" + "="*80)
        print("✅ Training Complete!")
        print("="*80)
        print(f"\nResults saved to: {config.output_dir}")
        print(f"  Checkpoints: {config.output_dir}/checkpoints/")
        print(f"  Logs: {config.output_dir}/logs/")
        print(f"\nBest model saved at: {config.output_dir}/checkpoints/best_model.pth")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"Last checkpoint saved at: {config.output_dir}/checkpoints/last_model.pth")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============================================================================
    # OPTIONAL: EVALUATE ON TEST SET
    # ============================================================================

    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80)

    try:
        test_metrics = trainer.validate(test_loader)

        print("\n📊 Test Set Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {test_metrics['f1_score']:.4f}")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Density MAE: {test_metrics['density_mae']:.4f}")
        print(f"  Congestion MAE: {test_metrics['congestion_mae']:.4f}")
        print(f"  Flow Rate MAE: {test_metrics['flow_rate_mae']:.4f}")

    except Exception as e:
        print(f"  ⚠️  Could not evaluate on test set: {e}")

    print("\n" + "="*80)
    print("All Done! 🎉")
    print("="*80)


if __name__ == "__main__":
    main()
