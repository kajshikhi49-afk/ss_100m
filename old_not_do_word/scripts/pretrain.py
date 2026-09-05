"""
Script for pretraining the language model
"""

import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import CustomLMModel, CustomLMConfig, create_model
from src.dataset import PretrainingDataset, create_dataloader
from src.train import Trainer, TrainingConfig
from src.tokenizer import load_tokenizer
from src.utils import (
    load_config, 
    set_seed, 
    print_system_info, 
    print_model_info,
    print_gpu_memory,
    ensure_dir
)


def main():
    parser = argparse.ArgumentParser(description="Pretrain language model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA even if available"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("LANGUAGE MODEL PRETRAINING")
    print("="*60 + "\n")
    
    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    
    # Override with command line arguments
    if args.output_dir:
        training_config["output_dir"] = args.output_dir
    if args.resume_from_checkpoint:
        training_config["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.no_cuda:
        training_config["device"] = "cpu"
    
    # Set seed
    seed = training_config.get("seed", 42)
    set_seed(seed)
    
    # Print system info
    print_system_info()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_path = data_config.get("tokenizer_path", "tokenizer")
    try:
        tokenizer = load_tokenizer(tokenizer_path)
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print(f"\nPlease train tokenizer first:")
        print(f"  python scripts/train_tokenizer.py")
        return
    
    # Update vocab size in model config
    model_config["vocab_size"] = len(tokenizer)
    
    # Create model
    print("\nCreating model...")
    model = create_model(model_config)
    print_model_info(model)
    
    # Load datasets
    print("Loading datasets...")
    train_data_path = data_config.get("train_data_path", "data/processed/train")
    val_data_path = data_config.get("val_data_path", "data/processed/val")
    block_size = data_config.get("block_size", 2048)
    
    # Check if processed data exists
    if not Path(train_data_path).exists():
        print(f"❌ Error: Training data not found at {train_data_path}")
        print(f"\nPlease prepare data first:")
        print(f"  python scripts/prepare_data.py")
        return
    
    try:
        train_dataset = PretrainingDataset(
            data_path=train_data_path,
            tokenizer=tokenizer,
            block_size=block_size,
        )
        
        train_dataloader = create_dataloader(
            train_dataset,
            batch_size=training_config.get("batch_size", 32),
            shuffle=True,
            num_workers=data_config.get("num_workers", 2),
            pin_memory=data_config.get("pin_memory", True),
        )
        
        print(f"✓ Training dataset: {len(train_dataset)} examples")
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return
    
    # Load validation dataset (optional)
    eval_dataloader = None
    if Path(val_data_path).exists():
        try:
            val_dataset = PretrainingDataset(
                data_path=val_data_path,
                tokenizer=tokenizer,
                block_size=block_size,
            )
            
            eval_dataloader = create_dataloader(
                val_dataset,
                batch_size=training_config.get("batch_size", 32),
                shuffle=False,
                num_workers=data_config.get("num_workers", 2),
                pin_memory=data_config.get("pin_memory", True),
            )
            
            print(f"✓ Validation dataset: {len(val_dataset)} examples")
            
        except Exception as e:
            print(f"Warning: Could not load validation data: {e}")
    else:
        print("Note: No validation data found, skipping evaluation")
    
    # Create training config
    print("\nSetting up training...")
    train_config = TrainingConfig(
        model_name=model_config.get("model_name", "custom_lm"),
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        block_size=block_size,
        batch_size=training_config.get("batch_size", 32),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 3e-4),
        weight_decay=training_config.get("weight_decay", 0.1),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        num_train_epochs=training_config.get("num_train_epochs", 3),
        max_steps=training_config.get("max_steps", -1),
        warmup_steps=training_config.get("warmup_steps", 500),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", False),
        output_dir=training_config.get("output_dir", "outputs/pretrain"),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        resume_from_checkpoint=training_config.get("resume_from_checkpoint"),
        eval_steps=training_config.get("eval_steps", 500),
        logging_steps=training_config.get("logging_steps", 50),
        log_to_wandb=training_config.get("log_to_wandb", False),
        wandb_project=training_config.get("wandb_project", "bengali-lm-100m"),
        wandb_run_name=training_config.get("wandb_run_name"),
        device=training_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        seed=seed,
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )
    
    # Create output directory
    ensure_dir(train_config.output_dir)
    
    # Print training info
    print(f"\nTraining Configuration:")
    print(f"  Output directory: {train_config.output_dir}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Epochs: {train_config.num_train_epochs}")
    print(f"  Warmup steps: {train_config.warmup_steps}")
    print(f"  Device: {train_config.device}")
    print(f"  Mixed precision: {'FP16' if train_config.fp16 else 'BF16' if train_config.bf16 else 'FP32'}")
    
    # Print GPU memory before training
    if torch.cuda.is_available():
        print_gpu_memory()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=train_config,
        tokenizer=tokenizer,
    )
    
    # Resume from checkpoint if specified
    if train_config.resume_from_checkpoint:
        print(f"\nResuming from checkpoint: {train_config.resume_from_checkpoint}")
        trainer.load_checkpoint(train_config.resume_from_checkpoint)
    
    # Start training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    try:
        trainer.train()
        
        print("\n" + "="*60)
        print("✓ TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModel saved to: {train_config.output_dir}")
        print(f"Best model: {train_config.output_dir}/best_model")
        print(f"Final model: {train_config.output_dir}/final_model")
        
        print("\nNext steps:")
        print("  1. Evaluate the model:")
        print(f"     python scripts/evaluate.py --model_path {train_config.output_dir}/best_model")
        print("  2. Fine-tune with SFT:")
        print("     python scripts/sft.py")
        print("  3. Export for Android:")
        print("     python scripts/export.py")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"Checkpoints saved to: {train_config.output_dir}")
        print("\nTo resume training, run:")
        print(f"  python scripts/pretrain.py --resume_from_checkpoint {train_config.output_dir}/checkpoint-XXXX")
        
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
