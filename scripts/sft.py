"""
Script for supervised fine-tuning (SFT) of the language model
"""

import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import CustomLMModel, CustomLMConfig
from src.dataset import SFTDataset, create_dataloader
from src.train import Trainer, TrainingConfig
from src.tokenizer import load_tokenizer
from src.utils import (
    load_config, 
    set_seed, 
    print_model_info,
    ensure_dir
)


def main():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning of language model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pretrained model (overrides config)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to SFT data JSONL file (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA even if available"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SUPERVISED FINE-TUNING (SFT)")
    print("="*60 + "\n")
    
    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    
    sft_config = config.get("sft", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    
    # Override with command line arguments
    if args.output_dir:
        sft_config["output_dir"] = args.output_dir
    if args.data_path:
        sft_config["train_data_path"] = args.data_path
    if args.no_cuda:
        training_config["device"] = "cpu"
    
    # Set seed
    seed = training_config.get("seed", 42)
    set_seed(seed)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_path = data_config.get("tokenizer_path", "tokenizer")
    try:
        tokenizer = load_tokenizer(tokenizer_path)
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return
    
    # Load pretrained model
    print("\nLoading pretrained model...")
    model_path = args.model_path or "outputs/pretrain/best_model"
    
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        print(f"\nPlease train the model first:")
        print(f"  python scripts/pretrain.py")
        return
    
    try:
        model = CustomLMModel.from_pretrained(model_path)
        print(f"✓ Loaded model from: {model_path}")
        print_model_info(model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load SFT dataset
    print("\nLoading SFT dataset...")
    train_data_path = sft_config.get("train_data_path", "data/examples/sft_data.jsonl")
    max_length = sft_config.get("max_length", 2048)
    prompt_template = sft_config.get("prompt_template")
    
    if not Path(train_data_path).exists():
        print(f"❌ Error: SFT data not found at {train_data_path}")
        print(f"\nPlease create an SFT dataset in JSONL format:")
        print(f"  Each line should be a JSON object with 'instruction' and 'response' fields")
        print(f"\nExample:")
        print('  {"instruction": "বাংলাদেশের রাজধানী কী?", "response": "বাংলাদেশের রাজধানী ঢাকা।"}')
        return
    
    try:
        train_dataset = SFTDataset(
            data_path=train_data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            prompt_template=prompt_template,
        )
        
        train_dataloader = create_dataloader(
            train_dataset,
            batch_size=sft_config.get("batch_size", 16),
            shuffle=True,
            num_workers=data_config.get("num_workers", 2),
            pin_memory=data_config.get("pin_memory", True),
        )
        
        print(f"✓ Training dataset: {len(train_dataset)} examples")
        
    except Exception as e:
        print(f"❌ Error loading SFT data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load validation dataset (optional)
    eval_dataloader = None
    val_data_path = sft_config.get("val_data_path")
    if val_data_path and Path(val_data_path).exists():
        try:
            val_dataset = SFTDataset(
                data_path=val_data_path,
                tokenizer=tokenizer,
                max_length=max_length,
                prompt_template=prompt_template,
            )
            
            eval_dataloader = create_dataloader(
                val_dataset,
                batch_size=sft_config.get("batch_size", 16),
                shuffle=False,
                num_workers=data_config.get("num_workers", 2),
                pin_memory=data_config.get("pin_memory", True),
            )
            
            print(f"✓ Validation dataset: {len(val_dataset)} examples")
            
        except Exception as e:
            print(f"Warning: Could not load validation data: {e}")
    
    # Create training config for SFT
    print("\nSetting up fine-tuning...")
    train_config = TrainingConfig(
        output_dir=sft_config.get("output_dir", "outputs/sft"),
        batch_size=sft_config.get("batch_size", 16),
        gradient_accumulation_steps=sft_config.get("gradient_accumulation_steps", 2),
        learning_rate=sft_config.get("learning_rate", 1e-4),  # Lower LR for fine-tuning
        weight_decay=training_config.get("weight_decay", 0.1),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        num_train_epochs=sft_config.get("num_train_epochs", 3),
        max_steps=sft_config.get("max_steps", -1),
        warmup_steps=sft_config.get("warmup_steps", 100),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", False),
        save_steps=sft_config.get("save_steps", 200),
        save_total_limit=training_config.get("save_total_limit", 3),
        eval_steps=sft_config.get("eval_steps", 200),
        logging_steps=sft_config.get("logging_steps", 20),
        log_to_wandb=training_config.get("log_to_wandb", False),
        wandb_project=training_config.get("wandb_project", "bengali-lm-100m"),
        wandb_run_name=sft_config.get("wandb_run_name", "sft"),
        device=training_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        seed=seed,
    )
    
    # Create output directory
    ensure_dir(train_config.output_dir)
    
    # Print training info
    print(f"\nFine-tuning Configuration:")
    print(f"  Output directory: {train_config.output_dir}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Epochs: {train_config.num_train_epochs}")
    print(f"  Warmup steps: {train_config.warmup_steps}")
    print(f"  Device: {train_config.device}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=train_config,
        tokenizer=tokenizer,
    )
    
    # Start fine-tuning
    print("\n" + "="*60)
    print("Starting fine-tuning...")
    print("="*60 + "\n")
    
    try:
        trainer.train()
        
        print("\n" + "="*60)
        print("✓ FINE-TUNING COMPLETE!")
        print("="*60)
        print(f"\nModel saved to: {train_config.output_dir}")
        print(f"Best model: {train_config.output_dir}/best_model")
        print(f"Final model: {train_config.output_dir}/final_model")
        
        print("\nNext steps:")
        print("  1. Test the fine-tuned model:")
        print(f"     python scripts/generate.py --model_path {train_config.output_dir}/best_model")
        print("  2. Export for Android:")
        print("     python scripts/export.py")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Fine-tuning interrupted by user")
        print(f"Checkpoints saved to: {train_config.output_dir}")
        
    except Exception as e:
        print(f"\n\n❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
