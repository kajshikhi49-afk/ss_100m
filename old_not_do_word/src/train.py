"""
Training Pipeline for Language Model
Supports mixed precision, gradient accumulation, checkpointing, and logging
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Callable
from pathlib import Path
import json
from tqdm import tqdm
from dataclasses import dataclass, field, asdict


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Model
    model_name: str = "custom_lm"
    
    # Data
    train_data_path: str = "data/processed/train"
    val_data_path: str = "data/processed/val"
    block_size: int = 2048
    
    # Training hyperparameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    
    # Schedule
    num_train_epochs: int = 3
    max_steps: int = -1  # If > 0, overrides num_train_epochs
    warmup_steps: int = 500
    lr_scheduler_type: str = "cosine"  # cosine, linear, constant
    
    # Precision
    fp16: bool = False
    bf16: bool = False
    
    # Checkpointing
    output_dir: str = "outputs"
    save_steps: int = 500
    save_total_limit: int = 3  # Keep only N recent checkpoints
    resume_from_checkpoint: Optional[str] = None
    
    # Evaluation
    eval_steps: int = 500
    eval_accumulation_steps: int = 1
    
    # Logging
    logging_steps: int = 50
    log_to_wandb: bool = False
    wandb_project: str = "bengali-lm-100m"
    wandb_run_name: Optional[str] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Optimization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Gradient checkpointing (save memory)
    gradient_checkpointing: bool = False
    
    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Load config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def load(cls, path: str):
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class Trainer:
    """
    Custom trainer for language model training.
    Handles training loop, evaluation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        config: TrainingConfig,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data
            config: Training configuration
            tokenizer: Tokenizer (optional, for logging)
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.tokenizer = tokenizer
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup mixed precision
        self.scaler = None
        if config.fp16 or config.bf16:
            self.scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
            self.amp_dtype = torch.float16 if config.fp16 else torch.bfloat16
        else:
            self.amp_dtype = torch.float32
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Calculate total training steps
        self.total_steps = self._calculate_total_steps()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Setup logging
        self.wandb_run = None
        if config.log_to_wandb:
            self._setup_wandb()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Save config
        config.save(os.path.join(config.output_dir, "training_config.json"))
        
        print(f"Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {'fp16' if config.fp16 else 'bf16' if config.bf16 else 'fp32'}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Warmup steps: {config.warmup_steps}")
        print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    
    def _create_optimizer(self):
        """Create AdamW optimizer with weight decay."""
        # Separate parameters into weight decay and no weight decay groups
        no_decay = ["bias", "LayerNorm.weight", "ln_f.weight", "ln_1.weight", "ln_2.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
        
        return optimizer
    
    def _calculate_total_steps(self):
        """Calculate total training steps."""
        if self.config.max_steps > 0:
            return self.config.max_steps
        
        steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        return steps_per_epoch * self.config.num_train_epochs
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        def lr_lambda(current_step: int):
            # Warmup
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            
            # Decay
            if self.config.lr_scheduler_type == "cosine":
                progress = float(current_step - self.config.warmup_steps) / \
                          float(max(1, self.total_steps - self.config.warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            elif self.config.lr_scheduler_type == "linear":
                progress = float(current_step - self.config.warmup_steps) / \
                          float(max(1, self.total_steps - self.config.warmup_steps))
                return max(0.0, 1.0 - progress)
            
            else:  # constant
                return 1.0
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            import wandb
            
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.to_dict(),
            )
            print("W&B logging enabled")
        except ImportError:
            print("Warning: wandb not installed, logging disabled")
            self.config.log_to_wandb = False
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        self.model.train()
        
        train_loss = 0.0
        logging_loss = 0.0
        
        progress_bar = tqdm(total=self.total_steps, desc="Training")
        
        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(
                    enabled=(self.config.fp16 or self.config.bf16),
                    dtype=self.amp_dtype
                ):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                train_loss += loss.item()
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = (train_loss - logging_loss) / self.config.logging_steps
                        current_lr = self.scheduler.get_last_lr()[0]
                        
                        log_dict = {
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch,
                            "train/step": self.global_step,
                        }
                        
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{current_lr:.2e}"
                        })
                        
                        if self.config.log_to_wandb and self.wandb_run:
                            self.wandb_run.log(log_dict, step=self.global_step)
                        
                        logging_loss = train_loss
                    
                    # Evaluation
                    if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                        eval_loss = self.evaluate()
                        
                        log_dict = {
                            "eval/loss": eval_loss,
                            "eval/perplexity": math.exp(eval_loss),
                        }
                        
                        print(f"\n  Eval loss: {eval_loss:.4f} | Perplexity: {math.exp(eval_loss):.2f}")
                        
                        if self.config.log_to_wandb and self.wandb_run:
                            self.wandb_run.log(log_dict, step=self.global_step)
                        
                        # Save best model
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint("best_model")
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")
                        self._cleanup_checkpoints()
                    
                    # Check if max steps reached
                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        progress_bar.close()
                        print("\nMax steps reached!")
                        return
        
        progress_bar.close()
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        # Final evaluation
        if self.eval_dataloader:
            eval_loss = self.evaluate()
            print(f"Final eval loss: {eval_loss:.4f} | Perplexity: {math.exp(eval_loss):.2f}")
        
        # Save final model
        self.save_checkpoint("final_model")
    
    def evaluate(self):
        """Run evaluation on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        return avg_loss
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save tokenizer if available
        if self.tokenizer:
            self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        state_dict = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        
        if self.scaler:
            state_dict["scaler_state"] = self.scaler.state_dict()
        
        state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
        torch.save(state_dict, state_path)
        
        print(f"Checkpoint saved: {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint and training state."""
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load model
        self.model = self.model.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        
        # Load training state
        state_path = os.path.join(checkpoint_path, "trainer_state.pt")
        if os.path.exists(state_path):
            state_dict = torch.load(state_path, map_location=self.device)
            
            self.global_step = state_dict["global_step"]
            self.epoch = state_dict["epoch"]
            self.best_eval_loss = state_dict["best_eval_loss"]
            
            self.optimizer.load_state_dict(state_dict["optimizer_state"])
            self.scheduler.load_state_dict(state_dict["scheduler_state"])
            
            if self.scaler and "scaler_state" in state_dict:
                self.scaler.load_state_dict(state_dict["scaler_state"])
            
            print(f"Resumed from step {self.global_step}, epoch {self.epoch}")
        else:
            print("Warning: trainer_state.pt not found, starting fresh")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if self.config.save_total_limit <= 0:
            return
        
        # Get all checkpoint directories
        checkpoints = []
        for path in Path(self.config.output_dir).glob("checkpoint-*"):
            if path.is_dir():
                checkpoints.append(path)
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old checkpoints
        for checkpoint in checkpoints[self.config.save_total_limit:]:
            print(f"Removing old checkpoint: {checkpoint}")
            import shutil
            shutil.rmtree(checkpoint)


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    
    # Compute perplexity
    loss = torch.nn.functional.cross_entropy(
        torch.tensor(logits).view(-1, logits.shape[-1]),
        torch.tensor(labels).view(-1),
        ignore_index=-100,
    )
    
    perplexity = math.exp(loss.item())
    
    return {
        "perplexity": perplexity,
        "loss": loss.item(),
    }


if __name__ == "__main__":
    print("Training module loaded successfully!")
    print("\nExample usage:")
    print("""
    from src.model import CustomLMModel, CustomLMConfig
    from src.train import Trainer, TrainingConfig
    from src.dataset import PretrainingDataset, create_dataloader
    from src.tokenizer import load_tokenizer
    
    # Load tokenizer
    tokenizer = load_tokenizer("./tokenizer")
    
    # Create model
    model_config = CustomLMConfig()
    model = CustomLMModel(model_config)
    
    # Create datasets
    train_dataset = PretrainingDataset("data/raw", tokenizer, block_size=2048)
    train_dataloader = create_dataloader(train_dataset, batch_size=32)
    
    # Create trainer
    training_config = TrainingConfig(
        output_dir="./outputs",
        num_train_epochs=3,
        learning_rate=3e-4,
    )
    
    trainer = Trainer(model, train_dataloader, None, training_config, tokenizer)
    
    # Train
    trainer.train()
    """)
