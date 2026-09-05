"""
Stage 1: Pretraining script for 50M Bengali GPT.
Trains on data/corpus.txt for 2 epochs from scratch (100% unfrozen).
Saves checkpoints to checkpoints/stage_1/
"""

import os
import sys
import math
import time
import glob
import torch
from torch.cuda.amp import autocast, GradScaler

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import GPTConfig
from src.model import BengaliGPT
from src.dataset import BengaliDataset
from tokenizers import Tokenizer

def run_pretrain():
    config = GPTConfig()
    device = config.device
    os.makedirs(config.output_dir_stage1, exist_ok=True)

    print("=" * 65)
    print("🚀 [STAGE 1: PRETRAINING] Starting Language Foundation Training")
    print(f"  - Dataset: {config.data_path_stage1}")
    print(f"  - Target Epochs: {config.num_train_epochs_stage1}")
    print(f"  - Learning Rate: {config.learning_rate}")
    print(f"  - Context (Block Size): {config.block_size}")
    print(f"  - Checkpoint Dir: {config.output_dir_stage1}")
    print("=" * 65)

    if not os.path.exists("tokenizer.json"):
        raise FileNotFoundError("tokenizer.json not found! Train tokenizer first.")

    tokenizer = Tokenizer.from_file("tokenizer.json")
    dataset = BengaliDataset(
        corpus_path=config.data_path_stage1,
        tokenizer=tokenizer,
        block_size=config.block_size,
        split_ratio=0.9
    )

    tokens_per_step = config.batch_size * config.gradient_accumulation_steps * config.block_size
    steps_per_epoch = dataset.train_len // tokens_per_step
    total_steps = int(steps_per_epoch * config.num_train_epochs_stage1)
    print(f"✓ 1 Epoch = {steps_per_epoch:,} steps | Total Steps (2 Epochs) = {total_steps:,}")

    model = BengaliGPT(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model Initialized from Scratch! Total Parameters: {total_params/1e6:.2f}M (100% Unfrozen)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=config.weight_decay)
    scaler = GradScaler(enabled=('cuda' in device))

    # Auto-resume check
    start_step = 1
    existing_ckpts = glob.glob(os.path.join(config.output_dir_stage1, "stage1_step_*.pt"))
    if existing_ckpts:
        def get_step(p):
            try: return int(p.split('_step_')[-1].replace('.pt', ''))
            except: return 0
        latest_ckpt = sorted(existing_ckpts, key=get_step)[-1]
        s = get_step(latest_ckpt)
        if s > 0 and s < total_steps:
            model.load_state_dict(torch.load(latest_ckpt, map_location=device))
            start_step = s + 1
            print(f"🔄 Auto-Resuming Stage 1 from step {start_step:,} ({latest_ckpt})")

    def get_lr(it):
        warmup = config.warmup_iters
        if it < warmup: return config.learning_rate * it / warmup
        if it > total_steps: return config.min_lr
        decay = (it - warmup) / (total_steps - warmup)
        return config.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay)) * (config.learning_rate - config.min_lr)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    for step in range(start_step, total_steps + 1):
        lr = get_lr(step)
        for g in optimizer.param_groups: g['lr'] = lr

        accum_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            x, y = dataset.get_batch('train', batch_size=config.batch_size, device=device)
            with autocast(dtype=torch.float16, enabled=('cuda' in device)):
                _, loss = model(x, y)
                loss = loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % config.eval_interval == 0 or step == start_step:
            ep = (step * tokens_per_step) / dataset.train_len
            elapsed = time.time() - t0
            spd = (step - start_step + 1) / elapsed if elapsed > 0 else 0
            print(f"[Stage 1] Step {step:5d}/{total_steps} (Epoch {ep:.2f}) | Train Loss: {accum_loss:.4f} | LR: {lr:.2e} | Speed: {spd:.2f} it/s")

        if step % config.save_interval == 0 or step == total_steps:
            ckpt_path = os.path.join(config.output_dir_stage1, f"stage1_step_{step}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

    final_path = os.path.join(config.output_dir_stage1, "stage_1_final.pt")
    torch.save(model.state_dict(), final_path)
    print("=" * 65)
    print(f"🎉 Stage 1 Complete! Final model saved to: {final_path}")
    print("=" * 65)

if __name__ == '__main__':
    run_pretrain()
