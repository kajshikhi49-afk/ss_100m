"""
Stage 2: Supervised Fine-Tuning (SFT) script for 50M Bengali GPT.
Loads Stage 1 pretrained model from checkpoints/stage_1/stage_1_final.pt.
Trains on data/sft_data.txt for strictly 1 epoch at lr=1e-4.
Saves checkpoints to checkpoints/stage_2/
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

def run_sft():
    config = GPTConfig()
    device = config.device
    os.makedirs(config.output_dir_stage2, exist_ok=True)

    print("=" * 65)
    print("🎯 [STAGE 2: SUPERVISED FINE-TUNING (SFT)] Q&A Chatbot Training")
    print(f"  - Dataset: {config.data_path_stage2}")
    print(f"  - Target Epochs: {config.num_train_epochs_stage2} (Strictly 1 epoch)")
    print(f"  - Learning Rate: {config.sft_learning_rate} (1e-4)")
    print(f"  - Context (Block Size): {config.block_size}")
    print(f"  - Checkpoint Dir: {config.output_dir_stage2}")
    print("=" * 65)

    stage1_ckpt = config.pretrained_stage1_ckpt
    if not os.path.exists(stage1_ckpt):
        # Fallback to check if any stage1_step_*.pt exists
        stage1_ckpts = glob.glob(os.path.join(config.output_dir_stage1, "stage1_step_*.pt"))
        if stage1_ckpts:
            def get_step(p):
                try: return int(p.split('_step_')[-1].replace('.pt', ''))
                except: return 0
            stage1_ckpt = sorted(stage1_ckpts, key=get_step)[-1]
        else:
            raise FileNotFoundError(f"Stage 1 checkpoint not found at {stage1_ckpt}! Please run Stage 1 pretraining first.")

    tokenizer = Tokenizer.from_file("tokenizer.json")
    dataset = BengaliDataset(
        corpus_path=config.data_path_stage2,
        tokenizer=tokenizer,
        block_size=config.block_size,
        split_ratio=0.9
    )

    tokens_per_step = config.batch_size * config.gradient_accumulation_steps * config.block_size
    steps_per_epoch = dataset.train_len // tokens_per_step
    total_steps = int(steps_per_epoch * config.num_train_epochs_stage2)
    print(f"✓ 1 Epoch = {steps_per_epoch:,} steps | Total SFT Steps = {total_steps:,}")

    model = BengaliGPT(config).to(device)
    print(f"📥 Loading Stage 1 pretrained weights from: {stage1_ckpt}")
    model.load_state_dict(torch.load(stage1_ckpt, map_location=device))
    print("✓ Stage 1 foundation model successfully loaded!")

    # Lower learning rate (1e-4) with fresh optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.sft_learning_rate, betas=(0.9, 0.95), weight_decay=config.weight_decay)
    scaler = GradScaler(enabled=('cuda' in device))

    # Auto-resume check for Stage 2
    start_step = 1
    existing_ckpts = glob.glob(os.path.join(config.output_dir_stage2, "stage2_step_*.pt"))
    if existing_ckpts:
        def get_step(p):
            try: return int(p.split('_step_')[-1].replace('.pt', ''))
            except: return 0
        latest_ckpt = sorted(existing_ckpts, key=get_step)[-1]
        s = get_step(latest_ckpt)
        if s > 0 and s < total_steps:
            model.load_state_dict(torch.load(latest_ckpt, map_location=device))
            start_step = s + 1
            print(f"🔄 Auto-Resuming Stage 2 from step {start_step:,} ({latest_ckpt})")

    def get_sft_lr(it):
        warmup = config.sft_warmup_iters
        if it < warmup: return config.sft_learning_rate * it / warmup
        if it > total_steps: return config.sft_min_lr
        decay = (it - warmup) / (total_steps - warmup)
        return config.sft_min_lr + 0.5 * (1.0 + math.cos(math.pi * decay)) * (config.sft_learning_rate - config.sft_min_lr)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    for step in range(start_step, total_steps + 1):
        lr = get_sft_lr(step)
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
            print(f"[Stage 2] Step {step:5d}/{total_steps} (Epoch {ep:.2f}) | SFT Loss: {accum_loss:.4f} | LR: {lr:.2e} | Speed: {spd:.2f} it/s")

        if step % config.save_interval == 0 or step == total_steps:
            ckpt_path = os.path.join(config.output_dir_stage2, f"stage2_step_{step}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

    final_product_path = os.path.join(config.output_dir_stage2, "checkpoint_stage_2_final.pt")
    torch.save(model.state_dict(), final_product_path)
    print("=" * 65)
    print(f"🎉 Stage 2 Complete! Final product model saved to: {final_product_path}")
    print("=" * 65)

if __name__ == '__main__':
    run_sft()
