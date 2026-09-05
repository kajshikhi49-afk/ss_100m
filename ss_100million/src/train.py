"""
Training pipeline for 50M parameter Bengali GPT with 2048 context length.
Features: Gradient accumulation, Mixed Precision, QLoRA, and Auto-Resume.
"""

import os
import math
import time
import glob
import torch
from src.config import GPTConfig
from src.model import BengaliGPT
from src.dataset import BengaliDataset
from src.lora import apply_lora, save_lora


def _lr(step, config):
    if step < config.warmup_iters:
        return config.learning_rate * step / config.warmup_iters
    decay = (step - config.warmup_iters) / max(1, config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * decay))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def _eval_loss(model, ds, config):
    model.eval()
    losses = {}
    for split in ('train', 'val'):
        L = []
        for _ in range(config.eval_iters):
            x, y = ds.get_batch(split, config.batch_size, config.device)
            with torch.amp.autocast('cuda', dtype=torch.float16) if 'cuda' in config.device else torch.nullcontext():
                _, loss = model(x, y)
            L.append(loss.item())
        losses[split] = sum(L) / len(L)
    model.train()
    return losses


def _find_last_checkpoint(ckpt_dir, prefix):
    files = glob.glob(os.path.join(ckpt_dir, f"{prefix}_step_*.pt"))
    if not files:
        return None, 0

    def extract_step(path):
        try:
            return int(path.split('_step_')[-1].replace('.pt', ''))
        except ValueError:
            return -1

    # সংখ্যাগতভাবে (Numerically) সর্বোচ্চ স্টেপটি বাছাই করুন
    files = sorted(files, key=extract_step)
    latest = files[-1]
    step = extract_step(latest)
    if step <= 0:
        return None, 0
    return latest, step


def _save_checkpoint(model, optimizer, step, val_loss, config, prefix, is_best=False):
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'val_loss': val_loss,
        'config': {k: v for k, v in vars(config).items() if not k.startswith('_')},
    }
    path = os.path.join(config.checkpoint_dir, f"{prefix}_step_{step}.pt")
    torch.save(ckpt, path)
    print(f"  💾 চেকপয়েন্ট ড্রাইভে সেভ হয়েছে: {path}")
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, f"{prefix}_best.pt")
        torch.save(ckpt, best_path)
        print(f"  🌟 সেরা মডেল ড্রাইভে আপডেট হয়েছে: {best_path}")


def train_qlora(config=None, corpus_path="data/corpus.txt", base_checkpoint=None):
    if config is None:
        config = GPTConfig()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.manual_seed(config.seed)

    # মডেলের প্যারামিটার সংখ্যা অনুযায়ী প্রিফিক্স নির্ধারণ করা
    total_params = sum(p.numel() for p in __import__('src.model', fromlist=['BengaliGPT']).BengaliGPT(config).parameters())
    param_m = round(total_params / 1e6)
    PREFIX = f"qlora_{param_m}m"

    print("=" * 65)
    print(f"🚀 {param_m}M MULTILINGUAL GPT — QLoRA ফাইন-টিউনিং শুরু হচ্ছে ({config.block_size} কনটেক্সট লেন্থ)")
    print(f"ডিভাইস: {config.device} | ব্যাচ: {config.batch_size} (Grad Accum: {config.gradient_accumulation_steps}) | সর্বমোট স্টেপ: {config.max_iters}")
    print("=" * 65)

    # vocab_size কনফিগ থেকে BengaliDataset-এ পাঠানো হচ্ছে (5k বা 10k উভয়ক্ষেত্রে সঠিক থাকবে)
    ds = BengaliDataset(corpus_path, block_size=config.block_size, vocab_size=config.vocab_size)
    config.vocab_size = ds.tokenizer.vocab_size
    ds.tokenizer.save(os.path.join(config.checkpoint_dir, "tokenizer.json"))

    model = BengaliGPT(config).to(config.device)

    # যদি কোনো প্রি-ট্রেইন্ড বেস মডেল দেওয়া থাকে, তার ওজন লোড করুন
    if base_checkpoint and os.path.exists(base_checkpoint):
        print(f"📥 বেস মডেল থেকে জ্ঞান লোড হচ্ছে: {base_checkpoint}")
        ckpt_base = torch.load(base_checkpoint, map_location=config.device)
        model.load_state_dict(ckpt_base['model'], strict=False)
        print("✓ বেস মডেলের ওজন সফলভাবে লোড হয়েছে!")

    model, trainable = apply_lora(
        model,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        use_qlora=True
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    scaler = torch.amp.GradScaler('cuda', enabled=('cuda' in config.device))

    # Auto-resume from Drive
    last_ckpt, start_step = _find_last_checkpoint(config.checkpoint_dir, PREFIX)
    if last_ckpt:
        ckpt = torch.load(last_ckpt, map_location=config.device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"🔄 পূর্ববর্তী চেকপয়েন্ট পাওয়া গেছে! {start_step} নম্বর স্টেপ থেকে ট্রেনিং চালু হচ্ছে...")
    else:
        print("🌱 কোনো পুরানো চেকপয়েন্ট নেই, শুরু থেকে (স্টেপ ১) ট্রেনিং শুরু হচ্ছে...")

    best_val = float('inf')
    t0 = time.time()

    for step in range(start_step + 1, config.max_iters + 1):
        lr = _lr(step, config)
        for g in optimizer.param_groups:
            g['lr'] = lr

        optimizer.zero_grad(set_to_none=True)

        accum_loss = 0.0
        # Gradient accumulation loop for 2048 context
        for _ in range(config.gradient_accumulation_steps):
            x, y = ds.get_batch('train', config.batch_size, config.device)
            with torch.amp.autocast('cuda', dtype=torch.float16) if 'cuda' in config.device else torch.nullcontext():
                _, loss = model(x, y)
                loss = loss / config.gradient_accumulation_steps
            accum_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # প্রতি ২৫ স্টেপে ভিজ্যুয়াল প্রগ্রেস বার সহ লাইভ আপডেট
        if step % 25 == 0 and step % config.eval_interval != 0:
            elapsed = time.time() - t0
            steps_done = step - start_step
            speed = elapsed / max(1, steps_done)
            eta_mins = (config.max_iters - step) * speed / 60
            pct = (step / config.max_iters) * 100
            bar_len = 20
            filled = int(bar_len * step / config.max_iters)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"  [{bar}] {pct:5.1f}% | Step {step:4d}/{config.max_iters} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | গতি: {speed:.2f}s/step | বাকি: ~{eta_mins:.1f} মি.")

        if step % config.eval_interval == 0 or step == config.max_iters:
            losses = _eval_loss(model, ds, config)
            elapsed = time.time() - t0
            pct = (step / config.max_iters) * 100
            bar_len = 20
            filled = int(bar_len * step / config.max_iters)
            bar = '█' * filled + '░' * (bar_len - filled)
            print("─" * 70)
            print(f"  [{bar}] {pct:5.1f}% | Step {step:5d}/{config.max_iters} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | সময়: {elapsed/60:.1f} মি.")
            print("─" * 70)
            is_best = losses['val'] < best_val
            if is_best:
                best_val = losses['val']
            _save_checkpoint(model, optimizer, step, losses['val'], config, PREFIX, is_best)
        elif step % config.save_interval == 0:
            _save_checkpoint(model, optimizer, step, None, config, PREFIX)

    print(f"🎉 {param_m}M মডেলের ট্রেনিং সফলভাবে সম্পন্ন! সেরা ভ্যালিডেশন লস: {best_val:.4f}")
    return model


def train_pretrain(config=None, corpus_path="data/corpus.txt"):
    if config is None: config = GPTConfig()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.manual_seed(config.seed)

    # মডেলের প্যারামিটার সংখ্যা অনুযায়ী প্রিফিক্স নির্ধারণ
    total_params = sum(p.numel() for p in BengaliGPT(config).parameters())
    param_m = round(total_params / 1e6)
    PREFIX = f"pretrain_{param_m}m"

    print("=" * 65)
    print(f"🚀 {param_m}M MULTILINGUAL GPT — FULL PRETRAINING ({config.block_size} কনটেক্সট লেন্থ)")
    print("=" * 65)

    ds = BengaliDataset(corpus_path, block_size=config.block_size, vocab_size=config.vocab_size)
    config.vocab_size = ds.tokenizer.vocab_size
    ds.tokenizer.save(os.path.join(config.checkpoint_dir, "tokenizer.json"))

    model = BengaliGPT(config).to(config.device)
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ সর্বমোট প্যারামিটার: {total:,} (~{total/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    scaler = torch.amp.GradScaler('cuda', enabled=('cuda' in config.device))

    last_ckpt, start_step = _find_last_checkpoint(config.checkpoint_dir, PREFIX)
    if last_ckpt:
        ckpt = torch.load(last_ckpt, map_location=config.device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"🔄 পূর্ববর্তী চেকপয়েন্ট পাওয়া গেছে: {start_step} নম্বর স্টেপ থেকে রিজ্যুম হচ্ছে...")
    else:
        print("🌱 শুরু থেকে স্ক্র্যাচ ট্রেনিং হচ্ছে...")

    best_val = float('inf')
    t0 = time.time()

    for step in range(start_step + 1, config.max_iters + 1):
        lr = _lr(step, config)
        for g in optimizer.param_groups: g['lr'] = lr

        accum_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            x, y = ds.get_batch('train', config.batch_size, config.device)
            with torch.amp.autocast('cuda', dtype=torch.float16) if 'cuda' in config.device else torch.nullcontext():
                _, loss = model(x, y)
                loss = loss / config.gradient_accumulation_steps
            accum_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # প্রতি ২৫ স্টেপে ভিজ্যুয়াল প্রগ্রেস বার সহ লাইভ আপডেট
        if step % 25 == 0 and step % config.eval_interval != 0:
            elapsed = time.time() - t0
            steps_done = step - start_step
            speed = elapsed / max(1, steps_done)
            eta_mins = (config.max_iters - step) * speed / 60
            pct = (step / config.max_iters) * 100
            bar_len = 20
            filled = int(bar_len * step / config.max_iters)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"  [{bar}] {pct:5.1f}% | Step {step:4d}/{config.max_iters} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | গতি: {speed:.2f}s/step | বাকি: ~{eta_mins:.1f} মি.")

        if step % config.eval_interval == 0 or step == config.max_iters:
            losses = _eval_loss(model, ds, config)
            elapsed = time.time() - t0
            pct = (step / config.max_iters) * 100
            bar_len = 20
            filled = int(bar_len * step / config.max_iters)
            bar = '█' * filled + '░' * (bar_len - filled)
            print("─" * 70)
            print(f"  [{bar}] {pct:5.1f}% | Step {step:5d}/{config.max_iters} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | সময়: {elapsed/60:.1f} মি.")
            print("─" * 70)
            is_best = losses['val'] < best_val
            if is_best: best_val = losses['val']
            _save_checkpoint(model, optimizer, step, losses['val'], config, PREFIX, is_best)
        elif step % config.save_interval == 0:
            _save_checkpoint(model, optimizer, step, None, config, PREFIX)

    print(f"🎉 {param_m}M মডেলের ফুল প্রি-ট্রেইনিং সম্পন্ন! সেরা লস: {best_val:.4f}")
    return model
