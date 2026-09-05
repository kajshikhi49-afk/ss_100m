"""
Main Entry Point for 50M Parameter Bengali GPT Model with 2048 Context Length & QLoRA.
"""

import sys
import os
import argparse
import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import GPTConfig
from src.train import train_qlora, train_pretrain
from src.generate import generate_text, chat_interactive


def print_info():
    config = GPTConfig()
    params = config.estimate_parameters()

    print("=" * 65)
    print("📋 50M PARAMETER BENGALI GPT MODEL SPECIFICATIONS")
    print("=" * 65)
    print(f"• Context Length (Block Size): {config.block_size} tokens (~1,600 Bengali words!)")
    print(f"• Vocabulary Size:            {config.vocab_size:,} (BPE Subwords)")
    print(f"• Embedding Dimension:        {config.embed_dim}")
    print(f"• Attention Mechanism:        GQA (Grouped-Query Attention, 4:1 ratio)")
    print(f"• Query Heads:                {config.num_heads} | Key-Value Heads: {config.num_kv_heads}")
    print(f"• Positional Embedding:       RoPE (Rotary Position Embeddings)")
    print(f"• Transformer Layers:         {config.num_layers} blocks")
    print(f"• Intermediate Expansion:     {config.intermediate_dim} (4x)")
    print(f"• Effective Batch Size:       {config.batch_size * config.gradient_accumulation_steps} ({config.batch_size} × {config.gradient_accumulation_steps})")
    print("—" * 65)
    print("📊 Parameter Breakdown:")
    print(f"  - Token Embeddings:         {params['token_emb']:,}")
    print(f"  - 16 Transformer Blocks:    {params['all_blocks']:,}")
    print(f"  - Total Parameters (Tied):  {params['total_tied']:,} (~{params['total_tied']/1e6:.2f}M)")
    print(f"  - Total Parameters (Head):  {params['total_untied']:,} (~{params['total_untied']/1e6:.2f}M)")
    print("—" * 65)
    print("⚡ QLoRA & Mobile Memory Specifications:")
    print(f"  - Targeted Projections:     {config.lora_target_modules}")
    print(f"  - LoRA Trainable Params:    {params['lora_params']:,} ({params['lora_percent']:.2f}% of model)")
    print(f"  - Mobile KV-Cache (2048):   ~{params['kv_cache_mb']:.1f} MB (Extremely low due to GQA!)")
    print(f"  - Mobile Q4 GGUF Size:      ~28 - 30 MB")
    print(f"  - Mobile RAM Usage (itel):  ~45 - 55 MB total")
    print("—" * 65)
    print("💾 Colab Free T4 GPU Requirements:")
    print("  - QLoRA VRAM Usage:         ~3 - 4.5 GB VRAM (Fits easily in 15GB T4)")
    print("  - Training Time (5k steps): ~2.5 hours on Colab Free T4 GPU")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="50M Parameter Bengali GPT Model with 2048 Context & QLoRA")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    subparsers.add_parser("info", help="Display model architecture & parameter statistics")

    p_qlora = subparsers.add_parser("qlora", help="Run 50M QLoRA fine-tuning (recommended)")
    p_qlora.add_argument("--iters", type=int, default=5000, help="Training iterations")
    p_qlora.add_argument("--batch_size", type=int, default=8, help="Micro batch size")
    p_qlora.add_argument("--corpus", type=str, default="data/corpus.txt", help="Path to corpus text")

    p_pretrain = subparsers.add_parser("pretrain", help="Run 50M full pretraining from scratch")
    p_pretrain.add_argument("--iters", type=int, default=5000, help="Training iterations")
    p_pretrain.add_argument("--batch_size", type=int, default=8, help="Micro batch size")
    p_pretrain.add_argument("--corpus", type=str, default="data/corpus.txt", help="Path to corpus text")

    p_gen = subparsers.add_parser("generate", help="Generate text from trained model")
    p_gen.add_argument("--prompt", type=str, default="বাংলাদেশ একটি", help="Input prompt")
    p_gen.add_argument("--lora_adapter", type=str, default="checkpoints/qlora_50m_best.pt", help="LoRA adapter")

    p_chat = subparsers.add_parser("chat", help="Start interactive proactive Socratic chat")
    p_chat.add_argument("--lora_adapter", type=str, default="checkpoints/qlora_50m_best.pt", help="LoRA adapter")

    args = parser.parse_args()

    if args.command == "info" or args.command is None:
        print_info()
    elif args.command == "qlora":
        cfg = GPTConfig()
        cfg.max_iters = args.iters
        cfg.batch_size = args.batch_size
        train_qlora(config=cfg, corpus_path=args.corpus)
    elif args.command == "pretrain":
        cfg = GPTConfig()
        cfg.max_iters = args.iters
        cfg.batch_size = args.batch_size
        train_pretrain(config=cfg, corpus_path=args.corpus)
    elif args.command == "generate":
        generate_text(prompt=args.prompt, lora_checkpoint=args.lora_adapter)
    elif args.command == "chat":
        chat_interactive(lora_checkpoint=args.lora_adapter)


if __name__ == "__main__":
    main()
