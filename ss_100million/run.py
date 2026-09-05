"""
Main Entry Point for 100M Parameter Multilingual (Bangla + English + Math) GPT Model.
Context Length: 4096 tokens (4k Context) | Vocab: 10,000 BPE | GQA 3:1 | RoPE
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
    print("📋 100M PARAMETER MULTILINGUAL GPT (BANGLA + ENGLISH + MATH)")
    print("=" * 65)
    print(f"• Context Length (Block Size): {config.block_size} tokens (4k Context! ~3,200 words!)")
    print(f"• Vocabulary Size:            {config.vocab_size:,} (Bangla + English + Math BPE)")
    print(f"• Embedding Dimension:        {config.embed_dim}")
    print(f"• Attention Mechanism:        GQA (Grouped-Query Attention, 3:1 ratio)")
    print(f"• Query Heads:                {config.num_heads} | Key-Value Heads: {config.num_kv_heads}")
    print(f"• Positional Embedding:       RoPE (Rotary Position Embeddings)")
    print(f"• Transformer Layers:         {config.num_layers} blocks")
    print(f"• Intermediate Expansion:     {config.intermediate_dim} (4x)")
    print(f"• Effective Batch Size:       {config.batch_size * config.gradient_accumulation_steps} ({config.batch_size} × {config.gradient_accumulation_steps})")
    print("—" * 65)
    print("📊 Parameter Breakdown:")
    print(f"  - Token Embeddings:         {params['token_emb']:,}")
    print(f"  - 14 Transformer Blocks:    {params['all_blocks']:,}")
    print(f"  - Total Parameters (Tied):  {params['total_tied']:,} (~{params['total_tied']/1e6:.2f}M)")
    print(f"  - Total Parameters (Head):  {params['total_untied']:,} (~{params['total_untied']/1e6:.2f}M)")
    print("—" * 65)
    print("⚡ QLoRA & Mobile Memory Specifications:")
    print(f"  - Targeted Projections:     {config.lora_target_modules}")
    print(f"  - LoRA Trainable Params:    {params['lora_params']:,} ({params['lora_percent']:.2f}% of model)")
    print(f"  - Mobile KV-Cache (4096):   ~{params['kv_cache_mb']:.1f} MB (Extremely low due to GQA!)")
    print(f"  - Mobile Q4 GGUF Size:      ~55 - 60 MB")
    print(f"  - Mobile RAM Usage:         ~80 - 95 MB total")
    print("—" * 65)
    print("💾 Colab Free T4 GPU Requirements:")
    print("  - QLoRA VRAM Usage:         ~4.5 - 5.5 GB VRAM (Fits easily in 15GB T4)")
    print("  - Training Time (5k steps): ~3.5 hours on Colab Free T4 GPU")
    print("=================================================================")


def main():
    parser = argparse.ArgumentParser(description="100M Parameter Multilingual GPT Model with 4096 Context")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    subparsers.add_parser("info", help="Display 100M model architecture & parameter statistics")

    p_qlora = subparsers.add_parser("qlora", help="Run 100M QLoRA fine-tuning")
    p_qlora.add_argument("--iters", type=int, default=5000, help="Training iterations")
    p_qlora.add_argument("--batch_size", type=int, default=4, help="Micro batch size")
    p_qlora.add_argument("--corpus", type=str, default="data/corpus.txt", help="Path to corpus text")
    p_qlora.add_argument("--base_checkpoint", type=str, default=None, help="Path to base checkpoint if fine-tuning")

    p_pretrain = subparsers.add_parser("pretrain", help="Run 100M full pretraining from scratch")
    p_pretrain.add_argument("--iters", type=int, default=5000, help="Training iterations")
    p_pretrain.add_argument("--batch_size", type=int, default=4, help="Micro batch size")
    p_pretrain.add_argument("--corpus", type=str, default="data/corpus.txt", help="Path to corpus text")

    p_gen = subparsers.add_parser("generate", help="Generate text from trained model")
    p_gen.add_argument("--prompt", type=str, default="গণিতের মূল নিয়ম হলো", help="Input prompt")
    p_gen.add_argument("--ckpt", type=str, default="checkpoints/qlora_100m_best.pt", help="Checkpoint file")
    p_gen.add_argument("--tokens", type=int, default=150, help="Max tokens to generate")

    subparsers.add_parser("chat", help="Start interactive conversational chat")

    args = parser.parse_args()

    if args.command == "info" or args.command is None:
        print_info()
    elif args.command == "qlora":
        cfg = GPTConfig()
        cfg.max_iters = args.iters
        cfg.batch_size = args.batch_size
        train_qlora(cfg, corpus_path=args.corpus, base_checkpoint=args.base_checkpoint)
    elif args.command == "pretrain":
        cfg = GPTConfig()
        cfg.max_iters = args.iters
        cfg.batch_size = args.batch_size
        train_pretrain(cfg, corpus_path=args.corpus)
    elif args.command == "generate":
        generate_text(prompt=args.prompt, base_checkpoint=args.ckpt, max_new_tokens=args.tokens)
    elif args.command == "chat":
        chat_interactive()


if __name__ == "__main__":
    main()
