"""
Export script for 50M Bengali GPT.
Converts checkpoint_stage_2_final.pt into:
1. Hugging Face format (model.safetensors + config.json + tokenizer.json)
2. GGUF format for llama.cpp, Android, and offline inference (FP16 and INT4)
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import GPTConfig
from src.model import BengaliGPT


def export_to_safetensors(checkpoint_path, output_dir="exported_model_hf"):
    """Converts PyTorch checkpoint (.pt) to Hugging Face safetensors format."""
    print("=" * 60)
    print(f"📦 Converting {checkpoint_path} to Safetensors...")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("Installing safetensors...")
        os.system("pip install -q safetensors")
        from safetensors.torch import save_file

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Clean up prefixes if any
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("_orig_mod.", "").replace("module.", "")
        cleaned_state_dict[clean_k] = v.contiguous()

    # Save model.safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(cleaned_state_dict, safetensors_path)
    
    # Save config.json
    config_dict = {
        "architectures": ["BengaliGPT"],
        "model_type": "bengali_gpt",
        "vocab_size": GPTConfig.vocab_size,
        "hidden_size": GPTConfig.embed_dim,
        "intermediate_size": GPTConfig.intermediate_dim,
        "num_hidden_layers": GPTConfig.num_layers,
        "num_attention_heads": GPTConfig.num_heads,
        "num_key_value_heads": GPTConfig.num_kv_heads,
        "max_position_embeddings": GPTConfig.block_size,
        "rope_theta": GPTConfig.rope_theta,
        "torch_dtype": "float16"
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Copy tokenizer if exists
    if os.path.exists("tokenizer.json"):
        import shutil
        shutil.copy("tokenizer.json", os.path.join(output_dir, "tokenizer.json"))

    size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print(f"✓ Safetensors export successful!")
    print(f"  File: {safetensors_path} ({size_mb:.1f} MB)")
    print(f"  Config: {os.path.join(output_dir, 'config.json')}")
    return safetensors_path


def export_to_gguf(checkpoint_path, output_gguf_path="bengali_gpt_50m_f16.gguf", tokenizer_path="tokenizer.json"):
    """
    Exports PyTorch model directly to GGUF format using python gguf writer.
    Compatible with llama.cpp, Android (llama.cpp JNI/Flutter), and desktop.
    """
    print("=" * 60)
    print(f"🤖 Exporting to GGUF format: {output_gguf_path}...")
    print("=" * 60)

    try:
        import gguf
    except ImportError:
        print("Installing gguf package...")
        os.system("pip install -q gguf")
        import gguf

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Load tokenizer vocabulary
    vocab = []
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
            model_data = tok_data.get("model", {})
            vocab_dict = model_data.get("vocab", {})
            # Sort tokens by their ID
            vocab = [None] * len(vocab_dict)
            for token, idx in vocab_dict.items():
                if idx < len(vocab):
                    vocab[idx] = token
            vocab = [t if t is not None else "<UNK>" for t in vocab]

    # Initialize GGUF Writer (architecture: llama standard)
    writer = gguf.GGUFWriter(output_gguf_path, arch="llama")
    writer.add_name("bengali-gpt-50m")
    writer.add_context_length(GPTConfig.block_size)
    writer.add_embedding_length(GPTConfig.embed_dim)
    writer.add_block_count(GPTConfig.num_layers)
    writer.add_feed_forward_length(GPTConfig.intermediate_dim)
    writer.add_head_count(GPTConfig.num_heads)
    writer.add_head_count_kv(GPTConfig.num_kv_heads)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(GPTConfig.rope_theta)

    if vocab:
        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(vocab)

    # Tensor mapping from BengaliGPT to GGUF LLaMA standard naming
    mapping = {
        "tok_emb.weight": "token_embd.weight",
        "norm_f.weight": "output_norm.weight",
        "lm_head.weight": "output.weight"
    }

    for k, v in state_dict.items():
        clean_k = k.replace("_orig_mod.", "").replace("module.", "")
        tensor = v.float().numpy()

        if clean_k in mapping:
            gguf_name = mapping[clean_k]
        elif "blocks." in clean_k:
            # Format: blocks.{i}.norm1.weight -> blk.{i}.attn_norm.weight
            parts = clean_k.split(".")
            layer_idx = parts[1]
            sub = ".".join(parts[2:])

            sub_map = {
                "norm1.weight": f"blk.{layer_idx}.attn_norm.weight",
                "attn.q_proj.weight": f"blk.{layer_idx}.attn_q.weight",
                "attn.k_proj.weight": f"blk.{layer_idx}.attn_k.weight",
                "attn.v_proj.weight": f"blk.{layer_idx}.attn_v.weight",
                "attn.out_proj.weight": f"blk.{layer_idx}.attn_output.weight",
                "norm2.weight": f"blk.{layer_idx}.ffn_norm.weight",
                "mlp.fc1.weight": f"blk.{layer_idx}.ffn_up.weight",
                "mlp.fc2.weight": f"blk.{layer_idx}.ffn_down.weight"
            }
            gguf_name = sub_map.get(sub, None)
        else:
            gguf_name = None

        if gguf_name is not None:
            writer.add_tensor(gguf_name, tensor)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(output_gguf_path) / (1024 * 1024)
    print(f"✓ GGUF export successful: {output_gguf_path} ({size_mb:.1f} MB)")
    return output_gguf_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/content/drive/MyDrive/bengali_gpt_50m_checkpoints/checkpoint_stage_2_final.pt")
    parser.add_argument("--output_dir", type=str, default="exported_model")
    args = parser.parse_args()

    if os.path.exists(args.checkpoint):
        export_to_safetensors(args.checkpoint, os.path.join(args.output_dir, "safetensors"))
        export_to_gguf(args.checkpoint, os.path.join(args.output_dir, "bengali_gpt_50m.gguf"))
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
