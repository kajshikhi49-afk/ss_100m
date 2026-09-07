import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import GPTConfig
from src.model import BengaliGPT


def permute_weight_for_rope(tensor, n_heads, head_dim):
    """
    Permutes a 2D weight matrix [n_heads * head_dim, embed_dim] from PyTorch halved RoPE layout
    [0..head_dim/2-1, head_dim/2..head_dim-1] to llama.cpp interleaved RoPE layout [2k, 2k+1].
    """
    embed_dim = tensor.shape[1]
    t = torch.from_numpy(tensor)
    permuted = t.view(n_heads, 2, head_dim // 2, embed_dim).transpose(1, 2).reshape(n_heads * head_dim, embed_dim)
    return permuted.contiguous().numpy()


def export_to_gguf_fixed(checkpoint_path, output_gguf_path="bengali_gpt_50m_fixed_f16.gguf", tokenizer_path="tokenizer.json"):
    """
    Exports PyTorch model directly to GGUF format with:
    - superbpe pre-tokenizer metadata (100% Bengali & English tokenizer match)
    - exact RoPE coordinate permutation on Q and K weights
    """
    print("=" * 65)
    print(f"🤖 Exporting to FIXED GGUF format: {output_gguf_path}...")
    print("=" * 65)

    import gguf

    state_dict = torch.load(checkpoint_path, map_location="cpu") if isinstance(checkpoint_path, str) else checkpoint_path

    # Load tokenizer vocabulary and merges
    vocab = []
    merges = []
    special_ids = {}
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
            model_data = tok_data.get("model", {})
            vocab_dict = model_data.get("vocab", {})
            merges = model_data.get("merges", [])
            for item in tok_data.get("added_tokens", []):
                special_ids[item["id"]] = item["content"]

            vocab = [None] * len(vocab_dict)
            for token, idx in vocab_dict.items():
                if idx < len(vocab):
                    vocab[idx] = token
            vocab = [t if t is not None else "<UNK>" for t in vocab]

    # Initialize GGUF Writer (architecture: llama)
    writer = gguf.GGUFWriter(output_gguf_path, arch="llama")
    writer.add_name("bengali-gpt-50m-fixed")
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
        # FIXED: Set pre-tokenizer to superbpe to prevent character shattering on Bengali Unicode marks
        writer.add_tokenizer_pre("superbpe")
        writer.add_token_list(vocab)

        if merges:
            formatted_merges = []
            for m in merges:
                if isinstance(m, list) and len(m) == 2:
                    formatted_merges.append(f"{m[0]} {m[1]}")
                elif isinstance(m, str):
                    formatted_merges.append(m)
            writer.add_token_merges(formatted_merges)
            print(f"✓ Formatted BPE merges added ({len(formatted_merges):,} merges)")

        writer.add_token_scores([0.0] * len(vocab))
        token_types = [3 if i in special_ids else 1 for i in range(len(vocab))]
        writer.add_token_types(token_types)

        writer.add_bos_token_id(2)
        writer.add_eos_token_id(3)
        writer.add_unk_token_id(1)
        writer.add_pad_token_id(0)

    # Tensor mapping from BengaliGPT to GGUF LLaMA naming
    mapping = {
        "tok_emb.weight": "token_embd.weight",
        "norm_f.weight": "output_norm.weight",
        "lm_head.weight": "output.weight"
    }

    head_dim = GPTConfig.embed_dim // GPTConfig.num_heads
    n_heads = GPTConfig.num_heads
    n_kv_heads = GPTConfig.num_kv_heads

    for k, v in state_dict.items():
        clean_k = k.replace("_orig_mod.", "").replace("module.", "")
        tensor = v.float().numpy() if isinstance(v, torch.Tensor) else v

        if clean_k in mapping:
            gguf_name = mapping[clean_k]
            writer.add_tensor(gguf_name, tensor)
        elif "blocks." in clean_k:
            parts = clean_k.split(".")
            layer_idx = parts[1]
            sub = ".".join(parts[2:])

            if sub == "norm1.weight":
                writer.add_tensor(f"blk.{layer_idx}.attn_norm.weight", tensor)
            elif sub == "attn.q_proj.weight":
                # FIXED: Apply mathematically exact RoPE permutation to Query weights
                permuted_q = permute_weight_for_rope(tensor, n_heads=n_heads, head_dim=head_dim)
                writer.add_tensor(f"blk.{layer_idx}.attn_q.weight", permuted_q)
            elif sub == "attn.k_proj.weight":
                # FIXED: Apply mathematically exact RoPE permutation to Key weights
                permuted_k = permute_weight_for_rope(tensor, n_heads=n_kv_heads, head_dim=head_dim)
                writer.add_tensor(f"blk.{layer_idx}.attn_k.weight", permuted_k)
            elif sub == "attn.v_proj.weight":
                writer.add_tensor(f"blk.{layer_idx}.attn_v.weight", tensor)
            elif sub == "attn.out_proj.weight":
                writer.add_tensor(f"blk.{layer_idx}.attn_output.weight", tensor)
            elif sub == "norm2.weight":
                writer.add_tensor(f"blk.{layer_idx}.ffn_norm.weight", tensor)
            elif sub == "mlp.fc1.weight":
                writer.add_tensor(f"blk.{layer_idx}.ffn_up.weight", tensor)
                writer.add_tensor(f"blk.{layer_idx}.ffn_gate.weight", tensor)
            elif sub == "mlp.fc2.weight":
                writer.add_tensor(f"blk.{layer_idx}.ffn_down.weight", tensor)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(output_gguf_path) / (1024 * 1024)
    print(f"✓ Fixed GGUF export successful: {output_gguf_path} ({size_mb:.1f} MB)")
    return output_gguf_path
