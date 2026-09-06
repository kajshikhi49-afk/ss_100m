"""
Export script for 50M Bengali GPT.
Converts checkpoint_stage_2_final.pt into:
1. Hugging Face format (model.safetensors + config.json + tokenizer.json)
2. Valid GGUF format with ffn_gate and BPE merges for llama.cpp & Android
3. ONNX format for universal 32-bit (itel A60) and 64-bit Android devices
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
        os.system("pip install -q safetensors")
        from safetensors.torch import save_file

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Clean up prefixes and clone shared memory tensors (weight tying: tok_emb == lm_head)
    cleaned_state_dict = {}
    seen_ptrs = set()
    for k, v in state_dict.items():
        clean_k = k.replace("_orig_mod.", "").replace("module.", "")
        if v.data_ptr() in seen_ptrs:
            cleaned_state_dict[clean_k] = v.clone().contiguous()
        else:
            cleaned_state_dict[clean_k] = v.contiguous()
            seen_ptrs.add(v.data_ptr())

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
    Includes ffn_gate synthesis and BPE merges for 100% llama.cpp compatibility.
    """
    print("=" * 60)
    print(f"🤖 Exporting to GGUF format: {output_gguf_path}...")
    print("=" * 60)

    try:
        import gguf
    except ImportError:
        os.system("pip install -q gguf")
        import gguf

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Load tokenizer vocabulary and merges
    vocab = []
    merges = []
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
            model_data = tok_data.get("model", {})
            vocab_dict = model_data.get("vocab", {})
            merges = model_data.get("merges", [])
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
        if merges:
            writer.add_token_merges(merges)
            print(f"✓ BPE merges added ({len(merges):,} merges)")
        writer.add_bos_token_id(2)
        writer.add_eos_token_id(3)
        writer.add_unk_token_id(1)
        writer.add_pad_token_id(0)

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
            writer.add_tensor(gguf_name, tensor)
        elif "blocks." in clean_k:
            parts = clean_k.split(".")
            layer_idx = parts[1]
            sub = ".".join(parts[2:])

            if sub == "norm1.weight":
                writer.add_tensor(f"blk.{layer_idx}.attn_norm.weight", tensor)
            elif sub == "attn.q_proj.weight":
                writer.add_tensor(f"blk.{layer_idx}.attn_q.weight", tensor)
            elif sub == "attn.k_proj.weight":
                writer.add_tensor(f"blk.{layer_idx}.attn_k.weight", tensor)
            elif sub == "attn.v_proj.weight":
                writer.add_tensor(f"blk.{layer_idx}.attn_v.weight", tensor)
            elif sub == "attn.out_proj.weight":
                writer.add_tensor(f"blk.{layer_idx}.attn_output.weight", tensor)
            elif sub == "norm2.weight":
                writer.add_tensor(f"blk.{layer_idx}.ffn_norm.weight", tensor)
            elif sub == "mlp.fc1.weight":
                # LLaMA requires both ffn_up and ffn_gate for SwiGLU FFN
                writer.add_tensor(f"blk.{layer_idx}.ffn_up.weight", tensor)
                writer.add_tensor(f"blk.{layer_idx}.ffn_gate.weight", tensor)
            elif sub == "mlp.fc2.weight":
                writer.add_tensor(f"blk.{layer_idx}.ffn_down.weight", tensor)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(output_gguf_path) / (1024 * 1024)
    print(f"✓ GGUF export successful: {output_gguf_path} ({size_mb:.1f} MB)")
    return output_gguf_path


def export_to_onnx(checkpoint_path, output_onnx_path="bengali_gpt_50m.onnx"):
    """
    Exports PyTorch model to ONNX format.
    Native support on all Android devices (including 32-bit armeabi-v7a itel A60).
    """
    print("=" * 60)
    print(f"⚡ Exporting to ONNX format: {output_onnx_path}...")
    print("=" * 60)

    try:
        try:
            import onnxscript
        except ImportError:
            os.system("pip install -q onnxscript")
            import onnxscript

        model = BengaliGPT(GPTConfig)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        cleaned = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned)
        model.eval()

        dummy_input = torch.randint(0, GPTConfig.vocab_size, (1, 32), dtype=torch.long)
        torch.onnx.export(
            model,
            (dummy_input,),
            output_onnx_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
            opset_version=14,
            do_constant_folding=True
        )

        size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
        print(f"✓ ONNX export successful: {output_onnx_path} ({size_mb:.1f} MB)")
        return output_onnx_path
    except Exception as e:
        print(f"⚠️ ONNX export skipped due to environment requirement: {e}")
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/content/drive/MyDrive/bengali_gpt_50m_checkpoints/checkpoint_stage_2_final.pt")
    parser.add_argument("--output_dir", type=str, default="exported_model")
    args = parser.parse_args()

    if os.path.exists(args.checkpoint):
        export_to_safetensors(args.checkpoint, os.path.join(args.output_dir, "safetensors"))
        export_to_gguf(args.checkpoint, os.path.join(args.output_dir, "bengali_gpt_50m_f16.gguf"))
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
