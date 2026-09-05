"""
Configuration for 50M Parameter Bengali GPT Model.
Features: 2048 Context Length, GQA (Grouped-Query Attention), RoPE, and 4-bit QLoRA.
"""

import torch


class GPTConfig:
    """Model & Training hyperparameters for 50M Parameter Bengali Model with 2048 context length."""

    # Architecture (~54.3M parameters)
    vocab_size    = 10000      # 10,000-Vocab ByteLevel BPE Tokenizer (1.8-1.9 tokens/word!)
    embed_dim     = 512        # Hidden dimension
    num_heads     = 8          # Query attention heads (512 / 8 = 64 head_dim)
    num_kv_heads  = 2          # Key-Value heads (GQA 4:1 ratio -> cuts mobile KV-Cache by 75%!)
    num_layers    = 16         # 16 Transformer layers
    intermediate_dim = 2048    # 4x MLP expansion
    block_size    = 2048       # 2048 Context Length (holds ~1,100 Bengali words with 10k vocab!)
    dropout       = 0.1
    bias          = False

    # RoPE (Rotary Position Embeddings)
    rope_theta    = 10000.0    # Base frequency for RoPE

    # Training Hyperparameters (Tuned for Colab Free T4 GPU)
    learning_rate = 3e-4
    min_lr        = 3e-5
    batch_size    = 4          # Micro-batch size (Safe and fast on T4)
    gradient_accumulation_steps = 4  # Effective batch size = 16 (4 * 4)
    max_iters     = 5000       # Training iterations
    warmup_iters  = 250        # Warmup steps
    eval_interval = 500        # Evaluate every 500 steps
    eval_iters    = 25
    weight_decay  = 0.1
    beta1, beta2  = 0.9, 0.95
    grad_clip     = 1.0

    # Enhanced LoRA fine-tuning parameters (4x capacity for fast language learning)
    lora_rank           = 32
    lora_alpha          = 64
    lora_dropout        = 0.05
    lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj']

    # Checkpoint settings (Drive synced)
    checkpoint_dir  = 'checkpoints'
    save_interval   = 500

    # System & Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed   = 42

    # Generation defaults
    temperature    = 0.75
    top_k          = 40
    max_new_tokens = 200

    @classmethod
    def estimate_parameters(cls):
        tok_emb = cls.vocab_size * cls.embed_dim
        head_dim = cls.embed_dim // cls.num_heads

        q_proj = cls.embed_dim * (cls.num_heads * head_dim)
        k_proj = cls.embed_dim * (cls.num_kv_heads * head_dim)
        v_proj = cls.embed_dim * (cls.num_kv_heads * head_dim)
        out_proj = (cls.num_heads * head_dim) * cls.embed_dim
        attn = q_proj + k_proj + v_proj + out_proj

        mlp = 2 * (cls.embed_dim * cls.intermediate_dim)
        norms = 2 * cls.embed_dim
        per_block = attn + mlp + norms
        all_blocks = cls.num_layers * per_block

        final_norm = cls.embed_dim
        lm_head = cls.vocab_size * cls.embed_dim

        total_tied = tok_emb + all_blocks + final_norm
        total_untied = total_tied + lm_head

        lora_params = cls.num_layers * len(cls.lora_target_modules) * (2 * cls.lora_rank * cls.embed_dim)

        kv_cache_bytes = 2 * cls.num_layers * cls.num_kv_heads * head_dim * cls.block_size * 2
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)

        return {
            'token_emb': tok_emb,
            'per_block': per_block,
            'all_blocks': all_blocks,
            'total_tied': total_tied,
            'total_untied': total_untied,
            'lora_params': lora_params,
            'lora_percent': (lora_params / total_untied) * 100,
            'kv_cache_mb': kv_cache_mb,
        }
