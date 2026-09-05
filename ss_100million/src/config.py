"""
Configuration for 100M Parameter Multilingual (Bangla + English + Math) GPT Model.
Features: 4096 Context Length (4k Context), GQA (Grouped-Query Attention 3:1), RoPE, and 10k Vocab.
"""

import torch


class GPTConfig:
    """Model & Training hyperparameters for 100M Parameter Model with 4096 context length."""

    # Architecture (~96M to 103.5M parameters)
    vocab_size    = 10000      # 10,000-Vocab BPE Subword Tokenizer (Bangla + English + Math)
    embed_dim     = 768        # Hidden dimension
    num_heads     = 12         # Query attention heads (768 / 12 = 64 head_dim)
    num_kv_heads  = 4          # Key-Value heads (GQA 3:1 ratio -> cuts mobile KV-Cache by 66.7%!)
    num_layers    = 14         # 14 Transformer layers
    intermediate_dim = 3072    # 4x MLP expansion (768 * 4)
    block_size    = 4096       # 4096 Context Length (4k Context! holds ~3,200 words!)
    dropout       = 0.1
    bias          = False

    # RoPE (Rotary Position Embeddings)
    rope_theta    = 10000.0    # Base frequency for RoPE

    # Training Hyperparameters (Tuned for Colab Free T4 GPU)
    learning_rate = 3e-4
    min_lr        = 3e-5
    batch_size    = 4          # Micro-batch size (4 sequences of 4096 tokens)
    gradient_accumulation_steps = 4  # Effective batch size = 16 (4 * 4)
    max_iters     = 5000       # Total training iterations
    warmup_iters  = 250        # Warmup steps
    eval_interval = 500        # Evaluate every 500 steps
    eval_iters    = 20
    weight_decay  = 0.1
    beta1, beta2  = 0.9, 0.95
    grad_clip     = 1.0

    # LoRA fine-tuning parameters (Scaled for 100M)
    lora_rank           = 16
    lora_alpha          = 32
    lora_dropout        = 0.05
    lora_target_modules = ['q_proj', 'v_proj']

    # Checkpoint settings (Drive synced)
    checkpoint_dir  = 'checkpoints'
    save_interval   = 500

    # System & Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed   = 42

    # Generation defaults
    temperature    = 0.75
    top_k          = 40
    max_new_tokens = 250

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

        # LoRA parameters across target projections
        # q_proj: 2 * lora_rank * embed_dim; v_proj: 2 * lora_rank * (num_kv_heads * head_dim)
        lora_params = cls.num_layers * (
            (2 * cls.lora_rank * cls.embed_dim) +
            (cls.lora_rank * cls.embed_dim + cls.lora_rank * (cls.num_kv_heads * head_dim))
        )

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
