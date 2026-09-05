"""
Modern Decoder-Only Transformer with:
- RoPE (Rotary Position Embeddings)
- GQA (Grouped-Query Attention for 75% KV-cache reduction)
- RMSNorm
- PyTorch SDPA (FlashAttention)
- 2048 Context Length (~50 Million parameters)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import GPTConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def apply_rotary_emb(x, cos, sin):
    B, H, T, D = x.shape
    x1 = x[..., : D // 2]
    x2 = x[..., D // 2 :]
    cos = cos[:T, :].unsqueeze(0).unsqueeze(1)
    sin = sin[:T, :].unsqueeze(0).unsqueeze(1)
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.cat([rx1, rx2], dim=-1)


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x.unsqueeze(2).expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_heads
        self.n_kv_heads = config.num_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.embed_dim, self.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.embed_dim, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.embed_dim, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, config.embed_dim, bias=config.bias)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.intermediate_dim, bias=config.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_dim, config.embed_dim, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.embed_dim)
        self.attn = GroupedQueryAttention(config)
        self.norm2 = RMSNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class BengaliGPT(nn.Module):
    """50 Million Parameter Bengali Model with 2048 Context Length."""
    def __init__(self, config=None):
        super().__init__()
        if config is None: config = GPTConfig()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm_f = RMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.tok_emb.weight = self.lm_head.weight

        head_dim = config.embed_dim // config.num_heads
        cos, sin = precompute_freqs_cis(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds {self.config.block_size}"

        cos = self.freqs_cos[:T].to(idx.device)
        sin = self.freqs_sin[:T].to(idx.device)

        x = self.drop(self.tok_emb(idx))

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return None, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=250, temperature=0.75, top_k=40, repetition_penalty=1.2, eos_id=3):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # Repetition penalty প্রয়োগ
            if repetition_penalty != 1.0:
                for b in range(logits.shape[0]):
                    for token_id in set(idx[b].tolist()):
                        logits[b, token_id] /= repetition_penalty

            logits = logits / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            next_token = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, next_token], dim=1)

            # EOS টোকেন আসলে জেনারেশন থামানো
            if eos_id is not None and (next_token == eos_id).all():
                break

        return idx


# Alias for backward compatibility and notebook imports
GPT = BengaliGPT

