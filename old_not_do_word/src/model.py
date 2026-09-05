"""
Custom Transformer Language Model Implementation
~100M parameters decoder-only architecture
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class CustomLMConfig(PretrainedConfig):
    """Configuration class for CustomLM model."""
    
    model_type = "custom_lm"
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache
        self._set_cos_sin_cache(max_seq_len)
    
    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(1)  # [seq_len, 1, dim]
    sin = sin.unsqueeze(1)  # [seq_len, 1, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function for feed-forward network."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
    
    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking and RoPE."""
    
    def __init__(self, config: CustomLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = config.attention_dropout
        
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        kv_seq_len = k.shape[2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]
        
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Handle past key values for caching
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        past_key_value = (k, v) if use_cache else None
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value


class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-norm formulation."""
    
    def __init__(self, config: CustomLMConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, present_key_value = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(mlp_output)
        
        return hidden_states, present_key_value


class CustomLMModel(PreTrainedModel):
    """
    Custom Language Model - Decoder-only Transformer
    ~100M parameters for Bengali language modeling
    """
    
    config_class = CustomLMConfig
    
    def __init__(self, config: CustomLMConfig):
        super().__init__(config)
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights between embedding and output
        self.lm_head.weight = self.embed_tokens.weight
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def _prepare_causal_attention_mask(
        self,
        batch_size: int,
        seq_len: int,
        past_key_values_length: int = 0,
        device: torch.device = None,
    ):
        """Create causal attention mask."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        
        if past_key_values_length > 0:
            mask = torch.cat([
                torch.zeros((seq_len, past_key_values_length), device=device),
                mask
            ], dim=-1)
        
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: bool = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get past length
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        # Prepare causal mask
        causal_mask = self._prepare_causal_attention_mask(
            batch_size,
            seq_len,
            past_length,
            device
        )
        
        # Pass through transformer blocks
        present_key_values = () if use_cache else None
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present = layer(
                hidden_states,
                attention_mask=causal_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                present_key_values += (present,)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,) + (present_key_values,) if use_cache else (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values,
            hidden_states=None,
            attentions=None,
        )
    
    def generate_text(
        self,
        tokenizer,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        device: str = "cuda",
    ):
        """Generate text from a prompt."""
        self.eval()
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(input_ids, use_cache=False)
                logits = outputs.logits
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Check for EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode and return
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total_params,
            "trainable": trainable_params,
            "total_millions": total_params / 1e6,
            "trainable_millions": trainable_params / 1e6,
        }


def create_model(config_dict: dict = None):
    """Helper function to create model from config dictionary."""
    if config_dict is None:
        config = CustomLMConfig()
    else:
        config = CustomLMConfig(**config_dict)
    
    model = CustomLMModel(config)
    params = model.count_parameters()
    
    print(f"Model created successfully!")
    print(f"Total parameters: {params['total_millions']:.2f}M")
    print(f"Trainable parameters: {params['trainable_millions']:.2f}M")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    model = create_model()
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output logits shape: {outputs.logits.shape}")
        print(f"Expected shape: ({batch_size}, {seq_len}, 32000)")
    
    print("\nModel test passed! ✓")
