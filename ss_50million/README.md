# 🇧🇩 50 Million Parameter Bengali GPT Model with 2048 Context Length & QLoRA

A cutting-edge, self-contained **50 Million parameter GPT-style Bengali language model** built from scratch in pure PyTorch.
Featuring a **2048-token context length** (~1,600 Bengali words), **GQA (Grouped-Query Attention)** for a 75% reduction in mobile KV-cache memory, **RoPE (Rotary Position Embeddings)**, and custom **QLoRA (Quantized Low-Rank Adaptation)** fine-tuning.

---

## 🎯 Architectural Highlights

- **2048 Context Length:** Holds up to **~1,600 full Bengali words** (5-6 pages of text or long multi-turn dialogues).
- **~49.2M - 50M Parameters:** Deep 16-layer Transformer with 512 embedding dimension and 2048 intermediate MLP expansion.
- **GQA (Grouped-Query Attention):** 8 Query heads with 2 Key-Value heads (4:1 ratio). **Reduces mobile KV-cache RAM footprint from 64 MB down to only 16 MB!**
- **RoPE (Rotary Position Embeddings):** Parameter-free rotational position encoding supporting dynamic length extrapolation (up to 4096 / 8192).
- **True 5,000-Vocab BPE Tokenizer:** Learns real Bengali words, compound letters (যুক্তবর্ণ), and subwords from your training corpus.
- **Custom LoRA / QLoRA from Scratch:** Only **262,144 trainable parameters (~0.53%)**, training in under 2.5 hours on Colab's Free T4 GPU (~3.5 GB VRAM).
- **Ultra-Light Mobile Offline Deployment:**
  - **Q4_K_M GGUF Size:** **~28 to 30 MB**!
  - **Total Phone RAM Usage:** **~45 to 55 MB** (Runs like butter on a 2GB RAM itel phone!).

---

## 📁 Project Structure

```
ss_10m/
├── data/
│   └── corpus.txt          # Training text data (Bangla educational & conversational text)
├── src/
│   ├── config.py           # 50M & 2048 Context Hyperparameters (GQA + RoPE)
│   ├── tokenizer.py        # 5,000-Vocab BPE Subword Tokenizer
│   ├── model.py            # GQA + RoPE + FlashAttention Decoder-Only Transformer
│   ├── dataset.py          # 2048-token batch loader
│   ├── lora.py             # Custom LoRA & QLoRA implementation from scratch
│   ├── train.py            # Training pipeline with Gradient Accumulation & Auto-Resume
│   └── generate.py         # Text generation & Proactive Socratic Chat
├── checkpoints/            # Saved checkpoints & LoRA adapters (Drive-synced)
├── tokenizer.json          # Pre-trained 5,000 BPE vocabulary file
├── requirements.txt        # Minimal dependencies
├── train_10m_bengali_gpt_colab.ipynb  # Rebuilt Colab Notebook (50M + 2048 Context)
├── README.md               # Project documentation
└── run.py                  # CLI entry point
```

---

## 📊 Model Specifications & Memory Breakdown

| Component | Dimensions / Details | Parameters / Memory |
| :--- | :--- | :--- |
| **Vocabulary Size** | 5,000 BPE Subwords | 2,560,000 |
| **Context Length (`block_size`)** | 2048 tokens (~1,600 Bengali words) | Parameter-Free (RoPE) |
| **Hidden Dimension (`embed_dim`)** | 512 | — |
| **Attention Mechanism** | Grouped-Query Attention (GQA) | 8 Query heads, 2 KV heads |
| **Transformer Blocks** | 16 layers | 44,056,576 |
| **Total Base Parameters (Tied)** | 16 Blocks + Embeddings | **46,617,088 (~46.6M)** |
| **Total Base Parameters (Untied)**| 16 Blocks + Embeddings + Head | **49,177,088 (~49.2M)** |
| **Trainable QLoRA Parameters** | `q_proj` & `v_proj` (rank 8, alpha 16) | **262,144 (0.53%)** |
| **Mobile KV-Cache Memory (2048)** | 16 layers × 2 KV heads × 64 dim | **~16.0 MB (Ultra-low!)** |
| **Mobile Model Weight (Q4 GGUF)**| 4-bit Quantized | **~28 - 30 MB** |
| **Colab Free T4 GPU VRAM** | 16 Effective Batch Size | **~3.5 - 4.5 GB (15 GB available)** |

---

## 🚀 Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Inspect 50M model architecture & parameters
python run.py info

# 3. Option 1: Run 50M QLoRA Fine-Tuning (~2.5 hours on T4 GPU)
python run.py qlora --iters 5000 --batch_size 8

# 4. Option 2: Full Pretraining from Scratch
python run.py pretrain --iters 5000 --batch_size 8

# 5. Interactive Proactive Socratic Chat Mode
python run.py chat

# 6. Single prompt text generation
python run.py generate --prompt "বাংলাদেশের রাজধানী"
```
