# Quick Start Guide

Get your 100M parameter Bengali language model running in minutes!

## 🚀 Choose Your Path

### Option A: Google Colab (Recommended for Beginners)
**Time: 5 minutes setup + 4-8 hours training**

1. Open the notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/bengali-lm-100m/blob/main/colab_notebook.ipynb)

2. Run all cells (Runtime → Run all)

3. Wait for training to complete

4. Download your trained model

**That's it!** Everything is automated in the notebook.

---

### Option B: Local Machine
**Time: 10 minutes setup + variable training time**

#### Prerequisites
- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 10GB+ free disk space

#### 5-Minute Setup

```bash
# 1. Clone and enter directory
git clone https://github.com/yourusername/bengali-lm-100m.git
cd bengali-lm-100m

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# Done! Ready to train.
```

#### Add Your Data

```bash
# Place Bengali text files (.txt) in data/raw/
# Example:
cp your_bengali_book.txt data/raw/
```

#### Train in 3 Commands

```bash
# 1. Train tokenizer (2-5 minutes)
python scripts/train_tokenizer.py

# 2. Prepare data (5-10 minutes)
python scripts/prepare_data.py

# 3. Start training (4-8 hours on T4 GPU)
python scripts/pretrain.py
```

#### Monitor Training

Watch for these signs of good training:
- Loss decreasing steadily
- No "Out of Memory" errors
- Checkpoints being saved every 500 steps

#### Test Your Model

```bash
# Generate Bengali text
python scripts/generate.py \
    --model_path outputs/pretrain/best_model \
    --prompt "একবার এক"
```

---

## 🎯 What Happens Behind the Scenes

```
Your Text Files
       ↓
[Train Tokenizer] → Converts text to numbers
       ↓
[Prepare Data] → Creates training batches
       ↓
[Pretrain Model] → Learns language patterns (takes hours)
       ↓
[Trained Model] → Can generate Bengali text!
```

---

## 📊 Expected Results

### With 10MB of text:
- Loss: 5-7
- Quality: Basic sentences
- Recommendation: Get more data

### With 100MB of text:
- Loss: 3-5
- Quality: Coherent paragraphs
- Recommendation: Good for testing

### With 1GB+ of text:
- Loss: 2-4
- Quality: High-quality generation
- Recommendation: Production-ready

---

## 🔧 Common Issues & Solutions

### "CUDA out of memory"
```bash
# Edit configs/model_config.yaml
# Change: batch_size: 16 (instead of 32)
# Change: gradient_accumulation_steps: 8 (instead of 4)
```

### "No text files found"
```bash
# Make sure files are in data/raw/ with .txt extension
ls data/raw/*.txt  # Should show your files
```

### "Tokenizer not found"
```bash
# Train tokenizer first
python scripts/train_tokenizer.py
```

### Training is slow
**Normal speeds:**
- CPU: Very slow (days), not recommended
- T4 GPU: 500-1000 tokens/second
- V100 GPU: 2000-3000 tokens/second
- A100 GPU: 5000+ tokens/second

---

## 🎓 Next Steps After Training

### 1. Fine-tune for Instructions
```bash
# Create instruction data (see data/examples/sft_data.jsonl)
python scripts/sft.py --model_path outputs/pretrain/best_model
```

### 2. Export for Android
```bash
python scripts/export.py --model_path outputs/pretrain/best_model
```

### 3. Improve Your Model
- Add more diverse training data
- Train for more epochs
- Tune hyperparameters in `configs/model_config.yaml`

---

## 💡 Pro Tips

1. **Start Small**: Test with a small dataset first to verify everything works

2. **Use Colab for First Try**: It's free and requires zero setup

3. **Monitor GPU Memory**: Run `nvidia-smi` to check usage

4. **Save Checkpoints**: Training is long, checkpoints let you resume

5. **Experiment**: Try different generation parameters (temperature, top_k, top_p)

---

## 📚 Learn More

- **Full Documentation**: See [README.md](README.md)
- **Data Guide**: See [data/README.md](data/README.md)
- **Project Details**: See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- **Code Examples**: Check the Colab notebook

---

## 🆘 Need Help?

1. **Check the logs**: Error messages are usually informative
2. **Read the README**: Comprehensive troubleshooting section
3. **Review the notebook**: Working example with explanations
4. **Open an issue**: On GitHub if you're stuck

---

## ⏱️ Time Investment

| Task | Time Required |
|------|--------------|
| Setup | 5-10 minutes |
| Tokenizer training | 2-5 minutes |
| Data preparation | 5-10 minutes |
| Model pretraining | 4-8 hours |
| Fine-tuning (optional) | 1-2 hours |
| Export | 5-10 minutes |
| **Total** | **~5-11 hours** |

**Most time is training - you can do other things while it runs!**

---

## 🎯 Minimum Dataset Requirements

For meaningful results:
- **Pretraining**: 50MB+ of Bengali text (100-1000+ MB recommended)
- **Fine-tuning**: 100+ instruction-response pairs (1000+ recommended)

**Don't have data?**
```python
# Download from Hugging Face
from datasets import load_dataset
dataset = load_dataset("oscar", "unshuffled_deduplicated_bn", split="train[:10000]")

# Save to file
with open('data/raw/oscar.txt', 'w', encoding='utf-8') as f:
    for item in dataset:
        f.write(item['text'] + '\n')
```

---

## ✅ Success Checklist

- [ ] Environment set up (Colab or local)
- [ ] Training data added to data/raw/
- [ ] Tokenizer trained successfully
- [ ] Data prepared and split
- [ ] Model training started
- [ ] Training loss decreasing
- [ ] Model generates coherent text
- [ ] Model exported (optional)

---

## 🎉 You're Ready!

Start with the Colab notebook or run the 3 commands above. In a few hours, you'll have your own Bengali language model!

**Good luck! 🚀**

---

*Questions? Check the full [README.md](README.md) or [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)*
