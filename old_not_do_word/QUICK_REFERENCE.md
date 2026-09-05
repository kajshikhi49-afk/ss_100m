# 🚀 Quick Reference - Qwen2.5-0.5B Fine-tuning

## ⚡ Fast Start (5 Steps)

### 1️⃣ Download from GitHub
```
📂 Download these files:
✅ colab_qwen_finetune.ipynb
✅ data/nctb_dataset_500plus.jsonl
```
**Link**: https://github.com/kajshikhi49-afk/ss_100m

---

### 2️⃣ Upload Dataset to Google Drive
```
📁 Google Drive
  └── MyDrive/
      └── nctb_dataset_500plus.jsonl  ← Put here
```

---

### 3️⃣ Open Colab & Enable GPU
1. Go to: https://colab.research.google.com
2. Upload notebook: `colab_qwen_finetune.ipynb`
3. **Runtime** → **Change runtime type** → **T4 GPU** ✅

---

### 4️⃣ Update Dataset Path (Cell 4)
```python
DATASET_PATH = "/content/drive/MyDrive/nctb_dataset_500plus.jsonl"
```
Change if your file is in a different location.

---

### 5️⃣ Run All Cells
Click **Runtime** → **Run all**

**Expected Time**: 30-40 minutes ⏱️

---

## 📊 What You'll See

### ✅ Successful Output:

**Cell 1**: GPU Info
```
Tesla T4, 15360MiB
```

**Cell 4**: Dataset Loading
```
✅ সফলভাবে লোড হয়েছে: 408 টি
🎉 সব ডেটা পারফেক্ট!
```

**Cell 9**: Training
```
✅ Trainer imports successful!
🚀 ট্রেনিং শুরু হচ্ছে...
Epoch 1/3: 100%|██████| 25/25 [05:23<00:00]
```

**Cell 10**: Model Saved
```
✅ মডেল সফলভাবে সেভ হয়েছে!
```

---

## 🎯 Key Settings

| Setting | Value | Location |
|---------|-------|----------|
| Base Model | Qwen2.5-0.5B-Instruct | Cell 5 |
| Dataset Size | 408 Q&A pairs | Cell 4 |
| Batch Size | 4 | Cell 8 |
| Gradient Accumulation | 4 | Cell 8 |
| Epochs | 3 | Cell 8 |
| Learning Rate | 2e-4 | Cell 8 |
| Quantization | 4-bit (QLoRA) | Cell 5 |

---

## 📂 Output Files

After training, you'll have:

```
📁 Google Drive/MyDrive/
├── qwen_nctb_finetuned/      ← Training checkpoints
│   ├── checkpoint-100/
│   └── checkpoint-200/
└── qwen_nctb_final/           ← Final model ⭐
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files...
```

**Download**: Right-click folder in Drive → Download

---

## 🧪 Testing Your Model

**Cell 11**: Automatic tests with 5 sample questions

**Cell 12**: Custom questions
```python
my_question = "আপনার প্রশ্ন এখানে লিখুন"
```

---

## ⚠️ Common Issues & Solutions

### ❌ No GPU available
**Fix**: Runtime → Change runtime type → T4 GPU

### ❌ File not found
**Fix**: Check `DATASET_PATH` in Cell 4 matches your Drive location

### ❌ CUDA out of memory
**Fix**: Cell 8, change:
```python
per_device_train_batch_size=2  # Was 4
```

### ❌ Training too slow
**Check**: Is GPU enabled? Should see ~12-15 sec/step

---

## 📈 Training Progress

| Time | What's Happening |
|------|------------------|
| 0-5 min | Package installation |
| 5-7 min | Dataset loading & model loading |
| 7-40 min | Training (main time) ⏳ |
| 40-42 min | Model saving |
| 42-45 min | Testing |

**Total**: ~45 minutes

---

## 🎓 Expected Results

### Good Training Signs:
- ✅ Loss decreases: 4.5 → 3.2 → 2.1 → 1.5
- ✅ No errors in any cell
- ✅ Test answers make sense
- ✅ Model saved successfully

### If Loss Doesn't Decrease:
- Dataset might be too small
- Learning rate might be wrong
- Check if GPU is being used

---

## 💡 Pro Tips

### Tip 1: Save Colab Session
File → Save a copy in Drive (so you don't lose progress)

### Tip 2: Monitor Training
Watch the loss value - it should go down over time

### Tip 3: Extend Dataset
Add more Q&A pairs to improve model quality:
```json
{"instruction": "নতুন প্রশ্ন", "output": "উত্তর"}
```

### Tip 4: Use Trained Model
The model in `qwen_nctb_final/` can be used with:
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "qwen_nctb_final/")
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `FINAL_WORKING_SOLUTION.md` | Complete solution overview |
| `COLAB_USAGE_GUIDE.md` | Detailed step-by-step guide (Bengali) |
| `BUGFIX_LOG.md` | All bugs fixed during development |
| `VALIDATION_REPORT.md` | Dataset validation results |
| `QUICK_REFERENCE.md` | This file (quick start) |

---

## 🔗 Important Links

- **GitHub Repo**: https://github.com/kajshikhi49-afk/ss_100m
- **Google Colab**: https://colab.research.google.com
- **Qwen Model**: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

---

## ✅ Checklist Before Starting

- [ ] Downloaded notebook and dataset
- [ ] Uploaded dataset to Google Drive
- [ ] Opened notebook in Colab
- [ ] Enabled T4 GPU
- [ ] Updated DATASET_PATH
- [ ] Have 45 minutes available

**If all checked, you're ready to go! 🚀**

---

## 🎉 Success!

After training, your model can answer:
- ✅ NCTB curriculum questions (Class 1-12)
- ✅ বাংলা grammar and literature
- ✅ English basics
- ✅ গণিত problems
- ✅ বিজ্ঞান concepts
- ✅ বাংলাদেশ ও বিশ্বপরিচয়
- ✅ ICT basics
- ✅ ধর্ম ও নৈতিক শিক্ষা

**Your own Bengali AI tutor! 🎓**

---

**Version**: 4.0 (Final)  
**Status**: ✅ Production Ready  
**Last Updated**: September 3, 2026  
**Created by**: Kiro AI 🤖
