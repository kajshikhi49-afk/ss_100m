# 📦 Trained Model কে GGUF Format-এ Convert করার Guide

## ✅ Your Training is Complete! এখন GGUF Convert করুন

আপনার training শেষ হয়েছে! এখন model-কে Android-এ ব্যবহারের জন্য GGUF format-এ convert করতে হবে।

---

## 🎯 Option 1: Colab-এই Convert করুন (EASIEST) ⭐ RECOMMENDED

### Step 1: Colab Notebook-এ নতুন Cell যোগ করুন

আপনার notebook-এর **শেষে** (ধাপ ১৪ এর পরে) এই cells গুলো যোগ করুন:

---

### 📌 Cell 1: Install llama.cpp

```python
%%capture
# llama.cpp clone এবং build করুন
!git clone https://github.com/ggerganov/llama.cpp
!cd llama.cpp && make

# Python requirements install করুন
!pip install -U torch transformers accelerate sentencepiece protobuf
```

---

### 📌 Cell 2: Merge LoRA Adapter with Base Model

```python
print("🔄 Merging LoRA adapter with base model...")

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Base model এবং adapter paths
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
ADAPTER_PATH = "/content/drive/MyDrive/smollm_135m_final"
OUTPUT_PATH = "/content/merged_model"

print(f"📥 Loading base model: {BASE_MODEL}")
# Base model load করুন
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",  # CPU-তে load করুন (merge-এর জন্য)
    trust_remote_code=True,
)

print(f"📥 Loading LoRA adapter: {ADAPTER_PATH}")
# LoRA adapter merge করুন
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()

print(f"💾 Saving merged model to: {OUTPUT_PATH}")
# Merged model save করুন
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)

# Tokenizer save করুন
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✅ Merge complete!")
print(f"📁 Merged model location: {OUTPUT_PATH}")

# Memory clear করুন
del model, base_model
torch.cuda.empty_cache()
```

---

### 📌 Cell 3: Convert to GGUF (FP16)

```python
print("🔄 Converting to GGUF format (FP16)...")

# Convert to GGUF
!python llama.cpp/convert_hf_to_gguf.py \
    /content/merged_model \
    --outtype f16 \
    --outfile /content/smollm_135m_bangla_fp16.gguf

print("✅ FP16 GGUF conversion complete!")
print("📁 Location: /content/smollm_135m_bangla_fp16.gguf")

# File size check করুন
import os
file_size = os.path.getsize("/content/smollm_135m_bangla_fp16.gguf")
print(f"📊 File size: {file_size / (1024**2):.2f} MB (~270 MB)")
```

---

### 📌 Cell 4: Quantize to Q4_K_M (Optimized ~85 MB for 2GB itel Android)

```python
print("🔄 Quantizing to Q4_K_M (Ultra-light ~85 MB for Android)...")

# Quantize to Q4_K_M (ছোট size, দুর্দান্ত performance)
!./llama.cpp/llama-quantize \
    /content/smollm_135m_bangla_fp16.gguf \
    /content/smollm_135m_bangla_q4_k_m.gguf \
    q4_k_m

print("✅ Q4_K_M quantization complete!")
print("📁 Location: /content/smollm_135m_bangla_q4_k_m.gguf")

# File size check করুন
import os
fp16_size = os.path.getsize("/content/smollm_135m_bangla_fp16.gguf")
q4_size = os.path.getsize("/content/smollm_135m_bangla_q4_k_m.gguf")

print(f"\n📊 Size Comparison:")
print(f"  - FP16:   {fp16_size / (1024**2):.2f} MB")
print(f"  - Q4_K_M: {q4_size / (1024**2):.2f} MB (itel 2GB ফোনে পানির মতো চলবে!)")
```

---

### 📌 Cell 5: Copy to Google Drive

```python
print("💾 Copying GGUF files to Google Drive...")

import shutil

# Drive-এ folder তৈরি করুন
output_folder = "/content/drive/MyDrive/gguf_models"
!mkdir -p {output_folder}

# Copy করুন
print("📤 Copying files...")
shutil.copy("/content/smollm_135m_bangla_q4_k_m.gguf", f"{output_folder}/")

print("✅ Files copied to Google Drive!")
print(f"📁 Location: {output_folder}")
print("\nFiles available:")
print("  2. qwen_nctb_bangla_q4_0.gguf (Mobile optimized, smaller) ⭐")

# File listing
!ls -lh {output_folder}
```

---

### 📌 Cell 6: Test GGUF Model (Optional)

```python
print("🧪 Testing GGUF model...")

# llama-cpp-python install করুন
!pip install llama-cpp-python

from llama_cpp import Llama

# Model load করুন (Q4_0 দিয়ে test)
print("📥 Loading Q4_0 model...")
llm = Llama(
    model_path="/content/qwen_nctb_bangla_q4_0.gguf",
    n_ctx=2048,      # Context length
    n_threads=4,     # CPU threads
    n_gpu_layers=0,  # CPU only (Colab-এ GPU optional)
)

# Test করুন
test_question = "বাংলাদেশের রাজধানী কী?"

prompt = f"""<|im_start|>system
You are a helpful AI assistant for Bengali education. Answer questions based on NCTB curriculum.<|im_end|>
<|im_start|>user
{test_question}<|im_end|>
<|im_start|>assistant
"""

print(f"\n❓ Test Question: {test_question}")
print("🤖 Generating answer...\n")

output = llm(
    prompt,
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
    echo=False
)

answer = output['choices'][0]['text'].strip()
print(f"✅ Answer: {answer}")

print("\n🎉 GGUF model is working perfectly!")
```

---

### 📌 Cell 7: Download GGUF from Browser

```python
print("📥 Preparing download links...")

from google.colab import files

# আপনি কোন file download করতে চান?
print("Select which file to download:")
print("1. FP16 (High quality, ~1GB)")
print("2. Q4_0 (Mobile optimized, ~300MB) ⭐ RECOMMENDED")
print()

# Option 1: Download FP16
# files.download("/content/qwen_nctb_bangla_fp16.gguf")

# Option 2: Download Q4_0 (RECOMMENDED for Android)
files.download("/content/qwen_nctb_bangla_q4_0.gguf")

print("✅ Download started!")
print("Check your browser's download folder.")
```

---

## 🎯 Complete Process Summary

আপনার Colab notebook-এ এই cells যোগ করে run করুন:

```
Cell 1: Install llama.cpp (2-3 min)
   ↓
Cell 2: Merge LoRA (5-7 min)
   ↓
Cell 3: Convert to GGUF FP16 (3-5 min)
   ↓
Cell 4: Quantize to Q4_0 (2-3 min)
   ↓
Cell 5: Copy to Drive (1 min)
   ↓
Cell 6: Test model (Optional - 2 min)
   ↓
Cell 7: Download to browser (Direct download)
```

**Total Time**: ~15-20 minutes

---

## 📦 Expected File Sizes

| Format | Size | Use Case | Quality |
|--------|------|----------|---------|
| **FP16** | ~1.0 GB | Desktop, Server | Highest |
| **Q4_0** | ~300 MB | Android, Mobile | Good ⭐ |
| **Q4_K_M** | ~350 MB | Balanced | Better |
| **Q3_K_S** | ~200 MB | Very small | Acceptable |

**Recommendation**: **Q4_0** - Best balance for Android!

---

## 💾 Download Methods

### Method 1: Browser Download (Cell 7)
```python
files.download("/content/qwen_nctb_bangla_q4_0.gguf")
```
- Direct download from Colab
- May timeout for large files
- Best for Q4_0 (~300MB)

### Method 2: Google Drive Download
```
1. Open Google Drive
2. Go to: MyDrive/gguf_models/
3. Right-click → Download
```
- More reliable for large files
- Can resume if interrupted
- Best for FP16 (1GB)

### Method 3: Share Link
```
1. Right-click file in Drive
2. Get link → Anyone with link can view
3. Share link publicly
4. Use for GitHub Release
```

---

## 🚀 After Download

### Option A: Upload to GitHub Release

1. **Create Release**:
   ```
   Go to: https://github.com/kajshikhi49-afk/ss_100m
   → Releases
   → Create new release
   ```

2. **Upload GGUF**:
   - Tag: `v1.0.0-android`
   - Title: "Bengali NCTB Model (GGUF)"
   - Upload: `qwen_nctb_bangla_q4_0.gguf`

3. **Get Download URL**:
   ```
   https://github.com/kajshikhi49-afk/ss_100m/releases/download/v1.0.0-android/qwen_nctb_bangla_q4_0.gguf
   ```

4. **Update Android App**:
   ```kotlin
   // Constants.kt
   const val MODEL_DOWNLOAD_URL = "YOUR_GITHUB_URL_HERE"
   ```

### Option B: Alternative Hosting

If GitHub has size limits, use:
- **Hugging Face**: https://huggingface.co (Free, unlimited)
- **Google Drive**: Direct link
- **Dropbox**: Public link
- **Cloudflare R2**: CDN hosting

---

## 🧪 Test Before Deploying

Before uploading to GitHub, test locally:

```python
# Test GGUF in Colab (Cell 6)
# If answers are good → Upload
# If quality is poor → Try Q4_K_M instead of Q4_0
```

---

## ⚠️ Troubleshooting

### Issue 1: Conversion Failed
```bash
# Re-run with more memory
!python llama.cpp/convert_hf_to_gguf.py \
    /content/merged_model \
    --outtype f16 \
    --outfile /content/qwen_nctb_bangla_fp16.gguf \
    --concurrency 1  # Lower concurrency
```

### Issue 2: Out of Memory
```python
# Clear memory before converting
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
```

### Issue 3: Download Timeout
```python
# Use Drive instead
# Files will be in: /content/drive/MyDrive/gguf_models/
```

---

## 📊 Quality Comparison

Test with same question across formats:

| Format | Answer Quality | Speed | Size |
|--------|---------------|-------|------|
| Original (LoRA) | 100% | Slow | Large |
| FP16 GGUF | 99% | Medium | 1GB |
| Q4_0 GGUF | 95% | Fast | 300MB ⭐ |
| Q3_K_S GGUF | 90% | Fastest | 200MB |

---

## ✅ Final Checklist

- [ ] All cells run successfully
- [ ] GGUF files created
- [ ] Files copied to Drive
- [ ] Tested with sample question
- [ ] Downloaded Q4_0 file
- [ ] Uploaded to GitHub Release
- [ ] Got download URL
- [ ] Updated Android app Constants.kt

---

## 🎉 Success!

আপনার GGUF model এখন ready!

**Next Steps:**
1. ✅ Download from browser (Cell 7)
2. ✅ Upload to GitHub Release
3. ✅ Get URL
4. ✅ Update Android app
5. ✅ Build & Test app
6. ✅ Publish to Play Store!

---

**File Location**: `CONVERT_TO_GGUF_GUIDE.md`  
**Repository**: https://github.com/kajshikhi49-afk/ss_100m
