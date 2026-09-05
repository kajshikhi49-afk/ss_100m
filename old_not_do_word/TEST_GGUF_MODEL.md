# 🧪 GGUF Model Testing Guide

## ❌ আমি সরাসরি model run করতে পারি না

**কেন?**
- আমার কাছে local file access নেই
- GGUF models run করার জন্য llama.cpp library দরকার
- আমি শুধু code generate করতে পারি, execute করতে পারি না

---

## ✅ কিন্তু আপনি নিজে test করতে পারবেন!

আমি আপনাকে **3টি উপায়** দিচ্ছি যেগুলো দিয়ে আপনি GGUF model test করতে পারবেন:

---

# 🎯 Method 1: Colab-এ Test করুন (EASIEST) ⭐

## Setup (2 minutes)

আপনার Colab notebook-এ এই cell যোগ করুন:

```python
# ============================================
# 🧪 Test GGUF Model in Colab
# ============================================

# Step 1: Install llama-cpp-python
print("📦 Installing llama-cpp-python...")
!pip install -q llama-cpp-python

# Step 2: Load model
from llama_cpp import Llama

print("📥 Loading GGUF model...")
llm = Llama(
    model_path="/content/qwen_nctb_bangla_q4_0.gguf",  # আপনার GGUF file
    n_ctx=2048,          # Context length
    n_threads=4,         # CPU threads
    n_gpu_layers=0,      # CPU only (set to 35 for GPU)
    verbose=False,
)

print("✅ Model loaded successfully!\n")

# Step 3: Test function
def ask_question(question):
    """Ask a question and get answer"""
    
    prompt = f"""<|im_start|>system
You are a helpful AI assistant for Bengali education. Answer questions based on NCTB curriculum.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    
    print(f"❓ Question: {question}")
    print("🤖 Generating answer...\n")
    
    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        echo=False,
        stop=["<|im_end|>", "<|im_start|>"],
    )
    
    answer = output['choices'][0]['text'].strip()
    print(f"✅ Answer: {answer}\n")
    print("-" * 60)
    
    return answer

# Step 4: Test করুন!
print("🧪 Testing model with sample questions...\n")
print("=" * 60)

# Test 1
ask_question("বাংলাদেশের রাজধানী কী?")

# Test 2
ask_question("বাংলা বর্ণমালায় মোট কয়টি বর্ণ আছে?")

# Test 3
ask_question("সালোকসংশ্লেষণ কাকে বলে?")

# Test 4
ask_question("পদ্মা সেতু কোথায় অবস্থিত?")

# Test 5
ask_question("২ + ৩ = কত?")

print("\n🎉 All tests complete!")

# Step 5: Interactive testing
print("\n💡 You can now ask custom questions:")

# আপনার নিজের প্রশ্ন করুন
my_question = "মুক্তিযুদ্ধ কত সালে হয়েছিল?"  # ✏️ Change this
ask_question(my_question)
```

---

# 🎯 Method 2: Local Machine-এ Test করুন

## Windows/Linux/Mac

### Step 1: Install Python packages

```bash
pip install llama-cpp-python
```

### Step 2: Create test script

**File: `test_gguf.py`**

```python
#!/usr/bin/env python3
"""Test GGUF model locally"""

from llama_cpp import Llama
import sys

def load_model(model_path):
    """Load GGUF model"""
    print(f"📥 Loading model: {model_path}")
    
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )
    
    print("✅ Model loaded!\n")
    return llm

def ask(llm, question):
    """Ask question and get answer"""
    
    prompt = f"""<|im_start|>system
You are a helpful AI assistant for Bengali education. Answer questions based on NCTB curriculum.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    
    print(f"❓ Q: {question}")
    print("🤖 Generating...\n")
    
    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        stop=["<|im_end|>"],
    )
    
    answer = output['choices'][0]['text'].strip()
    print(f"✅ A: {answer}\n")
    print("-" * 60)
    
    return answer

def main():
    # Load model
    model_path = "qwen_nctb_bangla_q4_0.gguf"  # Change to your path
    llm = load_model(model_path)
    
    # Test questions
    questions = [
        "বাংলাদেশের রাজধানী কী?",
        "বাংলা বর্ণমালায় কয়টি বর্ণ?",
        "সালোকসংশ্লেষণ কী?",
        "পিথাগোরাসের উপপাদ্য কী?",
        "What is the capital of Bangladesh?",
    ]
    
    print("🧪 Testing model...\n")
    print("=" * 60)
    
    for q in questions:
        ask(llm, q)
    
    print("\n🎉 Tests complete!")
    
    # Interactive mode
    print("\n💡 Interactive mode (type 'exit' to quit):")
    while True:
        try:
            question = input("\n❓ Your question: ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
            ask(llm, question)
        except KeyboardInterrupt:
            break
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
```

### Step 3: Run

```bash
python test_gguf.py
```

---

# 🎯 Method 3: Web UI দিয়ে Test করুন

## Option A: LM Studio (EASIEST!) ⭐

### Download & Install:
- Website: https://lmstudio.ai
- Available for: Windows, Mac, Linux
- Free!

### Steps:
1. **Download LM Studio**
2. **Open LM Studio**
3. **Load your GGUF file**: File → Load Model → Select `qwen_nctb_bangla_q4_0.gguf`
4. **Chat!** 💬

**Features:**
```
✅ Beautiful UI
✅ Easy to use
✅ GPU acceleration
✅ No coding needed!
```

---

## Option B: Text Generation WebUI

### Install:
```bash
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
./start_linux.sh  # Linux
./start_windows.bat  # Windows
./start_macos.sh  # Mac
```

### Use:
1. Open browser: `http://localhost:7860`
2. Upload GGUF file
3. Load model
4. Chat!

---

## Option C: llama.cpp Server

### Setup:
```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build
make

# Run server
./llama-server \
    -m qwen_nctb_bangla_q4_0.gguf \
    -c 2048 \
    --port 8080

# Open browser: http://localhost:8080
```

---

# 🎯 Method 4: Android App-এ Direct Test

## আপনার Android app-এই test করুন!

যখন app build করবেন:

1. **Model download করুন** app-এ
2. **Test questions** করুন:
   ```
   - বাংলাদেশের রাজধানী কী?
   - বাংলা বর্ণমালায় কয়টি বর্ণ?
   - ২ + ২ = কত?
   ```
3. **Check answers** - সঠিক উত্তর আসছে কিনা

---

# 📊 Expected Output

## Good Model Performance:

```
❓ Q: বাংলাদেশের রাজধানী কী?
✅ A: বাংলাদেশের রাজধানী ঢাকা।

❓ Q: বাংলা বর্ণমালায় কয়টি বর্ণ?
✅ A: বাংলা বর্ণমালায় মোট ৫০টি বর্ণ আছে। এর মধ্যে স্বরবর্ণ ১১টি এবং ব্যঞ্জনবর্ণ ৩৯টি।

❓ Q: ২ + ২ = কত?
✅ A: ২ + ২ = ৪
```

## Bad Model Performance:

```
❌ Gibberish output
❌ Wrong answers
❌ English when asked in Bengali
❌ Very short/incomplete answers
```

---

# 🔍 Debugging Issues

## Issue 1: Model loading failed

```python
# Error: Failed to load model
# Solution: Check file path and file integrity

import os
print(f"File exists: {os.path.exists('qwen_nctb_bangla_q4_0.gguf')}")
print(f"File size: {os.path.getsize('qwen_nctb_bangla_q4_0.gguf') / (1024**2):.2f} MB")
```

## Issue 2: Slow inference

```python
# Solution: Use GPU layers
llm = Llama(
    model_path="qwen_nctb_bangla_q4_0.gguf",
    n_gpu_layers=35,  # Use GPU! (if available)
    n_ctx=2048,
)
```

## Issue 3: Poor answers

```python
# Solution: Adjust generation parameters
output = llm(
    prompt,
    max_tokens=256,      # Increase if answers are cut off
    temperature=0.7,     # Lower (0.3) = more deterministic
    top_p=0.9,          # Adjust sampling
    repeat_penalty=1.2,  # Higher = less repetition
)
```

---

# 📝 Quick Test Checklist

- [ ] Model loads without error
- [ ] Bengali questions work
- [ ] English questions work (if trained)
- [ ] Math questions work
- [ ] Answers are correct
- [ ] Answers are complete (not cut off)
- [ ] No gibberish output
- [ ] Reasonable speed (5-10 seconds/answer)

---

# 🎯 Summary

## আমি কী করতে পারি না:
```
❌ আপনার GGUF file সরাসরি run করতে
❌ Model test করতে আপনার হয়ে
❌ File access করতে local machine-এ
```

## আমি কী করতে পারি:
```
✅ Testing script লিখে দিতে (Python)
✅ Colab code দিতে (যা আপনি run করবেন)
✅ Debugging help করতে
✅ Android integration code দিতে
✅ Best practices guide করতে
```

---

# 💡 My Recommendation

**Best way to test YOUR model:**

1. **Colab-এ test করুন** (easiest, free, immediate) ⭐
2. **Or LM Studio download করুন** (beautiful UI, no coding)
3. **Or আপনার Android app-এ test করুন** (real-world test)

**আমি script দিয়ে help করতে পারি, কিন্তু run করতে হবে আপনাকে!** 😊

---

## 🚀 Next Steps

1. Colab-এর test cell run করুন (উপরে দেওয়া আছে)
2. যদি answers ভালো হয়:
   - ✅ Model upload করুন (Hugging Face/Drive)
   - ✅ Android app-এ integrate করুন
3. যদি answers খারাপ হয়:
   - ⚠️ Model re-train করুন
   - ⚠️ Dataset improve করুন

---

**File**: `TEST_GGUF_MODEL.md`  
**Repository**: https://github.com/kajshikhi49-afk/ss_100m
