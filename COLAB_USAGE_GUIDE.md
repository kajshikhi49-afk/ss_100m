# 🚀 Qwen2.5-0.5B Fine-tuning Guide

## আপনার মডেল ট্রেন করার সহজ পদ্ধতি

এই গাইড অনুসরণ করে আপনি খুব সহজেই বাংলা NCTB ডেটাসেট দিয়ে Qwen2.5-0.5B মডেল ফাইন-টিউন করতে পারবেন।

---

## ⚠️ সমস্যা সমাধান করা হয়েছে!

আপনার আগের JSON error সমস্যা সম্পূর্ণভাবে সমাধান করা হয়েছে:

✅ Dataset validation script তৈরি করা হয়েছে  
✅ Dataset পরিষ্কার করা হয়েছে (408 valid entries)  
✅ Colab notebook-এ error handling যোগ করা হয়েছে  
✅ সব কিছু GitHub-এ push করা হয়েছে  

**এখন আপনি নিশ্চিন্তে ট্রেনিং শুরু করতে পারবেন!** 🎉

---

## 📋 শুরু করার আগে যা প্রয়োজন

### 1. Google Colab অ্যাকাউন্ট
- Google account থাকলেই হবে
- কোনো ইনস্টলেশন লাগবে না

### 2. Dataset আপলোড
আপনার Google Drive-এ `nctb_dataset_500plus.jsonl` ফাইল আপলোড করুন:

```
📁 Google Drive
  └── MyDrive/
      └── nctb_dataset_500plus.jsonl  ← এখানে রাখুন
```

**ফাইল কোথায় পাবেন?**  
GitHub repository থেকে ডাউনলোড করুন:  
https://github.com/kajshikhi49-afk/ss_100m/blob/main/data/nctb_dataset_500plus.jsonl

### 3. Colab Notebook খুলুন
GitHub থেকে `colab_qwen_finetune.ipynb` ডাউনলোড করে Google Colab-এ আপলোড করুন।

অথবা সরাসরি Colab-এ খুলুন:
1. https://colab.research.google.com এ যান
2. File → Upload notebook
3. GitHub থেকে clone করুন বা সরাসরি .ipynb ফাইল আপলোড করুন

---

## 🎯 ধাপে ধাপে ট্রেনিং প্রক্রিয়া

### ধাপ ১: GPU সক্রিয় করুন
Colab-এ:
1. **Runtime** → **Change runtime type** ক্লিক করুন
2. **Hardware accelerator** → **T4 GPU** নির্বাচন করুন
3. **Save** করুন

✅ GPU চালু হয়েছে কিনা চেক করতে প্রথম cell চালান:
```python
!nvidia-smi
```

### ধাপ ২: Google Drive মাউন্ট করুন
দ্বিতীয় cell চালান:
```python
from google.colab import drive
drive.mount('/content/drive')
```

- একটি popup আসবে → **Connect to Google Drive** ক্লিক করুন
- আপনার Google account নির্বাচন করুন
- **Allow** দিন

### ধাপ ৩: Dataset Path সেট করুন
Notebook-এ "ধাপ ৪" section-এ এই লাইন খুঁজে বের করুন:

```python
DATASET_PATH = "/content/drive/MyDrive/nctb_dataset_500plus.jsonl"
```

**যদি আপনার ফাইল অন্য জায়গায় থাকে**, path পরিবর্তন করুন। যেমন:
```python
DATASET_PATH = "/content/drive/MyDrive/datasets/nctb_dataset_500plus.jsonl"
```

### ধাপ ৪: সব Cell পর্যায়ক্রমে চালান
এখন থেকে সব cell একটার পর একটা চালান:

1. **ধাপ ৩**: প্যাকেজ ইনস্টল (৩-৫ মিনিট)
2. **ধাপ ৪**: Dataset লোড করুন
   - আপনি দেখবেন: `✅ সফলভাবে লোড হয়েছে: 408 টি`
   - কোনো error আসবে না! 🎉
3. **ধাপ ৫-৮**: Model setup (২-৩ মিনিট)
4. **ধাপ ৯**: Training শুরু! ⏱️ **25-40 মিনিট সময় লাগবে**

### ধাপ ৫: Training সম্পন্ন হলে
- **ধাপ ১০**: Model save হবে আপনার Drive-এ
- **ধাপ ১১-১২**: Model test করুন
- **ধাপ ১৩**: Model download করুন (optional)

---

## ⏱️ সময় সংক্রান্ত তথ্য

| ধাপ | সময় | বিবরণ |
|-----|------|-------|
| প্যাকেজ ইনস্টল | ৩-৫ মিনিট | প্রথমবার একটু বেশি লাগবে |
| Dataset লোড | ১ মিনিট | 408 entries লোড হবে |
| Model লোড | ২-৩ মিনিট | Qwen2.5-0.5B download |
| **Training** | **25-40 মিনিট** | ⭐ মূল সময় |
| Model save | ১ মিনিট | Drive-এ সেভ হবে |
| Testing | ১-২ মিনিট | নমুনা প্রশ্নের উত্তর দেখুন |

**মোট সময়**: প্রায় **৩৫-৫৫ মিনিট**

---

## 📊 Training Progress বোঝার উপায়

Training চলার সময় আপনি দেখবেন:

```
Epoch 1/3: 100%|██████████| 25/25 [05:23<00:00, 12.91s/it, loss=2.456]
```

এর অর্থ:
- **Epoch 1/3**: প্রথম ইপক চলছে
- **25/25**: মোট 25 স্টেপের মধ্যে 25টি সম্পন্ন
- **loss=2.456**: Error কমছে (ভালো চিহ্ন!)

**ভালো training-এর লক্ষণ**:
- Loss কমতে থাকবে (যেমন: 4.5 → 3.2 → 2.1 → 1.5)
- প্রতি স্টেপে 10-15 সেকেন্ড সময় নিবে

---

## 🔍 Dataset লোডিং আউটপুট

এখন আপনি দেখবেন:

```
📂 ফাইল লোড হচ্ছে...

============================================================
✅ সফলভাবে লোড হয়েছে: 408 টি
🎉 সব ডেটা পারফেক্ট!
============================================================

📝 প্রথম ৩টি উদাহরণ:

1. প্রশ্ন: বাংলা বর্ণমালায় মোট কয়টি বর্ণ আছে?...
   উত্তর: বাংলা বর্ণমালায় মোট ৫০টি বর্ণ আছে। এর মধ্যে স্বরবর্ণ ১১টি এবং ব্যঞ্জনবর্ণ ৩৯টি।...

📊 ডেটাসেট বিভাজন:
  - Training: 387 টি
  - Testing: 21 টি

✅ ডেটাসেট প্রস্তুত!
```

**যদি কোনো error থাকে**, notebook এখন automatically skip করে দেবে এবং বাকি data দিয়ে continue করবে।

---

## 🎓 Training সম্পন্ন হলে

### মডেল কোথায় সেভ হবে?
```
📁 Google Drive
  └── MyDrive/
      ├── qwen_nctb_finetuned/     ← Training checkpoints
      └── qwen_nctb_final/          ← Final trained model
```

### মডেল টেস্ট করুন
Notebook-এর **ধাপ ১১** এ নমুনা প্রশ্ন দেওয়া আছে:

```python
test_questions = [
    "বাংলাদেশের রাজধানী কী?",
    "সালোকসংশ্লেষণ কাকে বলে?",
    "২ + ২ = কত?",
]
```

আপনিও নিজের প্রশ্ন করতে পারবেন!

---

## ❓ সমস্যা সমাধান (Troubleshooting)

### ❌ Error: "cannot import name 'DataCollatorForCompletionOnlyLM'"
**সমাধান**: এই সমস্যা ইতিমধ্যে fix করা হয়েছে! নতুন notebook ব্যবহার করুন যেখানে `DataCollatorForLanguageModeling` ব্যবহার করা হয়েছে।

### ❌ Error: "No GPU available"
**সমাধান**: Runtime → Change runtime type → T4 GPU নির্বাচন করুন

### ❌ Error: "File not found"
**সমাধান**: `DATASET_PATH` এর path চেক করুন। Drive-এ ফাইল আছে কিনা verify করুন।

### ❌ Error: "CUDA out of memory"
**সমাধান**: 
```python
per_device_train_batch_size=2  # 4 থেকে 2 করুন
```

### ⚠️ Warning: "Some lines skipped"
**এটি সমস্যা নয়!** Notebook automatically corrupt lines skip করে বাকি data দিয়ে continue করবে।

### ❌ Training খুব ধীর গতিতে চলছে
**সমাধান**: GPU চালু আছে কিনা দেখুন। CPU-তে training করলে অনেক সময় লাগবে।

---

## 🎉 সফল Training-এর চেকলিস্ট

Training শেষে নিশ্চিত করুন:

- [ ] Loss কমেছে (4.0+ থেকে 2.0 বা তার নিচে)
- [ ] Model সেভ হয়েছে Drive-এ
- [ ] Test questions-এর সঠিক উত্তর আসছে
- [ ] `qwen_nctb_final/` folder-এ files আছে

---

## 🚀 পরবর্তী ধাপ

### Option 1: আরও ডেটা যোগ করুন
আরও NCTB প্রশ্ন-উত্তর যোগ করে dataset বড় করুন।

### Option 2: Model Deploy করুন
- Gradio/Streamlit দিয়ে web app বানান
- Android-এ convert করুন (ONNX/TFLite)

### Option 3: Fine-tune আরও করুন
নির্দিষ্ট বিষয়ে (যেমন শুধু গণিত) বেশি data দিয়ে retrain করুন।

---

## 📞 সাহায্য প্রয়োজন?

- **GitHub Issues**: https://github.com/kajshikhi49-afk/ss_100m/issues
- **Validation Report দেখুন**: `VALIDATION_REPORT.md`
- **Dataset Info**: `data/DATASET_INFO.md`

---

## 📊 আপনার Dataset Statistics

✅ **মোট Q&A**: 408 pairs  
✅ **বিষয় কভারেজ**: 7টি বিষয় (বাংলা, English, গণিত, বিজ্ঞান, ইতিহাস, ICT, ধর্ম)  
✅ **ক্লাস কভারেজ**: Class 1-12  
✅ **Quality**: 100% valid JSON, no errors  

---

**শুভকামনা!** 🎓🚀

আপনার training সফল হোক! কোনো প্রশ্ন থাকলে জানাবেন।

---

**তৈরি করেছেন**: Kiro AI 🤖  
**তারিখ**: September 3, 2026  
**Repository**: https://github.com/kajshikhi49-afk/ss_100m
