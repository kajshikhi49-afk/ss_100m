# NCTB Dataset Information

## 📊 Dataset Statistics

- **Total Questions**: 408 (validated)
- **Format**: JSONL (JSON Lines)
- **Language**: Bengali (Bangla) + English
- **Grade Level**: Class 1-12 (NCTB Curriculum)

---

## 📝 Dataset Details

### Content Coverage

| Subject | Approximate Questions |
|---------|----------------------|
| বাংলা ব্যাকরণ ও সাহিত্য | ~120 |
| English Grammar & Literature | ~60 |
| গণিত (Mathematics) | ~100 |
| বিজ্ঞান (Science) | ~80 |
| বাংলাদেশ ও বিশ্বপরিচয় | ~30 |
| তথ্য ও যোগাযোগ প্রযুক্তি | ~10 |
| ধর্ম ও নৈতিক শিক্ষা | ~8 |

### Statistics

- **Average instruction length**: 20 characters
- **Average output length**: 72 characters
- **Shortest instruction**: 9 characters
- **Longest instruction**: 80+ characters
- **Data Quality**: 100% validated (0 errors)

---

## 🔍 Validation Results

✅ **All 408 entries are valid**
- Proper JSON format
- Both `instruction` and `output` fields present
- No empty fields
- UTF-8 encoding verified

---

## 📋 Sample Data

### Example 1 (Bengali Grammar)
```json
{
  "instruction": "বাংলা বর্ণমালায় মোট কয়টি বর্ণ আছে?",
  "output": "বাংলা বর্ণমালায় মোট ৫০টি বর্ণ আছে। এর মধ্যে স্বরবর্ণ ১১টি এবং ব্যঞ্জনবর্ণ ৩৯টি।"
}
```

### Example 2 (Mathematics)
```json
{
  "instruction": "পিথাগোরাসের উপপাদ্য কী?",
  "output": "সমকোণী ত্রিভুজে অতিভুজের বর্গ অপর দুই বাহুর বর্গের সমষ্টির সমান। অর্থাৎ, c² = a² + b²।"
}
```

### Example 3 (English)
```json
{
  "instruction": "What is a noun?",
  "output": "A noun is a word that names a person, place, thing, or idea. For example: boy, Dhaka, book, happiness."
}
```

### Example 4 (Science)
```json
{
  "instruction": "সালোকসংশ্লেষণ কাকে বলে?",
  "output": "যে প্রক্রিয়ায় সবুজ উদ্ভিদ সূর্যালোক, পানি এবং কার্বন ডাইঅক্সাইড ব্যবহার করে খাদ্য তৈরি করে তাকে সালোকসংশ্লেষণ বলে।"
}
```

---

## 🎯 Use Cases

This dataset is perfect for:
1. ✅ Fine-tuning Bengali language models
2. ✅ Educational AI assistants
3. ✅ Question-answering systems
4. ✅ Student learning apps
5. ✅ NCTB curriculum-based chatbots

---

## 📦 Files

- **Main Dataset**: `nctb_dataset_500plus.jsonl` (408 entries)
- **Clean Version**: `nctb_dataset_clean.jsonl` (same, backup)
- **Validator Script**: `../scripts/validate_dataset.py`

---

## 🔧 Validation Command

To validate the dataset yourself:

```bash
python scripts/validate_dataset.py data/nctb_dataset_500plus.jsonl
```

---

## 📚 Data Format

Each line in the JSONL file contains:

```json
{
  "instruction": "প্রশ্ন বা নির্দেশনা",
  "output": "উত্তর বা প্রতিক্রিয়া"
}
```

**Required Fields**:
- `instruction`: The question or instruction (string)
- `output`: The answer or response (string)

**Encoding**: UTF-8 (supports Bengali unicode)

---

## ⚠️ Note

This is a **curated educational dataset** based on NCTB curriculum. 
One entry was removed during validation due to JSON formatting issues.

**Original**: 409 entries  
**Validated**: 408 entries  
**Success Rate**: 99.75%

---

## 📄 License

This dataset is for educational purposes. The questions are based on publicly available NCTB curriculum content.

---

**Last Updated**: 2024  
**Maintained by**: Kiro AI Project
