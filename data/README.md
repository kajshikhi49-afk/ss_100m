# Data Directory

This directory contains training data for the Bengali language model.

## Directory Structure

```
data/
├── raw/                       # Raw text files for pretraining
├── processed/                 # Processed training data (created by scripts)
│   ├── train/                # Training split
│   ├── val/                  # Validation split
│   └── metadata.json         # Dataset metadata
├── examples/                  # Sample data files
│   ├── sample_bengali_text.txt
│   └── sft_data.jsonl
└── README.md                 # This file
```

## Data Preparation Guide

### 1. Pretraining Data (data/raw/)

Add your Bengali text files here. Supported formats:
- Plain text files (`.txt`)
- UTF-8 encoding required

**Recommended Sources:**

1. **OSCAR Dataset** (Large web-crawled corpus):
```python
from datasets import load_dataset
dataset = load_dataset("oscar", "unshuffled_deduplicated_bn", split="train")
```

2. **CC-100** (Common Crawl):
```python
dataset = load_dataset("cc100", lang="bn")
```

3. **Bengali Wikipedia**:
```python
dataset = load_dataset("wikipedia", "20220301.bn")
```

4. **Bengali Books**:
- NCTB textbooks
- Classic Bengali literature
- Contemporary novels and stories

5. **News Articles**:
- Prothom Alo archive
- Daily Star Bengali
- Other news sources

**Data Quality Tips:**
- Remove duplicate content
- Filter out very short texts (< 50 words)
- Clean HTML/markup if scraping web data
- Ensure proper sentence boundaries
- Remove non-Bengali text (unless multilingual model)

### 2. Supervised Fine-Tuning Data (data/examples/)

Create instruction-response pairs in JSONL format.

**Format:**
```json
{
  "instruction": "প্রশ্ন বা নির্দেশনা",
  "response": "উত্তর বা সম্পূর্ণ প্রতিক্রিয়া",
  "input": ""  // Optional: Additional context
}
```

**Example Categories:**

1. **Question Answering**:
```json
{"instruction": "বাংলাদেশের রাজধানী কী?", "response": "বাংলাদেশের রাজধানী ঢাকা।"}
```

2. **Text Completion**:
```json
{"instruction": "এই বাক্যটি সম্পূর্ণ করুন: শিক্ষা হল", "response": "শিক্ষা হল জাতির মেরুদণ্ড এবং উন্নয়নের মূল চালিকা শক্তি।"}
```

3. **Summarization**:
```json
{"instruction": "এই প্রবন্ধটির সারাংশ লিখুন।", "input": "দীর্ঘ প্রবন্ধ...", "response": "সংক্ষিপ্ত সারাংশ..."}
```

4. **Translation** (if applicable):
```json
{"instruction": "ইংরেজিতে অনুবাদ করুন: আমি ভাত খাই।", "response": "I eat rice."}
```

5. **Creative Writing**:
```json
{"instruction": "একটি ছোট গল্প লিখুন।", "response": "একবার এক গ্রামে..."}
```

### 3. Data Size Recommendations

| Dataset Type | Minimum | Recommended | Optimal |
|--------------|---------|-------------|---------|
| Pretraining | 10 MB | 100 MB | 1+ GB |
| SFT | 100 examples | 1,000 examples | 10,000+ examples |

**Note**: More data generally leads to better results, but ensure quality over quantity.

### 4. Data Preprocessing

The training pipeline automatically handles:
- Tokenization with trained BPE tokenizer
- Chunking into fixed-length sequences (2048 tokens)
- Train/validation splitting (95/5 by default)

**Run preprocessing:**
```bash
# Train tokenizer first
python scripts/train_tokenizer.py --data_dir data/raw

# Then prepare training data
python scripts/prepare_data.py --raw_data_dir data/raw
```

## Sample Data Provided

### sample_bengali_text.txt
Contains Bengali text covering various topics:
- Language and literature
- Geography and culture
- History and politics
- Technology and science
- Society and education

**Usage**: Great for testing the pipeline, but insufficient for production model.

### sft_data.jsonl
Contains 20 instruction-response pairs in Bengali:
- General knowledge questions
- Facts about Bangladesh
- Cultural information

**Usage**: Demonstrates SFT format, but add 100+ more examples for real fine-tuning.

## Data Collection Best Practices

1. **Copyright Compliance**:
   - Use public domain texts
   - Respect copyright laws
   - Attribute sources when required

2. **Quality Control**:
   - Review random samples
   - Fix encoding issues
   - Remove inappropriate content

3. **Diversity**:
   - Include multiple domains (news, literature, science, etc.)
   - Mix formal and informal language
   - Cover various topics

4. **Privacy**:
   - Remove personal information (names, emails, phone numbers)
   - Anonymize sensitive data
   - Comply with data protection laws

## Troubleshooting

### Issue: "No text files found"
- Ensure files have `.txt` extension
- Check files are in `data/raw/` directory
- Verify UTF-8 encoding

### Issue: "Tokenizer fails on certain characters"
- Clean special characters
- Ensure consistent encoding
- Remove control characters

### Issue: "Dataset too small"
- Add more text files
- Download larger corpora
- Use data augmentation (paraphrasing)

### Issue: "Out of memory during preprocessing"
- Reduce `block_size` in config
- Process data in smaller batches
- Use streaming dataset mode

## Need More Data?

**Public Bengali Datasets:**
- Hugging Face Datasets Hub
- Common Crawl Bengali
- Bengali Wikipedia dumps
- Project Gutenberg (Bengali books)
- IndicNLP datasets

**Data Collection Tools:**
- Web scraping (BeautifulSoup, Scrapy)
- PDF extraction (PyPDF2)
- Document conversion (pandoc)

## License

Please ensure you have appropriate licenses for any data you use for training. The model itself is licensed under MIT, but training data may have different licenses.

## Questions?

If you need help with data preparation:
1. Check the main README.md
2. Review the Colab notebook
3. Open an issue on GitHub
