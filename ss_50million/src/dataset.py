"""
Memory-efficient dataset loader with uint16 binary memmap caching.
Supports 700MB+ / 10GB+ corpora on Colab Free Tier without RAM crashes (< 100MB RAM!).
"""

import os
import numpy as np
import torch
from src.tokenizer import BengaliTokenizer


class BengaliDataset:
    """
    Streaming, memory-mapped dataset for massive corpora.
    Converts corpus.txt into a compact uint16 binary file in chunks without RAM spikes.
    Uses np.memmap for zero-RAM batch sampling.
    """

    def __init__(self, corpus_path="data/corpus.txt", bin_cache_dir=None, tokenizer=None, split_ratio=0.9, block_size=2048):
        self.corpus_path = corpus_path
        self.block_size = block_size
        self.split_ratio = split_ratio

        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        file_size_mb = os.path.getsize(corpus_path) / (1024 * 1024)
        print(f"📖 বাংলা কর্পাস ফাইল: {corpus_path} ({file_size_mb:.1f} MB)")

        # টোকেনাইজার লোড
        if tokenizer is None:
            self.tokenizer = BengaliTokenizer(vocab_size=10000)
            if os.path.exists("tokenizer.json"):
                self.tokenizer.load("tokenizer.json")
            elif os.path.exists("/content/drive/MyDrive/bengali_gpt_50m_checkpoints/tokenizer.json"):
                self.tokenizer.load("/content/drive/MyDrive/bengali_gpt_50m_checkpoints/tokenizer.json")
            else:
                self.tokenizer.build_vocab(corpus_path)
                self.tokenizer.save("tokenizer.json")
        else:
            self.tokenizer = tokenizer

        # লোকাল ডিস্কে বাইনারি ক্যাশ ফাইল তৈরি (কোল্যাব র‍্যাম যাতে ০% ব্যবহার হয়)
        if bin_cache_dir is None:
            bin_cache_dir = os.path.dirname(corpus_path)
            if bin_cache_dir.startswith("/content/drive"):
                bin_cache_dir = "/content/ss_10million/data"
        if not bin_cache_dir:
            bin_cache_dir = "data"
        os.makedirs(bin_cache_dir, exist_ok=True)

        bin_file = os.path.join(bin_cache_dir, "corpus_tokens.bin")

        # বাইনারি ক্যাশ আগে তৈরি না থাকলে চাঙ্ক আকারে (Chunked) প্রসেস করুন
        if not os.path.exists(bin_file) or os.path.getsize(bin_file) == 0:
            print(f"⚡ মেমরি-সেফ চাঙ্কড টোকেনাইজেশন শুরু হচ্ছে (RAM সুরক্ষিত থাকবে)...")
            total_tokens = 0
            chunk_lines = []
            chunk_chars = 0
            CHUNK_LIMIT = 500_000  # প্রতি চাঙ্কে ~৫ লক্ষ অক্ষর (মাত্র কয়েক MB RAM খরচ)

            with open(corpus_path, "r", encoding="utf-8") as f_in, open(bin_file, "wb") as f_out:
                for line in f_in:
                    chunk_lines.append(line)
                    chunk_chars += len(line)
                    if chunk_chars >= CHUNK_LIMIT:
                        text_chunk = "".join(chunk_lines)
                        encoded = self.tokenizer.encode(text_chunk)
                        ids = encoded.ids if hasattr(encoded, "ids") else encoded
                        np.array(ids, dtype=np.uint16).tofile(f_out)
                        total_tokens += len(ids)
                        chunk_lines = []
                        chunk_chars = 0
                        if total_tokens % 2_000_000 < len(ids):
                            print(f"  ✍️ {total_tokens:,} টোকেন প্রক্রিয়াজাত হয়েছে (RAM নিরাপদ)...")

                if chunk_lines:
                    text_chunk = "".join(chunk_lines)
                    encoded = self.tokenizer.encode(text_chunk)
                    ids = encoded.ids if hasattr(encoded, "ids") else encoded
                    np.array(ids, dtype=np.uint16).tofile(f_out)
                    total_tokens += len(ids)

            print(f"✓ বাইনারি ক্যাশ সম্পন্ন: {bin_file} (মোট টোকেন: {total_tokens:,})")

        # Memmap দিয়ে সরাসরি ডিস্ক থেকে মেমরি ম্যাপিং (র‍্যাম খরচ মাত্র ~১০ MB!)
        self.data = np.memmap(bin_file, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)

        self.train_len = int(self.split_ratio * self.total_tokens)
        self.val_len = self.total_tokens - self.train_len

        print(f"⚡ Dataset Memmap Ready: {self.total_tokens:,} Tokens")
        print(f"  - Train: {self.train_len:,} টোকেন ({self.split_ratio*100:.0f}%)")
        print(f"  - Val:   {self.val_len:,} টোকেন ({(1-self.split_ratio)*100:.0f}%)")
        print(f"  - Sequences (at {block_size} context): ~{self.train_len // block_size:,}")

    def get_batch(self, split="train", batch_size=8, device="cpu"):
        if split == "train":
            max_start = self.train_len - self.block_size - 1
            offset = 0
        else:
            max_start = self.val_len - self.block_size - 1
            offset = self.train_len

        if max_start <= 0:
            raise ValueError("Dataset too small for context length.")

        ix = np.random.randint(offset, offset + max_start, size=(batch_size,))
        x_list = [torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64)) for i in ix]
        y_list = [torch.from_numpy((self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64)) for i in ix]

        x = torch.stack(x_list)
        y = torch.stack(y_list)

        if device != "cpu":
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        return x, y
