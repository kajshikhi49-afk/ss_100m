"""
Memory-efficient dataset loader with uint16 binary memmap caching.
Supports 700MB+ / 10GB+ corpora on Colab Free Tier without RAM crashes (< 100MB RAM!).
Supports configurable vocab_size for multilingual (10k) and Bengali-only (5k) tokenizers.
"""

import os
import numpy as np
import torch
from src.tokenizer import MultilingualTokenizer


class BengaliDataset:
    """
    Streaming, memory-mapped dataset for massive corpora.
    Converts corpus.txt into a compact uint16 binary file in chunks without RAM spikes.
    Uses np.memmap for zero-RAM batch sampling.
    Supports configurable vocab_size — pass vocab_size=10000 for multilingual 100M model.
    """

    def __init__(self, corpus_path="data/corpus.txt", bin_cache_dir=None,
                 tokenizer=None, split_ratio=0.9, block_size=2048, vocab_size=10000):
        self.corpus_path = corpus_path
        self.block_size = block_size
        self.split_ratio = split_ratio

        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        file_size_mb = os.path.getsize(corpus_path) / (1024 * 1024)
        print(f"📖 কর্পাস ফাইল: {corpus_path} ({file_size_mb:.1f} MB)")

        # ─── টোকেনাইজার লোড বা তৈরি ───────────────────────────────────────────
        if tokenizer is None:
            self.tokenizer = MultilingualTokenizer(vocab_size=vocab_size)

            # অগ্রাধিকার ক্রমে টোকেনাইজার খোঁজা:
            # ১. লোকাল tokenizer.json (Colab /content/ss_100million/tokenizer.json)
            # ২. ড্রাইভের 100M চেকপয়েন্ট ফোল্ডার
            # ৩. ড্রাইভের 50M চেকপয়েন্ট ফোল্ডার (Fallback)
            # ৪. স্ক্র্যাচ থেকে নতুন তৈরি করা
            search_paths = [
                "tokenizer.json",
                "/content/drive/MyDrive/bengali_gpt_100m_checkpoints/tokenizer.json",
                "/content/drive/MyDrive/bengali_gpt_50m_checkpoints/tokenizer.json",
            ]
            loaded = False
            for tok_path in search_paths:
                if os.path.exists(tok_path):
                    self.tokenizer.load(tok_path)
                    loaded = True
                    break

            if not loaded:
                # কোনো টোকেনাইজার পাওয়া যায়নি → নতুন করে তৈরি করো
                print(f"⚡ কোনো পূর্ববর্তী tokenizer.json পাওয়া যায়নি। কর্পাস থেকে {vocab_size:,}-vocab BPE তৈরি হচ্ছে...")
                self.tokenizer.build_vocab(corpus_path)
                self.tokenizer.save("tokenizer.json")
                print(f"✓ নতুন {vocab_size:,}-vocab টোকেনাইজার তৈরি ও সেভ সম্পন্ন!")
        else:
            self.tokenizer = tokenizer

        # ─── লোকাল Colab ডিস্কে বাইনারি ক্যাশ ফাইল ────────────────────────────
        # গুগল ড্রাইভের পাথ হলে Colab লোকাল ডিস্কে ক্যাশ করো (ড্রাইভ I/O অনেক ধীর)
        if bin_cache_dir is None:
            corpus_dir = os.path.dirname(corpus_path)
            if corpus_dir.startswith("/content/drive"):
                bin_cache_dir = "/content/ss_100million/data"
            elif not corpus_dir:
                bin_cache_dir = "data"
            else:
                bin_cache_dir = corpus_dir

        os.makedirs(bin_cache_dir, exist_ok=True)
        bin_file = os.path.join(bin_cache_dir, "corpus_tokens.bin")

        # ─── চাঙ্কড স্ট্রিমিং টোকেনাইজেশন (RAM মাত্র ~৫০ MB ব্যবহার হয়!) ──────
        if not os.path.exists(bin_file) or os.path.getsize(bin_file) == 0:
            print(f"⚡ মেমরি-সেফ চাঙ্কড টোকেনাইজেশন শুরু হচ্ছে (RAM সুরক্ষিত থাকবে)...")
            print(f"  → ক্যাশ লোকেশন: {bin_file}")
            total_tokens = 0
            chunk_lines = []
            chunk_chars = 0
            CHUNK_LIMIT = 500_000  # প্রতি চাঙ্কে ~৫ লক্ষ অক্ষর

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
                            print(f"  ✍️ {total_tokens:,} টোকেন প্রক্রিয়াজাত (RAM নিরাপদ)...")

                if chunk_lines:
                    text_chunk = "".join(chunk_lines)
                    encoded = self.tokenizer.encode(text_chunk)
                    ids = encoded.ids if hasattr(encoded, "ids") else encoded
                    np.array(ids, dtype=np.uint16).tofile(f_out)
                    total_tokens += len(ids)

            print(f"✓ বাইনারি ক্যাশ সম্পন্ন: {bin_file}")
            print(f"  → মোট টোকেন: {total_tokens:,}")
        else:
            print(f"✓ বিদ্যমান বাইনারি ক্যাশ লোড হচ্ছে: {bin_file}")

        # ─── Memmap দিয়ে ডিস্ক থেকে সরাসরি মেমরি ম্যাপিং ─────────────────────
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
