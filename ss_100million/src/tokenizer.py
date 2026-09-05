"""
Multilingual 10,000-Vocabulary Byte-Pair Encoding (BPE) Tokenizer.
Covers Bengali + English + Mathematics and Logic.
Uses HuggingFace tokenizers ByteLevel BPE for lossless tokenization.
"""

import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class MultilingualTokenizer:
    """10,000-Vocab Byte-Pair Encoding (BPE) Tokenizer for Bangla + English + Mathematics."""

    PAD = '<PAD>'
    UNK = '<UNK>'
    BOS = '<BOS>'
    EOS = '<EOS>'
    SYSTEM = '<|system|>'
    USER = '<|user|>'
    ASSISTANT = '<|assistant|>'
    MATH = '<|math|>'

    SPECIAL_TOKENS = [PAD, UNK, BOS, EOS, SYSTEM, USER, ASSISTANT, MATH]

    def __init__(self, vocab_size=10000):
        self.target_vocab_size = vocab_size
        self._tokenizer = None
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.system_id = 4
        self.user_id = 5
        self.assistant_id = 6
        self.math_id = 7

    @property
    def vocab_size(self):
        if self._tokenizer is not None:
            return max(self._tokenizer.get_vocab_size(), self.target_vocab_size)
        return self.target_vocab_size

    def build_vocab(self, text_or_filepath):
        """Train 10,000-vocabulary BPE on Multilingual & Math corpus."""
        print(f"⚡ Training real {self.target_vocab_size:,}-vocab BPE tokenizer (Bangla + English + Math)...")
        
        temp_file = None
        if os.path.exists(text_or_filepath):
            train_file = text_or_filepath
        else:
            temp_file = "temp_corpus_for_tokenizer.txt"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(text_or_filepath)
            train_file = temp_file

        tok = Tokenizer(models.BPE(unk_token=self.UNK))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.target_vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            min_frequency=1,
            show_progress=False
        )

        tok.train([train_file], trainer)
        self._tokenizer = tok

        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

        print(f"✓ Real {self.target_vocab_size:,}-Vocab BPE Tokenizer successfully built!")
        print(f"  - Actual Vocab Size: {self._tokenizer.get_vocab_size():,} tokens")
        return self

    def encode(self, text, add_bos=False, add_eos=False):
        """Encode text string into token IDs."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded. Call build_vocab() or load().")

        encoded = self._tokenizer.encode(text)
        ids = list(encoded.ids)

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        return ids

    def decode(self, ids, skip_special_tokens=True):
        """Decode token IDs back into text."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded. Call build_vocab() or load().")

        clean_ids = []
        for i in ids:
            val = int(i)
            if skip_special_tokens and val in [self.pad_id, self.unk_id, self.bos_id, self.eos_id]:
                continue
            clean_ids.append(val)

        return self._tokenizer.decode(clean_ids)

    def save(self, filepath="tokenizer.json"):
        """Save tokenizer configuration to json file."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded.")
        self._tokenizer.save(filepath)
        print(f"✓ Tokenizer saved to {filepath}")

    def load(self, filepath="tokenizer.json"):
        """Load tokenizer from json file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tokenizer file not found: {filepath}")
        self._tokenizer = Tokenizer.from_file(filepath)
        print(f"✓ Tokenizer loaded from {filepath} (Vocab: {self.vocab_size:,})")
        return self


# Backward compatibility alias
BengaliTokenizer = MultilingualTokenizer
