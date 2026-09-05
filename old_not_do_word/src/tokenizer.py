"""
Custom BPE Tokenizer Training Module
Trains a Byte-Pair Encoding tokenizer for Bengali text
"""

import os
import json
from pathlib import Path
from typing import List, Union, Optional, Iterator
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
    trainers,
    normalizers,
    processors,
)
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace, Metaspace
from transformers import PreTrainedTokenizerFast


class BengaliTokenizerTrainer:
    """
    Trains a BPE tokenizer optimized for Bengali text.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        lowercase: bool = False,
    ):
        """
        Initialize tokenizer trainer.
        
        Args:
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for a token to be included
            special_tokens: List of special tokens to include
            lowercase: Whether to lowercase text during normalization
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Define special tokens
        if special_tokens is None:
            self.special_tokens = [
                "[PAD]",   # Padding token
                "[UNK]",   # Unknown token
                "[BOS]",   # Beginning of sequence
                "[EOS]",   # End of sequence
                "[MASK]",  # Mask token for MLM tasks
            ]
        else:
            self.special_tokens = special_tokens
        
        self.lowercase = lowercase
        self.tokenizer = None
    
    def create_tokenizer(self):
        """Create a new BPE tokenizer with proper configuration."""
        # Initialize BPE model
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        
        # Setup normalization
        normalizer_list = [NFD()]
        if self.lowercase:
            normalizer_list.append(Lowercase())
        tokenizer.normalizer = normalizers.Sequence(normalizer_list)
        
        # Setup pre-tokenization (split on whitespace)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            Whitespace(),
            # Metaspace helps with Bengali script
        ])
        
        # Setup decoder
        tokenizer.decoder = decoders.BPEDecoder()
        
        # Setup post-processor for BOS/EOS tokens
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[
                ("[BOS]", 2),
                ("[EOS]", 3),
            ],
        )
        
        self.tokenizer = tokenizer
        return tokenizer
    
    def train_from_files(
        self,
        files: Union[str, List[str]],
        output_dir: str,
        model_name: str = "bengali_tokenizer",
    ):
        """
        Train tokenizer on text files.
        
        Args:
            files: Path to text file(s) or list of paths
            output_dir: Directory to save the trained tokenizer
            model_name: Name for the tokenizer files
        """
        if isinstance(files, str):
            files = [files]
        
        print(f"Training tokenizer on {len(files)} file(s)...")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Special tokens: {self.special_tokens}")
        
        # Create tokenizer if not exists
        if self.tokenizer is None:
            self.create_tokenizer()
        
        # Setup trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
        )
        
        # Train on files
        self.tokenizer.train(files, trainer)
        
        print(f"\nTokenizer trained successfully!")
        print(f"Final vocabulary size: {self.tokenizer.get_vocab_size()}")
        
        # Save tokenizer
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer.json
        tokenizer_path = os.path.join(output_dir, f"{model_name}.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to: {tokenizer_path}")
        
        # Create HuggingFace compatible tokenizer
        self._save_hf_tokenizer(output_dir, model_name)
        
        return self.tokenizer
    
    def train_from_iterator(
        self,
        iterator: Iterator[str],
        output_dir: str,
        model_name: str = "bengali_tokenizer",
    ):
        """
        Train tokenizer from an iterator of text strings.
        
        Args:
            iterator: Iterator yielding text strings
            output_dir: Directory to save the trained tokenizer
            model_name: Name for the tokenizer files
        """
        print(f"Training tokenizer from iterator...")
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Create tokenizer if not exists
        if self.tokenizer is None:
            self.create_tokenizer()
        
        # Setup trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
        )
        
        # Train on iterator
        self.tokenizer.train_from_iterator(iterator, trainer)
        
        print(f"\nTokenizer trained successfully!")
        print(f"Final vocabulary size: {self.tokenizer.get_vocab_size()}")
        
        # Save tokenizer
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer.json
        tokenizer_path = os.path.join(output_dir, f"{model_name}.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to: {tokenizer_path}")
        
        # Create HuggingFace compatible tokenizer
        self._save_hf_tokenizer(output_dir, model_name)
        
        return self.tokenizer
    
    def _save_hf_tokenizer(self, output_dir: str, model_name: str):
        """Save HuggingFace compatible tokenizer configuration."""
        # Create PreTrainedTokenizerFast wrapper
        tokenizer_path = os.path.join(output_dir, f"{model_name}.json")
        
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            mask_token="[MASK]",
        )
        
        # Save HuggingFace tokenizer
        hf_tokenizer.save_pretrained(output_dir)
        print(f"Saved HuggingFace tokenizer to: {output_dir}")
        
        # Save tokenizer config
        config = {
            "vocab_size": self.vocab_size,
            "model_type": "BPE",
            "special_tokens": self.special_tokens,
            "min_frequency": self.min_frequency,
            "lowercase": self.lowercase,
        }
        
        config_path = os.path.join(output_dir, "tokenizer_training_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Saved training config to: {config_path}")
    
    def test_tokenizer(self, test_texts: List[str]):
        """Test the trained tokenizer on sample texts."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        
        print("\n" + "="*60)
        print("TOKENIZER TEST")
        print("="*60)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text[:100]}...")
            
            # Encode
            encoding = self.tokenizer.encode(text)
            tokens = encoding.tokens
            ids = encoding.ids
            
            print(f"  Tokens ({len(tokens)}): {tokens[:20]}...")
            print(f"  IDs ({len(ids)}): {ids[:20]}...")
            
            # Decode
            decoded = self.tokenizer.decode(ids)
            print(f"  Decoded: {decoded[:100]}...")
            
            # Check if decoding matches original (approximately)
            match = text.strip() == decoded.strip()
            print(f"  Match: {'✓' if match else '✗'}")
        
        print("\n" + "="*60)


def load_tokenizer(tokenizer_path: str):
    """
    Load a trained tokenizer.
    
    Args:
        tokenizer_path: Path to tokenizer directory or .json file
    
    Returns:
        PreTrainedTokenizerFast: Loaded tokenizer
    """
    if os.path.isdir(tokenizer_path):
        # Load HuggingFace tokenizer from directory
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        # Load from .json file
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            mask_token="[MASK]",
        )
    
    print(f"Tokenizer loaded from: {tokenizer_path}")
    print(f"Vocabulary size: {len(tokenizer)}")
    
    return tokenizer


def collect_texts_from_directory(data_dir: str, extensions: List[str] = None):
    """
    Collect all text files from a directory.
    
    Args:
        data_dir: Directory containing text files
        extensions: List of file extensions to include (default: ['.txt'])
    
    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = ['.txt']
    
    data_path = Path(data_dir)
    files = []
    
    for ext in extensions:
        files.extend(data_path.glob(f"**/*{ext}"))
    
    files = [str(f) for f in files]
    print(f"Found {len(files)} text files in {data_dir}")
    
    return files


def create_text_iterator(files: List[str], chunk_size: int = 1000):
    """
    Create an iterator that yields text chunks from files.
    
    Args:
        files: List of file paths
        chunk_size: Number of lines to yield at once
    
    Yields:
        Chunks of text
    """
    buffer = []
    
    for file_path in files:
        print(f"Reading: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        buffer.append(line)
                        
                        if len(buffer) >= chunk_size:
                            yield " ".join(buffer)
                            buffer = []
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    # Yield remaining buffer
    if buffer:
        yield " ".join(buffer)


def main():
    """Example usage of the tokenizer trainer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for Bengali text")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing text files")
    parser.add_argument("--output_dir", type=str, default="./tokenizer", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--min_frequency", type=int, default=2, help="Minimum token frequency")
    parser.add_argument("--model_name", type=str, default="bengali_tokenizer", help="Model name")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text")
    
    args = parser.parse_args()
    
    # Collect text files
    files = collect_texts_from_directory(args.data_dir)
    
    if not files:
        print("No text files found! Please add .txt files to the data directory.")
        return
    
    # Create trainer
    trainer = BengaliTokenizerTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        lowercase=args.lowercase,
    )
    
    # Train tokenizer
    trainer.train_from_files(
        files=files,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )
    
    # Test tokenizer with sample texts
    test_texts = [
        "এটি একটি পরীক্ষা বাক্য।",  # Bengali: "This is a test sentence."
        "বাংলা ভাষা প্রক্রিয়াকরণ",  # Bengali: "Bengali language processing"
    ]
    
    # Read first few lines from first file for testing
    if files:
        try:
            with open(files[0], 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    if line.strip():
                        test_texts.append(line.strip())
        except Exception as e:
            print(f"Could not read test texts: {e}")
    
    trainer.test_tokenizer(test_texts)
    
    print("\n✓ Tokenizer training complete!")
    print(f"Tokenizer saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
