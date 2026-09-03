"""
Script to train a BPE tokenizer for Bengali text
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.tokenizer import BengaliTokenizerTrainer, collect_texts_from_directory
from src.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for Bengali text")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing text files (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Vocabulary size (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    tokenizer_config = config.get("tokenizer_training", {})
    
    # Override with command line arguments
    data_dir = args.data_dir or tokenizer_config.get("data_dir", "data/raw")
    output_dir = args.output_dir or tokenizer_config.get("output_dir", "tokenizer")
    vocab_size = args.vocab_size or tokenizer_config.get("vocab_size", 32000)
    min_frequency = tokenizer_config.get("min_frequency", 2)
    model_name = tokenizer_config.get("model_name", "bengali_tokenizer")
    lowercase = tokenizer_config.get("lowercase", False)
    special_tokens = tokenizer_config.get("special_tokens")
    
    print("="*60)
    print("TOKENIZER TRAINING")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Min frequency: {min_frequency}")
    print(f"Model name: {model_name}")
    print("="*60 + "\n")
    
    # Collect text files
    print("Collecting text files...")
    files = collect_texts_from_directory(data_dir, extensions=['.txt'])
    
    if not files:
        print("\n❌ Error: No text files found!")
        print(f"Please add .txt files to: {data_dir}")
        print("\nExample file structure:")
        print("  data/raw/")
        print("    ├── book1.txt")
        print("    ├── book2.txt")
        print("    └── articles.txt")
        return
    
    print(f"\nFound {len(files)} text file(s)")
    
    # Create trainer
    trainer = BengaliTokenizerTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        lowercase=lowercase,
    )
    
    # Train tokenizer
    print("\nStarting tokenizer training...")
    trainer.train_from_files(
        files=files,
        output_dir=output_dir,
        model_name=model_name,
    )
    
    # Test tokenizer with sample texts
    test_texts = [
        "এটি একটি পরীক্ষা বাক্য।",  # Bengali: "This is a test sentence."
        "বাংলা ভাষা প্রক্রিয়াকরণ একটি গুরুত্বপূর্ণ বিষয়।",  # Bengali: "Bengali language processing is an important topic."
        "আমরা একটি ভাষা মডেল তৈরি করছি।",  # Bengali: "We are creating a language model."
    ]
    
    # Try to read sample from data files
    if files:
        try:
            with open(files[0], 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 2:
                        break
                    if line.strip():
                        test_texts.append(line.strip()[:200])
        except Exception as e:
            print(f"Could not read sample texts: {e}")
    
    print("\n" + "="*60)
    print("Testing tokenizer...")
    trainer.test_tokenizer(test_texts)
    
    print("\n" + "="*60)
    print("✓ Tokenizer training complete!")
    print(f"Tokenizer saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
