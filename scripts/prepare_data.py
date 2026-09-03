"""
Script to prepare and preprocess training data
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import prepare_pretraining_data
from src.tokenizer import load_tokenizer
from src.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=None,
        help="Directory with raw text files (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to trained tokenizer (overrides config)"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Sequence length (overrides config)"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.95,
        help="Proportion of data for training"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    data_config = config.get("data", {})
    
    # Override with command line arguments
    raw_data_dir = args.raw_data_dir or "data/raw"
    output_dir = args.output_dir
    tokenizer_path = args.tokenizer_path or data_config.get("tokenizer_path", "tokenizer")
    block_size = args.block_size or data_config.get("block_size", 2048)
    
    print("="*60)
    print("DATA PREPARATION")
    print("="*60)
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tokenizer path: {tokenizer_path}")
    print(f"Block size: {block_size}")
    print(f"Train split: {args.train_split}")
    print("="*60 + "\n")
    
    # Check if raw data exists
    raw_path = Path(raw_data_dir)
    if not raw_path.exists() or not list(raw_path.glob("**/*.txt")):
        print("❌ Error: No text files found in raw data directory!")
        print(f"Please add .txt files to: {raw_data_dir}")
        return
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = load_tokenizer(tokenizer_path)
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print(f"\nPlease train tokenizer first:")
        print(f"  python scripts/train_tokenizer.py")
        return
    
    # Prepare data
    print("\nPreparing pretraining data...")
    print("This may take a while for large datasets...\n")
    
    try:
        prepare_pretraining_data(
            raw_data_dir=raw_data_dir,
            output_dir=output_dir,
            tokenizer=tokenizer,
            block_size=block_size,
            train_split=args.train_split,
        )
        
        print("\n" + "="*60)
        print("✓ Data preparation complete!")
        print(f"Processed data saved to: {output_dir}")
        print("="*60)
        
        # Print next steps
        print("\nNext steps:")
        print("  1. Review the processed data")
        print("  2. Start pretraining:")
        print("     python scripts/pretrain.py")
        
    except Exception as e:
        print(f"\n❌ Error during data preparation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
