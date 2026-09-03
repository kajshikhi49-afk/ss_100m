"""
Script to generate text from trained model
"""

import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import CustomLMModel
from src.tokenizer import load_tokenizer
from src.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizer",
        help="Path to tokenizer"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    device = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = load_tokenizer(args.tokenizer_path)
        print(f"✓ Loaded tokenizer (vocab size: {len(tokenizer)})\n")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return
    
    # Load model
    print("Loading model...")
    try:
        model = CustomLMModel.from_pretrained(args.model_path)
        model.to(device)
        model.eval()
        print(f"✓ Loaded model from: {args.model_path}\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get prompt
    if args.prompt:
        prompts = [args.prompt]
    else:
        # Default Bengali prompts
        prompts = [
            "একবার এক",
            "বাংলাদেশ হল",
            "শিক্ষা জীবনের",
            "আমরা যখন",
        ]
        print("No prompt provided, using default prompts:\n")
    
    # Generate text
    print("="*60)
    print("TEXT GENERATION")
    print("="*60 + "\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: {prompt}")
        print("-"*60)
        
        for sample_num in range(args.num_samples):
            print(f"\nSample {sample_num + 1}:")
            
            try:
                generated = model.generate_text(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=device,
                )
                
                print(generated)
                
            except Exception as e:
                print(f"❌ Error during generation: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
