"""
Script to export trained model to Android-friendly formats
Supports ONNX, TensorFlow Lite, and GGUF formats
"""

import sys
import os
import argparse
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import CustomLMModel
from src.tokenizer import load_tokenizer
from src.utils import load_config, ensure_dir


def export_to_onnx(model, tokenizer, output_path, sample_input_length=128):
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        output_path: Output file path
        sample_input_length: Length of sample input for tracing
    """
    print("\n" + "="*60)
    print("Exporting to ONNX")
    print("="*60)
    
    try:
        import onnx
        
        model.eval()
        
        # Create dummy input
        batch_size = 1
        dummy_input = torch.randint(
            0, len(tokenizer), 
            (batch_size, sample_input_length),
            dtype=torch.long
        )
        
        print(f"Sample input shape: {dummy_input.shape}")
        
        # Export to ONNX
        print("Exporting model...")
        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=14,
            do_constant_folding=True,
        )
        
        # Verify ONNX model
        print("Verifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ ONNX export successful!")
        print(f"  File: {output_path}")
        print(f"  Size: {file_size:.2f} MB")
        
        return True
        
    except ImportError:
        print("❌ Error: onnx package not installed")
        print("Install with: pip install onnx")
        return False
    except Exception as e:
        print(f"❌ Error exporting to ONNX: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_to_tflite(model, tokenizer, output_path, quantize=True):
    """
    Export model to TensorFlow Lite format.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        output_path: Output file path
        quantize: Whether to apply dynamic quantization
    """
    print("\n" + "="*60)
    print("Exporting to TensorFlow Lite")
    print("="*60)
    
    try:
        # Try using ai-edge-torch (recommended for PyTorch to TFLite)
        try:
            import ai_edge_torch
            
            model.eval()
            
            # Create sample input
            sample_input = torch.randint(0, len(tokenizer), (1, 128), dtype=torch.long)
            
            print("Converting model with ai-edge-torch...")
            
            # Convert to TFLite
            edge_model = ai_edge_torch.convert(
                model.eval(), 
                (sample_input,)
            )
            
            # Export
            edge_model.export(output_path)
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✓ TFLite export successful!")
            print(f"  File: {output_path}")
            print(f"  Size: {file_size:.2f} MB")
            
            return True
            
        except ImportError:
            print("Note: ai-edge-torch not available, trying alternative method...")
            
            # Alternative: Export via ONNX and convert to TFLite
            try:
                import tensorflow as tf
                import onnx
                from onnx_tf.backend import prepare
                
                # First export to ONNX
                onnx_path = output_path.replace('.tflite', '.onnx')
                if export_to_onnx(model, tokenizer, onnx_path):
                    print("\nConverting ONNX to TensorFlow...")
                    
                    # Load ONNX model
                    onnx_model = onnx.load(onnx_path)
                    
                    # Convert to TensorFlow
                    tf_rep = prepare(onnx_model)
                    
                    # Export to SavedModel
                    saved_model_path = output_path.replace('.tflite', '_saved_model')
                    tf_rep.export_graph(saved_model_path)
                    
                    # Convert to TFLite
                    print("Converting to TFLite...")
                    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
                    
                    if quantize:
                        print("Applying dynamic quantization...")
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    
                    tflite_model = converter.convert()
                    
                    # Save TFLite model
                    with open(output_path, 'wb') as f:
                        f.write(tflite_model)
                    
                    file_size = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"✓ TFLite export successful!")
                    print(f"  File: {output_path}")
                    print(f"  Size: {file_size:.2f} MB")
                    
                    # Cleanup
                    if os.path.exists(onnx_path):
                        os.remove(onnx_path)
                    
                    return True
                else:
                    return False
                    
            except ImportError as e:
                print(f"❌ Required packages not installed: {e}")
                print("\nInstall with:")
                print("  pip install tensorflow onnx onnx-tf")
                return False
    
    except Exception as e:
        print(f"❌ Error exporting to TFLite: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_to_gguf(model_path, output_path, quantization_type="q4_0"):
    """
    Export model to GGUF format (for llama.cpp).
    
    Args:
        model_path: Path to trained model
        output_path: Output file path
        quantization_type: Quantization type (f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0)
    """
    print("\n" + "="*60)
    print("Exporting to GGUF (llama.cpp format)")
    print("="*60)
    
    print("\n⚠️  GGUF export requires llama.cpp to be installed and configured.")
    print("\nSteps to export manually:")
    print("="*60)
    print("1. Clone llama.cpp:")
    print("   git clone https://github.com/ggerganov/llama.cpp")
    print("   cd llama.cpp")
    print("")
    print("2. Convert model to GGUF:")
    print(f"   python convert.py {model_path} --outtype {quantization_type}")
    print("")
    print("3. Quantize (if needed):")
    print(f"   ./quantize {model_path}/ggml-model-f32.gguf {output_path} {quantization_type}")
    print("="*60)
    
    print("\nNote: This custom model architecture may require modifications to llama.cpp")
    print("      conversion scripts to handle the specific model structure.")
    
    return False


def save_model_info(model, tokenizer, output_dir):
    """Save model information for deployment."""
    import json
    
    info = {
        "model_type": "custom_lm",
        "vocab_size": len(tokenizer),
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "num_heads": model.config.num_attention_heads,
        "max_length": model.config.max_position_embeddings,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "special_tokens": {
            "pad_token": tokenizer.pad_token,
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
            "unk_token": tokenizer.unk_token,
        }
    }
    
    info_path = os.path.join(output_dir, "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ Model info saved to: {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Export model to Android-friendly formats")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "tflite", "gguf", "all"],
        default="all",
        help="Export format"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Apply quantization"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizer",
        help="Path to tokenizer"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MODEL EXPORT FOR ANDROID")
    print("="*60)
    
    # Load config
    config = load_config(args.config) if os.path.exists(args.config) else {}
    export_config = config.get("export", {})
    
    # Override with command line arguments
    model_path = args.model_path or export_config.get("model_path", "outputs/pretrain/best_model")
    output_dir = args.output_dir or export_config.get("output_dir", "exports")
    
    print(f"\nModel path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Export format: {args.format}")
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print("\nPlease train the model first:")
        print("  python scripts/pretrain.py")
        return
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = load_tokenizer(args.tokenizer_path)
        print(f"✓ Loaded tokenizer (vocab size: {len(tokenizer)})")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return
    
    # Load model
    print("\nLoading model...")
    try:
        model = CustomLMModel.from_pretrained(model_path)
        model.eval()
        print(f"✓ Loaded model")
        
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Save model info
    save_model_info(model, tokenizer, output_dir)
    
    # Export to requested formats
    success_count = 0
    
    if args.format in ["onnx", "all"]:
        onnx_path = os.path.join(output_dir, "model.onnx")
        if export_to_onnx(model, tokenizer, onnx_path):
            success_count += 1
    
    if args.format in ["tflite", "all"]:
        tflite_path = os.path.join(output_dir, "model.tflite")
        if export_to_tflite(model, tokenizer, tflite_path, quantize=args.quantize):
            success_count += 1
    
    if args.format in ["gguf", "all"]:
        gguf_path = os.path.join(output_dir, "model.gguf")
        export_to_gguf(model_path, gguf_path, 
                      quantization_type=export_config.get("gguf_type", "q4_0"))
    
    # Summary
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    print(f"Successfully exported: {success_count} format(s)")
    print(f"Output directory: {output_dir}")
    
    print("\nExported files:")
    for file in Path(output_dir).glob("*"):
        if file.is_file():
            size = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name} ({size:.2f} MB)")
    
    print("\n" + "="*60)
    print("ANDROID INTEGRATION GUIDE")
    print("="*60)
    print("\nFor ONNX Runtime:")
    print("  1. Add ONNX Runtime Mobile to your Android project")
    print("  2. Copy model.onnx to assets/")
    print("  3. Load with: session = OrtSession(\"model.onnx\")")
    
    print("\nFor TensorFlow Lite:")
    print("  1. Add TensorFlow Lite to your Android project")
    print("  2. Copy model.tflite to assets/")
    print("  3. Load with: Interpreter(\"model.tflite\")")
    
    print("\nFor GGUF (llama.cpp):")
    print("  1. Build llama.cpp for Android")
    print("  2. Copy model.gguf to device storage")
    print("  3. Use llama.cpp Android bindings")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
