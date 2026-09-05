"""
Utility Functions
Helper functions for configuration, logging, seed setting, and more
"""

import os
import yaml
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file (.yaml or .json)
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    print(f"Loaded config from: {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path (.yaml or .json)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix in ['.yaml', '.yml']:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif output_path.suffix == '.json':
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {output_path.suffix}")
    
    print(f"Saved config to: {output_path}")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA over CPU
    
    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    
    return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
    
    Returns:
        Dictionary with parameter counts
    """
    if trainable_only:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable = total
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
    }


def print_model_info(model: torch.nn.Module):
    """
    Print detailed model information.
    
    Args:
        model: PyTorch model
    """
    params = count_parameters(model)
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Total parameters: {params['total']:,} ({params['total_millions']:.2f}M)")
    print(f"Trainable parameters: {params['trainable']:,} ({params['trainable_millions']:.2f}M)")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    
    # Print model structure summary
    print("\nModel Structure:")
    print("-" * 60)
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:20s}: {module_params:12,} params")
    print("="*60 + "\n")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model size in memory.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with size information
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        "param_size_mb": param_size / 1024 / 1024,
        "buffer_size_mb": buffer_size / 1024 / 1024,
        "total_size_mb": total_size / 1024 / 1024,
    }


def create_run_name(prefix: str = "run") -> str:
    """
    Create a unique run name with timestamp.
    
    Args:
        prefix: Prefix for the run name
    
    Returns:
        Unique run name
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def ensure_dir(path: Union[str, Path]):
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Get GPU memory usage statistics.
    
    Returns:
        Dictionary with memory usage in GB
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "total_gb": total,
        "free_gb": total - reserved,
    }


def print_gpu_memory():
    """Print GPU memory usage."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    mem = get_gpu_memory_usage()
    print(f"\nGPU Memory Usage:")
    print(f"  Allocated: {mem['allocated_gb']:.2f} GB")
    print(f"  Reserved:  {mem['reserved_gb']:.2f} GB")
    print(f"  Total:     {mem['total_gb']:.2f} GB")
    print(f"  Free:      {mem['free_gb']:.2f} GB")


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics.
        
        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class Timer:
    """Simple timer for measuring elapsed time."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        """Start the timer."""
        self.start_time = datetime.now()
    
    def stop(self):
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            return 0
        
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        return self.elapsed
    
    def get_elapsed(self) -> float:
        """Get elapsed time without stopping."""
        if self.start_time is None:
            return 0
        
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_formatted(self) -> str:
        """Get formatted elapsed time."""
        return format_time(self.get_elapsed())


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configuration dictionaries.
    Override config takes precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
    
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def save_training_args(args: Dict, output_dir: str, filename: str = "training_args.json"):
    """
    Save training arguments to file.
    
    Args:
        args: Training arguments dictionary
        output_dir: Output directory
        filename: Filename for saving args
    """
    ensure_dir(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(args, f, indent=2, default=str)
    
    print(f"Training args saved to: {output_path}")


def load_checkpoint_safely(model: torch.nn.Module, checkpoint_path: str, strict: bool = True):
    """
    Safely load model checkpoint with error handling.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce key matching
    
    Returns:
        Loaded model
    """
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        
        print("✓ Checkpoint loaded successfully")
        
        return model
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    samples_per_second: float,
) -> Dict[str, float]:
    """
    Estimate total training time.
    
    Args:
        num_samples: Total number of training samples
        batch_size: Batch size
        num_epochs: Number of training epochs
        samples_per_second: Processing speed (samples/sec)
    
    Returns:
        Dictionary with time estimates
    """
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * num_epochs
    total_seconds = num_samples * num_epochs / samples_per_second
    
    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "estimated_seconds": total_seconds,
        "estimated_hours": total_seconds / 3600,
        "estimated_formatted": format_time(total_seconds),
    }


def print_system_info():
    """Print system and environment information."""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    # PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
    
    # Python packages
    try:
        import transformers
        print(f"\nTransformers version: {transformers.__version__}")
    except ImportError:
        print("\nTransformers: not installed")
    
    try:
        import datasets
        print(f"Datasets version: {datasets.__version__}")
    except ImportError:
        print("Datasets: not installed")
    
    print("="*60 + "\n")


def validate_config(config: Dict, required_keys: list) -> bool:
    """
    Validate that configuration has all required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_keys = []
    
    for key in required_keys:
        if '.' in key:
            # Handle nested keys
            keys = key.split('.')
            current = config
            for k in keys:
                if k not in current:
                    missing_keys.append(key)
                    break
                current = current[k]
        else:
            if key not in config:
                missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    return True


if __name__ == "__main__":
    print("Utility functions module loaded successfully!\n")
    
    # Test some utilities
    print_system_info()
    
    # Test seed setting
    set_seed(42)
    
    # Test device selection
    device = get_device()
    
    # Test GPU memory
    print_gpu_memory()
    
    # Test timer
    timer = Timer()
    timer.start()
    import time
    time.sleep(0.1)
    print(f"\nTimer test: {timer.get_formatted()}")
    
    # Test average meter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average meter test: avg={meter.avg:.2f}, count={meter.count}")
    
    print("\n✓ All utility tests passed!")
