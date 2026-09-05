"""
Dataset Loading and Preprocessing Module
Supports both pretraining and supervised fine-tuning (SFT) datasets
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from transformers import PreTrainedTokenizer
import numpy as np


class PretrainingDataset(Dataset):
    """
    Dataset for causal language modeling pretraining.
    Chunks text into fixed-length sequences for next-token prediction.
    """
    
    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        block_size: int = 2048,
        stride: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize pretraining dataset.
        
        Args:
            data_path: Path to text file(s) or directory
            tokenizer: Trained tokenizer
            block_size: Maximum sequence length
            stride: Stride for chunking (default: block_size, no overlap)
            cache_dir: Directory to cache processed data
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        
        print(f"Loading pretraining data from: {data_path}")
        
        # Load and tokenize data
        self.examples = []
        self._load_and_tokenize(data_path, cache_dir)
        
        print(f"Loaded {len(self.examples)} training examples")
    
    def _load_and_tokenize(self, data_path: Union[str, List[str]], cache_dir: Optional[str]):
        """Load text files and tokenize them into chunks."""
        # Collect text files
        if isinstance(data_path, str):
            if os.path.isfile(data_path):
                files = [data_path]
            elif os.path.isdir(data_path):
                files = list(Path(data_path).glob("**/*.txt"))
                files = [str(f) for f in files]
            else:
                raise ValueError(f"Invalid data_path: {data_path}")
        else:
            files = data_path
        
        print(f"Processing {len(files)} file(s)...")
        
        # Process each file
        all_token_ids = []
        
        for file_path in files:
            print(f"  Reading: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Tokenize
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                all_token_ids.extend(token_ids)
                
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
                continue
        
        print(f"Total tokens: {len(all_token_ids):,}")
        
        # Chunk into blocks
        for i in range(0, len(all_token_ids) - self.block_size + 1, self.stride):
            chunk = all_token_ids[i:i + self.block_size]
            if len(chunk) == self.block_size:
                self.examples.append(chunk)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get a training example.
        
        Returns:
            Dict with 'input_ids' and 'labels' (same as input_ids for CLM)
        """
        token_ids = self.examples[idx]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(token_ids, dtype=torch.long),
        }


class StreamingPretrainingDataset:
    """
    Memory-efficient streaming dataset for large corpora.
    Uses HuggingFace datasets library for efficient loading.
    """
    
    def __init__(
        self,
        data_files: Union[str, List[str], Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        block_size: int = 2048,
        streaming: bool = True,
        buffer_size: int = 10000,
    ):
        """
        Initialize streaming pretraining dataset.
        
        Args:
            data_files: Path to text file(s)
            tokenizer: Trained tokenizer
            block_size: Maximum sequence length
            streaming: Whether to stream data (recommended for large datasets)
            buffer_size: Buffer size for shuffling
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.buffer_size = buffer_size
        
        print(f"Loading streaming dataset...")
        
        # Load dataset
        self.dataset = load_dataset(
            'text',
            data_files=data_files,
            streaming=streaming,
            split='train'
        )
        
        # Tokenize and chunk
        self.dataset = self.dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=['text'],
        )
        
        if streaming:
            self.dataset = self.dataset.shuffle(buffer_size=buffer_size)
    
    def _tokenize_function(self, examples):
        """Tokenize text examples."""
        # Tokenize all texts
        tokenized = self.tokenizer(
            examples['text'],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        
        # Concatenate all texts
        concatenated = {k: sum(tokenized[k], []) for k in tokenized.keys()}
        total_length = len(concatenated['input_ids'])
        
        # Chunk into blocks
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        
        result = {
            k: [t[i:i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated.items()
        }
        
        # Add labels (same as input_ids for CLM)
        result['labels'] = result['input_ids'].copy()
        
        return result
    
    def get_dataloader(self, batch_size: int, num_workers: int = 0):
        """Create a DataLoader for this dataset."""
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )


class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning (SFT).
    Expects instruction-response pairs in JSONL format.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize SFT dataset.
        
        Args:
            data_path: Path to JSONL file with instruction-response pairs
            tokenizer: Trained tokenizer
            max_length: Maximum sequence length
            prompt_template: Template for formatting prompts
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Default prompt template for Bengali
        if prompt_template is None:
            self.prompt_template = (
                "নির্দেশ: {instruction}\n\n"
                "উত্তর: {response}"
            )
        else:
            self.prompt_template = prompt_template
        
        print(f"Loading SFT data from: {data_path}")
        
        # Load data
        self.examples = self._load_data(data_path)
        
        print(f"Loaded {len(self.examples)} SFT examples")
    
    def _load_data(self, data_path: str) -> List[Dict[str, str]]:
        """Load JSONL data file."""
        examples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    
                    # Validate required fields
                    if 'instruction' not in example or 'response' not in example:
                        print(f"Warning: Line {line_num} missing required fields, skipping")
                        continue
                    
                    examples.append(example)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Error parsing line {line_num}: {e}")
                    continue
        
        return examples
    
    def _format_example(self, example: Dict[str, str]) -> str:
        """Format an example using the prompt template."""
        # Add input field if available
        instruction = example['instruction']
        if 'input' in example and example['input']:
            instruction = f"{instruction}\n\nইনপুট: {example['input']}"
        
        formatted = self.prompt_template.format(
            instruction=instruction,
            response=example['response']
        )
        
        return formatted
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get a training example.
        
        Returns:
            Dict with 'input_ids', 'attention_mask', and 'labels'
        """
        example = self.examples[idx]
        
        # Format the example
        text = self._format_example(example)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For SFT, we typically want to compute loss only on the response
        # Here we use the same as input_ids, but you can mask the instruction part
        labels = input_ids.clone()
        
        # Optional: Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class ChatDataset(Dataset):
    """
    Dataset for multi-turn conversation fine-tuning.
    Expects conversations in JSONL format.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
    ):
        """
        Initialize chat dataset.
        
        Args:
            data_path: Path to JSONL file with conversations
            tokenizer: Trained tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loading chat data from: {data_path}")
        
        # Load data
        self.examples = self._load_data(data_path)
        
        print(f"Loaded {len(self.examples)} chat examples")
    
    def _load_data(self, data_path: str) -> List[List[Dict[str, str]]]:
        """Load JSONL data file with conversations."""
        examples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    
                    # Validate conversation format
                    if 'messages' not in example:
                        print(f"Warning: Line {line_num} missing 'messages' field, skipping")
                        continue
                    
                    examples.append(example['messages'])
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Error parsing line {line_num}: {e}")
                    continue
        
        return examples
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format a conversation into a single string."""
        formatted_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'user':
                formatted_parts.append(f"ব্যবহারকারী: {content}")
            elif role == 'assistant':
                formatted_parts.append(f"সহায়ক: {content}")
            elif role == 'system':
                formatted_parts.append(f"সিস্টেম: {content}")
        
        return "\n\n".join(formatted_parts)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get a training example."""
        messages = self.examples[idx]
        
        # Format the conversation
        text = self._format_conversation(messages)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader from a dataset.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (faster GPU transfer)
    
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def prepare_pretraining_data(
    raw_data_dir: str,
    output_dir: str,
    tokenizer: PreTrainedTokenizer,
    block_size: int = 2048,
    train_split: float = 0.95,
):
    """
    Prepare and split pretraining data.
    
    Args:
        raw_data_dir: Directory with raw text files
        output_dir: Directory to save processed data
        tokenizer: Trained tokenizer
        block_size: Sequence length
        train_split: Proportion of data for training
    """
    print("Preparing pretraining data...")
    
    # Load all data
    dataset = PretrainingDataset(
        data_path=raw_data_dir,
        tokenizer=tokenizer,
        block_size=block_size,
    )
    
    # Split into train/val
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    print(f"Splitting into train ({train_size}) and validation ({val_size})")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    
    # Save train data
    train_data = dataset.examples[:train_size]
    train_path = os.path.join(output_dir, 'train', 'data.pt')
    torch.save(train_data, train_path)
    print(f"Saved training data to: {train_path}")
    
    # Save val data
    val_data = dataset.examples[train_size:]
    val_path = os.path.join(output_dir, 'val', 'data.pt')
    torch.save(val_data, val_path)
    print(f"Saved validation data to: {val_path}")
    
    # Save metadata
    metadata = {
        'block_size': block_size,
        'train_size': train_size,
        'val_size': val_size,
        'vocab_size': len(tokenizer),
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    print("✓ Data preparation complete!")


def main():
    """Example usage of dataset classes."""
    from src.tokenizer import load_tokenizer
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer("./tokenizer")
    
    # Example 1: Pretraining dataset
    print("\n" + "="*60)
    print("Testing Pretraining Dataset")
    print("="*60)
    
    try:
        dataset = PretrainingDataset(
            data_path="./data/raw",
            tokenizer=tokenizer,
            block_size=128,
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample input_ids shape: {sample['input_ids'].shape}")
            print(f"Sample labels shape: {sample['labels'].shape}")
            print(f"First 20 tokens: {sample['input_ids'][:20].tolist()}")
    except Exception as e:
        print(f"Could not test pretraining dataset: {e}")
    
    # Example 2: SFT dataset
    print("\n" + "="*60)
    print("Testing SFT Dataset")
    print("="*60)
    
    try:
        sft_dataset = SFTDataset(
            data_path="./data/examples/sft_data.jsonl",
            tokenizer=tokenizer,
            max_length=512,
        )
        
        if len(sft_dataset) > 0:
            sample = sft_dataset[0]
            print(f"Sample input_ids shape: {sample['input_ids'].shape}")
            print(f"Sample attention_mask shape: {sample['attention_mask'].shape}")
            print(f"Sample labels shape: {sample['labels'].shape}")
    except Exception as e:
        print(f"Could not test SFT dataset: {e}")
    
    print("\n✓ Dataset tests complete!")


if __name__ == "__main__":
    main()
