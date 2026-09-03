Project: Build a 100M Parameter Language Model from Scratch
Goal
Create a complete, production-ready codebase for training a 100 million parameter language model from scratch. The final output should be a model that can be converted to Android-friendly format (like .tflite or .gguf).

Technical Stack
Python 3.10+

PyTorch 2.0+

Hugging Face Transformers & Tokenizers libraries

Google Colab for training (free T4 GPU)

Project Structure Expected
project/
├── configs/
│ └── model_config.yaml # Model hyperparameters
├── data/
│ ├── raw/ # Place your raw text files here
│ └── processed/ # Tokenized datasets
├── src/
│ ├── model.py # Transformer model definition
│ ├── tokenizer.py # Custom tokenizer training
│ ├── dataset.py # Data loading & preprocessing
│ ├── train.py # Training loop
│ └── utils.py # Helper functions
├── scripts/
│ ├── prepare_data.py # Data preparation script
│ ├── train_tokenizer.py # Tokenizer training script
│ ├── pretrain.py # Pretraining script
│ ├── sft.py # Supervised fine-tuning script
│ └── export.py # Export to Android format
├── requirements.txt
├── README.md
└── colab_notebook.ipynb # Ready-to-run Colab notebook

Detailed Implementation Requirements
1. Model Architecture (src/model.py)
Create a decoder-only Transformer model with:

Hidden size: 768

Number of layers: 6

Attention heads: 12

Vocabulary size: 32,000 (configurable)

Context length: 2,048 tokens

Total parameters: ~100M
Implement the following components from scratch using PyTorch:

Positional encoding (sinusoidal or learned)

Multi-head self-attention with causal masking

Feed-forward network with SwiGLU activation

Layer normalization (pre-norm formulation)

Residual connections
Use the PreTrainedModel class from Hugging Face Transformers for compatibility.

2. Tokenizer Training (src/tokenizer.py)
Train a BPE tokenizer on your Bengali text corpus using the tokenizers library:

Use the BPE model

Set vocab_size = 32,000

Include special tokens: [PAD], [UNK], [BOS], [EOS], [MASK]

Save the tokenizer as tokenizer.json and tokenizer_config.json

3. Data Preparation (src/dataset.py)
Create a dataset class that:

Loads raw text files from data/raw/

Tokenizes the text using the trained tokenizer

Chunks into sequences of length block_size (2048)

Uses datasets library for efficient streaming

Supports both pretraining (next-token prediction) and SFT (instruction-response) formats

4. Training Pipeline (src/train.py)
Implement training using Hugging Face's Trainer or a custom training loop:

Support for mixed precision (fp16/bf16)

Gradient accumulation

Learning rate scheduling (cosine decay with warmup)

Checkpoint saving and loading

Logging with wandb or tensorboard

Evaluation on validation set (perplexity)

5. Configuration (configs/model_config.yaml)
Create a YAML config file with all hyperparameters:

yaml

model:  
  vocab_size: 32000  
  hidden_size: 768  
  num_hidden_layers: 6  
  num_attention_heads: 12  
  intermediate_size: 2048  
  max_position_embeddings: 2048  
  dropout: 0.1  
training:  
  batch_size: 32  
  learning_rate: 3e-4  
  warmup_steps: 500  
  num_train_epochs: 3  
  max_steps: 5000  
  gradient_accumulation_steps: 4  
  fp16: true  
  save_steps: 500  
  logging_steps: 50  
data:  
  train_data_path: "data/processed/train"  
  val_data_path: "data/processed/val"  
  block_size: 2048  
 

**6. Colab Notebook (colab_notebook.ipynb)**  
Create a complete Jupyter notebook that:

- Mounts Google Drive for persistent storage  
- Installs all dependencies  
- Downloads and prepares the dataset  
- Trains the tokenizer (or loads a pre-trained one)  
- Runs pretraining with the specified config  
- Runs supervised fine-tuning (SFT)  
- Exports the model to `.tflite` or `.gguf` format for Android

**7. Export Script (scripts/export.py)**

- Convert the trained PyTorch model to ONNX  
- Use `ai-edge-torch` to convert to TensorFlow Lite (`.tflite`)  
- Or use `llama.cpp` to convert to GGUF format  
- Quantize the model (INT8 or INT4) for mobile deployment

**8. Requirements (requirements.txt)**  
Include all necessary packages:  
torch>=2.0.0  
transformers>=4.30.0  
tokenizers>=0.13.0  
datasets>=2.12.0  
accelerate>=0.20.0  
peft>=0.5.0  
bitsandbytes>=0.41.0  
sentencepiece>=0.1.99  
pyyaml>=6.0  
tqdm>=4.65.0  
wandb>=0.15.0  
ai-edge-torch>=0.1.0

**9. README.md**  
Write comprehensive documentation covering:

- Project overview and goals  
- Setup instructions  
- Data preparation steps  
- Training commands  
- Evaluation instructions  
- Export to Android guide  
- License information (MIT or Apache 2.0)

**10. Sample Training Data**  
Include a small sample dataset in `data/examples/` for testing:

- A few Bengali text files (from NCTB books or other sources)  
- An instruction-response JSONL file for SFT

**Success Criteria**

- □  
  Model trains without errors on Colab T4 GPU  
- □  
  Achieves reasonable loss (< 5.0 on validation)  
- □  
  Can generate coherent Bengali text  
- □  
  Exported model runs on Android device  
- □  
  Complete codebase is self-contained and well-documented

**Additional Notes**

- Use `torch.compile` for faster training when possible  
- Implement gradient checkpointing to save memory  
- The model should be trained on **Bengali text only** to keep it lightweight  
- After training, the model should be fully yours with no external license restrictions  
- All code should be原创 and not copy-pasted from other repositories
Kiro