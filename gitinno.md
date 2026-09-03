# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Training outputs
outputs/
checkpoints/
logs/
runs/
wandb/

# Data (keep structure, not actual data)
data/raw/*.txt
data/processed/
!data/examples/

# Model exports
exports/
*.onnx
*.tflite
*.gguf

# Tokenizer (keep examples)
tokenizer/*.json
tokenizer/*.txt
tokenizer/*.model

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# PyTorch
*.pt
*.pth
*.ckpt

# Logs
*.log

# Temporary files
tmp/
temp/
*.tmp

# OS
Thumbs.db
.DS_Store
