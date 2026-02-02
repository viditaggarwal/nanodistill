# Model Setup & Inference Guide

## Overview

NanoDistill uses a **student model** (the small model you fine-tune) and a **teacher model** (Claude, used to generate training data).

- **Teacher**: Claude Sonnet 4.5 (runs on Anthropic servers, no download needed)
- **Student**: Llama-3-8B Instruct 4-bit (downloaded automatically, runs locally)

## Student Model: Llama-3-8B-Instruct-4bit

### Automatic Download (Recommended)

When you run `distill()`, MLX-LM **automatically downloads and caches** the model:

```python
from nanodistill import distill

result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
    teacher="claude-sonnet-4-5",
    # student model is downloaded automatically on first use
)
```

**First run**: ~10-15 minutes (downloads ~4GB)
**Subsequent runs**: Instant (uses cached model)

### Model Location

Downloaded to: `~/.cache/huggingface/hub/`

```
~/.cache/huggingface/hub/
├── models--mlx-community--Llama-3-8B-Instruct-4bit/
│   └── snapshots/
│       └── [hash]/
│           ├── model.safetensors
│           ├── config.json
│           ├── tokenizer.json
│           └── ...
```

### Manual Pre-download (Optional)

If you want to download before running distillation:

```python
from mlx_lm import load

model_id = "mlx-community/Llama-3-8B-Instruct-4bit"
model, tokenizer = load(model_id)
print("✅ Model cached and ready")
```

Or using HuggingFace CLI:

```bash
huggingface-cli download mlx-community/Llama-3-8B-Instruct-4bit
```

### Alternative Student Models

You can use different MLX-compatible models:

```python
# Smaller (faster, less VRAM)
student="mlx-community/Llama-2-7b-chat-4bit"

# Larger (better quality, more VRAM)
student="mlx-community/Llama-3-70B-Instruct-4bit"

# From HuggingFace
student="mlx-community/Mistral-7B-Instruct-v0.1-4bit"
```

Browse available models: https://huggingface.co/mlx-community

## After Distillation: Using Your Fine-Tuned Model

After running `distill()`, your trained model is in:

```
./outputs/math-tutor-v1/
├── model/                    # Fine-tuned weights
│   ├── adapters.npz         # LoRA weights
│   ├── config.json
│   └── tokenizer.json
├── traces_cot.jsonl         # Original CoT traces
└── traces_amplified.jsonl   # Training data
```

### Option 1: Inference with MLX-LM (Fastest)

**Recommended for Apple Silicon. Native integration, fastest speed.**

```python
from mlx_lm import load, generate

# Load your fine-tuned model
model, tokenizer = load("./outputs/math-tutor-v1/model")

# Generate response
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="What is 5+3?",
    max_tokens=200,
)

print(response)
```

**Speed**: ~50-100 tokens/second on M1/M2/M3

### Option 2: Inference with HuggingFace Transformers

**More compatible, works on any platform.**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model_path = "./outputs/math-tutor-v1/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Use MPS backend on Apple Silicon
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

# Generate
inputs = tokenizer("What is 5+3?", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

**Speed**: ~10-20 tokens/second (slower due to PyTorch overhead)

### Option 3: Inference with Ollama (Easiest for Production)

**Best for production. Simple API, no code needed.**

#### Step 1: Install Ollama

Download from: https://ollama.ai

#### Step 2: Convert Model to GGUF Format

```bash
# Using llama.cpp (requires separate installation)
python llama.cpp/convert.py \
    ./outputs/math-tutor-v1/model \
    --outtype q4_k_m \
    --outfile math-tutor.gguf
```

Or use MLX tools:

```python
from mlx_lm.utils import convert_to_gguf

convert_to_gguf(
    model_path="./outputs/math-tutor-v1/model",
    output_path="./math-tutor.gguf"
)
```

#### Step 3: Create Modelfile

Create `Modelfile` in your project:

```
FROM ./math-tutor.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM You are a helpful math tutor. Show your step-by-step reasoning.
```

#### Step 4: Create and Run

```bash
# Create the model
ollama create math-tutor -f Modelfile

# Run interactively
ollama run math-tutor

# Or use the API
curl http://localhost:11434/api/generate -d '{
  "model": "math-tutor",
  "prompt": "What is 5+3?"
}'
```

**Speed**: ~50-100 tokens/second
**Memory**: Very efficient, easy to deploy

### Option 4: Inference with MLX Web Server

**For web applications.**

```bash
# Install mlx-lm with server support
pip install mlx-lm[server]

# Start server
mlx_lm serve ./outputs/math-tutor-v1/model --port 8000
```

Then use the API:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "local-model",
        "messages": [{"role": "user", "content": "What is 5+3?"}],
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

## Choosing the Right Option

| Option | Speed | Quality | Setup | Best For |
|--------|-------|---------|-------|----------|
| **MLX-LM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Easy | Local testing, best speed |
| **Transformers** | ⭐⭐ | ⭐⭐⭐⭐ | Medium | Multi-platform apps |
| **Ollama** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | Production servers |
| **Web Server** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | Web/API services |

## Troubleshooting

### Model won't download

**Error**: `Connection timeout`, `403 Forbidden`

**Solution**:
```bash
# Increase timeout
export HF_HUB_TIMEOUT=600
export HF_HUB_LOCAL_DIR_AUTO_SYMLINK_CACHE=1
```

### Out of memory during training

**Solution**: Use smaller model
```python
student="mlx-community/Llama-2-7B-Instruct-4bit"  # 7B instead of 8B
```

### Model loads slowly

**Solution**: Use MLX-LM instead of Transformers
```python
from mlx_lm import load, generate
```

### Inference quality is poor

**Possible causes**:
1. Seed data too small (use 20-50 examples)
2. Augmentation too aggressive (reduce `augment_factor` to 30)
3. Teacher model inconsistent (check Claude responses)

**Solution**: Retrain with better seed data

## Model Specifications

### Default Model: Llama-3-8B-Instruct-4bit

| Property | Value |
|----------|-------|
| **Parameters** | 8 billion |
| **Format** | 4-bit quantized |
| **Memory** | ~5GB VRAM |
| **Context** | 8K tokens |
| **Training** | 2 epochs |
| **Output** | Fine-tuned Llama |

### Model Card

- **HuggingFace**: [mlx-community/Llama-3-8B-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3-8B-Instruct-4bit)
- **Base Model**: Meta Llama 3
- **Quantization**: MLX 4-bit (reduces size by ~75%)
- **License**: Meta Llama License

## Hardware Requirements

| Device | Min RAM | Recommended | Works |
|--------|---------|-------------|-------|
| M1/M2/M3 (8GB) | ✅ | Slower | Yes |
| M1/M2/M3 (16GB) | ✅ | Good | Yes |
| M1 Pro/Max (32GB) | ✅ | Best | Yes |
| Intel Mac | ⚠️ | Not optimized | Yes (CPU only) |

## Next Steps

1. **Run distillation**: `python examples/basic_usage.py`
2. **Test inference**: See examples above
3. **Deploy**: Use Ollama or web server
4. **Evaluate**: Compare outputs with teacher (Claude)

## Resources

- [MLX-LM Docs](https://github.com/ml-explore/mlx-lm)
- [Ollama Docs](https://github.com/ollama/ollama)
- [HuggingFace Models](https://huggingface.co/mlx-community)
- [Llama 3 Details](https://www.llama.com/)
