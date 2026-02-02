# Quick Start: From Seed Data to Working Model

This guide walks you through the complete process in 5 steps.

## Prerequisites

- Python 3.9+
- Mac M1/M2/M3 (or Linux/Windows with adjustment)
- 16GB+ RAM
- Anthropic API key (get from https://console.anthropic.com)

## Step 1: Install NanoDistill

```bash
cd /path/to/nanodistill
pip install -e .
```

This installs all dependencies including MLX, MLX-LM, LiteLLM, etc.

## Step 2: Set Your API Key

```bash
export ANTHROPIC_API_KEY='sk-ant-...'

# Verify it's set
echo $ANTHROPIC_API_KEY
```

## Step 3: Prepare Seed Data

Create a file `seeds.json`:

```json
[
  {"input": "What is 2+2?", "output": "4"},
  {"input": "What is 3+5?", "output": "8"},
  {"input": "What is 10-4?", "output": "6"},
  {"input": "What is 5×3?", "output": "15"},
  {"input": "What is 20÷4?", "output": "5"},
  {"input": "What is 7+8?", "output": "15"},
  {"input": "What is 100-25?", "output": "75"},
  {"input": "What is 6×6?", "output": "36"},
  {"input": "What is 12÷3?", "output": "4"},
  {"input": "What is 9+9?", "output": "18"}
]
```

## Step 4: Run Distillation

Create `my_distillation.py`:

```python
from nanodistill import distill

result = distill(
    name="math-tutor-v1",
    seed="seeds.json",  # Or a Python list
    instruction="You are a helpful math tutor. Show your reasoning step-by-step.",
    teacher="claude-sonnet-4-5",
    augment_factor=50,  # 10 seeds → 500 examples
)

print(f"✅ Model saved to: {result.model_path}")
print(f"📊 Training examples: {result.metrics['training_examples']}")
```

Run it:

```bash
python my_distillation.py
```

**Timeline**:
- First run: 20-30 minutes (includes model download)
- Subsequent runs: 10-15 minutes (uses cached model)

**What happens**:
1. 🎓 Downloads Llama-3-8B model (~4GB) - cached for future use
2. 🎓 Generates reasoning traces using Claude
3. 📈 Extracts task policy from your examples
4. 📈 Generates 500 synthetic examples matching the pattern
5. 🔥 Fine-tunes the Llama model on Apple Silicon
6. 💾 Saves to `./outputs/math-tutor-v1/`

## Step 5: Use Your Model

### Quick Test (MLX - Recommended)

```python
from mlx_lm import load, generate

# Load your fine-tuned model
model, tokenizer = load("./outputs/math-tutor-v1/model")

# Ask it a question
prompt = "What is 7+8?"
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=200,
)

print(response)
```

Expected output:
```
Let me work through this step by step:
7 + 8
= 15

The answer is 15.
```

### Production Deployment (Ollama - Recommended)

```bash
# 1. Install Ollama (https://ollama.ai)

# 2. Create a file called "Modelfile"
cat > Modelfile << 'EOF'
FROM ./outputs/math-tutor-v1/model

PARAMETER temperature 0.7

SYSTEM You are a helpful math tutor. Show your reasoning.
EOF

# 3. Create the model
ollama create math-tutor -f Modelfile

# 4. Run it
ollama run math-tutor
# Then type: "What is 7+8?"
```

## File Organization

After running, you'll have:

```
.
├── my_distillation.py
├── seeds.json
├── Modelfile
└── outputs/
    └── math-tutor-v1/
        ├── model/                  # Fine-tuned weights
        │   ├── adapters.npz       # LoRA weights
        │   ├── config.json
        │   └── tokenizer.json
        ├── traces_cot.jsonl       # Generated reasoning
        └── traces_amplified.jsonl # Training data (500 examples)
```

## Common Use Cases

### Use Case 1: Customer Support Chatbot

```python
result = distill(
    name="support-chatbot-v1",
    seed=[
        {
            "input": "How do I reset my password?",
            "output": "Click Settings → Account → Reset Password"
        },
        # ... more support Q&As
    ],
    instruction="You are a helpful customer support agent.",
    augment_factor=30,
)
```

### Use Case 2: Domain-Specific QA

```python
result = distill(
    name="biology-tutor-v1",
    seed=[
        {"input": "What is photosynthesis?", "output": "..."},
        # ... biology Q&As
    ],
    instruction="You are a biology expert. Explain concepts clearly.",
    augment_factor=50,
)
```

### Use Case 3: Code Generation

```python
result = distill(
    name="python-helper-v1",
    seed=[
        {"input": "Write a function to reverse a string", "output": "def reverse(s): return s[::-1]"},
        # ... coding tasks
    ],
    instruction="You are an expert Python programmer.",
    augment_factor=50,
)
```

## Performance Expectations

### Generation Speed

| Hardware | Speed | Comments |
|----------|-------|----------|
| M1 | ~50 tok/s | Good |
| M1 Pro | ~70 tok/s | Better |
| M2 | ~80 tok/s | Optimal |
| M3 Max | ~120 tok/s | Best |

### Model Quality

| Seed Count | Model Quality | Recommendation |
|-----------|--------------|-----------------|
| 5-10 | Fair | Minimum viable |
| 10-20 | Good | Recommended |
| 20-50 | Very Good | Ideal |
| 50+ | Excellent | Maximum |

### Memory Usage

| Phase | Memory | Duration |
|-------|--------|----------|
| Download model | 4GB | ~2 min |
| CoT generation | 2GB | ~5 min |
| Data amplification | 3GB | ~5 min |
| Training | 6GB | ~3 min |

Total peak: ~6GB (fits on most Macs)

## Troubleshooting

### Error: API key not set

```
❌ ANTHROPIC_API_KEY not set
```

**Fix**:
```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

### Error: Out of memory

**Fix**: Use a smaller model
```python
student="mlx-community/Llama-2-7B-Instruct-4bit"
```

Or reduce augmentation:
```python
augment_factor=30  # Instead of 50
```

### Model loads but gives bad answers

**Likely causes**:
1. Seed data inconsistent
2. Seed data too small
3. Instruction too vague

**Fix**: Review and improve seed data, then retrain

### First run is slow

**Normal behavior**:
- First run downloads ~4GB model
- This only happens once
- Cached for future runs

## Next Steps

1. ✅ Prepare your seed data
2. ✅ Run distillation
3. ✅ Test outputs locally
4. ✅ Deploy with Ollama
5. ✅ Monitor performance
6. ✅ Iterate with new seed data

## Advanced Options

### Custom Models

```python
result = distill(
    name="my-model",
    seed=seed_data,
    instruction="...",
    teacher="gpt-4",  # Use OpenAI instead
    student="mlx-community/Mistral-7B-Instruct-v0.1-4bit",  # Different base model
    augment_factor=100,  # More synthetic examples
)
```

### Batch Processing

```python
from nanodistill import distill

for task in ["math", "biology", "history"]:
    result = distill(
        name=f"tutor-{task}",
        seed=load_seeds(f"{task}.json"),
        instruction=f"You are a {task} tutor.",
    )
    print(f"✅ {task} model saved")
```

## Support

- **Issues**: Report on GitHub
- **API Key Issues**: https://console.anthropic.com
- **Model Issues**: https://huggingface.co/mlx-community
- **Ollama Issues**: https://github.com/ollama/ollama

---

**Ready?** Start with Step 1: Install NanoDistill
