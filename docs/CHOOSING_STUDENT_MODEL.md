# Choosing Your Student Model

NanoDistill allows you to use any MLX-compatible model as your student. This guide helps you choose the best one for your use case.

## Available Models

### Qwen Models (Recommended for Apple Silicon)

Qwen is developed by Alibaba and optimized for efficiency.

| Model | Size | Memory | Speed | Quality | Best For |
|-------|------|--------|-------|---------|----------|
| **Qwen-0.5B** | 500MB | <1GB | ⭐⭐⭐⭐⭐ Super fast | ⭐⭐ Basic | Edge devices, fast responses |
| **Qwen-1.8B** | 1.8GB | 2-3GB | ⭐⭐⭐⭐ Very fast | ⭐⭐⭐ Good | M1/M2 with limited RAM |
| **Qwen-4B** | 4GB | 5-6GB | ⭐⭐⭐ Fast | ⭐⭐⭐⭐ Very good | Balanced option |
| **Qwen-7B** | 7GB | 8-9GB | ⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Excellent | Production quality |

### Llama Models

Meta's Llama family, widely used and well-tested.

| Model | Size | Memory | Speed | Quality | Best For |
|-------|------|--------|-------|---------|----------|
| **Llama-2-7B** | 7GB | 8-9GB | ⭐⭐ Moderate | ⭐⭐⭐⭐ Very good | Well-established baseline |
| **Llama-3-8B** ⭐ | 8GB | 9-10GB | ⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Excellent | **Default choice** |
| **Llama-3-70B** | 70GB | 75GB+ | ⭐ Slow | ⭐⭐⭐⭐⭐ Best | Very large machines only |

### Other Options

| Model | Size | Memory | Speed | Quality | Best For |
|-------|------|--------|-------|---------|----------|
| **Mistral-7B** | 7GB | 8-9GB | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Very good | Fast & capable |
| **Phi-2** | 3.5GB | 4-5GB | ⭐⭐⭐⭐ | ⭐⭐⭐ Good | Lightweight |
| **Yi-6B** | 6GB | 7-8GB | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Very good | Capable small model |

## How to Use Different Models

### Using Qwen-4B (Recommended)

```python
from nanodistill import distill

result = distill(
    name="my-model",
    seed=seed_data,
    instruction="You are a helpful assistant.",
    teacher="claude-sonnet-4-5",
    student="mlx-community/Qwen-4B-Chat-4bit",  # ← Change this
)
```

### Using Qwen-0.5B (Fastest)

```python
result = distill(
    name="my-model",
    seed=seed_data,
    instruction="...",
    student="mlx-community/Qwen-0.5B-Chat-4bit",  # Ultra-fast, minimal quality
)
```

### Using Qwen-7B (Best Quality)

```python
result = distill(
    name="my-model",
    seed=seed_data,
    instruction="...",
    student="mlx-community/Qwen-7B-Chat-4bit",  # Highest quality
)
```

### Using Llama-2 (Alternative)

```python
result = distill(
    name="my-model",
    seed=seed_data,
    instruction="...",
    student="mlx-community/Llama-2-7B-Chat-4bit",  # Alternative option
)
```

## Choosing the Right Model

### Decision Tree

```
Do you have 16GB RAM?
├─ YES (M1/M2 standard)
│  ├─ Need FAST responses?
│  │  └─ Use Qwen-1.8B (very fast, still good quality)
│  │
│  └─ Need GOOD quality?
│     └─ Use Qwen-4B (balanced option)
│
└─ NO, only 8GB RAM
   └─ Use Qwen-0.5B (smallest option)

Do you have 32GB+ RAM?
├─ YES (M1 Pro/Max, M2 Max)
│  ├─ Need BEST quality?
│  │  └─ Use Qwen-7B or Llama-3-8B
│  │
│  └─ Otherwise
│     └─ Use Qwen-4B (great balance)
```

### By Use Case

**Fast API/Chatbot**
- Want: Quick responses, low latency
- Use: Qwen-0.5B or Qwen-1.8B
- Speed: 200-300 tokens/sec

**Customer Support Bot**
- Want: Good quality, reasonable speed
- Use: Qwen-4B
- Speed: ~100 tokens/sec

**Educational Tutor**
- Want: Excellent reasoning, explanation quality
- Use: Qwen-7B or Llama-3-8B
- Speed: ~50 tokens/sec

**Domain Expert (Medical, Legal)**
- Want: Highest accuracy and reasoning
- Use: Qwen-7B or Llama-3-8B
- Speed: ~50 tokens/sec

**Offline Application**
- Want: Minimal storage (portable)
- Use: Qwen-0.5B (500MB total)
- Speed: 300+ tokens/sec

## Qwen vs Llama: Pros and Cons

### Qwen Advantages
✅ More efficient (smaller = faster)
✅ Better for limited resources
✅ Multiple size options (0.5B, 1.8B, 4B, 7B)
✅ Optimized for Asian languages
✅ Great inference speed
✅ Lower memory usage during training

### Qwen Disadvantages
❌ Less community support than Llama
❌ Fewer academic papers
❌ Smaller base model sizes
❌ Less tested on niche domains

### Llama Advantages
✅ Most tested and well-documented
✅ Largest community support
✅ Better for English-heavy tasks
✅ More research available
✅ Works great with standard tools
✅ Trusted by enterprises

### Llama Disadvantages
❌ Larger models (higher memory)
❌ Slower inference
❌ Not optimized for Asian languages
❌ Higher resource requirements

## Practical Recommendations

### For Most Users (M1/M2 with 16GB RAM)

**Use Qwen-4B**

```python
result = distill(
    name="my-model",
    seed=seed_data,
    instruction="Your task description...",
    teacher="claude-sonnet-4-5",
    student="mlx-community/Qwen-4B-Chat-4bit",
    augment_factor=50,
)
```

Why?
- Good balance between speed and quality
- Fits in 16GB RAM comfortably
- Training takes ~2-3 minutes
- Inference is reasonably fast (~100 tok/s)
- Gets excellent results with good seed data

### For Speed-Critical Applications

**Use Qwen-0.5B or Qwen-1.8B**

```python
# Super fast, minimal quality loss
student="mlx-community/Qwen-0.5B-Chat-4bit"  # 300+ tok/s

# Or balanced fast option
student="mlx-community/Qwen-1.8B-Chat-4bit"  # 200+ tok/s
```

### For Maximum Quality

**Use Qwen-7B or Llama-3-8B**

```python
# Qwen (more efficient)
student="mlx-community/Qwen-7B-Chat-4bit"

# Or Llama (more tested, English-optimized)
student="mlx-community/Llama-3-8B-Instruct-4bit"
```

### For Limited Resources (8GB RAM)

**Use Qwen-0.5B or Qwen-1.8B**

```python
# Only option for 8GB RAM
student="mlx-community/Qwen-0.5B-Chat-4bit"  # 500MB

# Or if you can manage
student="mlx-community/Qwen-1.8B-Chat-4bit"  # 1.8GB
```

## Available MLX Models

Browse all available models:
- https://huggingface.co/mlx-community

Filter by:
- Organization: `mlx-community`
- Type: Chat/Instruct models
- Quantization: 4-bit (look for `-4bit` in name)

## Complete List of Working Models

```python
MLX_MODELS = {
    # Qwen
    "mlx-community/Qwen-0.5B-Chat-4bit",
    "mlx-community/Qwen-1.8B-Chat-4bit",
    "mlx-community/Qwen-4B-Chat-4bit",
    "mlx-community/Qwen-7B-Chat-4bit",

    # Llama
    "mlx-community/Llama-2-7B-Chat-4bit",
    "mlx-community/Llama-3-8B-Instruct-4bit",

    # Mistral
    "mlx-community/Mistral-7B-Instruct-v0.1-4bit",

    # Other
    "mlx-community/Yi-6B-Chat-4bit",
    "mlx-community/Phi-2",
}
```

## Performance Comparison

### Training Time (on M1)

```
Model              Download  Training  Total (First Run)
──────────────────────────────────────────────────────
Qwen-0.5B          3 min     1 min     4 min
Qwen-1.8B          5 min     1.5 min   6.5 min
Qwen-4B            8 min     2 min     10 min
Qwen-7B            12 min    2.5 min   14.5 min
Llama-3-8B         15 min    3 min     18 min
```

### Inference Speed (tokens/sec)

```
Model              M1        M1 Pro    M2        M3 Max
────────────────────────────────────────────────────
Qwen-0.5B          300+      400+      350+      400+
Qwen-1.8B          200       250       250       300
Qwen-4B            100       120       120       150
Qwen-7B            50        70        80        100
Llama-3-8B         50        70        80        100
```

## Memory Usage During Training

```
Model              Peak Memory  Fits in 16GB?
──────────────────────────────────────────
Qwen-0.5B          3GB          ✅ Yes
Qwen-1.8B          4GB          ✅ Yes
Qwen-4B            6GB          ✅ Yes
Qwen-7B            9GB          ✅ Yes
Llama-3-8B         10GB         ✅ Yes (tight)
```

## Testing Different Models

### Quick Benchmark

```python
from nanodistill import distill

models = {
    "Qwen-0.5B": "mlx-community/Qwen-0.5B-Chat-4bit",
    "Qwen-4B": "mlx-community/Qwen-4B-Chat-4bit",
    "Llama-3-8B": "mlx-community/Llama-3-8B-Instruct-4bit",
}

for name, model_id in models.items():
    print(f"\nTesting {name}...")
    result = distill(
        name=f"benchmark-{name}",
        seed=seed_data,
        instruction="...",
        student=model_id,
    )
    print(f"✅ Done: {result.model_path}")
```

## Troubleshooting Model Choice

### Model won't download

**Solution**: Check if model ID is correct
```bash
# List available models
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
# This will show if model is available
AutoModelForCausalLM.from_pretrained('mlx-community/Qwen-4B-Chat-4bit')
"
```

### Model too slow

**Solution**: Use smaller model
```python
# If using Qwen-7B and it's slow:
student="mlx-community/Qwen-4B-Chat-4bit"  # 2x faster
```

### Out of memory

**Solution**: Use smaller model
```python
# If getting OOM with Qwen-7B:
student="mlx-community/Qwen-4B-Chat-4bit"  # Lower memory

# Or even smaller:
student="mlx-community/Qwen-1.8B-Chat-4bit"  # Minimal memory
```

### Poor quality

**Solution**: Use larger model
```python
# If quality is bad with Qwen-1.8B:
student="mlx-community/Qwen-4B-Chat-4bit"  # Better quality

# Or maximum quality:
student="mlx-community/Qwen-7B-Chat-4bit"  # Highest quality
```

## Recommendation Summary

| Situation | Recommended Model |
|-----------|------------------|
| New to distillation | Qwen-4B |
| Speed critical | Qwen-0.5B or 1.8B |
| Quality critical | Qwen-7B or Llama-3-8B |
| Limited RAM (8GB) | Qwen-0.5B |
| Standard M1/M2 | Qwen-4B ⭐ |
| M1 Pro/Max | Qwen-7B or Llama-3-8B |
| Want to match default | Llama-3-8B |

---

**Default**: `mlx-community/Llama-3-8B-Instruct-4bit` (well-tested, excellent quality)

**Our Recommendation**: `mlx-community/Qwen-4B-Chat-4bit` (best balance for most users)
