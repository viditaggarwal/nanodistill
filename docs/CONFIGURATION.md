# Configuration Reference

NanoDistill is designed to work out-of-the-box with sensible defaults, but provides full control over training, generation, and system parameters when you need it.

## Quick Start (No Configuration Needed)

```python
from nanodistill import distill

result = distill(
    name="my-model",
    seed=[
        {"input": "What is 2+2?", "output": "4"},
        {"input": "What is 3+5?", "output": "8"},
        # ... more examples
    ],
    instruction="You are a helpful math tutor. Show your reasoning.",
)
```

That's it! Everything else uses sensible defaults.

## Core Parameters (Required/Primary)

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `name` | str | Required | Run identifier (used in output paths) |
| `seed` | List[Dict] | Required | Training examples with `input` and `output` fields |
| `instruction` | str | Required | System prompt / task description |
| `teacher` | str | `"claude-sonnet-4-5"` | Teacher model (any LiteLLM-compatible model) |
| `student` | str | `"mlx-community/Llama-3-8B-Instruct-4bit"` | Student model (MLX-compatible model ID) |
| `augment_factor` | int | `50` | Multiply seed examples by this factor (range: 1-500) |
| `output_dir` | str | `"./outputs"` | Directory to save model and outputs |
| `response_model` | Pydantic model | `None` | Optional schema to enforce on synthetic examples |

## Optional Parameters (via kwargs)

All optional parameters are passed as keyword arguments to `distill()`:

```python
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",

    # Optional parameters here
    batch_size=2,
    learning_rate=5e-5,
    # ... etc
)
```

### Training Parameters

| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| `batch_size` | int | `1` | 1-32 | Batch size for training (reduce for memory-constrained systems) |
| `learning_rate` | float | `1e-5` | >0, <1 | Learning rate for fine-tuning (lower = more stable) |
| `num_train_epochs` | int | `1` | 1-10 | Number of training epochs |
| `max_seq_length` | int | `256` | 32-2048 | Maximum sequence length (lower = less memory) |

**Example: Memory-constrained M1 MacBook Air**
```python
distill(
    ...,
    batch_size=1,           # Minimal memory usage
    max_seq_length=256,     # Shorter sequences
    learning_rate=5e-6,     # Conservative training
)
```

### LoRA Parameters (New in v0.2.0)

| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| `lora_rank` | int | `8` | 1-64 | LoRA adapter rank (higher = more parameters, slower) |
| `lora_layers` | int | `4` | 1-32 | Number of layers to apply LoRA (higher = more expressive) |

**Example: High-quality model on powerful system**
```python
distill(
    ...,
    lora_rank=16,       # More expressive adapters
    lora_layers=8,      # More layers affected
)
```

### Generation Parameters (New in v0.2.0)

| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| `temperature` | float | `0.7` | 0.0-2.0 | Sampling temperature (0.0 = deterministic, 2.0 = very creative) |

**Example: Consistent outputs (lower temperature)**
```python
distill(
    ...,
    temperature=0.3,  # More deterministic
)
```

**Example: Diverse outputs (higher temperature)**
```python
distill(
    ...,
    temperature=1.2,  # More creative
)
```

### Data Split (New in v0.2.0)

| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| `val_split` | float | `0.2` | 0.0-0.5 | Validation set split ratio (0.2 = 20% validation, 80% training) |

**Example: 90/10 train/validation split**
```python
distill(
    ...,
    val_split=0.1,  # 10% validation, 90% training
)
```

### System Configuration Parameters (New in v0.2.0)

These parameters control system resource usage during training:

| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| `max_memory_gb` | int | Auto-detect, capped at 12 | 2-128 | Maximum RAM to use during training |
| `memory_hard_limit_gb` | int | Auto-detect, capped at 12 | 2-128 | Hard stop if memory exceeds this (prevents system crash) |
| `cpu_capacity_percent` | float | `0.8` | 0.1-1.0 | CPU/memory threshold before training pauses (0.8 = 80%) |

#### System Auto-Detection

If you don't specify system parameters, NanoDistill automatically detects your system and sets safe defaults:

```python
import psutil

# Auto-detection logic (runs at config creation):
total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)

# Conservative defaults:
max_memory_gb = min(total_memory_gb * 0.9, 12)          # 90% of RAM, max 12GB
memory_hard_limit_gb = min(total_memory_gb * 0.95, 12)  # 95% of RAM, max 12GB
cpu_capacity_percent = 0.8                              # Pause if > 80% loaded
```

#### Why Capped at 12GB?

The 12GB cap is intentionally conservative to prevent:
- **Out-of-Memory (OOM) crashes** on most consumer systems
- **Excessive swapping** that would slow training
- **System instability** when other applications are running

#### Examples by System

**M1 MacBook Air (8GB RAM)**
- Auto-detected: `max_memory_gb=7.2`, `memory_hard_limit_gb=7.6`
- No config needed - just call `distill()`
- To be extra safe: `max_memory_gb=4, memory_hard_limit_gb=5`

```python
# Auto-detected (safe)
result = distill(...)

# Extra conservative (slower but safer)
result = distill(..., max_memory_gb=4, memory_hard_limit_gb=5)
```

**M2 MacBook Pro (16GB RAM)**
- Auto-detected: `max_memory_gb=12`, `memory_hard_limit_gb=12` (capped)
- No config needed

**M3 Pro (18GB RAM)**
- Auto-detected: `max_memory_gb=12`, `memory_hard_limit_gb=12` (capped)
- Optionally increase limits:

```python
result = distill(
    ...,
    max_memory_gb=16,           # Use more memory
    memory_hard_limit_gb=18,    # Higher hard limit
)
```

**M3 Max (128GB RAM)**
- Auto-detected: `max_memory_gb=12`, `memory_hard_limit_gb=12` (capped)
- Recommended for dedicated training machine:

```python
result = distill(
    ...,
    max_memory_gb=48,           # Use 48GB
    memory_hard_limit_gb=64,    # Hard stop at 64GB (leaves 64GB for OS)
    cpu_capacity_percent=0.95,  # Aggressive: only stop if > 95% loaded
)
```

## Complete Usage Examples

### Example 1: Quick Prototyping (Fastest)

```python
result = distill(
    name="quick-test",
    seed=[...],  # 10-20 examples
    instruction="...",
    augment_factor=10,      # Small dataset
    num_train_epochs=1,     # One epoch
)
```

### Example 2: Memory-Constrained (M1 8GB)

```python
result = distill(
    name="m1-model",
    seed=[...],
    instruction="...",

    # Minimal memory usage
    batch_size=1,
    max_seq_length=256,
    lora_rank=4,
    lora_layers=2,

    # Conservative system config
    max_memory_gb=4,
    memory_hard_limit_gb=5,
)
```

### Example 3: High Performance (M3 Pro 36GB)

```python
result = distill(
    name="high-quality",
    seed=[...],
    instruction="...",

    # Larger dataset
    augment_factor=100,

    # Better training
    batch_size=4,
    max_seq_length=1024,
    learning_rate=5e-5,
    num_train_epochs=3,

    # More expressive model
    lora_rank=16,
    lora_layers=8,

    # System auto-detected, but can override
    # max_memory_gb=24 (optional)
)
```

### Example 4: Production Quality (Conservative)

```python
result = distill(
    name="production-model",
    seed=[...],  # 50+ high-quality examples
    instruction="...",

    # Larger dataset
    augment_factor=100,

    # Conservative training for consistency
    batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=2,
    temperature=0.3,  # Lower temp = more consistent

    # Good expressiveness
    lora_rank=12,
    lora_layers=6,
)
```

### Example 5: Dedicated Training Machine (M3 Max 128GB)

```python
result = distill(
    name="powerful",
    seed=[...],
    instruction="...",

    # Aggressive training
    batch_size=8,
    max_seq_length=2048,
    learning_rate=1e-4,
    num_train_epochs=5,
    augment_factor=200,

    # High expressiveness
    lora_rank=32,
    lora_layers=16,

    # Use system resources aggressively
    max_memory_gb=48,
    memory_hard_limit_gb=64,
    cpu_capacity_percent=0.95,
)
```

### Example 6: Structured Output (Schema Enforcement)

```python
from pydantic import BaseModel

class TaskOutput(BaseModel):
    answer: str
    reasoning: str
    confidence: float

result = distill(
    name="structured-model",
    seed=[...],
    instruction="...",
    response_model=TaskOutput,  # Enforce schema
    temperature=0.3,  # Lower temp for consistency
)
```

## What's Configurable vs. Fixed

### ✅ Fully Configurable (New in v0.2.0)

- ✅ Training parameters (batch_size, learning_rate, epochs, seq_length)
- ✅ LoRA settings (rank, layers)
- ✅ Generation temperature
- ✅ Train/validation split ratio
- ✅ System memory limits (with auto-detection)
- ✅ System CPU threshold
- ✅ Teacher model (any LiteLLM model)
- ✅ Student model (any MLX model)
- ✅ Data augmentation factor
- ✅ Output schema validation

### ❌ Fixed by Design

These are intentionally fixed to keep the pipeline simple and reliable:

| Component | Current | Rationale |
|-----------|---------|-----------|
| **Chain-of-Thought Format** | `<thinking>` XML tags | Universal across models, easy to parse reliably |
| **Data Format** | User → Assistant conversation | Aligns with standard LLM instruction tuning |
| **Optimizer** | AdamW | Best practice for LLM fine-tuning (via MLX-LM) |
| **Response Parsing** | 3 strategies (Markdown, colon-sep, JSON) | Handles most common teacher output formats |

Future versions (Phase 3+) may add:
- 🔮 Custom chain-of-thought templates
- 🔮 Custom prompt overrides
- 🔮 Additional optimizer options (requires MLX-LM update)

---

## Troubleshooting

### Issue: "Memory limit exceeded" or OOM errors

**Solution**: Reduce memory usage
```python
distill(
    ...,
    batch_size=1,           # Reduce batch size
    max_seq_length=256,     # Shorter sequences
    max_memory_gb=4,        # Lower limit
    memory_hard_limit_gb=5, # Lower hard limit
)
```

### Issue: Training is very slow

**Solution 1**: Increase batch size (if memory allows)
```python
distill(..., batch_size=4)
```

**Solution 2**: Reduce sequence length
```python
distill(..., max_seq_length=512)  # Default 256, try 512
```

**Solution 3**: Increase memory limits on powerful systems
```python
distill(
    ...,
    max_memory_gb=24,
    memory_hard_limit_gb=32,
)
```

### Issue: Training pauses frequently

**Solution**: Increase CPU threshold
```python
distill(..., cpu_capacity_percent=0.9)  # 90% instead of 80%
```

**Note**: Only do this on dedicated training machines.

### Issue: Training heats up my laptop (thermal throttling)

**Solution 1**: Use thermal-optimized config (recommended)
```python
distill(
    ...,
    student="mlx-community/Qwen2.5-3B-Instruct-4bit",  # Smaller model
    lora_rank=4,             # Reduced from 8
    lora_layers=2,           # Reduced from 4
    augment_factor=30,       # Fewer examples
)
```

**Solution 2**: External cooling
- Laptop cooling pad (~$20-40)
- Elevated laptop stand for airflow
- Train in a cooler room

**Note**: Thermal throttling is automatic on Apple Silicon when CPU exceeds ~85-95°C. Hardware-optimized defaults are now auto-detected for your chip.

### Issue: Generated examples are too different/similar

**Solution**: Adjust temperature during generation
```python
# Too similar (low diversity)?
distill(..., temperature=1.2)  # Increase

# Too different (not focused)?
distill(..., temperature=0.3)  # Decrease
```

### Issue: Model quality is poor

**Solution 1**: More seed examples
```python
# Need at least 10, ideally 20+
seed=[...]  # Add more diverse examples
```

**Solution 2**: Increase augmentation
```python
distill(..., augment_factor=100)  # More synthetic examples
```

**Solution 3**: Train longer
```python
distill(..., num_train_epochs=3)  # More training
```

**Solution 4**: More expressive model
```python
distill(
    ...,
    lora_rank=16,
    lora_layers=8,
)
```

## Migration from Earlier Versions

NanoDistill v0.2.0+ is fully backward compatible. All existing code continues to work with defaults:

```python
# v0.1.0 code (still works in v0.2.0+)
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
)

# v0.2.0+ can add new parameters
result = distill(
    name="my-model",
    seed=[...],
    instruction="...",
    temperature=0.5,        # NEW
    lora_rank=16,           # NEW
    max_memory_gb=24,       # NEW
)
```

No code changes required - everything is opt-in!

## Getting Help

- **Configuration issues**: See the troubleshooting section above
- **General questions**: Check the [README.md](../README.md)
- **Examples**: See [examples/configuration.py](../examples/configuration.py)
- **Architecture details**: See [CLAUDE.md](../CLAUDE.md)
