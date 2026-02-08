# NanoDistill - Knowledge distillation for small language models

<p align="center">
  <strong>10 examples. One API key. Your own model.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge" alt="build" />
  <img src="https://img.shields.io/badge/tests-passing-brightgreen?style=for-the-badge" alt="tests" />
  <img src="https://img.shields.io/badge/python-3.9+-green?style=for-the-badge" alt="python" />
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-green?style=for-the-badge" alt="platform" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="license" />
</p>

**NanoDistill** is a _pipeline_ that turns a small set of seed examples and a task instruction into a custom small language model you run locally. You give it around 10 input/output pairs and an API key; it uses a teacher model to generate reasoning traces and hundreds of synthetic examples, then fine-tunes a student model (e.g. Llama 3 8B) with MLX on Apple Silicon. The result is a model that follows your task without ongoing API calls.

If you want a small, local model that does one thing well from a handful of examples, this is it.

[Quick Start](docs/QUICK_START.md) · [Workflow](docs/WORKFLOW.md) · [Model Setup](docs/MODEL_SETUP.md) · [Contributing](docs/CONTRIBUTING.md)

Preferred setup: `uv pip install -e .` (or `pip install -e .`). Runs on **macOS with Apple Silicon (M1/M2/M3+)**, 16GB+ RAM.

**Required:** API key for the teacher model:
- [Anthropic API key](https://console.anthropic.com) for Claude (default)
- Or [OpenAI](https://platform.openai.com), [Google](https://ai.google.dev), Ollama, etc. via LiteLLM

**If using gated models:** [HuggingFace token](https://huggingface.co/settings/tokens) for downloading student models (e.g., Meta's Llama models). Set via `HF_TOKEN` environment variable or pass to `distill()`.

New install? Start here: [Getting started](docs/QUICK_START.md)

---

## Examples

### Install and run

```bash
export ANTHROPIC_API_KEY='sk-ant-...'

# Or add to .env file
# ANTHROPIC_API_KEY='sk-ant-...'
```

```python
from nanodistill import distill

result = distill(
    name='stock-sentiment',
    seed='seeds.json',  # list of {"input": "...", "output": "..."}
    instruction='You analyze financial news and headlines. Output sentiment (bullish/bearish/neutral) and brief reasoning.',
    teacher='claude-sonnet-4-5',
)
print(f'Model saved to: {result.model_path}')
```

### Seed file (`seeds.json`)

```json
[
  {"input": "Tesla down 8% today. Elon says 'best quarter ever coming'.", "output": "{\"sentiment\": \"bearish\", \"reasoning\": \"Price drop and skeptical tone outweigh positive statement.\"}"},
  {"input": "AAPL beats earnings by 12% but warns of supply chain issues. Stock flat after-hours.", "output": "{\"sentiment\": \"neutral\", \"reasoning\": \"Strong beat offset by guidance; flat reaction suggests mixed view.\"}"},
  {"input": "NVDA 200% YoY growth. IV crush post-earnings, premium sellers loving it.", "output": "{\"sentiment\": \"bullish\", \"reasoning\": \"Strong fundamentals; IV crush reflects reduced uncertainty.\"}"}
]
```

### Run the model locally

```python
from mlx_lm import load, generate

model, tokenizer = load('./outputs/stock-sentiment/model')
response = generate(model, tokenizer, 'META announces layoffs affecting 15% of staff.', max_tokens=150)
print(response)
```

### Pipeline in one picture

```
Your 10 examples
    ↓
    ├→ Generate reasoning with Claude
    ├→ Extract task pattern
    ├→ Create hundreds of synthetic examples
    └→ Fine-tune Llama-3-8B (MLX)
    ↓
Locally runnable model
```

---

## How it works

1. **Policy extraction** - Infers the task pattern from your seed data and optional Chain-of-Thought traces.
2. **Synthetic generation** - Uses the teacher (e.g. Claude) to produce many new examples that match the pattern.
3. **Data amplification** - Turns seed + synthetic examples into training data (optionally with CoT).
4. **Fine-tuning** - Trains a student model on Apple Silicon with MLX-LM (LoRA).

You can swap the teacher (Claude, GPT-4o, Gemini, Ollama via LiteLLM) and the student (e.g. different MLX community models). Optional Pydantic `response_model` support lets you distill structured outputs.

---

## Configuration

NanoDistill works out of the box with sensible defaults. Everything is optional—just provide your 10+ examples and task instruction, and training starts immediately.

### Essential Parameters

- `name` - Run identifier (used in output directory)
- `seed` - Training examples: List of `{"input": "...", "output": "..."}` dicts, or path to JSON/JSONL/CSV file
- `instruction` - System prompt / task description for teacher and student
- `teacher` - Teacher model (default: `"claude-sonnet-4-5"`, any LiteLLM model works)
- `student` - Student model (default: `"mlx-community/Llama-3-8B-Instruct-4bit"`, any MLX model)

### Advanced Parameters (Optional)

All of these are optional and can be tuned via kwargs:

**Training Parameters:**
- `batch_size` (default: 1) - Batch size (1-32, reduce for memory-constrained systems)
- `learning_rate` (default: 1e-5) - Learning rate
- `num_train_epochs` (default: 1) - Training epochs
- `max_seq_length` (default: 256) - Maximum sequence length (32-2048)

**Model Parameters (v0.2.0+):**
- `lora_rank` (default: 8) - LoRA adapter rank (1-64, higher = more parameters)
- `lora_layers` (default: 4) - Number of layers for LoRA (1-32)
- `temperature` (default: 0.7) - Sampling temperature for synthesis (0.0-2.0)

**Data & System (v0.2.0+):**
- `augment_factor` (default: 50) - Data multiplication factor (1-500)
- `output_dir` (default: "./outputs") - Output directory
- `val_split` (default: 0.2) - Validation split ratio
- `max_memory_gb` (auto-detect, capped at 12) - Maximum RAM to use
- `memory_hard_limit_gb` (auto-detect, capped at 12) - Hard memory limit
- `cpu_capacity_percent` (default: 0.8) - CPU threshold before pause

### Environment Variables

**Teacher Model:**
- `ANTHROPIC_API_KEY` - Required for Claude (get from [console.anthropic.com](https://console.anthropic.com))
- `OPENAI_API_KEY` - Optional, for GPT models
- `GOOGLE_API_KEY` - Optional, for Gemini models

**Student Model (HuggingFace):**
- `HF_TOKEN` - Required if using gated models like Meta's Llama (get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)). Set via:
  ```bash
  export HF_TOKEN='hf_...'
  ```
  Or pass to `distill()`: `distill(..., huggingface_token='hf_...')`

### Examples

**Minimal (works immediately):**
```python
from nanodistill import distill

result = distill(
    name="my-model",
    seed=[
        {"input": "example 1", "output": "answer 1"},
        {"input": "example 2", "output": "answer 2"},
        # ... 10+ examples
    ],
    instruction="Your task description here",
)
```

**Memory-constrained M1 MacBook:**
```python
result = distill(
    name="m1-model",
    seed=[...],
    instruction="...",
    batch_size=1,
    max_seq_length=256,
    lora_rank=4,
)
```

**High-performance M3 Pro/Max:**
```python
result = distill(
    name="pro-model",
    seed=[...],
    instruction="...",
    batch_size=4,
    max_seq_length=1024,
    lora_rank=16,
    lora_layers=8,
    num_train_epochs=3,
)
```

**Thermal-optimized (for laptops that heat up):**
```python
result = distill(
    name="cool-model",
    seed=[...],
    instruction="...",
    student="mlx-community/Qwen2.5-3B-Instruct-4bit",  # Smaller model
    lora_rank=4,        # Reduced from 8
    lora_layers=2,      # Reduced from 4
    augment_factor=30,  # Reduced from 50
)
```
For more examples and detailed guidance, see [**Configuration Reference**](docs/CONFIGURATION.md) and [**Configuration Examples**](examples/configuration.py).

---

## Output

Per run, under `{output_dir}/{name}/`:

- `model/` -Fine-tuned model (MLX)
- `model.gguf` -Quantized model
- Training logs and metrics

---

## Troubleshooting

**"ANTHROPIC_API_KEY not set"** - Export your key: `export ANTHROPIC_API_KEY='sk-ant-...'`

**"Access denied" or "gated model" error** - You need a HuggingFace token for gated models like Llama:
```bash
export HF_TOKEN='hf_...'  # Get from https://huggingface.co/settings/tokens
# OR accept the model license: https://huggingface.co/meta-llama/Llama-2-7b-hf
```

**Out of memory** - Lower `augment_factor` (e.g. 20–30), use a smaller student, or close other GPU-heavy apps.

**Laptop heating up during training** - Use a smaller model (Qwen 4B) and reduced LoRA rank (4 instead of 8). Hardware-optimized defaults are now auto-detected for your Apple Silicon chip.

**MLX** - Requires macOS 13+. See [MLX](https://github.com/ml-explore/mlx).

---

## Development

```bash
uv pip install -e ".[dev]"
pytest
black src/
mypy src/
```

**Layout**

```
src/nanodistill/
├── config.py       # Configuration
├── core.py         # Orchestrator
├── teacher/        # Teacher API (LiteLLM)
├── amplifier/      # Policy + synthetic data
├── distiller/      # MLX-LM training
├── data/           # Loaders and formatters
└── utils/          # Errors and helpers
```

---

## Roadmap

**Current:** MLX-LM on Apple Silicon, Claude Sonnet as default teacher, policy-based synthetic generation.

**Planned:** Cross-platform (e.g. Unsloth), more teachers (GPT-4o, Gemini, Ollama), richer amplification and evaluation, CLI.

---

## License

MIT

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines on how to get started, development workflow, and our standards. This project is governed by our [Code of Conduct](docs/CODE_OF_CONDUCT.md).

**Security concerns?** Please see [SECURITY.md](docs/SECURITY.md) for our security policy and responsible disclosure process.

---

## Citation

```bibtex
@software{nanodistill2026,
  title = {NanoDistill: Knowledge Distillation for Small Language Models},
  author = {NanoDistill Contributors},
  year = {2026},
  url = {https://github.com/viditaggarwal/nanodistill}
}
```
