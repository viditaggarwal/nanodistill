# NanoDistill

Convert 10 examples + instruction into a locally-runnable, reasoning-capable small language model.

## Core Promise

**Give us 10 examples and an API key. We give you a locally runnable, reasoning-capable model.**

## Quick Start

### Installation

```bash
pip install -e .
```

### Requirements

- **Python**: 3.9+
- **Hardware**: Mac M1/M2/M3+ (Apple Silicon) with 16GB+ RAM
- **API Key**: Anthropic API key (get from https://console.anthropic.com)

### 5-Minute Walkthrough

```bash
# 1. Set your API key
export ANTHROPIC_API_KEY='sk-ant-...'

# 2. Create seed data (seeds.json)
cat > seeds.json << 'EOF'
[
  {"input": "What is 2+2?", "output": "4"},
  {"input": "What is 3+5?", "output": "8"},
  {"input": "What is 10-4?", "output": "6"},
  ...10+ examples
]
EOF

# 3. Run distillation
python -c "
from nanodistill import distill
result = distill(
    name='math-tutor',
    seed='seeds.json',
    instruction='You are a helpful math tutor.',
    teacher='claude-sonnet-4-5'
)
print(f'✅ Model: {result.model_path}')
"

# 4. Test the model
python -c "
from mlx_lm import load, generate
model, tokenizer = load('./outputs/math-tutor/model')
response = generate(model, tokenizer, 'What is 7+8?', max_tokens=100)
print(response)
"
```

### How It Works

```
Your 10 Examples
    ↓
    ├→ Generate reasoning with Claude
    ├→ Extract task pattern
    ├→ Create 490 synthetic examples
    └→ Fine-tune Llama-3-8B
    ↓
Locally-runnable Model ✅
```

## How It Works

NanoDistill transforms your seed examples through 4 stages:

1. **🎓 Policy Extraction** - Analyzes seed data to extract the underlying task pattern
2. **🔄 Synthetic Generation** - Uses Claude to generate diverse new examples matching the pattern
3. **📚 Data Amplification** - Converts examples into Chain-of-Thought training data
4. **🔥 Model Fine-tuning** - Trains student model on Apple Silicon using MLX-LM

## Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Step-by-step from seed data to model
- **[Complete Workflow](docs/WORKFLOW.md)** - Detailed visual guide of each stage
- **[Model Setup & Inference](docs/MODEL_SETUP.md)** - How to download, train, and use models
- **[Implementation Plan](docs/IMPLEMENTATION_PLAN.md)** - Technical architecture details

## Configuration

### Environment Variables

- `ANTHROPIC_API_KEY` - Required for Claude teacher model (get from https://console.anthropic.com)
- `HF_HUB_TIMEOUT` - Optional: increase if slow internet (default: 300s)

### Configuration Parameters

- `name` - Identifier for this distillation run
- `seed` - Examples with `input` and `output` fields (min 1, recommend 10+)
- `instruction` - System prompt describing the task
- `teacher` - Teacher model (default: "claude-sonnet-4-5")
- `student` - Student model (default: "mlx-community/Llama-3-8B-Instruct-4bit")
- `augment_factor` - Multiply seed data by this factor (default: 50)
- `output_dir` - Where to save outputs (default: "./outputs")

## Output

Each distillation run creates:

- `{output_dir}/{name}/model/` - Fine-tuned model (MLX format)
- `{output_dir}/{name}/model.gguf` - Quantized model for inference
- Training logs and metrics

## Troubleshooting

### API Key Issues

```
❌ ANTHROPIC_API_KEY not set
```

Set your API key:

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

### Memory Issues

If you encounter out-of-memory errors:
- Reduce `augment_factor` to 20-30
- Use a smaller student model
- Ensure no other GPU-heavy applications are running

### MLX Installation

MLX requires macOS 13+. For more info: https://github.com/ml-explore/mlx

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Type checking
mypy src/
```

## Architecture

```
src/nanodistill/
├── config.py           # Configuration validation
├── core.py             # Main orchestrator
├── teacher/            # Claude API integration
├── amplifier/          # Policy extraction & synthetic generation
├── distiller/          # MLX-LM training
├── data/               # Dataset utilities
└── utils/              # Error handling & logging
```

## Roadmap

**Current (Phase 1)**:
- ✅ MLX-LM training on Apple Silicon
- ✅ Claude Sonnet as teacher
- ✅ Policy-based synthetic generation

**Post-MVP**:
- Cross-platform support (Unsloth for Linux/Windows)
- Multiple teacher model options (GPT-4o, Gemini, Ollama)
- Advanced amplification strategies
- Model evaluation harness
- CLI interface

## License

MIT

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon).

## Citation

If you use NanoDistill in your research, please cite:

```bibtex
@software{nanodistill2025,
  title = {NanoDistill: Knowledge Distillation for Small Language Models},
  author = {NanoDistill Contributors},
  year = {2025},
  url = {https://github.com/yourusername/nanodistill}
}
```
