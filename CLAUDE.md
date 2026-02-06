# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation & Setup
```bash
# Install package in development mode with dependencies
pip install -e ".[dev]"

# Verify installation
python -c "from nanodistill import distill; print('✓ Installation successful')"
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_config.py -v

# Run specific test function
pytest tests/test_config.py::test_config_name -v

# Run tests with coverage report
pytest --cov=src/nanodistill --cov-report=term-missing
```

### Code Quality
```bash
# Format code with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type checking with mypy
mypy src/
```

### Build & Quality Checks
```bash
# Run all quality checks before pushing
./scripts/build
```

This runs all checks in sequence and exits with error if any fail:
1. **Ruff** - Linting and code style
2. **mypy** - Type checking
3. **pytest** - Unit tests
4. **Import validation** - Package imports successfully

**Recommendation:** Run `./scripts/build` before `git push` to ensure code quality.

### Running the Pipeline
```bash
# See docs/QUICK_START.md for examples
# Requires ANTHROPIC_API_KEY set
python -c "
from nanodistill import distill
result = distill(name='model', seed=[...], instruction='...')
"
```

## Project Architecture

4-stage pipeline in specialized modules:

| Stage | Module | Key Components |
|-------|--------|-----------------|
| 1. CoT & Policy | `teacher/client.py` | `TeacherClient` (LiteLLM), `synthesize_cot()`, schemas |
| 2. Amplification | `amplifier/pipeline.py` | `AmplificationPipeline` - policy extraction & synthetic generation |
| 3. Fine-tuning | `distiller/trainer.py` | `MLXTrainer` - LoRA training on Apple Silicon |
| Entry/Config | `core.py`, `config.py` | `distill()` orchestrator, `DistillationConfig` validator |

**Data flow:** `seed` → CoT traces → amplified examples → fine-tuned model

**Caching:** If `traces_amplified.jsonl` exists, skips API calls and jumps to training.

## Key Patterns

- **LiteLLM** - Swap teachers: `distill(..., teacher="gpt-4o")` or `teacher="gemini-pro"`
- **Instructor** - Structured output via Pydantic with automatic retry & filtering
- **Resumable** - Cache check skips API calls, re-use amplified data with different training params
- **Schema validation** - Optional `response_model` parameter enforces structured outputs

## Quick Configs

```python
# Different teacher (requires respective API key)
distill(..., teacher="gpt-4o")

# Different student model
distill(..., student="mlx-community/Mistral-7B-Instruct-4bit")

# Memory-constrained: lower batch_size, max_seq_length, lora_rank
distill(..., batch_size=1, max_seq_length=256, lora_rank=4)

# Structured output validation
from pydantic import BaseModel
class Output(BaseModel):
    answer: str
    reasoning: str

distill(..., response_model=Output)
```

## Testing

Tests use mocked LiteLLM (no real API calls). See `tests/conftest.py` for fixtures.

```bash
pytest                                    # All tests
pytest tests/test_config.py -v           # Specific file
pytest --cov=src/nanodistill             # With coverage
```

## Requirements

- **Python:** 3.9+
- **macOS with Apple Silicon** (M1/M2/M3+) for MLX training
- **API Keys** (set via `.env` or environment):
  - `ANTHROPIC_API_KEY` - Claude (required by default)
  - `OPENAI_API_KEY` - GPT models (if used)
  - `GOOGLE_API_KEY` - Gemini (if used)

## File Organization Guidelines

### Markdown Files
**All `.md` documentation files go in the `docs/` folder**, except for:
- `README.md` - Root level (main entry point)
- `LICENSE` - Root level (license file)
- `CHANGELOG.md` - Root level (future, when added)

**Reason:** Keeps all documentation organized and searchable in one place.

**Examples:**
- ✅ `docs/QUICK_START.md` - User tutorial
- ✅ `docs/CONTRIBUTING.md` - Community guidelines
- ✅ `docs/CODE_OF_CONDUCT.md` - Code of conduct
- ❌ `CONTRIBUTING.md` - Wrong, should be in docs/
