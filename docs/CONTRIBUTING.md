# Contributing to NanoDistill

Thank you for your interest in contributing to NanoDistill! We welcome contributions from the community, whether they're bug fixes, feature requests, documentation improvements, or new ideas.

## Getting Started

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/nanodistill.git
   cd nanodistill
   ```

2. **Install in development mode with dependencies:**
   ```bash
   pip install -e ".[dev]"
   # or with uv:
   uv pip install -e ".[dev]"
   ```

3. **Verify installation:**
   ```bash
   python -c "from nanodistill import distill; print('✓ Installation successful')"
   ```

## Development Workflow

### Running Tests

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

We follow strict code quality standards. Before submitting a PR, run these checks:

```bash
# Format code with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type checking with mypy
mypy src/
```

All PRs must pass these checks. We recommend setting up pre-commit hooks locally:

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks (if configured in repo)
pre-commit install
```

### Project Structure

```
src/nanodistill/
├── config.py       # Configuration and validation
├── core.py         # Main orchestrator
├── teacher/        # Teacher API wrapper (LiteLLM)
├── amplifier/      # Policy extraction and synthetic data generation
├── distiller/      # MLX-LM fine-tuning
├── data/           # Data loaders and formatters
└── utils/          # Error handling and utilities

tests/              # Test suite
docs/               # Documentation
```

## Submitting Changes

### Before You Start

1. **Check existing issues** - Look for open issues or discussions that relate to your idea
2. **Create an issue first** (for non-trivial changes) - Describe what you want to do and why
3. **Discuss major changes** - For architectural changes or new features, open an issue to get feedback before implementing

### Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** - Keep commits focused and atomic

3. **Write tests** - Add tests for new functionality or bug fixes
   - Tests go in `tests/` directory
   - Follow naming convention: `test_*.py` for files
   - Use fixtures in `tests/conftest.py` for common test data

4. **Run the full test suite and linting:**
   ```bash
   black src/ tests/
   ruff check src/ tests/
   mypy src/
   pytest
   ```

5. **Update documentation** - If your change affects usage, update relevant docs in `docs/`

### Pull Request Process

1. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a PR** with:
   - Clear title describing the change
   - Description of what and why
   - Reference to any related issues (e.g., "Fixes #123")
   - Any testing notes or considerations

3. **Respond to feedback** - We may request changes during review. This is normal and collaborative.

4. **PR Checklist:**
   - [ ] Tests pass locally (`pytest`)
   - [ ] Code is formatted (`black src/ tests/`)
   - [ ] Linting passes (`ruff check src/ tests/`)
   - [ ] Type checking passes (`mypy src/`)
   - [ ] New tests added for new functionality
   - [ ] Documentation updated if needed

## What We're Looking For

**Good contributions:**
- Bug fixes with clear reproduction steps and tests
- New features with documentation and tests
- Performance improvements with benchmarks
- Documentation improvements and examples
- Test coverage improvements

**Help us help you:**
- Provide context for bug reports (OS, Python version, error messages)
- Link related issues or PRs
- Include reproducible examples
- Be patient with review feedback

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) (enforced by Black/Ruff)
- Use type hints for function signatures
- Keep functions focused and testable
- Use descriptive variable names
- Add docstrings for public functions and classes

### Example:

```python
from typing import List, Dict, Optional

def extract_policy(
    examples: List[Dict[str, str]],
    max_retries: int = 3,
) -> Optional[Dict[str, any]]:
    """Extract task policy from examples.

    Args:
        examples: List of {"input": "...", "output": "..."} dicts
        max_retries: Maximum number of API retries

    Returns:
        Dictionary containing extracted policy, or None on failure
    """
    pass
```

### Testing

- Avoid actual API calls in tests—use mocking (tests should be fast and deterministic)
- Test both success and failure cases
- Use descriptive test names: `test_config_validates_batch_size_range`
- Keep tests focused on one thing

## Areas Where We Need Help

- Cross-platform support (Windows/Linux in addition to macOS)
- Performance optimizations
- Additional teacher models (beyond LiteLLM)
- Better error messages and user guidance
- More example tasks and seed data
- Documentation improvements

## Questions or Need Help?

- Open an issue with the label `question`
- Check existing discussions for similar questions
- Look through the docs in `docs/` and examples

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to abide by its terms.

## License

By contributing to NanoDistill, you agree that your contributions will be licensed under its MIT License.

---

Thanks for contributing to NanoDistill! 🙌
