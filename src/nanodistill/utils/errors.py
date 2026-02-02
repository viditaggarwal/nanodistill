"""Custom exceptions and error handling for NanoDistill."""

import os


class NanoDistillError(Exception):
    """Base exception for all NanoDistill errors."""

    pass


class TeacherAPIError(NanoDistillError):
    """Raised when teacher model API fails."""

    pass


class ConfigError(NanoDistillError):
    """Raised when configuration is invalid."""

    pass


class AmplificationError(NanoDistillError):
    """Raised when dataset amplification fails."""

    pass


class TrainingError(NanoDistillError):
    """Raised when model training fails."""

    pass


class ExportError(NanoDistillError):
    """Raised when model export fails."""

    pass


def validate_teacher_api_key(teacher_model: str) -> None:
    """Validate that required API key is set for the teacher model.

    Args:
        teacher_model: Teacher model name (e.g., "claude-sonnet-4-5", "gpt-4")

    Raises:
        ConfigError: If required API key is not set

    Example:
        >>> validate_teacher_api_key("claude-sonnet-4-5")
        # Checks for ANTHROPIC_API_KEY environment variable
    """
    teacher_lower = teacher_model.lower().strip()

    # Claude models require ANTHROPIC_API_KEY
    if teacher_lower.startswith("claude"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ConfigError(
                "❌ ANTHROPIC_API_KEY not set\n\n"
                "Please set your Anthropic API key:\n"
                "  export ANTHROPIC_API_KEY='sk-ant-...'\n\n"
                "Get your API key from: https://console.anthropic.com"
            )

    # GPT models require OPENAI_API_KEY
    elif teacher_lower.startswith("gpt"):
        if not os.getenv("OPENAI_API_KEY"):
            raise ConfigError(
                "❌ OPENAI_API_KEY not set\n\n"
                "Please set your OpenAI API key:\n"
                "  export OPENAI_API_KEY='sk-...'\n\n"
                "Get your API key from: https://platform.openai.com/api-keys"
            )

    # Gemini models require GOOGLE_API_KEY
    elif teacher_lower.startswith("gemini"):
        if not os.getenv("GOOGLE_API_KEY"):
            raise ConfigError(
                "❌ GOOGLE_API_KEY not set\n\n"
                "Please set your Google API key:\n"
                "  export GOOGLE_API_KEY='...'\n\n"
                "Get your API key from: https://aistudio.google.com"
            )

    # Ollama models can run locally (no API key required)
    elif teacher_lower.startswith("ollama"):
        # Ollama runs locally, no API key needed
        pass

    # Unknown models - warn but don't fail
    else:
        # LiteLLM will handle validation
        pass


def validate_seed_count(seed_count: int, min_required: int = 1) -> None:
    """Validate that we have enough seed examples.

    Args:
        seed_count: Number of seed examples
        min_required: Minimum required (default: 1)

    Raises:
        ConfigError: If seed count is below minimum
    """
    if seed_count < min_required:
        raise ConfigError(
            f"❌ Insufficient seed examples: {seed_count}\n\n"
            f"Minimum required: {min_required}\n"
            f"This helps the teacher understand the task pattern."
        )


def validate_output_dir(output_dir: str) -> None:
    """Validate that output directory is writable.

    Args:
        output_dir: Path to output directory

    Raises:
        ConfigError: If directory is not writable
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Test write permission
        test_file = os.path.join(output_dir, ".nanodistill_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except (IOError, OSError) as e:
        raise ConfigError(
            f"❌ Output directory not writable: {output_dir}\n\n"
            f"Error: {e}\n"
            f"Please ensure the directory exists and you have write permissions."
        )
