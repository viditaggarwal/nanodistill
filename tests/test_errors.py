"""Tests for error handling and validation."""

import os
import pytest

from nanodistill.utils.errors import (
    ConfigError,
    validate_output_dir,
    validate_seed_count,
    validate_teacher_api_key,
)


def test_validate_teacher_api_key_claude_missing():
    """Test that Claude models require ANTHROPIC_API_KEY."""
    # Ensure env var is not set
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]

    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        validate_teacher_api_key("claude-sonnet-4-5")


def test_validate_teacher_api_key_gpt_missing():
    """Test that GPT models require OPENAI_API_KEY."""
    # Ensure env var is not set
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        validate_teacher_api_key("gpt-4")


def test_validate_teacher_api_key_ollama():
    """Test that Ollama models don't require API key."""
    # Should not raise
    validate_teacher_api_key("ollama/llama2")


def test_validate_seed_count_valid():
    """Test valid seed count."""
    # Should not raise
    validate_seed_count(10)
    validate_seed_count(1)


def test_validate_seed_count_insufficient():
    """Test that insufficient seed count raises error."""
    with pytest.raises(ConfigError):
        validate_seed_count(0)


def test_validate_output_dir_valid(tmp_path):
    """Test validation of writable output directory."""
    # Should not raise
    validate_output_dir(str(tmp_path))


def test_validate_output_dir_creates_directory(tmp_path):
    """Test that validate_output_dir creates directory if needed."""
    new_dir = tmp_path / "new_output_dir"
    assert not new_dir.exists()

    validate_output_dir(str(new_dir))

    assert new_dir.exists()
