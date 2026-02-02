"""Tests for config module."""

import pytest

from nanodistill.config import DistillationConfig
from nanodistill.utils.errors import ConfigError


def test_config_valid(sample_seed_data, sample_instruction):
    """Test creating valid DistillationConfig."""
    config = DistillationConfig(
        name="test-run",
        seed=sample_seed_data,
        instruction=sample_instruction,
    )

    assert config.name == "test-run"
    assert len(config.seed) == len(sample_seed_data)
    assert config.augment_factor == 50  # Default
    assert config.teacher == "claude-sonnet-4-5"  # Default


def test_config_missing_seed():
    """Test that config fails with missing seed."""
    with pytest.raises(ValueError):
        DistillationConfig(
            name="test",
            seed=[],  # Empty seed
            instruction="Test instruction",
        )


def test_config_missing_input_field():
    """Test that config fails if seed is missing 'input' field."""
    with pytest.raises(ValueError):
        DistillationConfig(
            name="test",
            seed=[
                {"output": "answer"},  # Missing 'input'
            ],
            instruction="Test instruction",
        )


def test_config_missing_output_field():
    """Test that config fails if seed is missing 'output' field."""
    with pytest.raises(ValueError):
        DistillationConfig(
            name="test",
            seed=[
                {"input": "question"},  # Missing 'output'
            ],
            instruction="Test instruction",
        )


def test_config_empty_input():
    """Test that config fails if seed has empty input."""
    with pytest.raises(ValueError):
        DistillationConfig(
            name="test",
            seed=[
                {"input": "", "output": "answer"},  # Empty input
            ],
            instruction="Test instruction",
        )


def test_config_custom_values(sample_seed_data, sample_instruction):
    """Test config with custom values."""
    config = DistillationConfig(
        name="custom-run",
        seed=sample_seed_data,
        instruction=sample_instruction,
        teacher="gpt-4",
        student="llama-2-7b",
        augment_factor=100,
        output_dir="/custom/path",
    )

    assert config.teacher == "gpt-4"
    assert config.student == "llama-2-7b"
    assert config.augment_factor == 100
    assert config.output_dir == "/custom/path"
