"""Pytest configuration and fixtures for NanoDistill tests."""

import pytest


@pytest.fixture
def sample_seed_data():
    """Sample seed data for testing."""
    return [
        {"input": "What is 2+2?", "output": "4"},
        {"input": "What is 3+5?", "output": "8"},
        {"input": "What is 10-4?", "output": "6"},
        {"input": "What is 5×3?", "output": "15"},
        {"input": "What is 20÷4?", "output": "5"},
    ]


@pytest.fixture
def sample_instruction():
    """Sample task instruction for testing."""
    return "You are a helpful math tutor. Show your reasoning step-by-step."
