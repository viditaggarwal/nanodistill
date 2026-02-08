"""Tests for the training_data bypass path."""

import json

import pytest

from nanodistill.data.loader import load_training_data
from nanodistill.teacher.schemas import ThinkingTrace
from nanodistill.utils.errors import ConfigError

# --- load_training_data: list input ---


def test_load_training_data_from_list():
    """Test loading training data from a list of dicts."""
    data = [
        {"input": "What is 2+2?", "thinking": "2+2=4", "output": "4"},
        {"input": "What is 3+5?", "thinking": "3+5=8", "output": "8"},
    ]
    traces = load_training_data(data)
    assert len(traces) == 2
    assert all(isinstance(t, ThinkingTrace) for t in traces)
    assert traces[0].input == "What is 2+2?"
    assert traces[0].thinking == "2+2=4"
    assert traces[0].output == "4"
    assert traces[0].confidence == 0.9  # default


def test_load_training_data_with_confidence():
    """Test that custom confidence values are preserved."""
    data = [
        {"input": "q", "thinking": "t", "output": "a", "confidence": 0.75},
    ]
    traces = load_training_data(data)
    assert traces[0].confidence == 0.75


def test_load_training_data_empty_list():
    """Test that empty list raises ConfigError."""
    with pytest.raises(ConfigError, match="at least 1 example"):
        load_training_data([])


def test_load_training_data_missing_thinking():
    """Test that missing 'thinking' field raises ConfigError."""
    data = [{"input": "q", "output": "a"}]
    with pytest.raises(ConfigError, match="missing required fields.*thinking"):
        load_training_data(data)


def test_load_training_data_missing_input():
    """Test that missing 'input' field raises ConfigError."""
    data = [{"thinking": "t", "output": "a"}]
    with pytest.raises(ConfigError, match="missing required fields.*input"):
        load_training_data(data)


def test_load_training_data_missing_output():
    """Test that missing 'output' field raises ConfigError."""
    data = [{"input": "q", "thinking": "t"}]
    with pytest.raises(ConfigError, match="missing required fields.*output"):
        load_training_data(data)


def test_load_training_data_non_dict_element():
    """Test that non-dict element raises ConfigError."""
    with pytest.raises(ConfigError, match="must be a dict"):
        load_training_data(["not a dict"])


def test_load_training_data_invalid_type():
    """Test that invalid type raises ConfigError."""
    with pytest.raises(ConfigError, match="must be a list of dicts"):
        load_training_data(123)


# --- load_training_data: file input ---


def test_load_training_data_from_jsonl(tmp_path):
    """Test loading training data from a JSONL file."""
    jsonl_file = tmp_path / "traces.jsonl"
    records = [
        {"input": "q1", "thinking": "t1", "output": "a1", "confidence": 0.9},
        {"input": "q2", "thinking": "t2", "output": "a2", "confidence": 0.8},
    ]
    with open(jsonl_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    traces = load_training_data(str(jsonl_file))
    assert len(traces) == 2
    assert traces[0].input == "q1"
    assert traces[1].confidence == 0.8


def test_load_training_data_file_not_found():
    """Test that missing file raises ConfigError."""
    with pytest.raises(ConfigError, match="not found"):
        load_training_data("/nonexistent/path/data.jsonl")


def test_load_training_data_unsupported_format(tmp_path):
    """Test that non-JSONL file raises ConfigError."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("input,thinking,output\nq,t,a\n")
    with pytest.raises(ConfigError, match="Unsupported training data format"):
        load_training_data(str(csv_file))


def test_load_training_data_empty_jsonl(tmp_path):
    """Test that empty JSONL file raises ConfigError."""
    jsonl_file = tmp_path / "empty.jsonl"
    jsonl_file.write_text("")
    with pytest.raises(ConfigError, match="empty"):
        load_training_data(str(jsonl_file))


def test_load_training_data_from_path_object(tmp_path):
    """Test loading training data from a Path object."""
    from pathlib import Path

    jsonl_file = tmp_path / "traces.jsonl"
    record = {"input": "q", "thinking": "t", "output": "a", "confidence": 0.9}
    with open(jsonl_file, "w") as f:
        f.write(json.dumps(record) + "\n")

    traces = load_training_data(Path(jsonl_file))
    assert len(traces) == 1
    assert traces[0].input == "q"
