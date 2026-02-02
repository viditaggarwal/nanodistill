"""Tests for data loading and formatting."""

import json

import pytest

from nanodistill.data import load_seed_data, to_dict_list, to_hf_dataset
from nanodistill.teacher.schemas import ThinkingTrace
from nanodistill.utils.errors import ConfigError


def test_load_seed_data_list(sample_seed_data):
    """Test loading seed data from Python list."""
    loaded = load_seed_data(sample_seed_data)
    assert len(loaded) == len(sample_seed_data)
    assert loaded == sample_seed_data


def test_load_seed_data_json_file(tmp_path, sample_seed_data):
    """Test loading seed data from JSON file."""
    json_file = tmp_path / "seeds.json"
    with open(json_file, "w") as f:
        json.dump(sample_seed_data, f)

    loaded = load_seed_data(str(json_file))
    assert len(loaded) == len(sample_seed_data)


def test_load_seed_data_jsonl_file(tmp_path, sample_seed_data):
    """Test loading seed data from JSONL file."""
    jsonl_file = tmp_path / "seeds.jsonl"
    with open(jsonl_file, "w") as f:
        for example in sample_seed_data:
            f.write(json.dumps(example) + "\n")

    loaded = load_seed_data(str(jsonl_file))
    assert len(loaded) == len(sample_seed_data)


def test_load_seed_data_missing_input():
    """Test that loading invalid seed data raises error."""
    invalid_seed = [{"output": "answer"}]  # Missing 'input'

    with pytest.raises(ConfigError):
        load_seed_data(invalid_seed)


def test_load_seed_data_missing_output():
    """Test that loading seed without output raises error."""
    invalid_seed = [{"input": "question"}]  # Missing 'output'

    with pytest.raises(ConfigError):
        load_seed_data(invalid_seed)


def test_to_hf_dataset():
    """Test converting ThinkingTrace to HuggingFace Dataset."""
    traces = [
        ThinkingTrace(
            input="What is 2+2?",
            thinking="Let me think... 2 plus 2 equals 4",
            output="4",
        ),
        ThinkingTrace(
            input="What is 3+5?",
            thinking="3 plus 5 equals 8",
            output="8",
        ),
    ]

    dataset = to_hf_dataset(traces)

    assert len(dataset) == 2
    assert "input" in dataset.column_names
    assert "thinking" in dataset.column_names
    assert "output" in dataset.column_names


def test_to_dict_list():
    """Test converting ThinkingTrace to dict list."""
    traces = [
        ThinkingTrace(
            input="Q1",
            thinking="reasoning1",
            output="A1",
        ),
    ]

    dict_list = to_dict_list(traces)

    assert len(dict_list) == 1
    assert dict_list[0]["input"] == "Q1"
    assert dict_list[0]["thinking"] == "reasoning1"
    assert dict_list[0]["output"] == "A1"
