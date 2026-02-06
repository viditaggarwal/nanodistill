"""Data loading utilities for NanoDistill."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Union

from datasets import Dataset

from ..teacher.schemas import ThinkingTrace
from ..utils.errors import ConfigError


def load_seed_data(
    seed: Union[List[Dict[str, str]], str, Path],
) -> List[Dict[str, str]]:
    """Load seed data from various formats.

    Supports:
    - Python list of dicts
    - JSON files (.json)
    - JSONL files (.jsonl) - one JSON object per line
    - CSV files (.csv)

    Args:
        seed: Seed data as list, file path, or file path string

    Returns:
        List of examples with 'input' and 'output' fields

    Raises:
        ConfigError: If seed format is invalid or file cannot be loaded
    """
    # Already a list
    if isinstance(seed, list):
        _validate_seed_list(seed)
        return seed

    # File path
    if isinstance(seed, (str, Path)):
        path = Path(seed)

        if not path.exists():
            raise ConfigError(f"Seed file not found: {path}")

        try:
            if path.suffix == ".json":
                with open(path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        _validate_seed_list(data)
                        return data
                    else:
                        raise ConfigError("JSON file must contain a list of examples")

            elif path.suffix == ".jsonl":
                data = []
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                _validate_seed_list(data)
                return data

            elif path.suffix == ".csv":
                data = []
                with open(path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        data.append(dict(row))
                _validate_seed_list(data)
                return data

            else:
                raise ConfigError(
                    f"Unsupported file format: {path.suffix}. " f"Supported: .json, .jsonl, .csv"
                )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ConfigError(f"Failed to load seed file {path}: {str(e)}") from e

    raise ConfigError(
        f"seed must be a list of dicts, file path string, or Path object, " f"got {type(seed)}"
    )


def _validate_seed_list(seed: List[Dict[str, str]]) -> None:
    """Validate seed list format.

    Args:
        seed: Seed data to validate

    Raises:
        ConfigError: If seed format is invalid
    """
    if not isinstance(seed, list):
        raise ConfigError(f"seed must be a list, got {type(seed)}")

    if not seed:
        raise ConfigError("seed must contain at least 1 example")

    for i, example in enumerate(seed):
        if not isinstance(example, dict):
            raise ConfigError(f"seed[{i}] must be a dict, got {type(example)}")

        if "input" not in example:
            raise ConfigError(f"seed[{i}] missing required 'input' field")

        if "output" not in example:
            raise ConfigError(f"seed[{i}] missing required 'output' field")

        if not example["input"] or not str(example["input"]).strip():
            raise ConfigError(f"seed[{i}] has empty 'input' field")

        if not example["output"] or not str(example["output"]).strip():
            raise ConfigError(f"seed[{i}] has empty 'output' field")


def to_hf_dataset(
    traces: List[ThinkingTrace],
) -> Dataset:
    """Convert ThinkingTrace objects to HuggingFace Dataset.

    Args:
        traces: List of Chain-of-Thought traces

    Returns:
        HuggingFace Dataset with 'input', 'thinking', and 'output' columns
    """
    if not traces:
        raise ConfigError("Cannot create dataset from empty traces list")

    data = {
        "input": [t.input for t in traces],
        "thinking": [t.thinking for t in traces],
        "output": [t.output for t in traces],
        "confidence": [t.confidence for t in traces],
    }

    dataset = Dataset.from_dict(data)
    return dataset


def to_dict_list(traces: List[ThinkingTrace]) -> List[Dict[str, str]]:
    """Convert ThinkingTrace objects to list of dicts.

    Args:
        traces: List of Chain-of-Thought traces

    Returns:
        List of dicts with 'input', 'thinking', 'output', 'confidence'
    """
    return [
        {
            "input": t.input,
            "thinking": t.thinking,
            "output": t.output,
            "confidence": str(t.confidence),
        }
        for t in traces
    ]


def save_traces_to_jsonl(
    traces: List[ThinkingTrace],
    output_path: Union[str, Path],
) -> None:
    """Save ThinkingTrace objects to JSONL file.

    Args:
        traces: List of Chain-of-Thought traces
        output_path: Path to output JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for trace in traces:
            f.write(trace.model_dump_json() + "\n")


def append_traces_to_jsonl(
    traces: List[ThinkingTrace],
    output_path: Union[str, Path],
) -> None:
    """Append ThinkingTrace objects to JSONL file.

    Creates file if it doesn't exist, otherwise appends to existing file.

    Args:
        traces: List of Chain-of-Thought traces to append
        output_path: Path to output JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use append mode if file exists, write mode if new
    mode = "a" if output_path.exists() else "w"
    with open(output_path, mode) as f:
        for trace in traces:
            f.write(trace.model_dump_json() + "\n")


def load_traces_from_jsonl(
    input_path: Union[str, Path],
) -> List[ThinkingTrace]:
    """Load ThinkingTrace objects from JSONL file.

    Args:
        input_path: Path to input JSONL file

    Returns:
        List of ThinkingTrace objects
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise ConfigError(f"Input file not found: {input_path}")

    traces = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                traces.append(ThinkingTrace(**data))

    return traces
