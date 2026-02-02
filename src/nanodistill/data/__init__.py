"""Data module for loading and formatting training datasets."""

from .formatter import (
    extract_thinking_from_response,
    format_for_inference,
    format_for_training,
    format_traces_for_training,
    merge_thinking_output,
)
from .loader import (
    load_seed_data,
    load_traces_from_jsonl,
    save_traces_to_jsonl,
    to_dict_list,
    to_hf_dataset,
)

__all__ = [
    "load_seed_data",
    "to_hf_dataset",
    "to_dict_list",
    "save_traces_to_jsonl",
    "load_traces_from_jsonl",
    "format_for_training",
    "format_traces_for_training",
    "format_for_inference",
    "merge_thinking_output",
    "extract_thinking_from_response",
]
