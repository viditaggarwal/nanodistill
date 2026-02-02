"""Data formatting utilities for training."""

from typing import Dict, List

from transformers import PreTrainedTokenizer

from ..teacher.schemas import ThinkingTrace


def format_for_training(
    trace: ThinkingTrace,
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, List[int]]:
    """Format Chain-of-Thought trace into training format.

    Combines input and thinking/output into a conversation format using
    the model's chat template.

    Args:
        trace: ThinkingTrace to format
        tokenizer: HuggingFace tokenizer with chat template

    Returns:
        Dictionary with 'input_ids' and 'labels' for training
    """
    # Build conversation format
    conversation = [
        {"role": "user", "content": trace.input},
        {
            "role": "assistant",
            "content": f"<thinking>\n{trace.thinking}\n</thinking>\n\n{trace.output}",
        },
    ]

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=False,
    )

    # Return as training format
    return {
        "input_ids": formatted,
        "labels": formatted,
    }


def format_traces_for_training(
    traces: List[ThinkingTrace],
    tokenizer: PreTrainedTokenizer,
) -> List[Dict[str, List[int]]]:
    """Format multiple traces for training.

    Args:
        traces: List of ThinkingTrace objects
        tokenizer: HuggingFace tokenizer with chat template

    Returns:
        List of formatted training examples
    """
    return [format_for_training(trace, tokenizer) for trace in traces]


def format_for_inference(
    input_text: str,
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, List[int]]:
    """Format input for model inference.

    Args:
        input_text: User input
        tokenizer: HuggingFace tokenizer with chat template

    Returns:
        Dictionary with 'input_ids' for inference
    """
    conversation = [
        {"role": "user", "content": input_text},
    ]

    formatted = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
    )

    return {
        "input_ids": formatted,
    }


def merge_thinking_output(thinking: str, output: str) -> str:
    """Merge thinking and output into a single response.

    Args:
        thinking: Chain-of-thought reasoning
        output: Final answer

    Returns:
        Merged response string
    """
    return f"<thinking>\n{thinking}\n</thinking>\n\n{output}"


def extract_thinking_from_response(response: str) -> tuple[str, str]:
    """Extract thinking and output from model response.

    Args:
        response: Full model response

    Returns:
        Tuple of (thinking, output)
    """
    # Try to extract thinking from tags
    if "<thinking>" in response and "</thinking>" in response:
        start = response.find("<thinking>") + len("<thinking>")
        end = response.find("</thinking>")
        thinking = response[start:end].strip()
        output = response[end + len("</thinking>") :].strip()
        return thinking, output

    # Fallback: return response as output, empty thinking
    return "", response
