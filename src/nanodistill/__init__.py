"""NanoDistill: Convert 10 examples + instruction into a reasoning-capable small language model.

Main entry point for the NanoDistill library. Use the `distill()` function to convert
seed examples into a fine-tuned, locally-runnable model optimized for Apple Silicon.

Example:
    >>> from nanodistill import distill
    >>> result = distill(
    ...     name="math-tutor",
    ...     seed=[
    ...         {"input": "What is 2+2?", "output": "4"},
    ...         {"input": "What is 3+5?", "output": "8"},
    ...     ],
    ...     instruction="You are a helpful math tutor. Show your reasoning.",
    ... )
    >>> print(f"Model ready: {result.model_path}")
"""

from .core import DistillationResult, distill
from .config import DistillationConfig

__version__ = "0.1.0"

__all__ = [
    "distill",
    "DistillationResult",
    "DistillationConfig",
]
