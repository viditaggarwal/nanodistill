"""Data amplification pipeline for NanoDistill.

Orchestrates policy extraction and synthetic example generation to expand
a small seed dataset into a large training dataset.

Optimizations:
- Skips redundant CoT API calls when response_model provides embedded reasoning
- Filters low-confidence examples to improve training data quality
- Deduplicates near-identical synthetic examples
"""

import json
import logging
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Type

from pydantic import BaseModel

from ..data.loader import append_traces_to_jsonl, load_traces_from_jsonl, save_traces_to_jsonl
from ..teacher.client import TeacherClient
from ..teacher.schemas import TaskPolicy, ThinkingTrace

# Minimum confidence threshold for including traces in training data
DEFAULT_MIN_CONFIDENCE = 0.4

logger = logging.getLogger(__name__)


def _has_embedded_reasoning(response_model: Optional[Type[BaseModel]]) -> bool:
    """Check if a Pydantic response model already contains a reasoning field.

    When the output schema includes reasoning/explanation fields, we can skip
    the separate CoT synthesis API call and construct ThinkingTraces directly
    from the structured output, halving the number of API calls.

    Args:
        response_model: Optional Pydantic model to inspect

    Returns:
        True if model has a reasoning-like field
    """
    if response_model is None:
        return False

    reasoning_field_names = {"reasoning", "explanation", "rationale", "thinking", "analysis"}
    model_fields = set(response_model.model_fields.keys())
    return bool(model_fields & reasoning_field_names)


def _extract_reasoning_from_output(output_json: str, response_model: Type[BaseModel]) -> str:
    """Extract reasoning text from a structured JSON output.

    Args:
        output_json: JSON string of the structured output
        response_model: Pydantic model with reasoning field

    Returns:
        Extracted reasoning text, or generic fallback
    """
    reasoning_field_names = ["reasoning", "explanation", "rationale", "thinking", "analysis"]
    try:
        data = json.loads(output_json)
        for field_name in reasoning_field_names:
            if field_name in data and data[field_name]:
                return str(data[field_name])
    except (json.JSONDecodeError, TypeError):
        pass

    return "Structured output generated from task policy."


def _deduplicate_examples(
    examples: List[Dict[str, str]],
    existing_inputs: Optional[set] = None,
) -> List[Dict[str, str]]:
    """Remove duplicate examples based on normalized input text.

    Args:
        examples: List of examples to deduplicate
        existing_inputs: Optional set of already-seen input texts

    Returns:
        Deduplicated list of examples
    """
    seen = existing_inputs.copy() if existing_inputs else set()
    unique = []

    for ex in examples:
        normalized = ex["input"].strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(ex)

    removed = len(examples) - len(unique)
    if removed > 0:
        logger.info(f"Deduplication removed {removed} duplicate examples")

    return unique


class AmplificationPipeline:
    """Pipeline for amplifying seed data through synthetic generation.

    Two-phase approach:
    1. Extract task policy from seed data and CoT traces
    2. Generate new synthetic examples constrained by policy

    Optimizations:
    - Skips redundant CoT for structured outputs with embedded reasoning
    - Filters low-confidence traces before adding to training data
    - Deduplicates near-identical synthetic examples

    Attributes:
        teacher_client: Teacher client for API calls
        min_confidence: Minimum confidence threshold for filtering traces
    """

    def __init__(
        self,
        teacher_client: TeacherClient,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ):
        """Initialize amplification pipeline.

        Args:
            teacher_client: Initialized TeacherClient for API calls
            min_confidence: Minimum confidence for including traces (0.0-1.0)
        """
        self.teacher = teacher_client
        self.min_confidence = min_confidence

    def amplify(
        self,
        seed_examples: List[Dict[str, str]],
        cot_traces: List[ThinkingTrace],
        instruction: str,
        augment_factor: int,
        output_path: Optional[Path] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Generator[Tuple[int, int, int, int], None, Tuple[List[ThinkingTrace], TaskPolicy]]:
        """Amplify seed data into larger training dataset with incremental checkpointing.

        Generates synthetic examples in batches (one batch per augment unit),
        saving progress after each batch. Resumes from checkpoint if interrupted.

        When response_model has embedded reasoning fields (e.g., 'reasoning',
        'explanation'), skips the separate CoT synthesis API call and constructs
        ThinkingTraces directly from the structured output, halving API costs.

        Args:
            seed_examples: Original seed examples
            cot_traces: Generated Chain-of-Thought traces for seeds
            instruction: Task instruction / system prompt
            augment_factor: Target multiplication factor (e.g., 50x)
            output_path: Optional path to checkpoint file (traces_amplified.jsonl)
            response_model: Optional Pydantic model to enforce schema on synthetic outputs

        Yields:
            Tuple of (batch_num, total_batches, current_synthetic, total_synthetic)
            - batch_num: Current batch number (1-indexed)
            - total_batches: Total number of batches needed
            - current_synthetic: Total synthetic examples generated so far
            - total_synthetic: Total synthetic examples needed

        Returns:
            Tuple of (amplified traces, extracted policy)
            - List of original + synthetic ThinkingTrace objects
            - TaskPolicy describing the learned task pattern
        """
        # Start with original traces
        amplified_traces = cot_traces.copy()

        # Check if we can skip separate CoT synthesis for structured outputs
        skip_cot = _has_embedded_reasoning(response_model)
        if skip_cot:
            logger.info(
                "Response model has embedded reasoning — skipping separate CoT API calls "
                "(2x fewer API calls)"
            )

        # Collect existing inputs for deduplication
        existing_inputs = {ex["input"].strip().lower() for ex in seed_examples}

        # Phase 1: Extract policy from seeds
        policy = self._extract_policy(seed_examples, cot_traces, instruction)

        # Calculate batch parameters
        seed_count = len(seed_examples)
        total_synthetic = seed_count * (augment_factor - 1)
        batch_size = seed_count
        num_batches = augment_factor - 1

        # Skip if no synthetic examples needed
        if num_batches == 0:
            return amplified_traces, policy

        # Check for existing progress if checkpoint path provided
        existing_synthetic = 0
        completed_batches = 0
        if output_path and Path(output_path).exists():
            try:
                all_traces = load_traces_from_jsonl(output_path)
                existing_synthetic = max(0, len(all_traces) - seed_count)
                amplified_traces = all_traces.copy()
                completed_batches = existing_synthetic // batch_size
                # Add existing inputs for deduplication
                for trace in all_traces:
                    existing_inputs.add(trace.input.strip().lower())
            except Exception:
                # If checkpoint is corrupted, start fresh
                existing_synthetic = 0
                completed_batches = 0

        # Calculate which batches to generate
        start_batch = completed_batches + 1

        # Phase 2: Generate synthetic examples in batches
        for batch_num in range(start_batch, num_batches + 1):
            # Generate batch of synthetic examples
            batch_examples = self._generate_synthetic_examples(
                policy, batch_size, instruction, seed_count, response_model
            )

            # Deduplicate against existing data
            batch_examples = _deduplicate_examples(batch_examples, existing_inputs)

            # Update dedup set
            for ex in batch_examples:
                existing_inputs.add(ex["input"].strip().lower())

            # Generate CoT traces (or construct from structured output)
            if skip_cot and response_model is not None:
                batch_traces = self._construct_traces_from_structured(
                    batch_examples, response_model
                )
            else:
                batch_traces = self._synthesize_cot_for_synthetic(
                    batch_examples, instruction
                )

            # Filter low-confidence traces
            batch_traces = self._filter_by_confidence(batch_traces)

            # Save checkpoint
            if output_path:
                output_path = Path(output_path)
                if batch_num == 1 and completed_batches == 0:
                    # First batch: write seeds + first batch
                    save_traces_to_jsonl(amplified_traces + batch_traces, output_path)
                else:
                    # Subsequent batches: append only
                    append_traces_to_jsonl(batch_traces, output_path)

            # Update in-memory list
            amplified_traces.extend(batch_traces)

            # Calculate progress
            current_synthetic = existing_synthetic + (batch_num - completed_batches) * batch_size

            # Yield progress
            yield batch_num, num_batches, current_synthetic, total_synthetic

        # Return final results
        return amplified_traces, policy

    def _extract_policy(
        self,
        seed_examples: List[Dict[str, str]],
        cot_traces: List[ThinkingTrace],
        instruction: str,
    ) -> TaskPolicy:
        """Extract task policy from seed data.

        Args:
            seed_examples: Original seed examples
            cot_traces: Chain-of-Thought traces
            instruction: Task instruction

        Returns:
            TaskPolicy describing the task pattern
        """
        policy = self.teacher.extract_policy(seed_examples, cot_traces, instruction)
        return policy

    def _generate_synthetic_examples(
        self,
        policy: TaskPolicy,
        num_examples: int,
        instruction: str,
        seed_count: int,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> List[Dict[str, str]]:
        """Generate synthetic examples matching the policy.

        Args:
            policy: Task policy extracted from seed data
            num_examples: Number of examples to generate
            instruction: Task instruction
            seed_count: Number of original seed examples
            response_model: Optional Pydantic model to enforce schema

        Returns:
            List of generated examples with 'input' and 'output' fields
        """
        examples = self.teacher.generate_synthetic_examples(
            policy, num_examples, instruction, seed_count, response_model=response_model
        )
        return examples

    def _synthesize_cot_for_synthetic(
        self,
        synthetic_examples: List[Dict[str, str]],
        instruction: str,
    ) -> List[ThinkingTrace]:
        """Generate Chain-of-Thought traces for synthetic examples.

        Args:
            synthetic_examples: Generated synthetic examples
            instruction: Task instruction

        Returns:
            List of ThinkingTrace objects
        """
        traces = self.teacher.synthesize_cot(synthetic_examples, instruction)
        return traces

    def _construct_traces_from_structured(
        self,
        examples: List[Dict[str, str]],
        response_model: Type[BaseModel],
    ) -> List[ThinkingTrace]:
        """Construct ThinkingTraces from structured output without extra API calls.

        When the response_model includes reasoning fields, we extract the reasoning
        directly from the generated output JSON instead of making a separate API call.
        This halves the number of API calls per batch.

        Args:
            examples: Generated examples with 'input' and 'output' (JSON) fields
            response_model: Pydantic model with reasoning field

        Returns:
            List of ThinkingTrace objects
        """
        traces = []
        for ex in examples:
            reasoning = _extract_reasoning_from_output(ex["output"], response_model)
            trace = ThinkingTrace(
                input=ex["input"],
                thinking=reasoning,
                output=ex["output"],
                confidence=0.85,  # Default confidence for structured outputs
            )
            traces.append(trace)

        logger.info(
            f"Constructed {len(traces)} traces from structured output "
            f"(skipped {len(traces)} CoT API calls)"
        )
        return traces

    def _filter_by_confidence(
        self,
        traces: List[ThinkingTrace],
    ) -> List[ThinkingTrace]:
        """Filter traces below the minimum confidence threshold.

        Args:
            traces: List of ThinkingTrace objects

        Returns:
            Filtered list with only high-confidence traces
        """
        if self.min_confidence <= 0:
            return traces

        filtered = [t for t in traces if t.confidence >= self.min_confidence]
        removed = len(traces) - len(filtered)

        if removed > 0:
            logger.info(
                f"Confidence filter removed {removed}/{len(traces)} traces "
                f"(threshold: {self.min_confidence})"
            )

        return filtered
