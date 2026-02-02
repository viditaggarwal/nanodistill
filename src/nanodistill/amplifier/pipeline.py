"""Data amplification pipeline for NanoDistill.

Orchestrates policy extraction and synthetic example generation to expand
a small seed dataset into a large training dataset.
"""

from typing import Dict, List, Optional, Type

from pydantic import BaseModel

from ..teacher.client import TeacherClient
from ..teacher.schemas import TaskPolicy, ThinkingTrace


class AmplificationPipeline:
    """Pipeline for amplifying seed data through synthetic generation.

    Two-phase approach:
    1. Extract task policy from seed data and CoT traces
    2. Generate new synthetic examples constrained by policy

    Attributes:
        teacher_client: Teacher client for API calls
    """

    def __init__(self, teacher_client: TeacherClient):
        """Initialize amplification pipeline.

        Args:
            teacher_client: Initialized TeacherClient for API calls
        """
        self.teacher = teacher_client

    def amplify(
        self,
        seed_examples: List[Dict[str, str]],
        cot_traces: List[ThinkingTrace],
        instruction: str,
        augment_factor: int,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> List[ThinkingTrace]:
        """Amplify seed data into larger training dataset.

        Args:
            seed_examples: Original seed examples
            cot_traces: Generated Chain-of-Thought traces for seeds
            instruction: Task instruction / system prompt
            augment_factor: Target multiplication factor (e.g., 50x)
            response_model: Optional Pydantic model to enforce schema on synthetic outputs

        Returns:
            List of original + synthetic ThinkingTrace objects

        Raises:
            AmplificationError: If amplification fails
        """
        # Start with original traces
        amplified_traces = cot_traces.copy()

        # Phase 1: Extract policy from seeds
        policy = self._extract_policy(seed_examples, cot_traces, instruction)

        # Phase 2: Generate synthetic examples
        num_synthetic = len(seed_examples) * (augment_factor - 1)
        synthetic_examples = self._generate_synthetic_examples(
            policy, num_synthetic, instruction, len(seed_examples), response_model
        )

        # Convert synthetic examples to ThinkingTrace (generate CoT for each)
        # For synthetic examples, we use the teacher to generate reasoning
        synthetic_traces = self._synthesize_cot_for_synthetic(synthetic_examples, instruction)

        # Combine original and synthetic
        amplified_traces.extend(synthetic_traces)

        return amplified_traces

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
