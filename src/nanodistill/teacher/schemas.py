"""Pydantic schemas for teacher module outputs."""

from typing import List

from pydantic import BaseModel, Field


class ThinkingTrace(BaseModel):
    """Chain-of-Thought trace from teacher model.

    Represents a single example with the teacher's reasoning process.
    """

    input: str = Field(..., description="User input / question")

    thinking: str = Field(
        ..., description="Step-by-step reasoning inside <thinking> tags"
    )

    output: str = Field(..., description="Final answer inside <answer> tags")

    confidence: float = Field(
        default=0.9,
        description="Model confidence in the answer (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )


class TeacherResponse(BaseModel):
    """Response from teacher model API call.

    Contains the generated traces and metadata about the API call.
    """

    traces: List[ThinkingTrace] = Field(
        ..., description="Generated Chain-of-Thought traces"
    )

    model_used: str = Field(..., description="Teacher model name (e.g., claude-sonnet-4-5)")

    total_tokens: int = Field(
        default=0, description="Total tokens used in generation"
    )

    input_tokens: int = Field(
        default=0, description="Tokens used in input prompts"
    )

    output_tokens: int = Field(
        default=0, description="Tokens used in generated responses"
    )


class TaskPolicy(BaseModel):
    """Extracted policy from seed data and CoT traces.

    Used to guide synthetic example generation to match the task pattern.
    """

    task_description: str = Field(
        ..., description="What the task is about (extracted from instruction + examples)"
    )

    input_format: str = Field(
        ..., description="Description of expected input format and structure"
    )

    output_format: str = Field(
        ..., description="Description of expected output format and structure"
    )

    reasoning_style: str = Field(
        ..., description="How the model should reason (analytical, creative, etc.)"
    )

    key_constraints: List[str] = Field(
        default_factory=list,
        description="Important constraints or rules to follow",
    )

    difficulty_level: str = Field(
        ..., description="Difficulty level of examples (beginner, intermediate, advanced)"
    )

    reasoning_patterns: List[str] = Field(
        default_factory=list,
        description="Common reasoning patterns observed in seed data",
    )

    examples_summary: str = Field(
        ..., description="Brief summary of what the seed examples demonstrate"
    )

    input_length_range: str = Field(
        default="short to medium",
        description="Typical input length (short, medium, long)",
    )

    output_length_range: str = Field(
        default="short to medium",
        description="Typical output length (short, medium, long)",
    )
