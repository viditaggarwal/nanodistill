"""Tests for teacher module schemas."""

import pytest

from nanodistill.teacher.schemas import TaskPolicy, TeacherResponse, ThinkingTrace


def test_thinking_trace_creation():
    """Test creating a ThinkingTrace."""
    trace = ThinkingTrace(
        input="What is 2+2?",
        thinking="I need to add 2 and 2. That gives me 4.",
        output="4",
    )

    assert trace.input == "What is 2+2?"
    assert trace.thinking == "I need to add 2 and 2. That gives me 4."
    assert trace.output == "4"
    assert trace.confidence == 0.9  # Default


def test_thinking_trace_custom_confidence():
    """Test ThinkingTrace with custom confidence."""
    trace = ThinkingTrace(
        input="Q",
        thinking="reasoning",
        output="A",
        confidence=0.8,
    )

    assert trace.confidence == 0.8


def test_thinking_trace_invalid_confidence():
    """Test that confidence outside [0,1] raises error."""
    with pytest.raises(ValueError):
        ThinkingTrace(
            input="Q",
            thinking="reasoning",
            output="A",
            confidence=1.5,  # Invalid
        )


def test_teacher_response_creation():
    """Test creating a TeacherResponse."""
    traces = [
        ThinkingTrace(input="Q1", thinking="T1", output="A1"),
        ThinkingTrace(input="Q2", thinking="T2", output="A2"),
    ]

    response = TeacherResponse(
        traces=traces,
        model_used="claude-sonnet-4-5",
        total_tokens=1500,
    )

    assert len(response.traces) == 2
    assert response.model_used == "claude-sonnet-4-5"
    assert response.total_tokens == 1500


def test_task_policy_creation():
    """Test creating a TaskPolicy."""
    policy = TaskPolicy(
        task_description="Simple math problem solving",
        input_format="Math arithmetic questions",
        output_format="Numeric answers",
        reasoning_style="step-by-step calculation",
        difficulty_level="beginner",
        examples_summary="Basic addition and subtraction problems",
    )

    assert policy.task_description == "Simple math problem solving"
    assert policy.difficulty_level == "beginner"


def test_task_policy_with_constraints():
    """Test TaskPolicy with constraints and patterns."""
    policy = TaskPolicy(
        task_description="Math tutoring",
        input_format="Math questions",
        output_format="Numeric answers",
        reasoning_style="analytical",
        difficulty_level="intermediate",
        examples_summary="Various math operations",
        key_constraints=["Only positive numbers", "Show all steps"],
        reasoning_patterns=["Break into smaller parts", "Verify result"],
    )

    assert len(policy.key_constraints) == 2
    assert len(policy.reasoning_patterns) == 2
