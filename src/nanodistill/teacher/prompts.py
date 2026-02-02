"""Prompt templates for teacher model interactions."""

from typing import Dict, List

from .schemas import TaskPolicy, ThinkingTrace

# System prompt for generating Chain-of-Thought traces
COT_SYSTEM_PROMPT = """You are a reasoning-focused AI assistant. Your task is to demonstrate clear thinking process.

When answering:
1. Show your step-by-step thinking (this will be extracted as the 'thinking' field)
2. Break down complex problems into smaller parts
3. Consider multiple perspectives or approaches
4. Explain your reasoning clearly
5. Provide a clear final answer (this will be extracted as the 'output' field)

Provide your response in the following format:
- Start with your thinking process
- Then provide the final answer

Your thinking should be detailed and thorough."""


def build_cot_prompt(seed_example: Dict[str, str], instruction: str) -> str:
    """Build a prompt to generate Chain-of-Thought reasoning for a seed example.

    Args:
        seed_example: Dictionary with 'input' and optionally 'output' keys
        instruction: System instruction describing the task

    Returns:
        Formatted prompt string for the teacher model
    """
    example_input = seed_example.get("input", "").strip()
    example_output = seed_example.get("output", "").strip()

    prompt = f"""Task: {instruction}

Input: {example_input}

Generate your reasoning and answer following the format shown above."""

    # If we have a reference output, include it as a hint (but teacher should provide their own reasoning)
    if example_output:
        prompt += f"\n\nExpected output (for reference): {example_output}"

    return prompt


# System prompt for policy extraction
POLICY_EXTRACTION_SYSTEM_PROMPT = """You are an expert at analyzing task patterns and extracting implicit rules.

Your job is to analyze seed examples and their reasoning traces to understand:
1. What the task is fundamentally about
2. The structure of inputs and outputs
3. The reasoning approach needed
4. Key constraints or rules
5. Difficulty level and patterns

Be precise and specific - this policy will be used to generate new examples."""


def build_policy_extraction_prompt(
    seed_examples: List[Dict[str, str]],
    cot_traces: List[ThinkingTrace],
    instruction: str,
) -> str:
    """Build a prompt to extract task policy from seed data and CoT traces.

    Args:
        seed_examples: Original seed examples
        cot_traces: Generated Chain-of-Thought traces
        instruction: System instruction for the task

    Returns:
        Formatted prompt for policy extraction
    """
    examples_text = ""
    for i, (seed, trace) in enumerate(zip(seed_examples, cot_traces), 1):
        examples_text += f"\nExample {i}:\n"
        examples_text += f"  Input: {seed['input']}\n"
        examples_text += f"  Reasoning: {trace.thinking}\n"
        examples_text += f"  Output: {trace.output}\n"

    prompt = f"""Task Description: {instruction}

Seed Examples with Reasoning:
{examples_text}

Please analyze these examples and extract the underlying task policy. Consider:

1. **Task Description**: What is this task fundamentally about?
2. **Input Format**: What structure do inputs follow? What kind of information do they contain?
3. **Output Format**: What structure do outputs follow? What should be included?
4. **Reasoning Style**: What kind of reasoning is needed? (analytical, creative, systematic, etc.)
5. **Key Constraints**: What rules or constraints must be followed?
6. **Difficulty Level**: What is the difficulty level? (beginner, intermediate, advanced)
7. **Reasoning Patterns**: What reasoning patterns are used? (break into parts, check alternatives, verify, etc.)
8. **Examples Summary**: What do these examples demonstrate?
9. **Input Length**: Are inputs typically short, medium, or long?
10. **Output Length**: Are outputs typically short, medium, or long?

Respond in a structured format that can be used to generate new examples following the same pattern."""

    return prompt


# System prompt for synthetic example generation
SYNTHETIC_GENERATION_SYSTEM_PROMPT = """You are an expert at generating diverse examples that follow a specific task pattern.

Your job is to create new examples that:
1. Follow the exact pattern described in the policy
2. Are diverse and cover different aspects of the task
3. Are realistic and high-quality
4. Maintain consistent difficulty level
5. Respect all constraints mentioned in the policy

The generated examples will be used to fine-tune a language model, so quality is critical."""


def build_synthetic_generation_prompt(
    policy: TaskPolicy,
    num_examples: int,
    instruction: str,
    seed_count: int,
) -> str:
    """Build a prompt to generate synthetic examples matching the task policy.

    Args:
        policy: Extracted task policy
        num_examples: Number of examples to generate
        instruction: Original task instruction
        seed_count: Number of original seed examples

    Returns:
        Formatted prompt for synthetic generation
    """
    prompt = f"""You are generating synthetic examples to expand a training dataset.

Original Task: {instruction}

Task Policy (extracted from {seed_count} seed examples):
- Description: {policy.task_description}
- Input Format: {policy.input_format}
- Output Format: {policy.output_format}
- Reasoning Style: {policy.reasoning_style}
- Difficulty: {policy.difficulty_level}
- Key Constraints: {', '.join(policy.key_constraints) if policy.key_constraints else 'None specified'}
- Reasoning Patterns: {', '.join(policy.reasoning_patterns) if policy.reasoning_patterns else 'None specified'}
- Input Length: {policy.input_length_range}
- Output Length: {policy.output_length_range}

Task Summary: {policy.examples_summary}

Please generate {num_examples} NEW examples (not variations of the seeds) that:
1. Follow the exact pattern and constraints described above
2. Are diverse (don't just paraphrase)
3. Vary in specific details while maintaining the core pattern
4. Are realistic and of high quality
5. Maintain consistent difficulty level

For each example, provide:
- Input: [the question/task]
- Output: [the answer/response]

Generate the examples now:"""

    return prompt
