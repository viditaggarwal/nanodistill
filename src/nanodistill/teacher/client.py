"""Teacher model client using LiteLLM for API abstraction."""

import json
import re
from typing import Dict, List, Optional

import instructor
from litellm import completion

from ..utils.errors import TeacherAPIError, validate_teacher_api_key
from .prompts import (
    COT_SYSTEM_PROMPT,
    POLICY_EXTRACTION_SYSTEM_PROMPT,
    SYNTHETIC_GENERATION_SYSTEM_PROMPT,
    build_cot_prompt,
    build_policy_extraction_prompt,
    build_synthetic_generation_prompt,
)
from .schemas import TaskPolicy, TeacherResponse, ThinkingTrace


class TeacherClient:
    """Client for interacting with teacher models via LiteLLM.

    Supports any LiteLLM-compatible model (Claude, GPT, Gemini, Ollama, etc.)
    by simply changing the model name.

    Attributes:
        model: Teacher model name (e.g., "claude-sonnet-4-5")
        api_base: Optional custom API base URL
    """

    def __init__(
        self,
        model: str,
        api_base: Optional[str] = None,
        max_retries: int = 3,
    ):
        """Initialize teacher client.

        Args:
            model: Teacher model name (LiteLLM-compatible)
            api_base: Optional custom API base URL
            max_retries: Number of retries on API failures

        Raises:
            TeacherAPIError: If API key validation fails
        """
        # Validate API key is set
        validate_teacher_api_key(model)

        self.model = model
        self.api_base = api_base
        self.max_retries = max_retries

        # Initialize instructor for structured output
        self.client = instructor.from_litellm(completion)

    def synthesize_cot(
        self,
        seed_examples: List[Dict[str, str]],
        instruction: str,
    ) -> List[ThinkingTrace]:
        """Generate Chain-of-Thought traces for seed examples.

        Args:
            seed_examples: List of examples with 'input' and 'output' fields
            instruction: Task instruction / system prompt

        Returns:
            List of generated ThinkingTrace objects

        Raises:
            TeacherAPIError: If API call fails
        """
        traces = []
        total_tokens = 0

        for i, example in enumerate(seed_examples, 1):
            try:
                prompt = build_cot_prompt(example, instruction)

                # Use raw completion and parse manually
                response = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": COT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_retries=self.max_retries,
                    api_base=self.api_base,
                )

                # Extract response text
                response_text = response.choices[0].message.content

                # Parse thinking and output from response
                thinking = ""
                output = ""

                # Try to extract thinking (anything before the answer)
                if "thinking" in response_text.lower() or "work through" in response_text.lower():
                    # Split by common answer indicators
                    parts = re.split(r"\n(?:answer|final answer|output|json):", response_text, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        thinking = parts[0].strip()
                        output = parts[1].strip()
                    else:
                        thinking = response_text.strip()
                        output = response_text.strip()
                else:
                    # If no clear thinking section, treat whole response as output
                    thinking = response_text.strip()
                    output = response_text.strip()

                # Extract token usage if available
                if hasattr(response, "usage"):
                    total_tokens += response.usage.total_tokens

                trace = ThinkingTrace(
                    input=example["input"],
                    thinking=thinking,
                    output=output,
                    confidence=0.9
                )
                traces.append(trace)

            except Exception as e:
                raise TeacherAPIError(
                    f"Failed to generate CoT trace for example {i}: {str(e)}"
                ) from e

        return traces

    def extract_policy(
        self,
        seed_examples: List[Dict[str, str]],
        cot_traces: List[ThinkingTrace],
        instruction: str,
    ) -> TaskPolicy:
        """Extract task policy from seed examples and CoT traces.

        Args:
            seed_examples: Original seed examples
            cot_traces: Generated Chain-of-Thought traces
            instruction: Task instruction

        Returns:
            TaskPolicy object describing the task pattern

        Raises:
            TeacherAPIError: If API call fails
        """
        try:
            prompt = build_policy_extraction_prompt(
                seed_examples, cot_traces, instruction
            )

            response = self.client.chat.completions.create(
                model=self.model,
                response_model=TaskPolicy,
                messages=[
                    {
                        "role": "system",
                        "content": POLICY_EXTRACTION_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                max_retries=self.max_retries,
                api_base=self.api_base,
            )

            return response

        except Exception as e:
            raise TeacherAPIError(
                f"Failed to extract task policy: {str(e)}"
            ) from e

    def generate_synthetic_examples(
        self,
        policy: TaskPolicy,
        num_examples: int,
        instruction: str,
        seed_count: int,
    ) -> List[Dict[str, str]]:
        """Generate synthetic examples matching the task policy.

        Args:
            policy: Task policy to constrain generation
            num_examples: Number of examples to generate
            instruction: Original task instruction
            seed_count: Number of original seed examples

        Returns:
            List of generated examples with 'input' and 'output' fields

        Raises:
            TeacherAPIError: If API call fails or parsing fails
        """
        try:
            prompt = build_synthetic_generation_prompt(
                policy, num_examples, instruction, seed_count
            )

            # For synthetic generation, we use raw text output and parse it
            # (More flexible than structured output for list generation)
            response = completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYNTHETIC_GENERATION_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,  # Some diversity
                max_retries=self.max_retries,
                api_base=self.api_base,
            )

            # Parse response text to extract examples
            response_text = response.choices[0].message.content

            examples = self._parse_synthetic_examples(response_text)

            # Verify we got enough examples
            if len(examples) < num_examples:
                raise TeacherAPIError(
                    f"Generated only {len(examples)} examples, expected {num_examples}"
                )

            return examples[:num_examples]

        except TeacherAPIError:
            raise
        except Exception as e:
            raise TeacherAPIError(
                f"Failed to generate synthetic examples: {str(e)}"
            ) from e

    def _parse_synthetic_examples(
        self, response_text: str
    ) -> List[Dict[str, str]]:
        """Parse synthetic examples from teacher response text.

        Handles multiple formats:
        - Input: ... / Output: ...
        - Example: ... / Answer: ...
        - **Input**: ... / **Output**: ...
        - Numbered examples

        Args:
            response_text: Raw response text from teacher model

        Returns:
            List of parsed examples

        Raises:
            TeacherAPIError: If parsing fails
        """
        examples = []

        # Try multiple parsing strategies

        # Strategy 1: Simple split by marked examples and extraction
        if re.search(r"#{1,2}\s+Example\s+\d+", response_text, re.IGNORECASE):
            # Split by example headers
            blocks = re.split(r"#{1,2}\s+Example\s+\d+", response_text, flags=re.IGNORECASE)

            for block in blocks[1:]:  # Skip first element (before first Example)
                try:
                    # Find **Input:** marker
                    input_start = block.find("**Input:**")
                    if input_start == -1:
                        continue

                    # Find **Output:** marker
                    output_start = block.find("**Output:**", input_start)
                    if output_start == -1:
                        continue

                    # Extract everything between Input and Output
                    input_section = block[input_start + len("**Input:**"):output_start].strip()

                    # Remove quotes if present
                    if input_section.startswith('"') and '"\n' in input_section:
                        input_section = input_section[1:]
                        input_section = input_section[:input_section.find('"\n')].strip()
                    elif input_section.startswith('"') and '"' in input_section[1:]:
                        input_section = input_section[1:input_section.rfind('"')]

                    input_text = input_section.strip()

                    # Extract output (between Output marker and next Example or backticks)
                    output_section = block[output_start + len("**Output:**"):].strip()

                    # If in code block, extract JSON
                    if "```" in output_section:
                        json_start = output_section.find("```") + 3
                        if "json" in output_section[:json_start]:
                            json_start = output_section.find("\n", json_start) + 1
                        json_end = output_section.find("```", json_start)
                        output_text = output_section[json_start:json_end].strip()
                    else:
                        # Plain text - get until next newline or section
                        output_text = output_section.split("\n\n")[0].strip()

                    if input_text and output_text and len(input_text) > 5 and "{" in output_text:
                        examples.append({"input": input_text, "output": output_text})
                except Exception:
                    # Skip malformed blocks
                    continue

        # Strategy 2: Split by "Input:" / "Output:" pattern
        if not examples:
            blocks = re.split(r"(?:^|\n)(?:-\s*)?(?:Input|Question|Prompt):", response_text, flags=re.IGNORECASE)
            for block in blocks[1:]:
                lines = block.strip().split("\n")
                if not lines:
                    continue

                input_text = lines[0].strip()

                # Find output
                output_text = ""
                for line in lines[1:]:
                    if re.match(r"^\s*(?:-\s*)?(?:Output|Answer|Response):", line, re.IGNORECASE):
                        output_text = re.sub(r"^\s*(?:-\s*)?(?:Output|Answer|Response):\s*", "", line, flags=re.IGNORECASE).strip()
                        break

                if input_text and output_text and len(input_text) > 5:
                    examples.append({"input": input_text, "output": output_text})

        # Strategy 3: Look for JSON array of examples
        if not examples:
            json_match = re.search(r"\[\s*\{.*?\}\s*\]", response_text, re.DOTALL)
            if json_match:
                try:
                    json_data = json.loads(json_match.group())
                    for item in json_data:
                        if isinstance(item, dict):
                            input_key = next((k for k in item.keys() if "input" in k.lower() or "question" in k.lower()), None)
                            output_key = next((k for k in item.keys() if "output" in k.lower() or "answer" in k.lower()), None)
                            if input_key and output_key:
                                examples.append({"input": str(item[input_key]), "output": str(item[output_key])})
                except (json.JSONDecodeError, TypeError):
                    pass

        if not examples:
            # Log first 500 chars for debugging
            snippet = response_text[:500] if len(response_text) > 500 else response_text
            raise TeacherAPIError(
                f"Could not parse any examples from teacher response. "
                f"Response format didn't match expected patterns. "
                f"First 500 chars: {snippet}"
            )

        return examples
