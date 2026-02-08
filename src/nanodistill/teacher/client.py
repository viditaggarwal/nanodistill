"""Teacher model client using LiteLLM for API abstraction.

Supports both synchronous and asynchronous (concurrent) API calls for performance.
"""

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Type, cast

import instructor
from litellm import acompletion, completion
from pydantic import BaseModel, Field, create_model

from ..utils.errors import TeacherAPIError, validate_teacher_api_key
from ..utils.schema import filter_extra_fields
from .prompts import (
    COT_SYSTEM_PROMPT,
    POLICY_EXTRACTION_SYSTEM_PROMPT,
    SYNTHETIC_GENERATION_SYSTEM_PROMPT,
    build_cot_prompt,
    build_policy_extraction_prompt,
    build_synthetic_generation_prompt,
)
from .schemas import TaskPolicy, ThinkingTrace

if TYPE_CHECKING:
    from ..config import DistillationConfig

# Default concurrency limit for async API calls
DEFAULT_MAX_CONCURRENT = 5


def _clean_json_string(text: str) -> str:
    """Remove wrapping tags, code blocks, and whitespace from JSON strings.

    Handles:
    - <answer>...</answer> tags
    - ```json ... ``` code blocks
    - Extra whitespace and newlines

    Args:
        text: Potentially wrapped JSON string

    Returns:
        Clean JSON string ready for parsing
    """
    text = text.strip()

    # Remove <answer> tags
    if text.startswith("<answer>"):
        text = text[8:]  # Remove opening tag
    if text.endswith("</answer>"):
        text = text[:-9]  # Remove closing tag

    # Remove markdown code blocks (```json ... ``` or just ``` ... ```)
    if text.startswith("```"):
        # Remove opening backticks and optional 'json' language identifier
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Remove closing backticks
        text = re.sub(r"\n?```$", "", text)

    return text.strip()


def _create_synthetic_example_wrapper(output_model: Type[BaseModel]) -> Type[BaseModel]:
    """Dynamically create a wrapper model for synthetic example generation.

    Args:
        output_model: The Pydantic model for the output structure

    Returns:
        A new Pydantic model with 'input' (str) and 'output' (output_model) fields
    """
    wrapper = create_model(
        f"SyntheticExample_{output_model.__name__}",
        input=(str, Field(description="The input text or question to analyze")),
        output=(output_model, Field(description="The structured output response")),
        __base__=BaseModel,
    )
    return cast(Type[BaseModel], wrapper)


class TeacherClient:
    """Client for interacting with teacher models via LiteLLM.

    Supports any LiteLLM-compatible model (Claude, GPT, Gemini, Ollama, etc.)
    by simply changing the model name.

    Attributes:
        model: Teacher model name (e.g., "claude-sonnet-4-5")
        api_base: Optional custom API base URL
        config: DistillationConfig for accessing parameters like temperature
    """

    def __init__(
        self,
        model: str,
        config: Optional["DistillationConfig"] = None,
        api_base: Optional[str] = None,
        max_retries: int = 3,
    ):
        """Initialize teacher client.

        Args:
            model: Teacher model name (LiteLLM-compatible)
            config: Optional DistillationConfig for parameter access
            api_base: Optional custom API base URL
            max_retries: Number of retries on API failures

        Raises:
            TeacherAPIError: If API key validation fails
        """
        # Validate API key is set
        validate_teacher_api_key(model)

        self.model = model
        self.config = config
        self.api_base = api_base
        self.max_retries = max_retries

        # Initialize instructor for structured output (sync)
        self.client = instructor.from_litellm(completion)

        # Initialize async instructor client for concurrent calls
        self.async_client = instructor.from_litellm(acompletion)

        # Concurrency limit for async API calls
        self.max_concurrent = DEFAULT_MAX_CONCURRENT

    def synthesize_cot(
        self,
        seed_examples: List[Dict[str, str]],
        instruction: str,
    ) -> List[ThinkingTrace]:
        """Generate Chain-of-Thought traces for seed examples using concurrent API calls.

        Uses async concurrency to process multiple examples simultaneously,
        significantly reducing wall-clock time for large seed sets.

        Args:
            seed_examples: List of examples with 'input' and 'output' fields
            instruction: Task instruction / system prompt

        Returns:
            List of generated ThinkingTrace objects (in original order)

        Raises:
            TeacherAPIError: If API call fails
        """
        # Use async concurrency for multiple examples
        if len(seed_examples) > 1:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # We're inside an existing event loop; fall back to sequential
                return self._synthesize_cot_sequential(seed_examples, instruction)

            return asyncio.run(
                self._synthesize_cot_async(seed_examples, instruction)
            )

        # Single example: just run synchronously
        return self._synthesize_cot_sequential(seed_examples, instruction)

    async def _synthesize_cot_async(
        self,
        seed_examples: List[Dict[str, str]],
        instruction: str,
    ) -> List[ThinkingTrace]:
        """Async implementation of CoT synthesis with concurrency control.

        Args:
            seed_examples: List of examples with 'input' and 'output' fields
            instruction: Task instruction / system prompt

        Returns:
            List of generated ThinkingTrace objects (in original order)
        """
        logger = logging.getLogger(__name__)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _generate_one(i: int, example: Dict[str, str]) -> ThinkingTrace:
            async with semaphore:
                try:
                    prompt = build_cot_prompt(example, instruction)

                    result = await self.async_client.chat.completions.create(
                        model=self.model,
                        response_model=ThinkingTrace,
                        messages=[
                            {"role": "system", "content": COT_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        max_retries=self.max_retries,
                        api_base=self.api_base,
                    )

                    trace = cast(ThinkingTrace, result)
                    trace.input = example["input"]
                    trace.output = _clean_json_string(trace.output)

                    logger.debug(f"Generated CoT trace {i}/{len(seed_examples)}")
                    return trace

                except Exception as e:
                    raise TeacherAPIError(
                        f"Failed to generate CoT trace for example {i}: {str(e)}"
                    ) from e

        tasks = [
            _generate_one(i, example) for i, example in enumerate(seed_examples, 1)
        ]
        traces = await asyncio.gather(*tasks)
        return list(traces)

    def _synthesize_cot_sequential(
        self,
        seed_examples: List[Dict[str, str]],
        instruction: str,
    ) -> List[ThinkingTrace]:
        """Sequential fallback for CoT synthesis (original implementation).

        Used when running inside an existing event loop or for single examples.

        Args:
            seed_examples: List of examples with 'input' and 'output' fields
            instruction: Task instruction / system prompt

        Returns:
            List of generated ThinkingTrace objects
        """
        traces = []
        logger = logging.getLogger(__name__)

        for i, example in enumerate(seed_examples, 1):
            try:
                prompt = build_cot_prompt(example, instruction)

                trace = self.client.chat.completions.create(
                    model=self.model,
                    response_model=ThinkingTrace,
                    messages=[
                        {"role": "system", "content": COT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_retries=self.max_retries,
                    api_base=self.api_base,
                )

                trace.input = example["input"]
                trace.output = _clean_json_string(trace.output)

                traces.append(trace)
                logger.debug(f"Generated CoT trace {i}/{len(seed_examples)}")

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
            prompt = build_policy_extraction_prompt(seed_examples, cot_traces, instruction)

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

            return cast(TaskPolicy, response)

        except Exception as e:
            raise TeacherAPIError(f"Failed to extract task policy: {str(e)}") from e

    def generate_synthetic_examples(
        self,
        policy: TaskPolicy,
        num_examples: int,
        instruction: str,
        seed_count: int,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> List[Dict[str, str]]:
        """Generate synthetic examples matching the task policy.

        When response_model is provided, uses instructor for structured output
        with automatic field filtering. Otherwise uses text parsing.

        Args:
            policy: Task policy to constrain generation
            num_examples: Number of examples to generate
            instruction: Original task instruction
            seed_count: Number of original seed examples
            response_model: Optional Pydantic model to enforce schema on output

        Returns:
            List of generated examples with 'input' and 'output' fields.
            When response_model provided, output fields are JSON strings matching schema.

        Raises:
            TeacherAPIError: If API call fails or parsing fails
        """
        try:
            # Use schema-based generation if response_model provided
            if response_model is not None:
                return self._generate_with_schema(
                    policy, num_examples, instruction, seed_count, response_model
                )

            # Fallback to text parsing for backward compatibility
            return self._generate_with_text_parsing(policy, num_examples, instruction, seed_count)

        except TeacherAPIError:
            raise
        except Exception as e:
            raise TeacherAPIError(f"Failed to generate synthetic examples: {str(e)}") from e

    def _generate_with_text_parsing(
        self,
        policy: TaskPolicy,
        num_examples: int,
        instruction: str,
        seed_count: int,
    ) -> List[Dict[str, str]]:
        """Generate using raw text output and manual parsing (backward compatible).

        Args:
            policy: Task policy to constrain generation
            num_examples: Number of examples to generate
            instruction: Original task instruction
            seed_count: Number of original seed examples

        Returns:
            List of generated examples with 'input' and 'output' fields
        """
        prompt = build_synthetic_generation_prompt(
            policy, num_examples, instruction, seed_count
        )

        # Get LiteLLM kwargs from config (temperature + user-provided params)
        litellm_kwargs = {}
        if self.config:
            litellm_kwargs = self.config.get_litellm_synthesis_kwargs()

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
            max_retries=self.max_retries,
            api_base=self.api_base,
            **litellm_kwargs,  # Pass temperature and any other LiteLLM params
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

    def _generate_with_schema(
        self,
        policy: TaskPolicy,
        num_examples: int,
        instruction: str,
        seed_count: int,
        response_model: Type[BaseModel],
    ) -> List[Dict[str, str]]:
        """Generate synthetic examples using instructor with automatic wrapper model.

        Creates a dynamic wrapper model with 'input' and 'output' fields, where
        'output' is constrained to the provided response_model. This ensures
        full Pydantic validation on both input and structured output.

        Args:
            policy: Task policy to constrain generation
            num_examples: Number of examples to generate
            instruction: Original task instruction
            seed_count: Number of original seed examples
            response_model: Pydantic model to enforce on outputs

        Returns:
            List of generated examples with validated outputs
        """
        logger = logging.getLogger(__name__)
        examples: List[Dict[str, str]] = []

        prompt = build_synthetic_generation_prompt(
            policy, num_examples, instruction, seed_count, response_model=response_model
        )

        # Get LiteLLM kwargs from config (temperature + user-provided params)
        litellm_kwargs = {}
        if self.config:
            litellm_kwargs = self.config.get_litellm_synthesis_kwargs()

        # Create wrapper model: SyntheticExample with input (str) + output (response_model)
        wrapper_model = _create_synthetic_example_wrapper(response_model)

        try:
            # Use instructor with List[WrapperModel] to generate multiple examples
            response = self.client.chat.completions.create(
                model=self.model,
                response_model=List[wrapper_model],  # type: ignore[valid-type]
                messages=[
                    {
                        "role": "system",
                        "content": SYNTHETIC_GENERATION_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                max_retries=self.max_retries,
                api_base=self.api_base,
                **litellm_kwargs,
            )

            # Extract input/output pairs from validated wrapper instances
            for instance in response:
                if len(examples) >= num_examples:
                    break

                try:
                    # instance.input is str, instance.output is response_model
                    # If output is a string (fallback), clean and parse it
                    if isinstance(instance.output, str):
                        # Output field is a string - shouldn't happen but handle it
                        cleaned = _clean_json_string(instance.output)
                        output_dict = json.loads(cleaned)
                    else:
                        # Output field is properly parsed Pydantic model
                        output_dict = instance.output.model_dump(mode="json")

                    # Filter to only schema fields
                    filtered_dict = filter_extra_fields(output_dict, response_model, logger)

                    example = {
                        "input": instance.input,
                        "output": json.dumps(filtered_dict),
                    }
                    examples.append(example)

                except Exception as e:
                    logger.warning(f"Failed to process instance: {str(e)}")
                    continue

            # Log generation summary
            logger.info(f"Generated {len(examples)}/{num_examples} valid schema-compliant examples")

            # Verify we got enough valid examples
            if len(examples) < num_examples:
                raise TeacherAPIError(
                    f"Generated only {len(examples)} valid schema-compliant examples, "
                    f"expected {num_examples}. Try adjusting augment_factor or instruction."
                )

            return examples[:num_examples]

        except TeacherAPIError:
            raise
        except Exception as e:
            logger.warning(
                f"Schema-based generation failed, falling back to text parsing: {str(e)}"
            )
            # Fallback to text parsing if instructor fails
            return self._generate_with_text_parsing(policy, num_examples, instruction, seed_count)

    def _parse_synthetic_examples(self, response_text: str) -> List[Dict[str, str]]:
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
                    input_section = block[input_start + len("**Input:**") : output_start].strip()

                    # Remove quotes if present
                    if input_section.startswith('"') and '"\n' in input_section:
                        input_section = input_section[1:]
                        input_section = input_section[: input_section.find('"\n')].strip()
                    elif input_section.startswith('"') and '"' in input_section[1:]:
                        input_section = input_section[1 : input_section.rfind('"')]

                    input_text = input_section.strip()

                    # Extract output (between Output marker and next Example or backticks)
                    output_section = block[output_start + len("**Output:**") :].strip()

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
            blocks = re.split(
                r"(?:^|\n)(?:-\s*)?(?:Input|Question|Prompt):", response_text, flags=re.IGNORECASE
            )
            for block in blocks[1:]:
                lines = block.strip().split("\n")
                if not lines:
                    continue

                input_text = lines[0].strip()

                # Find output - handle multi-line outputs
                output_text = ""
                for idx, line in enumerate(lines[1:], start=1):
                    if re.match(r"^\s*(?:-\s*)?(?:Output|Answer|Response):", line, re.IGNORECASE):
                        # Extract text after the Output: marker on the same line
                        same_line_text = re.sub(
                            r"^\s*(?:-\s*)?(?:Output|Answer|Response):\s*",
                            "",
                            line,
                            flags=re.IGNORECASE,
                        ).strip()

                        if same_line_text:
                            # Output is on the same line
                            output_text = same_line_text
                        else:
                            # Output starts on next line - collect until next example
                            output_lines = []
                            for subsequent_line in lines[idx + 1 :]:
                                # Stop at next example marker
                                if re.match(
                                    r"^\s*(?:\*\*)?Example\s*\d+|^---+$|^\s*(?:-\s*)?(?:Input|Question|Prompt):",
                                    subsequent_line,
                                    re.IGNORECASE,
                                ):
                                    break
                                output_lines.append(subsequent_line)
                            output_text = "\n".join(output_lines).strip()
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
                            input_key = next(
                                (
                                    k
                                    for k in item.keys()
                                    if "input" in k.lower() or "question" in k.lower()
                                ),
                                None,
                            )
                            output_key = next(
                                (
                                    k
                                    for k in item.keys()
                                    if "output" in k.lower() or "answer" in k.lower()
                                ),
                                None,
                            )
                            if input_key and output_key:
                                examples.append(
                                    {"input": str(item[input_key]), "output": str(item[output_key])}
                                )
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
