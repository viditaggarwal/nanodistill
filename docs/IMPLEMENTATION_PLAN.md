# NanoDistill Implementation Plan

## Overview

**Goal**: Build an MVP that implements the core `distill()` function to transform 10 examples into a reasoning-capable small language model.

**Core Promise**: "Give us 10 examples and an API key. We give you a locally runnable, reasoning-capable model."

**User Priorities**:
- ✅ Speed to MVP - Get basic pipeline working fast
- ✅ Claude Sonnet as primary teacher model
- ✅ Optimized for Apple Silicon (Mac M1+)
- ✅ Full src-layout structure from start

---

## Project Structure

```
nanodistill/
├── pyproject.toml                    # Modern Python packaging
├── README.md                         # Quick start guide
├── LICENSE
├── .gitignore
├── src/
│   └── nanodistill/
│       ├── __init__.py              # Expose main distill() function
│       ├── core.py                  # Main orchestrator
│       ├── config.py                # Pydantic configuration
│       ├── teacher/
│       │   ├── __init__.py
│       │   ├── client.py           # LiteLLM wrapper
│       │   ├── prompts.py          # CoT prompt templates
│       │   └── schemas.py          # Pydantic output models
│       ├── amplifier/
│       │   ├── __init__.py
│       │   ├── strategies.py       # Amplification strategies
│       │   └── pipeline.py         # Amplification orchestration
│       ├── distiller/
│       │   ├── __init__.py
│       │   ├── trainer.py          # MLX-LM training wrapper (Phase 1)
│       │   ├── hyperparams.py      # Auto-configuration for Apple Silicon
│       │   └── export.py           # GGUF export utilities
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py           # Dataset loading
│       │   └── formatter.py        # Format conversions
│       └── utils/
│           ├── __init__.py
│           ├── logging.py          # Rich progress bars
│           └── errors.py           # Custom exceptions
├── tests/
│   ├── test_teacher.py
│   ├── test_amplifier.py
│   ├── test_distiller.py
│   └── fixtures/
│       └── sample_seeds.py
└── examples/
    └── basic_usage.py
```

---

## MVP Implementation Phases

### Phase 1: Foundation (Core Infrastructure)

**Goal**: Set up project structure and basic configuration

**Training Framework**: Using **MLX-LM** (Apple Silicon native optimization)

#### Tasks:
1. **Project initialization**
   - Create `pyproject.toml` with MLX-LM dependencies
   - Set up src-layout structure
   - Create `.gitignore`
   - Initialize basic README

2. **Core dependencies** (`pyproject.toml`):
   ```toml
   dependencies = [
       "litellm>=1.50.0",           # Teacher API (Claude support)
       "instructor>=1.3.0",          # Structured output
       "pydantic>=2.0.0",            # Data validation
       "mlx>=0.0.8",                 # MLX framework (Apple Silicon native)
       "mlx-lm>=0.0.2",              # MLX language models + training
       "datasets>=2.18.0",           # Data handling
       "typer>=0.9.0",               # CLI
       "rich>=13.0.0",               # Beautiful output
   ]
   ```

   **Note**: MLX-LM includes built-in support for model loading, LoRA, and training. No need for torch, transformers, or unsloth.

3. **Configuration module** (`src/nanodistill/config.py`):
   - Define `DistillationConfig` Pydantic model
   - Input validation (seed format, required fields)
   - Auto-detect Apple Silicon (MLX handles this)
   - No API key field (handled by environment vars via LiteLLM)

   ```python
   class DistillationConfig(BaseModel):
       name: str
       seed: List[Dict[str, str]]  # Min 1 example
       instruction: str
       teacher: str = "claude-sonnet-4-5"  # LiteLLM model name
       student: str = "mlx-community/Llama-3-8B-Instruct-4bit"  # MLX model
       augment_factor: int = 50
       output_dir: str = "./outputs"
   ```

4. **API Key validation** (`src/nanodistill/utils/errors.py`):
   - Teacher model validation (upfront, before pipeline starts)
   - Check for required env vars based on teacher model name
   - Provide helpful error messages with setup instructions

   ```python
   def validate_teacher_api_key(teacher_model: str) -> None:
       """Validate that required API key is set for teacher model"""
       if teacher_model.startswith("claude"):
           if not os.getenv("ANTHROPIC_API_KEY"):
               raise NanoDistillError(
                   "❌ ANTHROPIC_API_KEY not set\n\n"
                   "Please set your API key:\n"
                   "  export ANTHROPIC_API_KEY='sk-ant-...'"
               )
       elif teacher_model.startswith("gpt"):
           if not os.getenv("OPENAI_API_KEY"):
               raise NanoDistillError(
                   "❌ OPENAI_API_KEY not set\n\n"
                   "Please set your API key:\n"
                   "  export OPENAI_API_KEY='sk-...'"
               )
       elif teacher_model.startswith("gemini"):
           if not os.getenv("GEMINI_API_KEY"):
               raise NanoDistillError("❌ GEMINI_API_KEY not set")
       elif teacher_model.startswith("ollama"):
           # Local model, no auth needed
           pass
       else:
           # Unknown model type
           raise NanoDistillError(
               f"❌ Unknown teacher model: {teacher_model}\n"
               "Supported: claude-*, gpt-*, gemini-*, ollama/*"
           )
   ```

   - Call early in `distill()` before pipeline starts:
   ```python
   def distill(...):
       validate_teacher_api_key(teacher)  # Fail fast with clear message
       # ... rest of pipeline
   ```

5. **Error handling base** (`src/nanodistill/utils/errors.py`):
   ```python
   class NanoDistillError(Exception): pass
   class TeacherAPIError(NanoDistillError): pass
   class AmplificationError(NanoDistillError): pass
   class TrainingError(NanoDistillError): pass
   ```

**Deliverable**: Installable package with validated configuration and upfront API key validation

---

### Phase 2: Teacher Module (Data Synthesis)

**Goal**: Generate Chain-of-Thought traces using Claude Sonnet via LiteLLM

#### Tasks:

1. **Teacher schemas** (`src/nanodistill/teacher/schemas.py`):
   ```python
   from pydantic import BaseModel, Field

   class ThinkingTrace(BaseModel):
       input: str
       thinking: str = Field(description="Step-by-step reasoning")
       output: str = Field(description="Final answer")
       confidence: float = Field(ge=0.0, le=1.0, default=0.9)

   class TeacherResponse(BaseModel):
       traces: List[ThinkingTrace]
       model_used: str
       total_tokens: int
   ```

2. **CoT prompts** (`src/nanodistill/teacher/prompts.py`):
   ```python
   COT_SYSTEM_PROMPT = """You are a reasoning-focused AI assistant.
   For each input, you MUST:
   1. Show your step-by-step thinking inside <thinking> tags
   2. Provide a clear final answer inside <answer> tags

   Example:
   <thinking>
   Let me break this down:
   1. First consideration...
   2. Second point...
   3. Therefore...
   </thinking>
   <answer>
   [Your final answer]
   </answer>
   """

   def build_cot_prompt(seed_example: Dict[str, str], instruction: str) -> str:
       """Build prompt that forces CoT reasoning"""
       return f"""{instruction}

       Input: {seed_example['input']}

       Provide your reasoning and answer following the format above."""
   ```

3. **LiteLLM client** (`src/nanodistill/teacher/client.py`):
   ```python
   import instructor
   from litellm import completion

   class TeacherClient:
       def __init__(self, model: str):
           """Initialize teacher client.

           LiteLLM will automatically detect API keys from environment variables:
           - ANTHROPIC_API_KEY for claude-* models
           - OPENAI_API_KEY for gpt-* models
           - GEMINI_API_KEY for gemini-* models
           - etc.
           """
           self.model = model
           self.client = instructor.from_litellm(completion)

       def synthesize_cot(
           self,
           seed: List[Dict[str, str]],
           instruction: str
       ) -> List[ThinkingTrace]:
           """Generate CoT traces for seed examples"""
           traces = []
           for example in seed:
               prompt = build_cot_prompt(example, instruction)

               # Use Instructor for structured output
               # LiteLLM automatically uses env vars for auth
               response = self.client.chat.completions.create(
                   model=self.model,
                   response_model=ThinkingTrace,
                   messages=[
                       {"role": "system", "content": COT_SYSTEM_PROMPT},
                       {"role": "user", "content": prompt}
                   ],
                   max_retries=3,  # Auto-retry on failures
               )
               traces.append(response)
           return traces
   ```

4. **Testing strategy**:
   - Mock LiteLLM calls in tests
   - Validate prompt formatting
   - Test API key loading
   - Test retry logic on failures

**Deliverable**: Working teacher module that generates CoT traces from Claude

---

### Phase 3: Data Module (Loading & Formatting)

**Goal**: Handle various input formats and convert to training-ready datasets

#### Tasks:

1. **Data loader** (`src/nanodistill/data/loader.py`):
   ```python
   from datasets import Dataset

   def load_seed_data(seed: Union[List[Dict], str, Path]) -> List[Dict[str, str]]:
       """Load seed data from various formats"""
       if isinstance(seed, list):
           return seed
       elif isinstance(seed, (str, Path)):
           path = Path(seed)
           if path.suffix == '.json':
               return json.load(path.open())
           elif path.suffix == '.jsonl':
               return [json.loads(line) for line in path.open()]
           elif path.suffix == '.csv':
               return pd.read_csv(path).to_dict('records')
       raise ValueError(f"Unsupported seed format: {type(seed)}")

   def to_hf_dataset(traces: List[ThinkingTrace]) -> Dataset:
       """Convert traces to HuggingFace Dataset"""
       data = {
           "input": [t.input for t in traces],
           "thinking": [t.thinking for t in traces],
           "output": [t.output for t in traces],
       }
       return Dataset.from_dict(data)
   ```

2. **Format converter** (`src/nanodistill/data/formatter.py`):
   ```python
   def format_for_training(trace: ThinkingTrace, tokenizer) -> Dict:
       """Format trace into Llama-3 chat template"""
       conversation = [
           {"role": "user", "content": trace.input},
           {"role": "assistant", "content": f"<thinking>\n{trace.thinking}\n</thinking>\n\n{trace.output}"}
       ]

       # Apply chat template
       formatted = tokenizer.apply_chat_template(
           conversation,
           tokenize=True,
           add_generation_prompt=False
       )
       return {"input_ids": formatted, "labels": formatted}
   ```

**Deliverable**: Flexible data loading and HuggingFace Dataset conversion

---

### Phase 4: Policy Extraction & Amplifier

**Goal**: Extract task policy from seed data, then generate synthetic examples constrained by that policy

#### Phase 4A: Policy Extraction

**Purpose**: Analyze seed examples + CoT traces to understand the underlying task pattern

1. **Policy extraction** (`src/nanodistill/amplifier/policy.py`):
   ```python
   from pydantic import BaseModel

   class TaskPolicy(BaseModel):
       """Extracted product policy from seed data"""
       task_description: str  # "Solve basic arithmetic problems"
       input_format: str      # "Natural language math questions"
       output_format: str     # "Numeric answer"
       reasoning_style: str   # "Step-by-step calculation"
       key_constraints: List[str]  # ["Single operation or simple multi-step"]
       difficulty_level: str  # "elementary"
       reasoning_patterns: List[str]  # ["Identify operation", "Calculate", "Answer"]
       examples_summary: str  # Summary of what inputs/outputs look like

   def extract_policy(
       seed_examples: List[Dict],
       cot_traces: List[ThinkingTrace],
       instruction: str
   ) -> TaskPolicy:
       """Analyze seed data to extract underlying task policy"""

       # Prompt teacher to analyze the patterns
       analysis_prompt = f"""
       You are a task analyst. Here are {len(seed_examples)} examples with reasoning:

       Instruction: {instruction}

       Examples:
       {format_examples_for_analysis(seed_examples, cot_traces)}

       Based on these examples, extract the underlying PRODUCT POLICY:

       Respond with JSON containing:
       - task_description: One sentence describing what the task does
       - input_format: How inputs are structured
       - output_format: How outputs are structured
       - reasoning_style: What type of reasoning is shown
       - key_constraints: Important rules or limits
       - difficulty_level: Difficulty rating
       - reasoning_patterns: Common reasoning steps observed
       - examples_summary: Brief summary of example types
       """

       policy_json = teacher.get_json_response(analysis_prompt)
       return TaskPolicy(**policy_json)
   ```

#### Phase 4B: Synthetic Generation

**Purpose**: Generate new diverse examples following the extracted policy

2. **Policy-based generation** (`src/nanodistill/amplifier/generator.py`):
   ```python
   class SyntheticGenerator:
       def __init__(self, teacher_client: TeacherClient):
           self.teacher = teacher_client

       def generate_synthetic_examples(
           self,
           policy: TaskPolicy,
           num_examples: int,
           instruction: str,
           seed_count: int
       ) -> List[ThinkingTrace]:
           """Generate new examples constrained by policy"""

           generation_prompt = f"""
           You are a synthetic data generator. Using this product policy,
           generate {num_examples} NEW, DIVERSE examples.

           PRODUCT POLICY:
           - Task: {policy.task_description}
           - Input format: {policy.input_format}
           - Output format: {policy.output_format}
           - Reasoning style: {policy.reasoning_style}
           - Constraints: {', '.join(policy.key_constraints)}
           - Difficulty: {policy.difficulty_level}

           INSTRUCTION: {instruction}

           For EACH example, generate:
           1. A NEW input (different from seed data, but following the input_format)
           2. The correct output (following output_format)
           3. Step-by-step reasoning (following reasoning_style and patterns)

           Requirements:
           - Ensure DIVERSITY in the inputs (vary numbers, operators, domains, etc.)
           - Keep outputs CORRECT and CONSISTENT with task
           - Follow the reasoning_patterns: {', '.join(policy.reasoning_patterns)}
           - Maintain the difficulty level: {policy.difficulty_level}
           - DO NOT duplicate seed examples

           Output format: JSON array of objects with fields:
           {{"input": "...", "output": "...", "thinking": "..."}}
           """

           generated_data = self.teacher.get_json_response(generation_prompt)

           # Convert to ThinkingTrace objects
           traces = [
               ThinkingTrace(
                   input=item["input"],
                   thinking=item["thinking"],
                   output=item["output"],
                   confidence=0.95  # Teacher-generated has high confidence
               )
               for item in generated_data
           ]

           return traces
   ```

3. **Amplification pipeline** (`src/nanodistill/amplifier/pipeline.py`):
   ```python
   class AmplificationPipeline:
       def __init__(self, teacher_client: TeacherClient):
           self.policy_extractor = PolicyExtractor(teacher_client)
           self.generator = SyntheticGenerator(teacher_client)

       def amplify(
           self,
           seed_traces: List[ThinkingTrace],
           seed_examples: List[Dict],
           instruction: str,
           target_size: int
       ) -> List[ThinkingTrace]:
           """
           Two-phase amplification:
           1. Extract policy from seed data + CoT
           2. Generate synthetic examples following policy
           """

           # Phase 1: Extract policy
           policy = self.policy_extractor.extract_policy(
               seed_examples=seed_examples,
               cot_traces=seed_traces,
               instruction=instruction
           )

           console.print(f"\n📋 Extracted Task Policy:")
           console.print(f"   Task: {policy.task_description}")
           console.print(f"   Input format: {policy.input_format}")
           console.print(f"   Constraints: {', '.join(policy.key_constraints)}")

           # Phase 2: Generate synthetic data
           num_synthetic_needed = target_size - len(seed_traces)
           synthetic_traces = self.generator.generate_synthetic_examples(
               policy=policy,
               num_examples=num_synthetic_needed,
               instruction=instruction,
               seed_count=len(seed_traces)
           )

           # Combine: originals + synthetic
           all_traces = seed_traces + synthetic_traces

           console.print(f"\n✅ Generated {len(synthetic_traces)} synthetic examples")
           console.print(f"   Total dataset size: {len(all_traces)}")

           return all_traces
   ```

**Deliverable**: Policy-based synthetic data (10 seed → 500+ diverse, constrained examples)

---

### Phase 5: Distiller Module (Apple Silicon Training)

**Goal**: Fine-tune student model efficiently on Mac M1/M2/M3

**Training Framework**: Using **MLX-LM** (Primary choice for Phase 1)

Rationale: MLX-LM is purpose-built for Apple Silicon with automatic unified memory optimization. It provides the best performance and simplest implementation for the MVP.

| Aspect | MLX-LM (Primary) | Unsloth (Alternative) |
|--------|--------|---------|
| **Best For** | Apple Silicon native optimization | Cross-platform support |
| **Unified Memory** | ✅ Native support | ⚠️ Via PyTorch MPS |
| **Memory Efficiency** | ✅ ~70% savings | ✅ ~70% savings |
| **Setup Complexity** | ✅ Simpler | Moderate |
| **Model Availability** | ✅ Growing (good for Llama-3) | Extensive |
| **Community/Docs** | Growing | Large |
| **Inference Speed** | ✅ Optimized for Mac | ✅ Very fast |

#### Primary: MLX-LM (Phase 1 Implementation)

1. **MLX-based trainer** (`src/nanodistill/distiller/trainer.py` - MLX variant):
   ```python
   from mlx_lm import load, generate, lora
   from mlx_lm.tuning import lora_training

   class MLXTrainer:
       def __init__(self, student_model: str, config: dict):
           self.student_model = student_model
           self.config = config
           self.device = "mps"  # MLX optimized for Apple Silicon

       def train(self, dataset: Dataset) -> str:
           """Fine-tune student model using MLX"""

           # Load model (MLX handles unified memory automatically)
           model, tokenizer = load(self.student_model)

           # Configure LoRA (MLX style)
           lora_config = {
               "rank": 16,
               "alpha": 16,
               "layers": ["q_proj", "v_proj"],
               "freeze_model": True,
           }

           # MLX handles memory management automatically
           # No need for 4-bit quantization (unified memory handles it)
           trainer = lora_training.LoRATrainer(
               model=model,
               tokenizer=tokenizer,
               config={
                   "batch_size": 1,  # MLX adapts efficiently
                   "num_epochs": 2,
                   "learning_rate": 2e-4,
                   "max_seq_length": 2048,
               },
               lora_config=lora_config,
           )

           # Train (MLX optimizes memory usage automatically)
           trainer.train(dataset)

           # Save model
           model_path = f"./outputs/{self.config['name']}"
           model.save(model_path)
           tokenizer.save(model_path)

           return model_path
   ```

**Benefits**: MLX handles Apple Silicon optimization internally. No manual quantization needed. Better memory utilization.

---

#### Alternative (Post-MVP): Unsloth (Cross-Platform, for Linux/Windows support)

**When to use**: After Phase 1 MVP, if you want to add cross-platform support (Linux/Windows). Not needed for Phase 1.

1. **Device detection** (`src/nanodistill/distiller/hyperparams.py`):
   ```python
   import torch

   def detect_device():
       """Detect best available device"""
       if torch.backends.mps.is_available():
           return "mps"  # Apple Silicon
       elif torch.cuda.is_available():
           return "cuda"  # NVIDIA
       return "cpu"

   def get_apple_silicon_config():
       """Optimized config for Mac M1/M2/M3"""
       return {
           "load_in_4bit": True,  # QLoRA for memory efficiency
           "per_device_train_batch_size": 1,  # Conservative for unified memory
           "gradient_accumulation_steps": 8,  # Effective batch size = 8
           "max_seq_length": 2048,
           "learning_rate": 2e-4,
           "num_train_epochs": 2,
           "warmup_steps": 10,
           "logging_steps": 5,
           "optim": "adamw_8bit",  # Memory-efficient optimizer
           "fp16": False,  # MPS doesn't support fp16 well
           "bf16": False,  # Use default precision
       }
   ```

2. **Unsloth trainer** (`src/nanodistill/distiller/trainer.py` - Unsloth variant):
   ```python
   from unsloth import FastLanguageModel, is_bfloat16_supported
   from trl import SFTTrainer

   class UnslothTrainer:
       def __init__(self, student_model: str, config: dict):
           self.student_model = student_model
           self.config = config
           self.device = detect_device()

       def train(self, dataset: Dataset) -> str:
           """Fine-tune student model"""

           # Load model with Unsloth optimizations
           model, tokenizer = FastLanguageModel.from_pretrained(
               model_name=self.student_model,
               max_seq_length=self.config["max_seq_length"],
               load_in_4bit=self.config["load_in_4bit"],
               dtype=None,  # Auto-detect
           )

           # Configure LoRA
           model = FastLanguageModel.get_peft_model(
               model,
               r=16,  # LoRA rank
               lora_alpha=16,
               lora_dropout=0,
               target_modules=[
                   "q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"
               ],
               use_gradient_checkpointing="unsloth",
           )

           # Format dataset
           formatted_dataset = dataset.map(
               lambda x: format_for_training(x, tokenizer),
               remove_columns=dataset.column_names
           )

           # Create trainer
           trainer = SFTTrainer(
               model=model,
               tokenizer=tokenizer,
               train_dataset=formatted_dataset,
               args=TrainingArguments(
                   output_dir="./outputs/checkpoints",
                   **self.config
               ),
           )

           # Train
           trainer.train()

           # Save model
           model_path = f"./outputs/{self.config['name']}"
           model.save_pretrained(model_path)
           tokenizer.save_pretrained(model_path)

           return model_path
   ```

3. **Export utilities** (`src/nanodistill/distiller/export.py`):

   Both MLX and Unsloth models can be exported to GGUF format for portable inference:

   ```python
   import subprocess
   from pathlib import Path

   def export_to_gguf(model_path: str, quantization: str = "Q4_K_M") -> str:
       """Convert model to GGUF format using llama.cpp

       Works with both MLX and Unsloth trained models
       """

       gguf_path = f"{model_path}/model-{quantization}.gguf"

       # MLX models may be in .safetensors format
       # Convert via llama.cpp (handles both PyTorch and safetensors)
       subprocess.run([
           "python", "llama.cpp/convert.py",
           model_path,
           "--outtype", quantization,
           "--outfile", gguf_path
       ], check=True)

       return gguf_path
   ```

4. **Memory monitoring** (Apple Silicon specific):
   - For **MLX**: Automatically optimized by MLX framework
   - For **Unsloth**: Monitor via Activity Monitor, adjust batch size if needed
   - Unified memory usage tracked automatically on Mac

---

#### Choosing Your Trainer

**Use MLX-LM if**:
- Targeting Apple Silicon exclusively
- Want automatic memory optimization
- Prefer simpler setup
- Don't need cross-platform support

**Use Unsloth if**:
- Want cross-platform support (also use on Linux/Windows later)
- Prefer larger community/documentation
- Have experience with Transformers ecosystem
- Need more control over training parameters

Both produce compatible GGUF models for inference.

**Deliverable**: Working training pipeline optimized for Apple Silicon (choice of MLX or Unsloth)

---

### Phase 6: Core Orchestrator (Main Entry Point)

**Goal**: Tie all modules together into single `distill()` function

#### Implementation (`src/nanodistill/core.py`):

```python
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Union
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import DistillationConfig
from .teacher.client import TeacherClient
from .amplifier.pipeline import AmplificationPipeline
from .distiller.trainer import MLXTrainer  # Phase 1: MLX-LM for Apple Silicon
from .distiller.export import export_to_gguf
from .data.loader import load_seed_data, to_hf_dataset
from .utils.errors import NanoDistillError, validate_teacher_api_key

@dataclass
class DistillationResult:
    """Result of distillation process"""
    model_path: Path
    gguf_path: Optional[Path]
    metrics: Dict
    config: DistillationConfig

def distill(
    name: str,
    seed: Union[List[Dict[str, str]], str, Path],
    instruction: str,
    teacher: str = "claude-sonnet-4-5",
    student: str = "mlx-community/Llama-3-8B-Instruct-4bit",
    augment_factor: int = 50,
    output_dir: str = "./outputs",
    export_gguf: bool = True,
    **kwargs
) -> DistillationResult:
    """
    Main entry point: Convert seed examples into a reasoning-capable SLM.

    Args:
        name: Identifier for this distillation run
        seed: Training examples (list of dicts or file path)
        instruction: System prompt / task description
        teacher: Teacher model name (default: Claude Sonnet 4.5)
                 Supported: claude-*, gpt-*, gemini-*, ollama/*
        student: Student model to fine-tune (default: Llama-3-8B)
                 HuggingFace model identifier
        augment_factor: Multiply seed by this factor (default: 50)
        output_dir: Where to save outputs (default: ./outputs)
        export_gguf: Export to GGUF format (default: True)

    Environment Variables:
        ANTHROPIC_API_KEY: Required for claude-* models
        OPENAI_API_KEY: Required for gpt-* models
        GEMINI_API_KEY: Required for gemini-* models
        (No auth needed for ollama/* models)

    Returns:
        DistillationResult with paths and metrics

    Example:
        >>> # Set env var first
        >>> import os
        >>> os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
        >>>
        >>> result = distill(
        ...     name="math-tutor-v1",
        ...     seed=[{"input": "What is 2+2?", "output": "4"}, ...],
        ...     instruction="You are a patient math tutor.",
        ...     teacher="claude-sonnet-4-5"
        ... )
        >>> print(f"Model ready: {result.gguf_path}")
    """

    console = Console()

    try:
        # 1. Validate teacher API key upfront
        validate_teacher_api_key(teacher)

        # 2. Validate configuration
        seed_data = load_seed_data(seed)
        config = DistillationConfig(
            name=name,
            seed=seed_data,
            instruction=instruction,
            teacher=teacher,
            student=student,
            augment_factor=augment_factor,
            output_dir=output_dir,
            **kwargs
        )

        console.print(f"\n[bold cyan]🔬 NanoDistill: {name}[/bold cyan]")
        console.print(f"Teacher: {teacher}")
        console.print(f"Student: {student}")
        console.print(f"Seed examples: {len(seed_data)}")
        console.print(f"Target dataset size: {len(seed_data) * augment_factor}\n")

        # 3. Initialize components
        # TeacherClient uses env vars automatically via LiteLLM
        teacher_client = TeacherClient(config.teacher)
        amplifier = AmplificationPipeline(teacher_client)
        distiller = MLXTrainer(config.student, config)  # Phase 1: MLX-LM

        # 4. Execute pipeline with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            # Stage 1: Generate CoT traces
            task1 = progress.add_task(
                "[cyan]🎓 Synthesizing CoT traces from teacher...",
                total=None
            )
            cot_traces = teacher_client.synthesize_cot(
                config.seed,
                config.instruction
            )
            progress.update(task1, completed=True)
            console.print(f"✓ Generated {len(cot_traces)} CoT traces\n")

            # Stage 2: Amplify dataset (Extract Policy + Generate Synthetic)
            task2 = progress.add_task(
                "[yellow]📈 Extracting policy and generating synthetic data...",
                total=None
            )
            target_dataset_size = len(config.seed) * config.augment_factor
            amplified_traces = amplifier.amplify(
                seed_traces=cot_traces,
                seed_examples=config.seed,
                instruction=config.instruction,
                target_size=target_dataset_size
            )
            progress.update(task2, completed=True)
            console.print(f"✓ Amplified to {len(amplified_traces)} training examples\n")

            # Stage 3: Fine-tune student model
            task3 = progress.add_task(
                "[green]🔥 Fine-tuning student model...",
                total=None
            )
            training_dataset = to_hf_dataset(amplified_traces)
            model_path = distiller.train(training_dataset)
            progress.update(task3, completed=True)
            console.print(f"✓ Model trained and saved to {model_path}\n")

            # Stage 4: Export to GGUF
            gguf_path = None
            if export_gguf:
                task4 = progress.add_task(
                    "[blue]📦 Exporting to GGUF...",
                    total=None
                )
                gguf_path = export_to_gguf(model_path, quantization="Q4_K_M")
                progress.update(task4, completed=True)
                console.print(f"✓ GGUF model: {gguf_path}\n")

        # 5. Return result
        result = DistillationResult(
            model_path=Path(model_path),
            gguf_path=Path(gguf_path) if gguf_path else None,
            metrics=distiller.metrics,
            config=config
        )

        console.print("\n[bold green]✅ Distillation complete![/bold green]")
        console.print(f"Model: {result.gguf_path or result.model_path}")
        console.print(f"Training loss: {result.metrics.get('train_loss', 'N/A')}")

        return result

    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        raise NanoDistillError(f"Distillation failed: {e}") from e
```

**Package exports** (`src/nanodistill/__init__.py`):
```python
"""NanoDistill: Convert 10 examples into a reasoning-capable small language model."""

from .core import distill, DistillationResult
from .config import DistillationConfig

__version__ = "0.1.0"
__all__ = ["distill", "DistillationResult", "DistillationConfig"]
```

**Deliverable**: Working end-to-end pipeline

---

### Phase 7: Testing & Documentation

**Goal**: Ensure reliability and usability

#### Tasks:

1. **Unit tests**:
   - `tests/test_teacher.py` - Mock teacher API calls
   - `tests/test_amplifier.py` - Test amplification strategies
   - `tests/test_distiller.py` - Test training config generation
   - `tests/test_integration.py` - End-to-end smoke test

2. **Example script** (`examples/basic_usage.py`):
   ```python
   from nanodistill import distill

   # Minimal example
   seed_data = [
       {"input": "What is 2+2?", "output": "4"},
       {"input": "What is 3+5?", "output": "8"},
       # ... 8 more examples
   ]

   result = distill(
       name="math-tutor-v1",
       seed=seed_data,
       instruction="You are a helpful math tutor. Show your reasoning.",
       teacher="claude-sonnet-4-5",
       student="mlx-community/Llama-3-8B-Instruct-4bit",
       augment_factor=50
   )

   print(f"Model saved to: {result.gguf_path}")
   ```

3. **README** with:
   - Quick start guide
   - Installation instructions
   - API reference
   - Requirements (Mac M1+, 16GB+ RAM)
   - Troubleshooting tips

**Deliverable**: Tested, documented MVP

---

## Critical Files (Implementation Order)

1. **`src/nanodistill/config.py`** - Configuration and validation (foundational)
2. **`src/nanodistill/teacher/client.py`** - Claude API integration (first pipeline stage)
3. **`src/nanodistill/data/loader.py`** - Data loading utilities (needed by all stages)
4. **`src/nanodistill/amplifier/policy.py`** - Policy extraction from seed data
5. **`src/nanodistill/amplifier/generator.py`** - Synthetic data generation using policy
6. **`src/nanodistill/amplifier/pipeline.py`** - Orchestrates policy extraction + generation
7. **`src/nanodistill/distiller/trainer.py`** - MLX-LM training for Apple Silicon (Phase 1)
8. **`src/nanodistill/core.py`** - Main orchestrator (ties everything together)

---

## Apple Silicon Specific Considerations

### Memory Management (MLX-LM Handles This Automatically)
- Mac unified memory architecture: RAM = VRAM
- MLX-LM automatically optimizes for unified memory
- 4-bit quantization built into MLX (saves ~70% memory)
- Llama-3-8B-4bit uses ~4-5GB, leaving room for OS and other processes
- **No manual batch size tuning needed** - MLX adapts automatically

### MLX Framework Benefits
- ✅ Purpose-built for Apple Silicon
- ✅ Unified memory optimization built-in
- ✅ No separate GPU backend needed (uses Metal acceleration natively)
- ✅ Automatic quantization and memory management

### Model Size Recommendations
- **M1/M2 (16GB)**: Llama-3-8B-4bit (default, recommended)
- **M3 Pro/Max (32GB+)**: Llama-3-8B or Llama-3-70B
- **M3 Ultra (64GB+)**: Larger models supported

### Troubleshooting
- If training is slow: Check Activity Monitor, MLX adapts batch size automatically
- If memory issues: MLX handles memory management, contact MLX team if issues persist

---

## Success Metrics (Aligned with PRD)

✅ **First synthetic trace in 30 seconds**
- Claude API is fast, should see first trace immediately
- Show progress in real-time with Rich

✅ **Zero-crash training**
- Conservative batch size (1) for safety
- 4-bit quantization by default
- Memory estimation before training starts

✅ **Usable .gguf output**
- Export to GGUF by default
- Include inference example in README
- Test with llama.cpp before release

---

## Dependencies Summary

**Always Required**:
- `litellm>=1.50.0` - Teacher API (Claude support)
- `instructor>=1.3.0` - Structured output from Claude
- `pydantic>=2.0.0` - Data validation
- `datasets>=2.18.0` - Data handling
- `rich>=13.0.0` - Beautiful terminal UI

**Phase 1 Training Backend** (MLX-LM):
```toml
mlx>=0.0.8
mlx-lm>=0.0.2
```
- Purpose: Apple Silicon native, automatic memory optimization
- Includes: Model loading, LoRA training, inference

**Post-MVP Alternative** (Unsloth for cross-platform):
```toml
unsloth[colab-new]>=2024.1
transformers>=4.40.0
torch>=2.2.0
trl>=0.7.0
```
- Use: After Phase 1, if Linux/Windows support needed
- Includes: LoRA, optimization, trainer integration

**Optional**:
- `llama-cpp-python` - For GGUF conversion/testing
- `pytest` - Testing
- `typer` - CLI (post-MVP)

---

## Next Steps After MVP

**Post-MVP Enhancements**:
1. Implement caching for teacher responses (policy extraction + generation)
2. Add evaluation harness (test student vs teacher performance)
3. Support other teacher models (GPT-4o, local Ollama, Gemini)
4. Implement policy refinement loop (iterative policy updates)
5. CLI interface with Typer for easier file input/output
6. Dataset quality filtering and validation
7. Switch between MLX-LM and Unsloth at runtime
8. Multi-GPU support (for future Linux/Windows users)
9. Model registry (save/load distilled models with policies)

---

## Verification Plan

**Manual testing checklist**:
1. ✅ Install package on Mac M1/M2/M3
2. ✅ Run basic example with 10 seed examples
3. ✅ Verify Claude API calls work
4. ✅ Verify amplification produces expected dataset size
5. ✅ Verify training completes without OOM
6. ✅ Verify GGUF export succeeds
7. ✅ Test inference with llama.cpp or Ollama
8. ✅ Validate reasoning traces in output

**Automated tests**:
- Unit tests for each module (mock external dependencies)
- Integration test with small dataset (2-3 examples)
- Smoke test on CI (if GitHub Actions supports Mac runners)

---

## Implementation Timeline Estimate

**Phase 1 (Foundation)**: Project setup, config, errors, MLX-LM dependencies
**Phase 2 (Teacher)**: Claude integration with Instructor + LiteLLM
**Phase 3 (Data)**: Loading and formatting
**Phase 4 (Amplifier)**: Policy extraction + synthetic data generation
**Phase 5 (Distiller)**: MLX-LM training on Apple Silicon
**Phase 6 (Core)**: Main orchestrator
**Phase 7 (Testing)**: Tests, examples, README

**Total**: MVP should be achievable in focused development time. Prioritize getting `distill()` working end-to-end with MLX-LM, then iterate on quality and features.

---

## Notes

- **Phase 1 uses MLX-LM** - Apple Silicon native framework for optimal performance
- Start simple, iterate fast
- Test on real Mac hardware early (M1/M2/M3 recommended)
- MLX handles memory optimization automatically - no manual batch size tuning needed
- Save intermediate outputs (policy, synthetic data, trained weights) for debugging
- Post-MVP: Can switch to Unsloth for cross-platform support if needed
