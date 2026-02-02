"""Core orchestrator for NanoDistill distillation pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .amplifier.pipeline import AmplificationPipeline
from .config import DistillationConfig
from .data.loader import load_seed_data, load_traces_from_jsonl, save_traces_to_jsonl, to_hf_dataset
from .distiller.trainer import MLXTrainer
from .teacher.client import TeacherClient
from .utils.errors import (
    ConfigError,
    NanoDistillError,
    validate_output_dir,
    validate_seed_count,
    validate_teacher_api_key,
)


@dataclass
class DistillationResult:
    """Result of a distillation run.

    Attributes:
        model_path: Path to saved model directory
        metrics: Training metrics and statistics
        config: DistillationConfig used for this run
    """

    model_path: Path
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
    response_model: Optional[Type[BaseModel]] = None,
    **kwargs,
) -> DistillationResult:
    """Convert seed examples into a reasoning-capable small language model.

    Core entry point for NanoDistill. Transforms 10+ examples and an instruction
    into a fine-tuned, locally-runnable model optimized for Apple Silicon.

    Pipeline:
    1. 🎓 Policy Extraction - Analyze seed data to extract task pattern
    2. 🔄 Synthetic Generation - Use Claude to generate diverse new examples
    3. 📚 Data Amplification - Create Chain-of-Thought training data
    4. 🔥 Model Fine-tuning - Train student model on Apple Silicon

    Args:
        name: Identifier for this distillation run (used in output paths)
        seed: Training examples. Can be:
            - List of dicts with 'input' and 'output' keys
            - Path to JSON/JSONL/CSV file
            - File path string
        instruction: System prompt / task description to guide teacher and student
        teacher: Teacher model name (LiteLLM-compatible).
                Default: "claude-sonnet-4-5"
                Requires ANTHROPIC_API_KEY environment variable
        student: Student model to fine-tune (MLX-compatible model ID).
                Default: "mlx-community/Llama-3-8B-Instruct-4bit"
        augment_factor: Multiply seed examples by this factor for training dataset.
                       Default: 50 (10 seeds × 50 = 500 training examples)
        output_dir: Directory to save model and outputs.
                   Default: "./outputs"
        response_model: Optional Pydantic model to enforce schema on synthetic outputs.
                       When provided, uses instructor for structured generation
                       and filters extra fields automatically.
        **kwargs: Additional configuration options

    Returns:
        DistillationResult with model_path, metrics, and config

    Raises:
        ConfigError: If configuration is invalid
        TeacherAPIError: If teacher model API fails
        TrainingError: If model training fails
        NanoDistillError: For other distillation errors

    Example:
        >>> result = distill(
        ...     name="math-tutor-v1",
        ...     seed=[
        ...         {"input": "What is 2+2?", "output": "4"},
        ...         {"input": "What is 3+5?", "output": "8"},
        ...         # ... more examples
        ...     ],
        ...     instruction="You are a helpful math tutor. Show your reasoning.",
        ...     teacher="claude-sonnet-4-5",
        ... )
        >>> print(f"Model saved to: {result.model_path}")
    """
    console = Console()

    try:
        # Phase 0: Validate configuration
        console.print("\n[bold cyan]🔬 NanoDistill: Distillation Pipeline[/bold cyan]")
        console.print(f"Run: {name}")
        console.print(f"Teacher: {teacher}")
        console.print(f"Student: {student}")

        # Validate API key upfront
        validate_teacher_api_key(teacher)

        # Load and validate seed data
        seed_data = load_seed_data(seed)
        validate_seed_count(len(seed_data))
        validate_output_dir(output_dir)

        # Create configuration
        config = DistillationConfig(
            name=name,
            seed=seed_data,
            instruction=instruction,
            teacher=teacher,
            student=student,
            augment_factor=augment_factor,
            output_dir=output_dir,
            **kwargs,
        )

        console.print(f"Seed examples: {len(seed_data)}")
        console.print(f"Target dataset size: {len(seed_data) * augment_factor}\n")

        # Initialize components
        distiller = MLXTrainer(config.student, config)

        # Check if amplified data already exists
        amplified_path = Path(config.output_dir) / config.name / "traces_amplified.jsonl"
        use_existing_data = amplified_path.exists()

        if use_existing_data:
            console.print(f"[bold yellow]⚡ Found existing amplified data at {amplified_path}[/bold yellow]")
            console.print(f"[yellow]   Skipping API calls and going directly to fine-tuning...\n[/yellow]")

        # Execute pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            if not use_existing_data:
                # Stage 1: Generate CoT traces from seed
                task1 = progress.add_task(
                    "[cyan]🎓 Synthesizing CoT traces from teacher...",
                    total=None,
                )
                teacher_client = TeacherClient(config.teacher, config=config)
                cot_traces = teacher_client.synthesize_cot(seed_data, config.instruction)
                progress.update(task1, completed=True)
                console.print(f"✓ Generated {len(cot_traces)} CoT traces\n")

                # Save intermediate CoT traces
                traces_path = Path(config.output_dir) / config.name / "traces_cot.jsonl"
                traces_path.parent.mkdir(parents=True, exist_ok=True)
                save_traces_to_jsonl(cot_traces, traces_path)
                console.print(f"  Saved to: {traces_path}")

                # Stage 2: Amplify dataset through policy extraction and synthesis
                task2 = progress.add_task(
                    "[yellow]📈 Amplifying dataset (policy → synthesis)...",
                    total=None,
                )
                amplifier = AmplificationPipeline(teacher_client)
                amplified_traces = amplifier.amplify(
                    seed_data,
                    cot_traces,
                    config.instruction,
                    config.augment_factor,
                    response_model=response_model,
                )
                progress.update(task2, completed=True)
                console.print(f"✓ Amplified to {len(amplified_traces)} training examples\n")

                # Save amplified traces
                amplified_path.parent.mkdir(parents=True, exist_ok=True)
                save_traces_to_jsonl(amplified_traces, amplified_path)
                console.print(f"  Saved to: {amplified_path}")
            else:
                # Load existing amplified data
                console.print(f"📂 Loading existing amplified data...")
                amplified_traces = load_traces_from_jsonl(str(amplified_path))
                console.print(f"✓ Loaded {len(amplified_traces)} training examples\n")

            # Stage 3: Fine-tune student model
            task3 = progress.add_task(
                "[green]🔥 Fine-tuning student model...",
                total=None,
            )
            training_dataset = to_hf_dataset(amplified_traces)
            model_path = distiller.train(training_dataset)
            progress.update(task3, completed=True)
            console.print(f"✓ Model trained and saved to {model_path}\n")

        # Prepare results
        result = DistillationResult(
            model_path=Path(model_path),
            metrics={
                "seed_count": len(seed_data),
                "training_examples": len(amplified_traces),
                "augment_factor": config.augment_factor,
                "teacher_model": config.teacher,
                "student_model": config.student,
                **distiller.metrics,
            },
            config=config,
        )

        # Summary
        console.print("\n[bold green]✅ Distillation Complete![/bold green]")
        console.print(f"Model path: {result.model_path}")
        console.print(f"Training examples: {result.metrics['training_examples']}")
        console.print(f"\nNext steps:")
        console.print(f"1. Test the model locally with llama.cpp or MLX")
        console.print(f"2. Compare against teacher model (Claude) on test set")
        console.print(f"3. Iterate: adjust seed data or augment_factor as needed")

        return result

    except ConfigError as e:
        console.print(f"\n[bold red]❌ Configuration Error[/bold red]")
        console.print(str(e))
        raise
    except NanoDistillError as e:
        console.print(f"\n[bold red]❌ Distillation Error[/bold red]")
        console.print(str(e))
        raise
    except Exception as e:
        console.print(f"\n[bold red]❌ Unexpected Error[/bold red]")
        console.print(f"{type(e).__name__}: {str(e)}")
        raise NanoDistillError(f"Distillation failed: {str(e)}") from e
