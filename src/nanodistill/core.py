"""Core orchestrator for NanoDistill distillation pipeline."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .amplifier.pipeline import AmplificationPipeline
from .config import DistillationConfig
from .data.loader import (
    load_seed_data,
    load_traces_from_jsonl,
    load_training_data,
    save_traces_to_jsonl,
    to_hf_dataset,
)
from .distiller.trainer import MLXTrainer
from .teacher.client import TeacherClient
from .teacher.schemas import TaskPolicy
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
    seed: Union[List[Dict[str, str]], str, Path, None] = None,
    instruction: str = "",
    teacher: str = "claude-sonnet-4-5",
    student: str = "mlx-community/Llama-3-8B-Instruct-4bit",
    augment_factor: int = 50,
    output_dir: str = "./outputs",
    response_model: Optional[Type[BaseModel]] = None,
    training_data: Optional[Union[List[Dict[str, str]], str, Path]] = None,
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
        training_data: Pre-formatted training data to bypass CoT synthesis and
                      amplification stages. Can be:
                      - List of dicts with 'input', 'thinking', and 'output' keys
                      - Path to JSONL file with those fields
                      When provided, seed, instruction, and teacher API key are not required.
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

        # Bypass path: user-supplied training data skips CoT + amplification
        bypass_pipeline = training_data is not None

        if bypass_pipeline:
            console.print("[bold yellow]⚡ Using pre-formatted training data[/bold yellow]")
            console.print(f"Student: {student}")

            # Load and validate training data early
            amplified_traces = load_training_data(training_data)
            console.print(f"Training examples: {len(amplified_traces)}")

            validate_output_dir(output_dir)

            # Use placeholders for config fields that aren't needed
            seed_data = [{"input": "n/a", "output": "n/a"}]
            config = DistillationConfig(
                name=name,
                seed=seed_data,
                instruction="Training from pre-formatted data",
                teacher=teacher,
                student=student,
                augment_factor=augment_factor,
                output_dir=output_dir,
                **kwargs,
            )
        else:
            if seed is None:
                raise ConfigError("seed is required when training_data is not provided")
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

        # Check if adapters already exist (and are not empty)
        output_base = Path(config.output_dir) / config.name
        adapter_path = output_base / "adapters"
        adapter_files = (
            list(adapter_path.glob("*.safetensors")) + list(adapter_path.glob("*.npz"))
            if adapter_path.exists()
            else []
        )
        skip_training = len(adapter_files) > 0

        if skip_training:
            console.print("[bold yellow]⚠️  Fine-tuned adapters already exist![/bold yellow]")
            console.print(f"   Location: {adapter_path}")
            console.print(
                "[yellow]   Skipping training stage and proceeding to evaluation...[/yellow]"
            )
            console.print("\n[bold]💡 To train a new model, change the run name:[/bold]")
            console.print('   distill(name="math-tutor-v2", ...)')
            console.print("   OR")
            console.print('   distill(name="math-tutor-v1-retrain", ...)\n')

        # Initialize components
        distiller = MLXTrainer(config.student, config)

        # Check for existing amplified data and determine resume point
        amplified_path = Path(config.output_dir) / config.name / "traces_amplified.jsonl"
        use_existing_data = False
        resume_amplification = False

        if not bypass_pipeline and amplified_path.exists():
            try:
                all_traces = load_traces_from_jsonl(str(amplified_path))
                existing_synthetic = len(all_traces) - len(seed_data)
                total_synthetic = len(seed_data) * (config.augment_factor - 1)

                if existing_synthetic >= total_synthetic:
                    # Amplification complete
                    use_existing_data = True
                    console.print("[bold yellow]⚡ Found complete amplification data[/bold yellow]")
                    console.print(
                        "[yellow]   Skipping API calls and going directly to "
                        "fine-tuning...\n[/yellow]"
                    )
                else:
                    # Amplification incomplete - resume
                    resume_amplification = True
                    completed_batches = existing_synthetic // len(seed_data)
                    total_batches = config.augment_factor - 1
                    console.print(
                        f"[bold yellow]⚡ Found partial amplification: "
                        f"{existing_synthetic}/{total_synthetic} synthetic "
                        f"examples[/bold yellow]"
                    )
                    console.print(
                        f"[yellow]   Resuming from batch {completed_batches + 1}/"
                        f"{total_batches}...\n[/yellow]"
                    )
            except Exception:
                # If checkpoint is corrupted, start fresh
                console.print(
                    "[bold yellow]⚠️  Corrupted amplification checkpoint, "
                    "starting fresh[/bold yellow]\n"
                )

        # Execute pipeline
        # Initialize amplified_traces to empty list (will be populated below)
        if not bypass_pipeline:
            amplified_traces: List = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            if skip_training:
                # Skip stages 1-3, load existing model
                console.print(
                    "[bold cyan]Skipping stages 1-3 (CoT, Amplification, " "Training)[/bold cyan]\n"
                )
                model_path = str(output_base)

                # Try to load existing metrics from summary
                summary_path = output_base / "summary.json"
                summary_data: Dict = {}
                if summary_path.exists():
                    with open(summary_path) as f:
                        summary_data = json.load(f)

                existing_metrics = summary_data.get("training", {})

                # Load amplified_traces for evaluation purposes
                if amplified_path.exists():
                    amplified_traces = load_traces_from_jsonl(str(amplified_path))
                else:
                    # If no amplified data, use empty list
                    amplified_traces = []

            elif bypass_pipeline:
                # training_data was provided -- amplified_traces already loaded above
                pass

            elif not use_existing_data:
                # Stage 1: Generate CoT traces from seed (if not cached)
                traces_path = Path(config.output_dir) / config.name / "traces_cot.jsonl"
                if not resume_amplification and traces_path.exists():
                    # Load cached CoT traces
                    console.print("📂 Loading existing CoT traces...")
                    cot_traces = load_traces_from_jsonl(str(traces_path))
                    console.print(f"✓ Loaded {len(cot_traces)} CoT traces\n")
                    teacher_client = TeacherClient(config.teacher, config=config)
                else:
                    # Generate new CoT traces
                    task1 = progress.add_task(
                        "[cyan]🎓 Synthesizing CoT traces from teacher...",
                        total=None,
                    )
                    teacher_client = TeacherClient(config.teacher, config=config)
                    cot_traces = teacher_client.synthesize_cot(seed_data, config.instruction)
                    progress.update(task1, completed=True)
                    console.print(f"✓ Generated {len(cot_traces)} CoT traces\n")

                    # Save intermediate CoT traces
                    traces_path.parent.mkdir(parents=True, exist_ok=True)
                    save_traces_to_jsonl(cot_traces, traces_path)
                    console.print(f"  Saved to: {traces_path}")

                # Stage 2: Amplify dataset through policy extraction and synthesis (incremental)
                num_batches = max(0, config.augment_factor - 1)

                if num_batches > 0:
                    task2 = progress.add_task(
                        "[yellow]📈 Amplifying dataset (policy → synthesis)...",
                        total=num_batches,
                    )

                    amplifier = AmplificationPipeline(teacher_client)
                    amplify_gen = amplifier.amplify(
                        seed_data,
                        cot_traces,
                        config.instruction,
                        config.augment_factor,
                        output_path=amplified_path,
                        response_model=response_model,
                    )

                    # Consume generator to get final results and display batch progress
                    # Use manual iteration to properly capture StopIteration return
                    policy = None
                    try:
                        while True:
                            (
                                batch_num,
                                total_batches,
                                current_synthetic,
                                total_synthetic,
                            ) = next(amplify_gen)
                            progress.update(task2, completed=batch_num)
                            console.print(
                                f"  Batch {batch_num}/{total_batches} complete "
                                f"({current_synthetic}/{total_synthetic} "
                                f"synthetic examples)"
                            )
                    except StopIteration as e:
                        # Capture return value from generator
                        if e.value:
                            amplified_traces, policy = e.value
                        else:
                            # Fallback if generator ended without explicit return
                            amplified_traces = cot_traces.copy()
                            policy = None

                    console.print(f"✓ Amplified to {len(amplified_traces)} training examples\n")
                else:
                    # No amplification needed (augment_factor = 1)
                    amplified_traces = cot_traces.copy()
                    policy = None
                    console.print("✓ No amplification needed (augment_factor = 1)\n")

                # Save extracted task policy if available
                if policy:
                    policy_path = amplified_path.parent / "task_policy.json"
                    _save_policy(policy, policy_path)
                    console.print(f"  Policy saved to: {policy_path}")
            else:
                # Load existing amplified data
                console.print("📂 Loading existing amplified data...")
                amplified_traces = load_traces_from_jsonl(str(amplified_path))
                console.print(f"✓ Loaded {len(amplified_traces)} training examples\n")

            # Stage 3: Fine-tune student model (skip if using existing adapters)
            if not skip_training:
                task3 = progress.add_task(
                    "[green]🔥 Fine-tuning student model...",
                    total=None,
                )
                training_dataset = to_hf_dataset(amplified_traces)
                model_path = distiller.train(training_dataset)
                progress.update(task3, completed=True)
                console.print(f"✓ Model trained and saved to {model_path}\n")

        # Prepare results
        if skip_training:
            # Use existing metrics
            base_metrics = {
                "seed_count": len(seed_data) if not bypass_pipeline else 0,
                "augment_factor": config.augment_factor,
                "teacher_model": config.teacher,
                "student_model": config.student,
            }
            base_metrics.update(existing_metrics)
            result_metrics = base_metrics
        else:
            # Use newly computed metrics
            result_metrics = {
                "seed_count": len(seed_data) if not bypass_pipeline else 0,
                "training_examples": len(amplified_traces),
                "augment_factor": config.augment_factor,
                "teacher_model": config.teacher,
                "student_model": config.student,
                **distiller.metrics,
            }

        result = DistillationResult(
            model_path=Path(model_path),
            metrics=result_metrics,
            config=config,
        )

        # Stage 4: Optional baseline evaluation
        if config.evaluation_report:
            try:
                console.print("\n[bold cyan]📊 Running baseline evaluation...[/bold cyan]")
                from .evaluator import evaluate_baseline

                baseline_result = evaluate_baseline(
                    run_name=config.name,
                    output_dir=config.output_dir,
                )
                console.print(f"✓ Report saved: {baseline_result.html_path}\n")
                result.metrics["baseline"] = {
                    "exact_match_rate": baseline_result.metrics.exact_match_rate,
                    "avg_similarity": baseline_result.metrics.avg_similarity,
                    "total_examples": baseline_result.metrics.total_examples,
                    "exact_matches": baseline_result.metrics.exact_matches,
                }
            except Exception as e:
                console.print(f"[yellow]⚠️  Baseline evaluation failed: {e}[/yellow]\n")

        # Create summary report (skip if we skipped training)
        if not skip_training:
            summary_path = Path(config.output_dir) / config.name / "summary.json"
            _save_summary_report(result, summary_path, console)

        # Summary
        if skip_training:
            console.print("\n[bold green]✅ Ready for Evaluation![/bold green]")
            console.print("   (Training skipped - using existing adapters)")
        else:
            console.print("\n[bold green]✅ Distillation Complete![/bold green]")
        console.print(f"Model path: {result.model_path}")
        if "training_examples" in result.metrics:
            console.print(f"Training examples: {result.metrics['training_examples']}")
        console.print(f"Output directory: {Path(config.output_dir) / config.name}")

        if not skip_training:
            console.print("\nGenerated artifacts:")
            console.print("  • traces_cot.jsonl - Original Chain-of-Thought traces")
            console.print("  • task_policy.json - Extracted task pattern")
            console.print("  • traces_amplified.jsonl - Amplified training data")
            console.print("  • summary.json - Distillation metrics and statistics")
            console.print("\nNext steps:")
            console.print("1. Test the model locally with llama.cpp or MLX")
            console.print("2. Review task_policy.json to understand learned pattern")
            console.print("3. Compare against teacher model (Claude) on test set")
            console.print("4. Iterate: adjust seed data or augment_factor as needed")
        else:
            console.print("\nNext steps:")
            console.print("1. Review the baseline evaluation report")
            console.print("2. Optionally re-train with a new name if needed")

        return result

    except ConfigError as e:
        console.print("\n[bold red]❌ Configuration Error[/bold red]")
        console.print(str(e))
        raise
    except NanoDistillError as e:
        console.print("\n[bold red]❌ Distillation Error[/bold red]")
        console.print(str(e))
        raise
    except Exception as e:
        console.print("\n[bold red]❌ Unexpected Error[/bold red]")
        console.print(f"{type(e).__name__}: {str(e)}")
        raise NanoDistillError(f"Distillation failed: {str(e)}") from e


def _save_policy(policy: TaskPolicy, path: Path) -> None:
    """Save extracted task policy as JSON.

    Args:
        policy: TaskPolicy object to save
        path: Output file path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    policy_dict = policy.model_dump(mode="json")
    with open(path, "w") as f:
        json.dump(policy_dict, f, indent=2)


def _save_summary_report(result: DistillationResult, path: Path, console: Console) -> None:
    """Save comprehensive summary report with metrics and metadata.

    Args:
        result: DistillationResult from distillation
        path: Output file path
        console: Rich console for feedback
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "name": result.config.name,
        "model": {
            "teacher": result.config.teacher,
            "student": result.config.student,
            "model_path": str(result.model_path),
        },
        "data": {
            "seed_examples": result.metrics["seed_count"],
            "augment_factor": result.metrics["augment_factor"],
            "training_examples": result.metrics["training_examples"],
        },
        "training": {
            key: value
            for key, value in result.metrics.items()
            if key
            not in [
                "seed_count",
                "augment_factor",
                "training_examples",
                "teacher_model",
                "student_model",
            ]
        },
        "config": {
            "learning_rate": result.config.learning_rate,
            "num_train_epochs": result.config.num_train_epochs,
            "batch_size": result.config.batch_size,
            "max_seq_length": result.config.max_seq_length,
            "lora_rank": result.config.lora_rank,
        },
    }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"  Saved to: {path}")
