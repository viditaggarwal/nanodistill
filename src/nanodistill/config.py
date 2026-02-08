"""Configuration and validation for NanoDistill distillation runs.

Includes Apple Silicon auto-detection for optimal default configurations.
"""

import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def detect_apple_silicon() -> Optional[str]:
    """Detect Apple Silicon chip type from system info.

    Returns:
        Chip identifier string (e.g., 'M1', 'M1 Pro', 'M2 Max', 'M4 Pro')
        or None if not Apple Silicon.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        brand = result.stdout.strip()
        # Extract chip name: "Apple M4 Pro" -> "M4 Pro"
        if "Apple" in brand:
            return brand.replace("Apple ", "").strip()
    except Exception:
        pass

    return "Apple Silicon"


def get_hardware_preset(chip: Optional[str] = None) -> Dict[str, Any]:
    """Get recommended training configuration for detected hardware.

    Returns optimized defaults based on Apple Silicon chip variant.
    These are tuned for single-task distillation (e.g., stock sentiment).

    Args:
        chip: Optional chip identifier. If None, auto-detects.

    Returns:
        Dict of recommended config overrides for the hardware.
    """
    if chip is None:
        chip = detect_apple_silicon()

    if chip is None:
        # Not Apple Silicon — return empty (use defaults)
        return {}

    chip_lower = (chip or "").lower()

    # M4 Pro / M4 Max — 24-128GB, 12-16 core CPU, powerful GPU
    if "m4 pro" in chip_lower or "m4 max" in chip_lower:
        return {
            "batch_size": 4,
            "max_seq_length": 512,
            "learning_rate": 2e-4,
            "lora_rank": 16,
            "lora_layers": 8,
            "max_memory_gb": 18,
            "memory_hard_limit_gb": 22,
            "cpu_capacity_percent": 0.9,
        }

    # M4 (base) — 16-32GB
    if "m4" in chip_lower:
        return {
            "batch_size": 2,
            "max_seq_length": 512,
            "learning_rate": 2e-4,
            "lora_rank": 16,
            "lora_layers": 6,
            "max_memory_gb": 12,
            "memory_hard_limit_gb": 14,
            "cpu_capacity_percent": 0.85,
        }

    # M3 Pro / M3 Max — 18-128GB
    if "m3 pro" in chip_lower or "m3 max" in chip_lower:
        return {
            "batch_size": 4,
            "max_seq_length": 512,
            "learning_rate": 2e-4,
            "lora_rank": 16,
            "lora_layers": 8,
            "max_memory_gb": 16,
            "memory_hard_limit_gb": 20,
            "cpu_capacity_percent": 0.85,
        }

    # M3 (base) — 8-24GB
    if "m3" in chip_lower:
        return {
            "batch_size": 2,
            "max_seq_length": 512,
            "learning_rate": 1e-4,
            "lora_rank": 8,
            "lora_layers": 4,
            "max_memory_gb": 10,
            "memory_hard_limit_gb": 12,
            "cpu_capacity_percent": 0.8,
        }

    # M2 Pro / M2 Max
    if "m2 pro" in chip_lower or "m2 max" in chip_lower:
        return {
            "batch_size": 2,
            "max_seq_length": 512,
            "learning_rate": 1e-4,
            "lora_rank": 8,
            "lora_layers": 6,
            "max_memory_gb": 14,
            "memory_hard_limit_gb": 18,
            "cpu_capacity_percent": 0.85,
        }

    # M2 (base) / M1 Pro / M1 Max
    if "m2" in chip_lower or "m1 pro" in chip_lower or "m1 max" in chip_lower:
        return {
            "batch_size": 2,
            "max_seq_length": 256,
            "learning_rate": 1e-4,
            "lora_rank": 8,
            "lora_layers": 4,
            "max_memory_gb": 10,
            "memory_hard_limit_gb": 12,
            "cpu_capacity_percent": 0.8,
        }

    # M1 (base) — 8-16GB, most conservative
    if "m1" in chip_lower:
        return {
            "batch_size": 1,
            "max_seq_length": 256,
            "learning_rate": 1e-5,
            "lora_rank": 4,
            "lora_layers": 2,
            "max_memory_gb": 6,
            "memory_hard_limit_gb": 7,
            "cpu_capacity_percent": 0.8,
        }

    # Generic Apple Silicon fallback
    return {
        "batch_size": 2,
        "max_seq_length": 256,
        "learning_rate": 1e-4,
        "lora_rank": 8,
        "lora_layers": 4,
    }


class DistillationConfig(BaseModel):
    """Configuration for a NanoDistill distillation run.

    Automatically detects Apple Silicon hardware and applies optimized defaults
    when hardware-specific parameters are not explicitly provided.

    Attributes:
        name: Identifier for this distillation run (used in output folder names)
        seed: List of training examples with required 'input' and 'output' fields
        instruction: System prompt / task description to guide teacher and student
        teacher: Teacher model name (LiteLLM-compatible). Default: "claude-sonnet-4-5"
        student: Student model to fine-tune (MLX-compatible model ID).
                Default: "mlx-community/Llama-3-8B-Instruct-4bit"
        augment_factor: Multiply seed examples by this factor. Default: 50
                       (10 seeds × 50 = 500 training examples)
        output_dir: Directory to save outputs. Default: "./outputs"

    Example:
        >>> config = DistillationConfig(
        ...     name="math-tutor",
        ...     seed=[
        ...         {"input": "What is 2+2?", "output": "4"},
        ...         {"input": "What is 5×3?", "output": "15"},
        ...     ],
        ...     instruction="You are a patient math tutor. Show your reasoning.",
        ...     teacher="claude-sonnet-4-5",
        ...     student="mlx-community/Llama-3-8B-Instruct-4bit",
        ...     augment_factor=50,
        ... )
    """

    name: str = Field(
        ...,
        description="Identifier for this distillation run",
        min_length=1,
        max_length=100,
    )

    seed: List[Dict[str, str]] = Field(
        ...,
        description="List of training examples with 'input' and 'output' fields",
        min_items=1,
    )

    instruction: str = Field(
        ...,
        description="System prompt / task description",
        min_length=10,
    )

    teacher: str = Field(
        default="claude-sonnet-4-5",
        description="Teacher model name (LiteLLM-compatible)",
    )

    student: str = Field(
        default="mlx-community/Llama-3-8B-Instruct-4bit",
        description="Student model to fine-tune (MLX-compatible)",
    )

    augment_factor: int = Field(
        default=50,
        description="Multiply seed examples by this factor for training dataset",
        ge=1,
        le=500,
    )

    output_dir: str = Field(
        default="./outputs",
        description="Directory to save model outputs and intermediate results",
    )

    batch_size: int = Field(
        default=1,
        description="Training batch size (reduce for memory-constrained systems)",
        ge=1,
        le=32,
    )

    learning_rate: float = Field(
        default=1e-5,
        description="Learning rate for fine-tuning (lower for stability)",
        gt=0,
        lt=1,
    )

    num_train_epochs: int = Field(
        default=1,
        description="Number of training epochs",
        ge=1,
        le=10,
    )

    max_seq_length: int = Field(
        default=256,
        description="Maximum sequence length for tokenization (lower = less memory)",
        ge=32,
        le=2048,
    )

    # === v0.2.0+ New Parameters ===

    temperature: float = Field(
        default=0.7,
        description="Sampling temperature for synthetic generation (0.0-2.0)",
        ge=0.0,
        le=2.0,
    )

    lora_rank: int = Field(
        default=8,
        description="LoRA adapter rank (higher = more parameters, 1-64)",
        ge=1,
        le=64,
    )

    lora_layers: int = Field(
        default=4,
        description="Number of layers to apply LoRA adapters (1-32)",
        ge=1,
        le=32,
    )

    val_split: float = Field(
        default=0.2,
        description="Validation set split ratio (0.0-0.5)",
        ge=0.0,
        le=0.5,
    )

    max_memory_gb: int = Field(
        default=12,
        description="Maximum RAM to use during training (GB). Auto-detects if not set.",
        ge=2,
        le=128,
    )

    memory_hard_limit_gb: int = Field(
        default=12,
        description="Hard stop limit. Must be >= max_memory_gb.",
        ge=2,
        le=128,
    )

    cpu_capacity_percent: float = Field(
        default=0.8,
        description="CPU/memory threshold before training pauses (0.1-1.0, where 0.8 = 80%)",
        ge=0.1,
        le=1.0,
    )

    litellm_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional LiteLLM parameters (temperature, top_p, timeout, seed, etc.)",
    )

    mlx_lm_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional MLX-LM parameters (grad-accumulation-steps, optimizer, seed, etc.)",
    )

    save_checkpoints: bool = Field(
        default=True,
        description="Save model checkpoints incrementally during training for recovery. Default: True",
    )

    checkpoint_interval: int = Field(
        default=0,
        description="Save checkpoint every N iterations (0=auto: every epoch). Default: 0",
        ge=0,
        le=1000,
    )

    suppress_warnings: bool = Field(
        default=False,
        description="Suppress training warnings (e.g., truncation). Default: False",
    )

    @staticmethod
    def get_system_defaults() -> tuple[int, int]:
        """Auto-detect system capabilities and return safe defaults.

        Uses hardware-aware limits instead of fixed 12GB cap. On M4 Pro (24GB+),
        the old 12GB cap left significant performance on the table.

        Returns:
            (max_memory_gb, memory_hard_limit_gb) tuple
        """
        try:
            import psutil

            total_memory_gb = psutil.virtual_memory().total / (1024**3)

            # Hardware-aware: use chip detection for better defaults
            chip = detect_apple_silicon()
            preset = get_hardware_preset(chip)

            if preset and "max_memory_gb" in preset:
                return preset["max_memory_gb"], preset["memory_hard_limit_gb"]

            # Fallback: scale with available RAM (no fixed 12GB cap)
            max_memory = int(total_memory_gb * 0.75)
            hard_limit = int(total_memory_gb * 0.9)

            return max_memory, hard_limit
        except Exception:
            # Fallback if psutil fails
            return 12, 12

    def get_litellm_synthesis_kwargs(self) -> Dict[str, Any]:
        """Get kwargs dict for LiteLLM synthetic generation calls.

        Merges explicit temperature config with user-provided litellm_kwargs.
        User-provided kwargs override explicit config.

        Returns:
            Dictionary with all LiteLLM parameters for unpacking
        """
        base = {"temperature": self.temperature}
        # User-provided kwargs override base config
        base.update(self.litellm_kwargs)
        return base

    def get_mlx_lm_cli_args(self) -> List[str]:
        """Get CLI arguments for MLX-LM training.

        Converts mlx_lm_kwargs dict to command-line arguments.
        Example: {"grad_accumulation_steps": 4} -> ["--grad-accumulation-steps", "4"]

        Returns:
            List of CLI argument strings
        """
        args = []
        for key, value in self.mlx_lm_kwargs.items():
            # Convert underscore to hyphen (Python style to CLI style)
            cli_key = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    args.append(cli_key)
            else:
                args.extend([cli_key, str(value)])
        return args

    @field_validator("seed")
    @classmethod
    def validate_seed_format(cls, v: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Validate that each seed example has required 'input' and 'output' fields."""
        if not v:
            raise ValueError("seed must contain at least 1 example")

        for i, example in enumerate(v):
            if not isinstance(example, dict):
                raise ValueError(f"seed[{i}] must be a dictionary")

            if "input" not in example or not example["input"].strip():
                raise ValueError(f"seed[{i}] must have non-empty 'input' field")

            if "output" not in example or not example["output"].strip():
                raise ValueError(f"seed[{i}] must have non-empty 'output' field")

        return v

    @field_validator("teacher")
    @classmethod
    def validate_teacher_name(cls, v: str) -> str:
        """Validate that teacher model name looks reasonable."""
        v = v.strip().lower()

        # Check for obvious invalid patterns
        if not v or " " in v:
            raise ValueError(f"Invalid teacher model name: {v}")

        return v

    @field_validator("student")
    @classmethod
    def validate_student_name(cls, v: str) -> str:
        """Validate that student model name looks like an MLX model ID."""
        v = v.strip()

        if not v or " " in v:
            raise ValueError(f"Invalid student model name: {v}")

        # MLX models typically follow pattern: namespace/model-name
        # But we allow flexibility for custom models
        if not ("/" in v or v.startswith("mlx")):
            # Lenient check - just ensure it's not obviously wrong
            pass

        return v

    @field_validator("memory_hard_limit_gb")
    @classmethod
    def validate_memory_limits(cls, v: int, info) -> int:
        """Ensure hard limit >= max memory."""
        if info.data.get("max_memory_gb") and v < info.data["max_memory_gb"]:
            max_mem = info.data["max_memory_gb"]
            raise ValueError(f"memory_hard_limit_gb ({v}) must be >= max_memory_gb ({max_mem})")
        return v

    def model_post_init(self, __context) -> None:
        """Post-initialization: Apply hardware presets and create output directory.

        Detects Apple Silicon chip and applies optimized defaults for any
        training parameters that were left at their default values. This means
        explicit user values are never overridden, but unset parameters get
        hardware-optimized defaults.
        """
        # Apply hardware-aware defaults for parameters left at class defaults
        chip = detect_apple_silicon()
        if chip:
            preset = get_hardware_preset(chip)
            # These are the class-level defaults — only override if user didn't set explicitly
            class_defaults = {
                "batch_size": 1,
                "max_seq_length": 256,
                "learning_rate": 1e-5,
                "lora_rank": 8,
                "lora_layers": 4,
                "max_memory_gb": 12,
                "memory_hard_limit_gb": 12,
                "cpu_capacity_percent": 0.8,
            }

            for key, preset_value in preset.items():
                current_value = getattr(self, key, None)
                default_value = class_defaults.get(key)
                # Only apply preset if the current value matches the class default
                if current_value is not None and default_value is not None:
                    if current_value == default_value:
                        object.__setattr__(self, key, preset_value)

        # Ensure output_dir exists
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Export config as dictionary (sensitive values redacted)."""
        return {
            "name": self.name,
            "seed_count": len(self.seed),
            "instruction_length": len(self.instruction),
            "teacher": self.teacher,
            "student": self.student,
            "augment_factor": self.augment_factor,
            "output_dir": self.output_dir,
        }
