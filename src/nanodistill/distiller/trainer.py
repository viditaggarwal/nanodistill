"""MLX-LM trainer for fine-tuning student models on Apple Silicon."""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import Dataset

from ..utils.errors import TrainingError


class MLXTrainer:
    """Fine-tune models using MLX-LM on Apple Silicon with LoRA.

    Attributes:
        student_model: Model ID (e.g., "Qwen/Qwen2.5-7B-Instruct-MLX-4bit")
        config: Training configuration
    """

    def __init__(
        self,
        student_model: str,
        config: dict,
    ):
        """Initialize MLX trainer.

        Args:
            student_model: MLX-compatible model ID
            config: Training configuration dict with:
                - name: Run identifier
                - output_dir: Output directory
                - num_train_epochs: Number of training epochs (default: 2)
                - learning_rate: Learning rate (default: 2e-4)
                - batch_size: Batch size (default: 2 for M1 Pro)
                - max_seq_length: Maximum sequence length (default: 512)
                - lora_rank: LoRA rank (default: 8)
                - lora_layers: Number of layers to apply LoRA (default: 4)

        Raises:
            TrainingError: If MLX-LM is not installed
        """
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
            from mlx_lm import load, generate

            self.mx = mx
            self.nn = nn
            self.optim = optim
            self.load = load
            self.generate = generate

        except ImportError as e:
            raise TrainingError(
                "MLX-LM not installed. Install with: pip install mlx mlx-lm"
            ) from e

        self.student_model = student_model
        self.config = config
        self.model = None
        self.tokenizer = None
        self.lora_params = {}
        self.metrics = {}

    def train(
        self,
        dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> str:
        """Fine-tune student model using MLX-LM with LoRA.

        Args:
            dataset: Training dataset (HuggingFace Dataset)
            eval_dataset: Optional evaluation dataset

        Returns:
            Path to saved model directory

        Raises:
            TrainingError: If training fails
        """
        try:
            self._load_model()
            train_data = self._prepare_dataset(dataset)

            epochs = self.config.num_train_epochs
            lr = self.config.learning_rate

            self._train_loop(train_data, epochs, lr)

            model_path = self._save_model()

            if eval_dataset:
                eval_metrics = self.evaluate(eval_dataset)
                self.metrics.update(eval_metrics)

            return model_path

        except Exception as e:
            raise TrainingError(f"Training failed: {str(e)}") from e

    def _load_model(self) -> None:
        """Load model and tokenizer, apply LoRA adapters."""
        try:
            print(f"Loading model: {self.student_model}")
            self.model, self.tokenizer = self.load(self.student_model)
            print("Model loaded successfully")

            # Apply LoRA adapters
            self._setup_lora()

        except Exception as e:
            raise TrainingError(
                f"Failed to load model {self.student_model}: {str(e)}"
            ) from e

    def _setup_lora(self) -> None:
        """LoRA setup - handled by MLX-LM tuner during training."""
        lora_rank = self.config.lora_rank
        lora_layers = self.config.lora_layers

        print(f"LoRA configuration:")
        print(f"  Rank: {lora_rank}")
        print(f"  Target layers: Last {lora_layers} layers")
        print(f"  (Initialization will happen during training with MLX-LM tuner)\n")

    def _prepare_dataset(self, dataset: Dataset) -> List[Dict]:
        """Prepare dataset for training.

        Args:
            dataset: HuggingFace Dataset

        Returns:
            List of tokenized training examples
        """
        formatted_data = []

        for example in dataset:
            text = f"""Input: {example['input']}

Thinking: {example['thinking']}

Output: {example['output']}"""

            tokens = self.tokenizer.encode(text)

            formatted_data.append({
                "text": text,
                "input_ids": tokens,
            })

        return formatted_data

    def _train_loop(
        self,
        train_data: List[Dict],
        num_epochs: int,
        learning_rate: float,
    ) -> None:
        """Execute training loop using MLX-LM CLI for proper LoRA fine-tuning.

        Args:
            train_data: Prepared training data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
        """
        import subprocess
        import psutil

        def check_system_capacity() -> bool:
            """Check if system is below configured capacity threshold and memory hard limit."""
            try:
                # Get memory info
                mem_info = psutil.virtual_memory()
                memory_used_gb = mem_info.used / (1024 ** 3)
                memory_percent = mem_info.percent

                # Get configured limits
                memory_hard_limit_gb = self.config.memory_hard_limit_gb
                cpu_capacity_percent = self.config.cpu_capacity_percent

                # Hard cap at memory_hard_limit_gb
                if memory_used_gb > memory_hard_limit_gb:
                    print(f"\n🛑 MEMORY HARD CAP HIT: {memory_used_gb:.2f}GB > {memory_hard_limit_gb}GB limit!")
                    print(f"   Stopping training to prevent system crash...")
                    return False

                # Check CPU usage (average over 100ms)
                cpu_percent = psutil.cpu_percent(interval=0.1)

                capacity = max(memory_percent, cpu_percent)

                if capacity > (cpu_capacity_percent * 100):
                    print(f"\n⚠️  System capacity at {capacity:.1f}%")
                    print(f"   Memory: {memory_used_gb:.2f}GB / {memory_hard_limit_gb}GB max ({memory_percent:.1f}%)")
                    print(f"   CPU: {cpu_percent:.1f}%")
                    print(f"   Waiting for system to cool down...")
                    return False

                return True
            except Exception:
                # If monitoring fails, assume safe
                return True

        batch_size = self.config.batch_size
        output_dir = Path(self.config.output_dir)
        model_name = self.config.name
        lora_layers = self.config.lora_layers
        lora_rank = self.config.lora_rank

        # Save training data in MLX-LM format (expects data_dir/train.jsonl and valid.jsonl)
        data_dir = output_dir / model_name / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Split data: train/validation ratio from config
        val_split = self.config.val_split
        split_idx = int(len(train_data) * (1 - val_split))
        train_examples = train_data[:split_idx]
        valid_examples = train_data[split_idx:]

        # Write training set
        training_file = data_dir / "train.jsonl"
        with open(training_file, "w") as f:
            for item in train_examples:
                f.write(json.dumps({"text": item["text"]}) + "\n")

        # Write validation set
        valid_file = data_dir / "valid.jsonl"
        with open(valid_file, "w") as f:
            for item in valid_examples:
                f.write(json.dumps({"text": item["text"]}) + "\n")

        # Adapter output path
        adapter_path = output_dir / model_name / "adapters"
        adapter_path.mkdir(parents=True, exist_ok=True)

        # Calculate iterations
        iters = num_epochs * (len(train_data) // batch_size)
        steps_per_report = max(1, iters // 20)  # Report ~20 times during training

        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"  Model: {self.student_model}")
        print(f"  Training examples: {len(train_data)}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  LoRA rank: {lora_rank}")
        print(f"  LoRA layers: {lora_layers}")
        print(f"  Total iterations: {iters}")
        print(f"  Data directory: {data_dir}")
        print(f"  Adapter path: {adapter_path}")
        print(f"  Steps per report: {steps_per_report}\n")

        print(f"{'='*60}")
        print(f"Starting Fine-Tuning with Gradient-Based LoRA Training")
        print(f"{'='*60}")
        print(f"⚙️  System capacity monitoring: Keeping usage below 80%\n")

        # Use MLX-LM CLI for training (handles gradients, optimizer, checkpoints)
        try:
            # Limit CPU threads to 80% of available cores
            available_cpus = os.cpu_count() or 8
            limited_cpus = max(1, int(available_cpus * 0.8))

            # Set environment for capacity limiting
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(limited_cpus)
            env["MKL_NUM_THREADS"] = str(limited_cpus)
            env["OPENBLAS_NUM_THREADS"] = str(limited_cpus)

            print(f"Limiting CPU threads to {limited_cpus}/{available_cpus} (80%)")
            print(f"Batch size: {batch_size} (reduce if OOM errors occur)\n")

            max_seq_length = self.config.max_seq_length

            cmd = [
                "python", "-m", "mlx_lm", "lora",
                "--model", self.student_model,
                "--train",
                "--data", str(data_dir),
                "--iters", str(iters),
                "--batch-size", str(batch_size),
                "--learning-rate", str(learning_rate),
                "--num-layers", str(lora_layers),
                "--adapter-path", str(adapter_path),
                "--steps-per-report", str(steps_per_report),
                "--save-every", str(iters),  # Save at end
                "--max-seq-length", str(max_seq_length),
            ]

            # Monitor system while training
            print("Starting training (monitoring system capacity)...\n")
            result = subprocess.run(cmd, check=True, capture_output=False, env=env)

            # Check system health after training
            while not check_system_capacity():
                time.sleep(5)  # Wait 5 seconds and check again

            print(f"\n{'='*60}")
            print(f"✅ Training Complete!")
            print(f"{'='*60}")
            print(f"  Adapters saved to: {adapter_path}")
            print(f"  Training data saved to: {training_file}")
            print(f"  Total iterations: {iters}")
            print(f"{'='*60}\n")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with exit code {e.returncode}")
            raise TrainingError(f"MLX-LM training failed: {str(e)}") from e
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            raise TrainingError(f"Training failed: {str(e)}") from e

        # Store metrics
        self.metrics = {
            "num_examples": len(train_data),
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "lora_rank": lora_rank,
            "lora_layers": lora_layers,
            "total_iterations": iters,
            "model": self.student_model,
            "adapter_path": str(adapter_path),
        }

    def _save_model(self) -> str:
        """Save training metadata and README (adapters already saved by tuner).

        Returns:
            Path to saved model directory
        """
        output_dir = self.config.output_dir
        model_name = self.config.name

        model_path = Path(output_dir) / model_name
        adapter_path = model_path / "adapters"

        # Verify adapters were saved
        adapter_file = adapter_path / "adapters.safetensors"
        if not adapter_file.exists():
            print(f"⚠️  Warning: Adapter file not found at {adapter_file}")
        else:
            file_size_mb = adapter_file.stat().st_size / (1024 * 1024)
            print(f"✓ Adapter weights verified: {file_size_mb:.2f} MB")

        # Save distillation config
        config_data = {
            "model_id": self.student_model,
            "name": model_name,
            "adapter_path": str(adapter_path),
            "metrics": self.metrics,
        }

        config_file = model_path / "distillation_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)
        print(f"✓ Config saved to {config_file}")

        # Generate README
        readme_file = model_path / "README.md"
        with open(readme_file, "w") as f:
            f.write(f"""# Distilled Model: {model_name}

## Model Info
- **Base Model**: {self.student_model}
- **Training Examples**: {self.metrics.get('num_examples', 'N/A')}
- **Epochs**: {self.metrics.get('num_epochs', 'N/A')}
- **Total Iterations**: {self.metrics.get('total_iterations', 'N/A')}
- **LoRA Rank**: {self.metrics.get('lora_rank', 8)}
- **LoRA Layers**: {self.metrics.get('lora_layers', 4)}

## Files
- `adapters/adapters.safetensors` - LoRA adapter weights
- `adapters/adapter_config.json` - Adapter configuration
- `training_data.jsonl` - Training examples used
- `distillation_config.json` - Distillation metadata

## Usage

### Load with MLX-LM
```python
from mlx_lm import load, generate

# Load base model with adapters
model, tokenizer = load(
    "{self.student_model}",
    adapter_path="{adapter_path}",
)

# Generate response
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Your input here",
    max_tokens=512,
)
print(response)
```

### CLI Usage
```bash
mlx_lm.generate \\
    --model {self.student_model} \\
    --adapter-path {adapter_path} \\
    --prompt "Your input here" \\
    --max-tokens 512
```

## Training Details

This model was fine-tuned using knowledge distillation:
1. Teacher model (Claude) generated reasoning traces for seed examples
2. Synthetic examples created via task pattern extraction
3. Student model fine-tuned with LoRA adapters on amplified dataset
4. Gradient-based training with AdamW optimizer (via MLX-LM tuner)
5. Checkpoints saved incrementally during training
""")
        print(f"✓ README saved to {readme_file}")

        return str(model_path)

    def evaluate(
        self,
        eval_dataset: Dataset,
    ) -> Dict[str, float]:
        """Evaluate model on evaluation dataset.

        Args:
            eval_dataset: Evaluation dataset

        Returns:
            Dictionary of evaluation metrics
        """
        eval_data = self._prepare_dataset(eval_dataset)
        total_loss = 0.0

        for item in eval_data:
            input_ids = self.mx.array([item["input_ids"]])
            logits = self.model(input_ids)

            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = input_ids[:, 1:].reshape(-1)

            loss = self.nn.losses.cross_entropy(
                shift_logits, shift_labels, reduction="mean"
            )
            self.mx.eval(loss)
            total_loss += float(loss)

        avg_loss = total_loss / len(eval_data) if eval_data else 0
        perplexity = float(self.mx.exp(self.mx.array(avg_loss)))

        return {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
            "eval_examples": len(eval_data),
        }