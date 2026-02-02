"""Example of how to download the student model and use it for inference."""

import os
from pathlib import Path

# ============================================================================
# PART 1: Download & Setup
# ============================================================================

# Option A: Automatic Download (MLX-LM handles it)
# When you run distill(), MLX-LM automatically downloads the model
# The model is cached in: ~/.cache/huggingface/hub/

# Option B: Manual Pre-download
def download_student_model():
    """Pre-download the student model before running distill()."""
    try:
        from mlx_lm import load

        model_id = "mlx-community/Llama-3-8B-Instruct-4bit"

        print(f"📥 Downloading {model_id}...")
        print("   This is a one-time download (~4GB for 8B model)")
        print("   Cached to: ~/.cache/huggingface/hub/\n")

        # This downloads and caches the model
        model, tokenizer = load(model_id)

        print("✅ Model downloaded and cached successfully!")
        print(f"   Model type: {type(model)}")
        print(f"   Tokenizer type: {type(tokenizer)}")

        return model, tokenizer

    except ImportError:
        print("❌ MLX-LM not installed")
        print("   Install with: pip install mlx mlx-lm")
        raise


# Option C: Using HuggingFace CLI
def download_with_huggingface_cli():
    """Download using huggingface-cli (requires huggingface-hub)."""
    os.system(
        "huggingface-cli download mlx-community/Llama-3-8B-Instruct-4bit "
        "--local-dir ./models/llama-3-8b-instruct-4bit"
    )


# ============================================================================
# PART 2: Run Distillation (from basic_usage.py)
# ============================================================================

def run_distillation():
    """Run the distillation pipeline."""
    from nanodistill import distill

    seed_data = [
        {"input": "What is 2+2?", "output": "4"},
        {"input": "What is 3+5?", "output": "8"},
        {"input": "What is 10-4?", "output": "6"},
        {"input": "What is 5×3?", "output": "15"},
        {"input": "What is 20÷4?", "output": "5"},
        {"input": "What is 7+8?", "output": "15"},
        {"input": "What is 100-25?", "output": "75"},
        {"input": "What is 6×6?", "output": "36"},
        {"input": "What is 12÷3?", "output": "4"},
        {"input": "What is 9+9?", "output": "18"},
    ]

    result = distill(
        name="math-tutor-v1",
        seed=seed_data,
        instruction=(
            "You are a helpful math tutor. "
            "For each arithmetic question, show your step-by-step reasoning "
            "and then provide the final answer."
        ),
        teacher="claude-sonnet-4-5",  # Uses ANTHROPIC_API_KEY
        student="mlx-community/Llama-3-8B-Instruct-4bit",
        augment_factor=50,
        output_dir="./outputs",
    )

    return result


# ============================================================================
# PART 3: Use the Fine-Tuned Model for Inference
# ============================================================================

def inference_with_mlx(model_path: str, prompt: str):
    """Run inference using MLX-LM on the fine-tuned model."""
    try:
        from mlx_lm import load, generate

        print(f"📂 Loading model from: {model_path}")

        # Load the fine-tuned model
        model, tokenizer = load(model_path)

        print(f"💬 Generating response for: {prompt}\n")

        # Generate response
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=256,
            temperature=0.7,
        )

        print("Response:")
        print(response)

        return response

    except ImportError:
        print("❌ MLX-LM not installed")
        raise


def inference_with_transformers(model_path: str, prompt: str):
    """Run inference using HuggingFace transformers (slower, CPU friendly)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"📂 Loading model from: {model_path}")

        # Load the fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Move to appropriate device
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)

        print(f"🖥️  Using device: {device}")
        print(f"💬 Generating response for: {prompt}\n")

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Response:")
        print(response)

        return response

    except ImportError:
        print("❌ transformers not installed")
        print("   Install with: pip install transformers torch")
        raise


# ============================================================================
# PART 4: Use with Ollama (Easiest for local inference!)
# ============================================================================

def convert_to_gguf_for_ollama(model_path: str, output_path: str = None):
    """Convert MLX model to GGUF format for use with Ollama.

    This requires llama.cpp to be installed.
    """
    import subprocess

    if output_path is None:
        output_path = f"{model_path}/model-Q4_K_M.gguf"

    print(f"📦 Converting {model_path} to GGUF format...")
    print(f"   Output: {output_path}")
    print("   This may take a few minutes...\n")

    try:
        # Note: Requires llama.cpp to be set up
        # See: https://github.com/ggerganov/llama.cpp

        subprocess.run([
            "python3",
            "llama.cpp/convert.py",
            model_path,
            "--outtype", "q4_k_m",
            "--outfile", output_path,
        ], check=True)

        print(f"\n✅ GGUF conversion complete!")
        print(f"   Ready for Ollama: {output_path}")

    except FileNotFoundError:
        print("❌ llama.cpp not found")
        print("   Clone from: https://github.com/ggerganov/llama.cpp")
        raise


def use_with_ollama(model_path: str, model_name: str = "math-tutor"):
    """Create Modelfile and use model with Ollama."""

    modelfile = f"""FROM {model_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM You are a helpful math tutor. Show your reasoning step-by-step.
"""

    print(f"📋 Modelfile for Ollama:")
    print(modelfile)
    print("\n📝 To use with Ollama:")
    print(f"   1. Save the Modelfile above")
    print(f"   2. Run: ollama create {model_name} -f Modelfile")
    print(f"   3. Run: ollama run {model_name}")
    print(f"   4. Ask: 'What is 5+3?'")


# ============================================================================
# MAIN: Complete Workflow
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NanoDistill: Complete Workflow Example")
    print("=" * 70)

    # Step 1: Download student model (optional, auto-downloads during distill)
    print("\n📥 STEP 1: Download Student Model")
    print("   MLX models are auto-downloaded when first used.")
    print("   Location: ~/.cache/huggingface/hub/")
    print("   Size: ~4GB for 8B Instruct model\n")

    # Step 2: Run distillation
    print("🚀 STEP 2: Run Distillation")
    print("   Set ANTHROPIC_API_KEY before running:")
    print("   export ANTHROPIC_API_KEY='sk-ant-...'\n")

    print("   Then run:")
    print("   python examples/inference_example.py\n")

    # Uncomment to actually run:
    # result = run_distillation()
    # model_path = str(result.model_path)

    # Step 3: Inference
    print("💬 STEP 3: Inference Options\n")

    print("   Option A: MLX-LM (Fastest on Apple Silicon)")
    print("   ─────────────────────────────────────────")
    print("   from examples.inference_example import inference_with_mlx")
    print("   inference_with_mlx(model_path, 'What is 5+3?')\n")

    print("   Option B: HuggingFace Transformers (Compatible)")
    print("   ────────────────────────────────────────────────")
    print("   from examples.inference_example import inference_with_transformers")
    print("   inference_with_transformers(model_path, 'What is 5+3?')\n")

    print("   Option C: Ollama (Easiest for production)")
    print("   ─────────────────────────────────────────")
    print("   1. Convert to GGUF: convert_to_gguf_for_ollama(model_path)")
    print("   2. Create Modelfile: use_with_ollama(model_path)")
    print("   3. Run: ollama create my-tutor -f Modelfile")
    print("   4. Use: ollama run my-tutor\n")

    # Detailed setup instructions
    print("=" * 70)
    print("DETAILED SETUP INSTRUCTIONS")
    print("=" * 70)

    print("\n1️⃣  BEFORE RUNNING DISTILLATION:")
    print("   ─────────────────────────────────")
    print("   • Set your API key:")
    print("     export ANTHROPIC_API_KEY='sk-ant-...'")
    print("   • Optional: Pre-download model (will auto-download if skipped)")
    print("     python -c \"from examples.inference_example import download_student_model; download_student_model()\"")

    print("\n2️⃣  RUN DISTILLATION:")
    print("   ─────────────────────")
    print("   python examples/basic_usage.py")
    print("   ")
    print("   This will:")
    print("   • Generate CoT traces from Claude")
    print("   • Extract task policy")
    print("   • Generate 500 synthetic examples")
    print("   • Fine-tune Llama-3-8B on Apple Silicon")
    print("   • Save to: ./outputs/math-tutor-v1/")

    print("\n3️⃣  USE THE FINE-TUNED MODEL:")
    print("   ──────────────────────────────")
    print("   # Quick test with MLX")
    print("   python -c \"")
    print("   from examples.inference_example import inference_with_mlx")
    print("   inference_with_mlx('./outputs/math-tutor-v1', 'What is 7×8?')\"")

    print("\n4️⃣  DEPLOY WITH OLLAMA (Recommended):")
    print("   ──────────────────────────────────────")
    print("   # Install Ollama: https://ollama.ai")
    print("   # Then use the model")
    print("   ollama run math-tutor")

    print("\n" + "=" * 70)
