"""Example of using Qwen as the student model instead of Llama-3-8B."""

from nanodistill import distill

# ============================================================================
# QWEN MODELS AVAILABLE IN MLX-COMMUNITY
# ============================================================================

QWEN_MODELS = {
    # Small/Fast (Recommended for Apple Silicon)
    "qwen-0.5b": "mlx-community/Qwen-0.5B-Chat-4bit",
    "qwen-1.8b": "mlx-community/Qwen-1.8B-Chat-4bit",
    "qwen-4b": "mlx-community/Qwen-4B-Chat-4bit",

    # Medium (Balance speed/quality)
    "qwen-7b": "mlx-community/Qwen-7B-Chat-4bit",
    "qwen-14b": "mlx-community/Qwen-14B-Chat-4bit",

    # Other models for comparison
    "llama-3-8b": "mlx-community/Llama-3-8B-Instruct-4bit",
    "mistral-7b": "mlx-community/Mistral-7B-Instruct-v0.1-4bit",
}

# ============================================================================
# OPTION 1: Use Small Qwen (Recommended for Apple Silicon)
# ============================================================================

def distill_with_qwen_small():
    """Use Qwen-0.5B (smallest, fastest)."""

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
        name="math-tutor-qwen-0.5b",
        seed=seed_data,
        instruction="You are a helpful math tutor. Show your reasoning step-by-step.",
        teacher="claude-sonnet-4-5",
        student="mlx-community/Qwen-0.5B-Chat-4bit",  # ← Qwen 0.5B
        augment_factor=50,
    )

    return result


# ============================================================================
# OPTION 2: Use Medium Qwen (Best balance)
# ============================================================================

def distill_with_qwen_medium():
    """Use Qwen-7B (better quality than 0.5B, still reasonably fast)."""

    seed_data = [
        {"input": "What is photosynthesis?", "output": "Process where plants convert light into chemical energy"},
        {"input": "What is DNA?", "output": "Double helix molecule that carries genetic information"},
        # ... more biology examples
    ]

    result = distill(
        name="biology-tutor-qwen-7b",
        seed=seed_data,
        instruction="You are a biology expert. Explain concepts clearly and accurately.",
        teacher="claude-sonnet-4-5",
        student="mlx-community/Qwen-7B-Chat-4bit",  # ← Qwen 7B
        augment_factor=50,
    )

    return result


# ============================================================================
# OPTION 3: Compare Different Models
# ============================================================================

def compare_models():
    """Run distillation with different student models and compare."""

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

    models = {
        "Qwen 0.5B": "mlx-community/Qwen-0.5B-Chat-4bit",
        "Qwen 1.8B": "mlx-community/Qwen-1.8B-Chat-4bit",
        "Qwen 4B": "mlx-community/Qwen-4B-Chat-4bit",
        "Qwen 7B": "mlx-community/Qwen-7B-Chat-4bit",
        "Llama 3-8B": "mlx-community/Llama-3-8B-Instruct-4bit",
    }

    results = {}

    for model_name, model_id in models.items():
        print(f"\n📚 Training with {model_name}...")
        print(f"   Model: {model_id}")

        result = distill(
            name=f"math-tutor-{model_name.replace(' ', '-').lower()}",
            seed=seed_data,
            instruction="You are a helpful math tutor. Show your reasoning step-by-step.",
            teacher="claude-sonnet-4-5",
            student=model_id,
            augment_factor=50,
        )

        results[model_name] = {
            "model_path": result.model_path,
            "metrics": result.metrics,
        }

        print(f"✅ Done: {result.model_path}")

    return results


# ============================================================================
# MODEL COMPARISON TABLE
# ============================================================================

def print_model_comparison():
    """Print comparison of available student models."""

    comparison = """
    ┌────────────────────────────────────────────────────────────────────────┐
    │ STUDENT MODEL COMPARISON (All in MLX-Community)                        │
    ├─────────────────┬──────────┬──────────┬────────────┬──────────────────┤
    │ Model           │ Size     │ Speed    │ Quality    │ Best For         │
    ├─────────────────┼──────────┼──────────┼────────────┼──────────────────┤
    │ Qwen-0.5B       │ 500MB    │ ⭐⭐⭐⭐⭐ │ ⭐⭐       │ Fast inference   │
    │                 │          │ ~300 t/s │            │ Edge devices     │
    ├─────────────────┼──────────┼──────────┼────────────┼──────────────────┤
    │ Qwen-1.8B       │ 1.8GB    │ ⭐⭐⭐⭐  │ ⭐⭐⭐    │ Good balance     │
    │                 │          │ ~200 t/s │            │ Most M1/M2 Macs  │
    ├─────────────────┼──────────┼──────────┼────────────┼──────────────────┤
    │ Qwen-4B         │ 4GB      │ ⭐⭐⭐   │ ⭐⭐⭐⭐  │ Better quality   │
    │                 │          │ ~100 t/s │            │ More reasoning   │
    ├─────────────────┼──────────┼──────────┼────────────┼──────────────────┤
    │ Qwen-7B         │ 7GB      │ ⭐⭐    │ ⭐⭐⭐⭐⭐ │ Production use   │
    │                 │          │ ~50 t/s  │            │ Best quality     │
    ├─────────────────┼──────────┼──────────┼────────────┼──────────────────┤
    │ Llama-3-8B      │ 8GB      │ ⭐⭐    │ ⭐⭐⭐⭐⭐ │ Default choice   │
    │ (Default)       │          │ ~50 t/s  │            │ Well-tested      │
    ├─────────────────┼──────────┼──────────┼────────────┼──────────────────┤
    │ Mistral-7B      │ 7GB      │ ⭐⭐⭐  │ ⭐⭐⭐⭐  │ Fast & capable   │
    │                 │          │ ~80 t/s  │            │ Good alternative │
    └─────────────────┴──────────┴──────────┴────────────┴──────────────────┘

    Notes:
    • Speeds are approximate on M1/M2/M3
    • All models are 4-bit quantized
    • Download happens once, then cached
    • t/s = tokens per second

    RECOMMENDATIONS:
    ─────────────────

    M1/M2 with 16GB RAM:
    ✅ Start with Qwen-1.8B or Qwen-4B
       (Fast training, good quality, lower memory)

    M1 Pro/Max or M2 with 32GB RAM:
    ✅ Use Qwen-7B or Llama-3-8B
       (Better quality, still manages memory well)

    Want fastest inference?
    ✅ Use Qwen-0.5B
       (50GB+)

    Want best quality?
    ✅ Use Qwen-7B or Llama-3-8B
       (Takes more time but better reasoning)
    """

    print(comparison)


# ============================================================================
# INFERENCE WITH QWEN
# ============================================================================

def test_qwen_inference(model_path: str):
    """Test inference with a trained Qwen model."""

    from mlx_lm import load, generate

    print(f"📂 Loading Qwen model from: {model_path}")

    model, tokenizer = load(model_path)

    test_prompts = [
        "What is 7+8?",
        "What is 15-6?",
        "What is 3×4?",
    ]

    for prompt in test_prompts:
        print(f"\n💬 Prompt: {prompt}")

        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=200,
        )

        print(f"Response: {response}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("NanoDistill: Using Qwen as Student Model")
    print("=" * 70)

    # Show comparison
    print_model_comparison()

    # Choose option
    print("\n" + "=" * 70)
    print("CHOOSE YOUR OPTION:")
    print("=" * 70)
    print("\n1. Quick test with Qwen-0.5B (smallest, fastest)")
    print("   python -c \"from examples.using_qwen import distill_with_qwen_small; distill_with_qwen_small()\"")

    print("\n2. Better quality with Qwen-7B")
    print("   python -c \"from examples.using_qwen import distill_with_qwen_medium; distill_with_qwen_medium()\"")

    print("\n3. Compare all models")
    print("   python -c \"from examples.using_qwen import compare_models; compare_models()\"")

    print("\n4. View model comparison")
    print("   python examples/using_qwen.py")

    print("\n" + "=" * 70)
    print("\nOr modify distill() directly in your code:")
    print("""
    result = distill(
        name="my-model",
        seed=seed_data,
        instruction="...",
        student="mlx-community/Qwen-4B-Chat-4bit",  # Change this line
    )
    """)
