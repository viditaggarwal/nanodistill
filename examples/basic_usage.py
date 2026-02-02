"""Basic example of using NanoDistill to create a simple math tutor."""

from nanodistill import distill

# Example seed data - 10 simple math examples
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

# Run distillation
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
    augment_factor=50,  # 10 seeds × 50 = 500 training examples
    output_dir="./outputs",
)

print("\n" + "=" * 60)
print("✅ Distillation Complete!")
print("=" * 60)
print(f"Model saved to: {result.model_path}")
print(f"Training examples: {result.metrics['training_examples']}")
print(f"Teacher model: {result.metrics['teacher_model']}")
print(f"Student model: {result.metrics['student_model']}")
print("=" * 60)

# Next steps
print("\nNext steps:")
print("1. Test the model locally using MLX or llama.cpp")
print("2. Compare outputs against the teacher (Claude) on a test set")
print("3. Evaluate quality and adjust as needed")
print("\nExample inference code:")
print("""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = "{result.model_path}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Your inference code here
""")
