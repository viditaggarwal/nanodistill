from src.nanodistill.distiller.trainer import MLXTrainer
from types import SimpleNamespace

# Test config
config = SimpleNamespace(
    name="test-validation",
    output_dir="./test_output",
    num_train_epochs=1,
    learning_rate=2e-4,
    batch_size=2,
    max_seq_length=512,
    lora_rank=8,
    lora_layers=4,
)

# Initialize trainer
trainer = MLXTrainer(
    student_model="mlx-community/Qwen2.5-7B-Instruct-4bit",
    config=config,
)

print("✓ Trainer initialized successfully")

# Test model loading
trainer._load_model()
print("✓ Model loaded successfully")

# Test LoRA setup
trainer._setup_lora()
print("✓ LoRA setup completed")

# Create minimal training data
from datasets import Dataset
test_data = Dataset.from_dict({
    "input": ["Test input"],
    "thinking": ["Test thinking"],
    "output": ['{"test": "output"}'],
})

# Test data preparation
prepared = trainer._prepare_dataset(test_data)
print(f"✓ Data prepared: {len(prepared)} examples")

print("\n✅ All validation tests passed!")
print("Ready to run full training.")
