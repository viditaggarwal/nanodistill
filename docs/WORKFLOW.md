# Complete Workflow: From Seed Data to Deployment

Visual guide to the entire NanoDistill process.

## High-Level Flow

```
Your Seed Data (10 examples)
        ↓
        ├─→ [Download] Student Model (4GB, one-time)
        │   └─→ ~/.cache/huggingface/hub/
        │
        ├─→ [Generate] CoT Traces (Claude API)
        │   📊 Input: 10 examples
        │   📊 Output: 10 reasoning traces
        │
        ├─→ [Extract] Task Policy (Claude API)
        │   📊 Analyzes patterns in your examples
        │
        ├─→ [Generate] Synthetic Examples (Claude API)
        │   📊 Input: 10 original examples
        │   📊 Output: 490 new diverse examples
        │
        ├─→ [Merge] All Examples
        │   📊 500 total examples with reasoning
        │
        ├─→ [Train] Student Model (MLX-LM)
        │   📊 Fine-tunes Llama-3-8B
        │   📊 Uses LoRA for efficiency
        │
        └─→ Your Fine-Tuned Model ✅
            └─→ ./outputs/math-tutor-v1/
```

## Detailed Stage Breakdown

### Stage 1: Download Model (First Run Only)

```
┌─────────────────────────────────────────────────┐
│ distill() starts                                │
│ ↓                                               │
│ Check: Is model cached?                         │
│ ├─ YES → Skip download, use cached              │
│ └─ NO  → Download from HuggingFace              │
│         (4GB for Llama-3-8B-Instruct-4bit)     │
│         ↓                                       │
│         Save to ~/.cache/huggingface/hub/       │
│         ↓                                       │
│         ✅ Ready to use                         │
└─────────────────────────────────────────────────┘

First Run:  ~10-15 minutes (includes download)
Later Runs: < 1 minute (uses cache)
```

### Stage 2: Generate CoT Traces (Your Seeds)

```
┌──────────────────────────────────────────────┐
│ Your 10 Seed Examples                        │
├──────────────────────────────────────────────┤
│ {input: "What is 2+2?",  output: "4"}       │
│ {input: "What is 3+5?",  output: "8"}       │
│ ... (8 more examples)                        │
└──────────────────────────────────────────────┘
        ↓
        [Send to Claude]
        ↓
┌──────────────────────────────────────────────┐
│ Claude Generates Reasoning                   │
├──────────────────────────────────────────────┤
│ Input: "What is 2+2?"                        │
│ Thinking: "2 plus 2 equals 4"                │
│ Output: "4"                                  │
│                                              │
│ ... (for each of your 10 examples)          │
└──────────────────────────────────────────────┘
        ↓
        ✅ 10 Chain-of-Thought traces
        Saved to: traces_cot.jsonl
```

### Stage 3: Extract Task Policy

```
┌──────────────────────────────────────────────┐
│ Analyze Your 10 Examples + Reasoning         │
├──────────────────────────────────────────────┤
│ What patterns do we see?                     │
│ • Task: Simple arithmetic                    │
│ • Input: Basic math questions                │
│ • Output: Numeric answers                    │
│ • Reasoning: Step-by-step calculation        │
│ • Difficulty: Beginner level                 │
│ • Key rules: Show all steps, be clear       │
└──────────────────────────────────────────────┘
        ↓
        ✅ Task Policy Extracted
        (Used to guide synthetic generation)
```

### Stage 4: Generate Synthetic Examples

```
┌──────────────────────────────────────────────┐
│ Claude Generates 490 NEW Examples            │
│ (Constrained by extracted policy)            │
├──────────────────────────────────────────────┤
│ "Generate examples following this pattern:"  │
│ • Task: Simple arithmetic                    │
│ • Input: Basic math questions                │
│ • Output: Numeric answers                    │
│ • Similar difficulty to seeds                │
│ • But NEW and DIVERSE                        │
│                                              │
│ Result examples:                             │
│ {input: "What is 12+8?", output: "20"}     │
│ {input: "What is 25-10?", output: "15"}    │
│ {input: "What is 3×4?", output: "12"}      │
│ ... (487 more examples)                      │
└──────────────────────────────────────────────┘
        ↓
        ✅ 490 Synthetic Examples Generated
```

### Stage 5: Generate CoT for Synthetic Data

```
┌──────────────────────────────────────────────┐
│ For Each Synthetic Example                   │
├──────────────────────────────────────────────┤
│ Input: "What is 12+8?"                       │
│         ↓                                    │
│         [Claude generates reasoning]         │
│         ↓                                    │
│ Thinking: "12 + 8 = 20"                     │
│ Output: "20"                                 │
│                                              │
│ Repeat for all 490 synthetic examples        │
└──────────────────────────────────────────────┘
        ↓
        ✅ 490 Synthetic Examples with CoT
```

### Stage 6: Merge & Prepare Training Data

```
┌────────────────────────────────────────────────┐
│ Combine All Training Data                     │
├────────────────────────────────────────────────┤
│ Your Seeds:          10 examples              │
│ Synthetic:          490 examples              │
│                     ──────────                │
│ Total:              500 examples              │
│                                               │
│ Each example has:                             │
│ • Input (question)                            │
│ • Thinking (reasoning process)                │
│ • Output (answer)                             │
└────────────────────────────────────────────────┘
        ↓
        ✅ 500 Training Examples Ready
```

### Stage 7: Train Student Model

```
┌────────────────────────────────────────────────┐
│ MLX-LM Fine-tunes Llama-3-8B                  │
├────────────────────────────────────────────────┤
│ Loading model...                               │
│ Configuring LoRA (parameter efficiency)       │
│ Training on 500 examples                      │
│ Hardware: Apple Silicon (MPS backend)         │
│ Epochs: 2                                      │
│ Batch size: auto-optimized by MLX             │
│                                               │
│ Progress:                                      │
│ Epoch 1: [████████████      ] Loss: 0.45      │
│ Epoch 2: [████████████      ] Loss: 0.32      │
│                                               │
│ ✅ Training complete!                         │
└────────────────────────────────────────────────┘
        ↓
        ✅ Fine-tuned Model Saved
        ./outputs/math-tutor-v1/model/
```

## Complete Timeline

```
Time    Stage                          Duration   Cumulative
────────────────────────────────────────────────────────────
0:00    Start                          -          0:00
0:00    Download model (first run)     10 min     10:00
0:10    Generate CoT traces            5 min      15:00
0:15    Extract policy                 2 min      17:00
0:17    Generate synthetic examples    5 min      22:00
0:22    Generate CoT for synthetic     3 min      25:00
0:25    Prepare training data          1 min      26:00
0:26    Train model                    3 min      29:00
0:29    Save model                     1 min      30:00
0:30    Done!                          ✅         30:00

Notes:
- First run: ~30 minutes (includes 4GB download)
- Subsequent runs: ~20 minutes (cached model)
- All times on M1/M2 (M3 will be faster)
```

## File Structure After Completion

```
project/
├── my_distillation.py          # Your script
├── seeds.json                  # Your 10 examples
│
└── outputs/
    └── math-tutor-v1/
        ├── model/              # ← FINAL MODEL (for inference)
        │   ├── adapters.npz   # LoRA weights
        │   ├── config.json
        │   ├── tokenizer.json
        │   └── ...
        │
        ├── traces_cot.jsonl    # Original 10 + reasoning
        └── traces_amplified.jsonl  # All 500 training examples
```

## Usage After Training

### Option A: Quick Test (Local)

```
Your Trained Model
        ↓
    Load with MLX
        ↓
    Send prompt: "What is 7+8?"
        ↓
    Model generates response
        ↓
    Display: "The answer is 15"
```

### Option B: Serve with Ollama (Production)

```
Your Trained Model
        ↓
    Convert to GGUF format
        ↓
    Create Modelfile
        ↓
    Run: ollama create math-tutor -f Modelfile
        ↓
    Start server: ollama serve
        ↓
    Access via API or web interface
        ↓
    Query: curl http://localhost:11434/api/generate
```

### Option C: Web Application

```
Your Trained Model
        ↓
    Start MLX web server
        ↓
    Use REST API to query
        ↓
    Integrate into application
```

## Key Points

### Model Download
- ✅ Happens automatically on first use
- ✅ Only once (cached for reuse)
- ✅ No manual steps needed

### Training Process
- ✅ Uses your seed examples (10)
- ✅ Generates synthetic examples via Claude (490)
- ✅ Fine-tunes with MLX-LM on Apple Silicon
- ✅ Creates an efficient model (~5GB)

### After Training
- ✅ Model ready to use immediately
- ✅ Can test locally (MLX)
- ✅ Can deploy to production (Ollama)
- ✅ Can integrate into apps (API)

## Memory Usage Over Time

```
12GB ├────────────────────────────────────────────
     │                      ✓ Training
11GB ├─────────────         ├────────────
     │                      │            ✓ Model saved
10GB ├─────────────────────┤
     │  Downloaded model    │
9GB  ├──────────────────────┤
     │                      │
8GB  ├──────────────────────┼────────────── Final model uses 5GB
     │                      │
7GB  ├──────────────────────┤
     │
6GB  ├──────────────────────────────────┘
     │      Training peaks
5GB  ├──────────────────────
     │
     └─────────────────────────────────────────────
       0min    5min    10min   15min   20min   25min
```

## Failure Recovery

```
If training is interrupted:

Interrupt at:              Recovery:
─────────────────────────────────────────────
Download                   → Resume from ~/.cache
CoT generation            → Skip, regenerate
Policy extraction         → Skip, regenerate
Synthetic generation      → Skip, regenerate
CoT for synthetic          → Skip, regenerate
Training                   → Restart training
```

All data is saved, so you can resume!

---

**Ready to start?** See QUICK_START.md for step-by-step instructions.
