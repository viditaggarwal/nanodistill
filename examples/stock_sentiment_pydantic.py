"""
Stock sentiment distillation with Pydantic output.

Uses mlx-community/Llama-3-8B-Instruct-4bit as student and queries the
(distilled or base) model with structured output via StockAnalysis schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field
from enum import Enum
from typing import Literal

from nanodistill import distill, DistillationResult

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class Sentiment(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class StockAnalysis(BaseModel):
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 confidence score")
    reasoning: str = Field(description="Brief explanation")
    key_factors: list[str] = Field(description="Main factors influencing sentiment")
    price_direction: Literal["up", "down", "sideways"]
    risk_level: Literal["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Seed data (hard examples where small models often fail)
# ---------------------------------------------------------------------------

seed_data = [
    {
        "input": "Tesla down 8% today. Elon says 'everything is fine, best quarter ever coming'. Sure.",
        "output": json.dumps({
            "sentiment": "bearish",
            "confidence": 0.75,
            "reasoning": "Sarcastic tone suggests disbelief despite positive statement. Actual 8% drop is significant negative signal.",
            "key_factors": ["8% price drop", "sarcasm indicating skepticism", "disconnect between claim and reality"],
            "price_direction": "down",
            "risk_level": "high",
        }),
    },
    {
        "input": "AAPL beats earnings by 12% but warns of supply chain issues in China affecting Q2 guidance. Stock flat after-hours.",
        "output": json.dumps({
            "sentiment": "neutral",
            "confidence": 0.80,
            "reasoning": "Strong earnings beat offset by forward-looking concerns. Market reaction (flat) confirms mixed sentiment.",
            "key_factors": ["12% earnings beat", "Q2 guidance warning", "supply chain risks", "neutral market reaction"],
            "price_direction": "sideways",
            "risk_level": "medium",
        }),
    },
    {
        "input": "NVDA IV crush post-earnings despite 200% YoY growth. Premium sellers loving this.",
        "output": json.dumps({
            "sentiment": "bullish",
            "confidence": 0.70,
            "reasoning": "IV crush indicates volatility decreased but doesn't negate strong 200% growth. Bullish fundamentals despite options dynamics.",
            "key_factors": ["200% YoY growth", "strong fundamentals", "reduced volatility post-earnings"],
            "price_direction": "up",
            "risk_level": "medium",
        }),
    },
    {
        "input": "Goldman upgrades TSLA to buy ($300 target). Morgan Stanley downgrades to sell ($150 target). Current price: $240.",
        "output": json.dumps({
            "sentiment": "neutral",
            "confidence": 0.60,
            "reasoning": "Conflicting analyst opinions with wide target range. Current price between targets suggests uncertainty.",
            "key_factors": ["analyst disagreement", "wide price target range", "no clear consensus"],
            "price_direction": "sideways",
            "risk_level": "high",
        }),
    },
    {
        "input": "META announces 'cost optimization' and 'strategic workforce realignment' affecting 15% of staff.",
        "output": json.dumps({
            "sentiment": "bearish",
            "confidence": 0.85,
            "reasoning": "Corporate euphemisms for layoffs. 15% reduction signals serious issues despite positive framing.",
            "key_factors": ["15% workforce reduction", "euphemistic language hiding layoffs", "operational challenges"],
            "price_direction": "down",
            "risk_level": "high",
        }),
    },
    {
        "input": "AMZN up 3% on cloud revenue growth. Note: this is after falling 25% over past 3 months.",
        "output": json.dumps({
            "sentiment": "neutral",
            "confidence": 0.65,
            "reasoning": "Short-term positive signal but context shows deeper weakness. 3% gain doesn't reverse 25% decline trend.",
            "key_factors": ["3% daily gain", "25% decline over 3 months", "cloud growth positive but insufficient"],
            "price_direction": "sideways",
            "risk_level": "medium",
        }),
    },
    {
        "input": "CEO bought $2M in stock. CFO sold $8M. COO sold $5M. All transactions this week.",
        "output": json.dumps({
            "sentiment": "bearish",
            "confidence": 0.80,
            "reasoning": "Net insider selling significantly outweighs buying. Multiple executives selling is stronger signal than single CEO buy.",
            "key_factors": ["net $11M insider selling", "multiple executives selling", "selling outweighs CEO confidence signal"],
            "price_direction": "down",
            "risk_level": "high",
        }),
    },
    {
        "input": "Fed signals rate cuts. Tech stocks rally 2%, value stocks down 1.5%. MSFT up 2.5%.",
        "output": json.dumps({
            "sentiment": "bullish",
            "confidence": 0.85,
            "reasoning": "MSFT outperforming sector average during favorable macro environment. Rate cuts benefit tech growth stocks.",
            "key_factors": ["rate cut signals", "tech sector rally", "MSFT outperforming sector", "favorable macro"],
            "price_direction": "up",
            "risk_level": "low",
        }),
    },
    {
        "input": "GOOGL reports 5% ad revenue growth. Industry average is 12%. Stock down 4% pre-market.",
        "output": json.dumps({
            "sentiment": "bearish",
            "confidence": 0.90,
            "reasoning": "Underperforming sector by significant margin. Market reaction confirms disappointment with below-average growth.",
            "key_factors": ["5% growth vs 12% industry average", "underperformance", "negative market reaction"],
            "price_direction": "down",
            "risk_level": "medium",
        }),
    },
    {
        "input": "DOJ 'monitoring' AAPL app store practices. No formal investigation announced yet. Lawyers say 50/50 chance of action.",
        "output": json.dumps({
            "sentiment": "neutral",
            "confidence": 0.55,
            "reasoning": "Regulatory overhang creates uncertainty but no concrete action. 50/50 probability suggests balanced risk.",
            "key_factors": ["regulatory uncertainty", "no formal charges", "unclear timeline", "balanced probability"],
            "price_direction": "sideways",
            "risk_level": "medium",
        }),
    },
]

INSTRUCTION = """You are a financial analyst specializing in stock sentiment analysis.
Analyze the given text and extract structured sentiment information.
Consider: sarcasm, mixed signals, temporal context, comparative performance, and insider activity.
Output valid JSON matching the StockAnalysis schema."""

# ---------------------------------------------------------------------------
# JSON extraction from model output
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


def _extract_json_from_text(text: str) -> str:
    """Extract JSON string from model output (may include thinking, markdown, etc.)."""
    text = text.strip()
    # Try raw parse first
    try:
        parsed = json.loads(text)
        # If it's nested with sentiment as object, try to flatten it
        if isinstance(parsed, dict) and "sentiment" in parsed and isinstance(parsed["sentiment"], dict):
            parsed = _flatten_nested_json(parsed)
        return json.dumps(parsed)
    except json.JSONDecodeError:
        pass
    # Strip markdown code block
    if "```json" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            json_text = match.group(1).strip()
            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, dict) and "sentiment" in parsed and isinstance(parsed["sentiment"], dict):
                    parsed = _flatten_nested_json(parsed)
                return json.dumps(parsed)
            except json.JSONDecodeError:
                pass
    if "```" in text:
        match = re.search(r"```\s*([\s\S]*?)```", text)
        if match:
            json_text = match.group(1).strip()
            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, dict) and "sentiment" in parsed and isinstance(parsed["sentiment"], dict):
                    parsed = _flatten_nested_json(parsed)
                return json.dumps(parsed)
            except json.JSONDecodeError:
                pass
    # Find first { ... } balanced
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    depth = 0
    end_pos = -1
    for i, c in enumerate(text[start:], start=start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end_pos = i + 1
                break

    if end_pos == -1:
        # JSON is incomplete, try to complete it
        json_text = text[start:]
        # Count unclosed braces and close them
        depth = json_text.count("{") - json_text.count("}")
        json_text = json_text + "}" * depth
    else:
        json_text = text[start:end_pos]

    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict) and "sentiment" in parsed and isinstance(parsed["sentiment"], dict):
            parsed = _flatten_nested_json(parsed)
        return json.dumps(parsed)
    except json.JSONDecodeError:
        pass

    raise ValueError("No complete JSON object found in model output")


def _flatten_nested_json(data: dict) -> dict:
    """Flatten nested sentiment structure to match StockAnalysis schema."""
    if not isinstance(data, dict):
        return data

    result = {}

    # Extract sentiment from nested structure
    sentiment_obj = data.get("sentiment", {})
    if isinstance(sentiment_obj, dict):
        # Try to extract sentiment value - check multiple possible field names
        sentiment = (
            sentiment_obj.get("sentiment_direction") or
            sentiment_obj.get("sentiment") or
            data.get("sentiment")
        )
        # If sentiment is still negative (e.g., -0.5), interpret as bearish
        if sentiment is not None:
            if isinstance(sentiment, str):
                result["sentiment"] = sentiment.lower()
            elif isinstance(sentiment, (int, float)):
                result["sentiment"] = "bearish" if sentiment < 0 else ("bullish" if sentiment > 0 else "neutral")
    else:
        if isinstance(sentiment_obj, str):
            result["sentiment"] = sentiment_obj.lower()
        elif isinstance(sentiment_obj, (int, float)):
            result["sentiment"] = "bearish" if sentiment_obj < 0 else ("bullish" if sentiment_obj > 0 else "neutral")

    # Extract confidence
    result["confidence"] = data.get("confidence") or sentiment_obj.get("confidence", 0.5)
    if isinstance(result["confidence"], str):
        try:
            result["confidence"] = float(result["confidence"])
        except:
            result["confidence"] = 0.5

    # Extract reasoning
    result["reasoning"] = data.get("reasoning") or sentiment_obj.get("reasoning", "")

    # Extract key_factors - handle both list of strings and list of dicts
    key_factors = data.get("key_factors") or sentiment_obj.get("key_factors", [])
    if isinstance(key_factors, list):
        result["key_factors"] = []
        for factor in key_factors:
            if isinstance(factor, str):
                result["key_factors"].append(factor)
            elif isinstance(factor, dict):
                # Extract from dict - try common keys
                factor_str = factor.get("factor") or factor.get("key") or factor.get("description") or str(factor)
                result["key_factors"].append(factor_str)
    else:
        result["key_factors"] = []

    # Extract price_direction
    result["price_direction"] = data.get("price_direction") or sentiment_obj.get("price_direction", "sideways")

    # Extract risk_level
    result["risk_level"] = data.get("risk_level") or sentiment_obj.get("risk_level", "medium")

    return result


def _extract_final_output(full_response: str) -> str:
    """Get final answer from response, optionally after </thinking>."""
    if "<thinking>" in full_response and "</thinking>" in full_response:
        end_tag = full_response.find("</thinking>") + len("</thinking>")
        return full_response[end_tag:].strip()
    return full_response.strip()


# ---------------------------------------------------------------------------
# Query with Pydantic output
# ---------------------------------------------------------------------------


def query_stock_sentiment(
    model_path_or_id: str | Path,
    text: str,
    *,
    max_tokens: int = 150,
    temperature: float = 0.1,
) -> StockAnalysis:
    """Run the (distilled or base) model on input text and return parsed StockAnalysis.

    Args:
        model_path_or_id: Path to distilled model dir or HuggingFace model id
                         (e.g. "mlx-community/Llama-3-8B-Instruct-4bit")
        text: Input news/snippet to analyze
        max_tokens: Max tokens to generate (default 150 for JSON-only output)
        temperature: Sampling temperature (lower = more deterministic)

    Returns:
        Validated StockAnalysis instance

    Raises:
        ValueError: If model output cannot be parsed as StockAnalysis
    """
    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        raise ImportError("Install mlx and mlx-lm: uv pip install mlx mlx-lm") from None

    from pathlib import Path
    import json

    model_path = Path(model_path_or_id)
    adapter_path = None

    # Check if this is a distilled model directory
    if model_path.is_dir():
        config_file = model_path / "distillation_config.json"
        adapter_dir = model_path / "adapters"

        if config_file.exists():
            # Load base model ID from config
            with open(config_file, "r") as f:
                config = json.load(f)
            base_model_id = config.get("model_id")

            # Check for adapters
            if adapter_dir.exists() and (adapter_dir / "adapters.safetensors").exists():
                adapter_path = str(adapter_dir)
                print(f"Loading base model: {base_model_id}")
                print(f"Loading adapters from: {adapter_path}")
                model, tokenizer = load(base_model_id, adapter_path=adapter_path)
            else:
                print(f"⚠️  No adapters found, loading base model only")
                model, tokenizer = load(base_model_id)
        else:
            # Try loading as standard model directory
            model, tokenizer = load(str(model_path_or_id))
    else:
        # Load from HuggingFace model ID
        model, tokenizer = load(str(model_path_or_id))

    sampler = make_sampler(temp=temperature)

    # Explicitly request JSON-only output to reduce verbose generation
    prompt = f"{INSTRUCTION}\n\nInput to analyze:\n{text}\n\nRespond with ONLY valid JSON (no markdown, no explanation):"
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )

    print(f"Response: {response}")
    final = response.strip()  # Use response directly since we're requesting JSON-only
    try:
        json_str = _extract_json_from_text(final)
        return StockAnalysis.model_validate_json(json_str)
    except ValueError as e:
        raise ValueError(
            f"Model output could not be parsed as JSON. {e}\n"
            f"Raw output :\n{response}"
        ) from e


# ---------------------------------------------------------------------------
# Distillation + example usage
# ---------------------------------------------------------------------------


def run_distillation(output_dir: str = "./outputs") -> DistillationResult:
    """Run distillation for stock sentiment with Qwen2.5-3B-Instruct-4bit.

    Optimized for M4 Pro with hardware-auto-detected defaults:
    - Qwen 3B: ideal for single-task reasoning (smaller, faster, plenty of capacity)
    - augment_factor=30: generates 300 training examples from 10 seeds
    - response_model=StockAnalysis: schema enforcement + skips redundant CoT (2x fewer API calls)
    - lora_rank=16, lora_layers=8: higher expressiveness for nuanced sentiment
    - learning_rate=2e-4: standard LoRA learning rate
    - num_train_epochs=3: more passes over small focused dataset
    - batch_size=4: M4 Pro handles this easily with 24GB
    - max_seq_length=512: sufficient for stock analysis JSON output

    Note: batch_size, memory limits, and other hardware params auto-detect on M4 Pro.
    Explicit values here override auto-detection for reproducibility.
    """
    return distill(
        name="stock-sentiment-v4",
        seed=seed_data,
        instruction=INSTRUCTION,
        teacher="claude-sonnet-4-5",
        student="mlx-community/Qwen2.5-3B-Instruct-4bit",
        augment_factor=5,
        output_dir=output_dir,
        response_model=StockAnalysis,  # Schema enforcement + skip redundant CoT (2x speed)
        learning_rate=2e-4,  # Standard LoRA LR (1e-5 was too conservative)
        num_train_epochs=3,  # More epochs for smaller focused dataset
        batch_size=4,  # M4 Pro can handle this easily
        max_seq_length=512,  # Sufficient for stock analysis JSON
        lora_rank=16,  # Higher rank for nuanced sentiment reasoning
        lora_layers=8,  # Adapt more layers for better quality
    )


if __name__ == "__main__":
    import sys

    print("Stock sentiment with Pydantic output")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  # Run distillation (requires ANTHROPIC_API_KEY):")
        print("  python examples/stock_sentiment_pydantic.py distill")
        print("\n  # Query base or distilled model with Pydantic output:")
        print("  python examples/stock_sentiment_pydantic.py query <model_path_or_id> \"<text>\"")
        print("\nExample query:")
        print('  python examples/stock_sentiment_pydantic.py query mlx-community/Llama-3-8B-Instruct-4bit "Tesla down 5% after recall."')
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "distill":
        result = run_distillation()
        print(f"\nModel saved to: {result.model_path}")
        print("Query it with:")
        print(f'  python examples/stock_sentiment_pydantic.py query "{result.model_path}" "Your headline here"')
    elif cmd == "query":
        if len(sys.argv) < 4:
            print("Usage: python examples/stock_sentiment_pydantic.py query <model_path_or_id> \"<text>\"")
            sys.exit(1)
        model_path_or_id = sys.argv[2]
        text = sys.argv[3]
        analysis = query_stock_sentiment(model_path_or_id, text)
        print("\nStockAnalysis (Pydantic):")
        print(analysis.model_dump_json(indent=2))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
