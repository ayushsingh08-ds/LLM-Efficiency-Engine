from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SIMPLE_PROMPTS = [
    "Define polymorphism in one sentence.",
    "What is 15 percent of 320?",
    "Write a polite follow-up email subject line.",
    "Summarize this paragraph in 10 words.",
    "Translate 'good morning' to Spanish.",
]

MEDIUM_PROMPTS = [
    "Compare REST and GraphQL for public APIs.",
    "Write a Python function to merge overlapping intervals.",
    "Explain dropout in neural networks with an example.",
    "Design SQL indexes for a products table with category and price filters.",
    "Create a one-week meal plan for a vegetarian athlete.",
]

COMPLEX_PROMPTS = [
    "Design an event-driven architecture for real-time fraud detection with failure handling.",
    "Propose a retrieval-augmented generation system with evaluation metrics and fallback strategies.",
    "Analyze tradeoffs between model quantization and quality for low-latency inference.",
    "Write a secure multi-tenant API auth design with threat model and mitigations.",
    "Plan an online experiment framework for ranking model updates with guardrail metrics.",
]


def _expand_prompts(base_prompts: list[str], label: str, n: int) -> list[dict]:
    rows: list[dict] = []
    i = 1
    while len(rows) < n:
        template = base_prompts[(i - 1) % len(base_prompts)]
        variation = f"{template} Variant {i}."
        rows.append({"prompt": variation, "difficulty": label})
        i += 1
    return rows


def build_dataset(total_rows: int = 120) -> pd.DataFrame:
    if total_rows < 90:
        raise ValueError("total_rows should be at least 90 for balanced classes")

    per_class = total_rows // 3

    rows = []
    rows.extend(_expand_prompts(SIMPLE_PROMPTS, "Simple", per_class))
    rows.extend(_expand_prompts(MEDIUM_PROMPTS, "Medium", per_class))
    rows.extend(_expand_prompts(COMPLEX_PROMPTS, "Complex", total_rows - (2 * per_class)))

    df = pd.DataFrame(rows)
    df.insert(0, "prompt_id", range(1, len(df) + 1))
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock prompt dataset (100+) for contrastive training.")
    parser.add_argument("--rows", type=int, default=120, help="Number of rows to generate")
    parser.add_argument("--output", default="data/mock_training_prompts.csv", help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_dataset(total_rows=args.rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
