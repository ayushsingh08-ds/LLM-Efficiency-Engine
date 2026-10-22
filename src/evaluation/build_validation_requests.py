from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_validation_requests(output_csv: str, n_requests: int = 1000) -> pd.DataFrame:
    base_prompts = [
        "Summarize this customer support thread in 3 bullet points.",
        "Explain why a Kubernetes pod may restart repeatedly.",
        "Write a SQL query for monthly active users by region.",
        "Draft a concise project status update for executives.",
        "Design a feature flag rollout strategy for a payment flow.",
        "Compare BERT and GPT style models for classification tasks.",
        "Propose caching strategy for a read-heavy API.",
        "Create a test plan for a rate limiter implementation.",
        "Write secure OAuth callback validation logic.",
        "Analyze latency-cost tradeoff for LLM model selection.",
    ]

    rows: list[dict] = []
    for idx in range(1, n_requests + 1):
        prompt = base_prompts[(idx - 1) % len(base_prompts)]
        rows.append({"request_id": idx, "prompt": f"{prompt} Request {idx}."})

    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate validation request set for A/B testing.")
    parser.add_argument("--output", default="data/validation_requests.csv")
    parser.add_argument("--n", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_validation_requests(output_csv=args.output, n_requests=args.n)
    print(f"Wrote {len(df)} validation requests to {args.output}")


if __name__ == "__main__":
    main()
