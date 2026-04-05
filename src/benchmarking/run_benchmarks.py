from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.benchmarking.hallucination_checks import lightweight_hallucination_flag
from src.benchmarking.quality_scoring import llm_assisted_quality_score, manual_placeholder_score
from src.clients.client_factory import ClientFactory


def run_benchmarks(
    prompts_csv: str,
    model_config: str,
    output_csv: str,
    judge_provider: str | None = None,
) -> pd.DataFrame:
    load_dotenv()

    prompts_df = pd.read_csv(prompts_csv)
    required_cols = {"prompt_id", "category", "prompt"}
    missing = required_cols - set(prompts_df.columns)
    if missing:
        raise ValueError(f"Prompts CSV is missing required columns: {missing}")

    clients = ClientFactory.load_from_config(model_config)

    judge_client = None
    if judge_provider:
        for c in clients:
            if c.provider_name == judge_provider.lower():
                judge_client = c
                break

    rows: list[dict] = []

    for _, prompt_row in prompts_df.iterrows():
        prompt_id = int(prompt_row["prompt_id"])
        category = str(prompt_row["category"])
        prompt = str(prompt_row["prompt"])

        for client in clients:
            started = time.perf_counter()
            error_message = ""

            try:
                result = client.invoke(prompt)
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                response_len = len(result.text)
                cost_usd = client.estimate_cost_usd(result.input_tokens, result.output_tokens)
                hallucination_flag = lightweight_hallucination_flag(prompt, result.text)
                quality_manual = manual_placeholder_score()

                quality_llm, quality_llm_notes = (None, "judge_not_configured")
                if judge_client:
                    quality_llm, quality_llm_notes = llm_assisted_quality_score(
                        judge_client=judge_client,
                        prompt=prompt,
                        response=result.text,
                    )

                row = {
                    "prompt_id": prompt_id,
                    "category": category,
                    "prompt": prompt,
                    "provider_name": client.provider_name,
                    "model_name": client.model_name,
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "response_length": response_len,
                    "quality_score_manual_1_10": quality_manual,
                    "quality_score_llm_1_10": quality_llm,
                    "quality_score_llm_notes": quality_llm_notes,
                    "hallucination_flag": hallucination_flag,
                    "error": "",
                }
            except Exception as e:
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                error_message = str(e)
                row = {
                    "prompt_id": prompt_id,
                    "category": category,
                    "prompt": prompt,
                    "provider_name": client.provider_name,
                    "model_name": client.model_name,
                    "latency_ms": latency_ms,
                    "cost_usd": None,
                    "response_length": 0,
                    "quality_score_manual_1_10": None,
                    "quality_score_llm_1_10": None,
                    "quality_score_llm_notes": "",
                    "hallucination_flag": None,
                    "error": error_message,
                }

            rows.append(row)
            print(
                f"bench prompt={prompt_id} provider={client.provider_name} model={client.model_name} "
                f"latency_ms={latency_ms} error={'yes' if error_message else 'no'}"
            )

    out_df = pd.DataFrame(rows)
    Path(os.path.dirname(output_csv) or ".").mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark prompts across LLM providers.")
    parser.add_argument("--prompts", default="data/benchmark_prompts.csv", help="Path to prompts CSV")
    parser.add_argument("--models", default="configs/models.yaml", help="Path to model config YAML")
    parser.add_argument(
        "--output",
        default="outputs/benchmark_results/benchmark_results.csv",
        help="Output benchmark CSV path",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        help="Optional provider name to score quality with an LLM judge",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_benchmarks(
        prompts_csv=args.prompts,
        model_config=args.models,
        output_csv=args.output,
        judge_provider=args.judge_provider,
    )
    print(f"Wrote {len(df)} benchmark rows to {args.output}")


if __name__ == "__main__":
    main()
