from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_quality(row: pd.Series) -> float:
    manual = row.get("quality_score_manual_1_10")
    llm = row.get("quality_score_llm_1_10")

    if pd.notna(manual):
        return float(manual)
    if pd.notna(llm):
        return float(llm)
    return 5.0


def build_model_features(benchmark_csv: str) -> pd.DataFrame:
    df = pd.read_csv(benchmark_csv)
    if df.empty:
        raise ValueError("Benchmark CSV is empty")

    working = df.copy()
    working["quality_score"] = working.apply(_safe_quality, axis=1)
    working["error_flag"] = working["error"].fillna("").astype(str).str.len() > 0
    working["hallucination_flag"] = working["hallucination_flag"].fillna(False).astype(bool)

    success_only = working[~working["error_flag"]].copy()

    perf_grouped = (
        success_only.groupby(["provider_name", "model_name"], as_index=False)
        .agg(
            avg_cost_usd=("cost_usd", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            avg_quality=("quality_score", "mean"),
            avg_response_length=("response_length", "mean"),
            hallucination_rate=("hallucination_flag", "mean"),
        )
    )

    reliability_grouped = (
        working.groupby(["provider_name", "model_name"], as_index=False)
        .agg(
            error_rate=("error_flag", "mean"),
            total_runs=("prompt_id", "count"),
        )
    )

    grouped = reliability_grouped.merge(
        perf_grouped,
        on=["provider_name", "model_name"],
        how="left",
    )

    # If no successful runs exist for a model, assign conservative defaults.
    grouped["avg_cost_usd"] = grouped["avg_cost_usd"].fillna(1.0)
    grouped["avg_latency_ms"] = grouped["avg_latency_ms"].fillna(5000.0)
    grouped["avg_quality"] = grouped["avg_quality"].fillna(0.0)
    grouped["avg_response_length"] = grouped["avg_response_length"].fillna(0.0)
    grouped["hallucination_rate"] = grouped["hallucination_rate"].fillna(1.0)
    grouped["error_rate"] = grouped["error_rate"].fillna(1.0)

    grouped["cost_to_quality_ratio"] = grouped["avg_cost_usd"] / grouped["avg_quality"].clip(lower=0.1)

    min_latency = grouped["avg_latency_ms"].min()
    max_latency = grouped["avg_latency_ms"].max()
    if max_latency == min_latency:
        grouped["speed_score"] = 1.0
    else:
        grouped["speed_score"] = 1.0 - ((grouped["avg_latency_ms"] - min_latency) / (max_latency - min_latency))

    grouped["reliability_metric"] = 1.0 - ((grouped["hallucination_rate"] + grouped["error_rate"]) / 2.0)
    grouped["reliability_metric"] = grouped["reliability_metric"].clip(lower=0.0, upper=1.0)

    category_quality = (
        success_only.groupby(["provider_name", "model_name", "category"], as_index=False)
        .agg(category_quality=("quality_score", "mean"))
    )

    # Specialization weights: normalized mean quality per category for each model.
    cat_pivot = category_quality.pivot_table(
        index=["provider_name", "model_name"],
        columns="category",
        values="category_quality",
        fill_value=0.0,
    )

    cat_norm = cat_pivot.div(cat_pivot.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    cat_norm.columns = [f"spec_weight_{c}" for c in cat_norm.columns]

    result = grouped.merge(cat_norm.reset_index(), on=["provider_name", "model_name"], how="left")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-level features from benchmark logs.")
    parser.add_argument(
        "--input",
        default="outputs/benchmark_results/benchmark_results.csv",
        help="Benchmark CSV input path",
    )
    parser.add_argument(
        "--output",
        default="outputs/benchmark_results/model_features.csv",
        help="Model feature CSV output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_df = build_model_features(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.output, index=False)
    print(f"Wrote {len(features_df)} model feature rows to {args.output}")


if __name__ == "__main__":
    main()
