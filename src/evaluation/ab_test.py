from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.routing.knn_router import KNNRouter


def _select_baseline_row(model_features: pd.DataFrame, baseline_model: str) -> pd.Series:
    m = model_features[model_features["model_name"] == baseline_model]
    if m.empty:
        raise ValueError(f"Baseline model not found: {baseline_model}")
    return m.iloc[0]


def _with_baseline_fallback(
    row: pd.Series,
    baseline_model: str,
    baseline_cost_usd: float,
    baseline_latency_ms: float,
    baseline_quality_1_10: float,
) -> pd.Series:
    out = row.copy()
    if float(out.get("avg_quality", 0.0)) <= 0.0:
        out["avg_cost_usd"] = baseline_cost_usd
        out["avg_latency_ms"] = baseline_latency_ms
        out["avg_quality"] = baseline_quality_1_10
        out["provider_name"] = str(out.get("provider_name", "baseline"))
        out["model_name"] = baseline_model
    return out


def _pick_model_stats(model_features: pd.DataFrame, provider: str, model: str) -> pd.Series:
    row = model_features[(model_features["provider_name"] == provider) & (model_features["model_name"] == model)]
    if row.empty:
        raise ValueError(f"Model stats missing for {provider}/{model}")
    return row.iloc[0]


def run_ab_test(
    validation_csv: str,
    model_features_csv: str,
    model_embeddings_csv: str,
    prompt_encoder_weights: str,
    output_csv: str,
    summary_csv: str,
    baseline_model: str = "gpt-4o-mini",
    quality_threshold: float = 6.5,
    baseline_cost_usd: float = 0.02,
    baseline_latency_ms: float = 1800.0,
    baseline_quality_1_10: float = 8.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    requests_df = pd.read_csv(validation_csv)
    model_features = pd.read_csv(model_features_csv)

    required = {"request_id", "prompt"}
    missing = required - set(requests_df.columns)
    if missing:
        raise ValueError(f"Validation CSV missing columns: {missing}")

    baseline = _select_baseline_row(model_features, baseline_model)
    baseline = _with_baseline_fallback(
        baseline,
        baseline_model=baseline_model,
        baseline_cost_usd=baseline_cost_usd,
        baseline_latency_ms=baseline_latency_ms,
        baseline_quality_1_10=baseline_quality_1_10,
    )

    router = KNNRouter(
        model_embeddings_csv=model_embeddings_csv,
        prompt_encoder_weights=prompt_encoder_weights,
        k=3,
    )

    rows: list[dict[str, object]] = []
    for _, req in requests_df.iterrows():
        request_id = int(req["request_id"])
        prompt = str(req["prompt"])

        # Stable 50/50 assignment by request_id parity.
        arm = "smart" if (request_id % 2 == 0) else "baseline"

        if arm == "smart":
            decision = router.route(prompt=prompt, quality_threshold=quality_threshold)
            stats = _pick_model_stats(model_features, decision.provider_name, decision.model_name)
            provider = decision.provider_name
            model = decision.model_name
        else:
            stats = baseline
            provider = str(stats["provider_name"])
            model = str(stats["model_name"])

        rows.append(
            {
                "request_id": request_id,
                "arm": arm,
                "provider_name": provider,
                "model_name": model,
                "estimated_cost_usd": float(stats.get("avg_cost_usd", 0.0)),
                "estimated_latency_ms": float(stats.get("avg_latency_ms", 0.0)),
                "estimated_quality_1_10": float(stats.get("avg_quality", 0.0)),
            }
        )

    out_df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    summary = (
        out_df.groupby("arm", as_index=False)
        .agg(
            requests=("request_id", "count"),
            total_cost_usd=("estimated_cost_usd", "sum"),
            avg_latency_ms=("estimated_latency_ms", "mean"),
            avg_quality_1_10=("estimated_quality_1_10", "mean"),
        )
        .sort_values("arm")
    )
    summary.to_csv(summary_csv, index=False)

    return out_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 50/50 A/B test between smart router and static baseline model.")
    parser.add_argument("--validation", default="data/validation_requests.csv")
    parser.add_argument("--model-features", default="outputs/benchmark_results/model_features.csv")
    parser.add_argument("--model-embeddings", default="outputs/trained_models/model_embeddings.csv")
    parser.add_argument("--prompt-encoder", default="outputs/trained_models/prompt_encoder.pt")
    parser.add_argument("--output", default="outputs/reports/ab_test_detailed.csv")
    parser.add_argument("--summary", default="outputs/reports/ab_test_summary.csv")
    parser.add_argument("--baseline-model", default="gpt-4o-mini")
    parser.add_argument("--quality-threshold", type=float, default=6.5)
    parser.add_argument("--baseline-cost-usd", type=float, default=0.02)
    parser.add_argument("--baseline-latency-ms", type=float, default=1800.0)
    parser.add_argument("--baseline-quality", type=float, default=8.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary = run_ab_test(
        validation_csv=args.validation,
        model_features_csv=args.model_features,
        model_embeddings_csv=args.model_embeddings,
        prompt_encoder_weights=args.prompt_encoder,
        output_csv=args.output,
        summary_csv=args.summary,
        baseline_model=args.baseline_model,
        quality_threshold=args.quality_threshold,
        baseline_cost_usd=args.baseline_cost_usd,
        baseline_latency_ms=args.baseline_latency_ms,
        baseline_quality_1_10=args.baseline_quality,
    )

    print("A/B Summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
