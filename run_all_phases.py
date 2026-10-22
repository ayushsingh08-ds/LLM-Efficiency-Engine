from __future__ import annotations

import argparse
import subprocess
from typing import Sequence


def _run(cmd: Sequence[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_python(module: str, args: list[str], docker: bool) -> None:
    if docker:
        _run(["docker", "compose", "exec", "fastapi", "python", "-m", module, *args])
    else:
        _run(["python", "-m", module, *args])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all LLM routing phases end-to-end.")
    parser.add_argument("--docker", action="store_true", help="Execute inside docker compose fastapi service")
    parser.add_argument("--epochs", type=int, default=5, help="Contrastive training epochs")
    parser.add_argument("--requests", type=int, default=1000, help="Validation requests for A/B")
    parser.add_argument("--quality-threshold", type=float, default=6.5)
    parser.add_argument("--baseline-model", default="gpt-4o-mini")
    parser.add_argument("--baseline-cost-usd", type=float, default=0.02)
    parser.add_argument("--baseline-latency-ms", type=float, default=1800.0)
    parser.add_argument("--baseline-quality", type=float, default=8.5)
    args = parser.parse_args()

    if args.docker:
        _run(["docker", "compose", "up", "-d", "--build", "fastapi"])

    # Phase 1
    _run_python(
        "src.benchmarking.run_benchmarks",
        [
            "--prompts", "data/benchmark_prompts.csv",
            "--models", "configs/models.yaml",
            "--output", "outputs/benchmark_results/benchmark_results.csv",
        ],
        docker=args.docker,
    )
    _run_python(
        "src.features.feature_engineering",
        [
            "--input", "outputs/benchmark_results/benchmark_results.csv",
            "--output", "outputs/benchmark_results/model_features.csv",
        ],
        docker=args.docker,
    )
    _run_python(
        "src.registry.load_registry",
        [
            "--features", "outputs/benchmark_results/model_features.csv",
            "--db-path", "outputs/registry/model_registry.sqlite",
            "--schema", "src/registry/schema.sql",
        ],
        docker=args.docker,
    )

    # Phase 2
    _run_python(
        "src.training.build_mock_dataset",
        ["--rows", "120", "--output", "data/mock_training_prompts.csv"],
        docker=args.docker,
    )
    _run_python(
        "src.training.train_contrastive",
        [
            "--prompts", "data/mock_training_prompts.csv",
            "--model-features", "outputs/benchmark_results/model_features.csv",
            "--output-dir", "outputs/trained_models",
            "--epochs", str(args.epochs),
        ],
        docker=args.docker,
    )

    # Phase 3
    _run_python(
        "src.evaluation.validate_quality",
        [
            "--input", "data/quality_eval_samples.csv",
            "--output", "outputs/reports/quality_scores.csv",
        ],
        docker=args.docker,
    )
    _run_python(
        "src.evaluation.build_validation_requests",
        ["--output", "data/validation_requests.csv", "--n", str(args.requests)],
        docker=args.docker,
    )
    _run_python(
        "src.evaluation.ab_test",
        [
            "--validation", "data/validation_requests.csv",
            "--model-features", "outputs/benchmark_results/model_features.csv",
            "--model-embeddings", "outputs/trained_models/model_embeddings.csv",
            "--prompt-encoder", "outputs/trained_models/prompt_encoder.pt",
            "--output", "outputs/reports/ab_test_detailed.csv",
            "--summary", "outputs/reports/ab_test_summary.csv",
            "--baseline-model", args.baseline_model,
            "--quality-threshold", str(args.quality_threshold),
            "--baseline-cost-usd", str(args.baseline_cost_usd),
            "--baseline-latency-ms", str(args.baseline_latency_ms),
            "--baseline-quality", str(args.baseline_quality),
        ],
        docker=args.docker,
    )
    _run_python(
        "src.dashboard.tradeoff_plot",
        [
            "--input", "outputs/benchmark_results/model_features.csv",
            "--output", "outputs/reports/cost_quality_tradeoff.png",
            "--quality-threshold", str(args.quality_threshold),
        ],
        docker=args.docker,
    )

    print("All phases completed successfully.")


if __name__ == "__main__":
    main()
