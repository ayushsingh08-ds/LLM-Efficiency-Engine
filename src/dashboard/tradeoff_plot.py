from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_tradeoff(model_features_csv: str, output_path: str, quality_threshold: float = 6.5) -> None:
    df = pd.read_csv(model_features_csv)
    required = {"provider_name", "model_name", "avg_quality", "avg_cost_usd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Model features missing required columns: {missing}")

    plt.figure(figsize=(10, 6))

    for _, row in df.iterrows():
        x = float(row["avg_quality"])
        y = float(row["avg_cost_usd"])
        label = f"{row['provider_name']}/{row['model_name']}"

        color = "#0b6e4f" if x >= quality_threshold else "#8d3b72"
        plt.scatter(x, y, s=120, alpha=0.9, color=color)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.axvline(quality_threshold, linestyle="--", color="#1f2937", linewidth=1.2, label="Quality threshold")
    plt.title("LLM Cost vs Quality Tradeoff")
    plt.xlabel("Quality (1-10)")
    plt.ylabel("Cost per request (USD)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cost-quality scatter plot for model tradeoffs.")
    parser.add_argument("--input", default="outputs/benchmark_results/model_features.csv")
    parser.add_argument("--output", default="outputs/reports/cost_quality_tradeoff.png")
    parser.add_argument("--quality-threshold", type=float, default=6.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_tradeoff(
        model_features_csv=args.input,
        output_path=args.output,
        quality_threshold=args.quality_threshold,
    )
    print(f"Saved tradeoff plot to {args.output}")


if __name__ == "__main__":
    main()
