from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.benchmarking.quality_scoring import llm_assisted_quality_score
from src.clients.client_factory import ClientFactory


def _lcs_length(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_f1(reference: str, candidate: str) -> float:
    reference = reference.strip().lower()
    candidate = candidate.strip().lower()
    if not reference or not candidate:
        return 0.0

    lcs = _lcs_length(reference, candidate)
    prec = lcs / len(candidate)
    rec = lcs / len(reference)
    if (prec + rec) == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def score_to_1_10(score_0_1: float) -> float:
    return round(1.0 + 9.0 * max(0.0, min(1.0, score_0_1)), 2)


@dataclass
class QualityScoreResult:
    request_id: int
    metric_score_1_10: float
    llm_judge_score_1_10: float | None
    final_score_1_10: float


def score_quality_file(
    input_csv: str,
    output_csv: str,
    model_config: str,
    judge_provider: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"request_id", "prompt", "response", "reference"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in quality input: {missing}")

    judge_client = None
    if judge_provider:
        for client in ClientFactory.load_from_config(model_config):
            if client.provider_name == judge_provider.lower():
                judge_client = client
                break

    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        metric = score_to_1_10(rouge_l_f1(str(row["reference"]), str(row["response"])))

        judge_score = None
        if judge_client is not None:
            judge_score, _ = llm_assisted_quality_score(
                judge_client=judge_client,
                prompt=str(row["prompt"]),
                response=str(row["response"]),
            )

        if judge_score is None:
            final = metric
        else:
            final = round((metric + float(judge_score)) / 2.0, 2)

        rows.append(
            {
                "request_id": int(row["request_id"]),
                "metric_score_1_10": metric,
                "llm_judge_score_1_10": judge_score,
                "final_score_1_10": final,
            }
        )

    out_df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score quality (1-10) via ROUGE-L + optional LLM judge.")
    parser.add_argument("--input", default="data/quality_eval_samples.csv")
    parser.add_argument("--output", default="outputs/reports/quality_scores.csv")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--judge-provider", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = score_quality_file(
        input_csv=args.input,
        output_csv=args.output,
        model_config=args.models,
        judge_provider=args.judge_provider,
    )
    print(f"Wrote {len(out)} scored rows to {args.output}")


if __name__ == "__main__":
    main()
