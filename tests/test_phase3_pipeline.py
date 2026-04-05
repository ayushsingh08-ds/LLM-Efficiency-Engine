from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.dashboard.tradeoff_plot import plot_tradeoff
from src.evaluation.ab_test import run_ab_test
from src.evaluation.validate_quality import score_quality_file


class TestPhase3Pipeline(unittest.TestCase):
    def test_quality_scoring_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            input_csv = base / "quality_input.csv"
            out_csv = base / "reports" / "quality_scores.csv"

            pd.DataFrame(
                [
                    {
                        "request_id": 1,
                        "prompt": "Summarize this.",
                        "response": "Optimize cost and quality.",
                        "reference": "Optimize both quality and cost.",
                    }
                ]
            ).to_csv(input_csv, index=False)

            result = score_quality_file(
                input_csv=str(input_csv),
                output_csv=str(out_csv),
                model_config="configs/models.yaml",
                judge_provider=None,
            )

            self.assertEqual(len(result), 1)
            self.assertTrue(out_csv.exists())
            self.assertGreater(float(result.iloc[0]["final_score_1_10"]), 0.0)

    def test_ab_test_generates_balanced_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            validation_csv = base / "validation.csv"
            model_features_csv = base / "model_features.csv"
            output_csv = base / "reports" / "ab_detail.csv"
            summary_csv = base / "reports" / "ab_summary.csv"

            pd.DataFrame(
                [{"request_id": i, "prompt": f"Prompt {i}"} for i in range(1, 11)]
            ).to_csv(validation_csv, index=False)

            pd.DataFrame(
                [
                    {
                        "provider_name": "openai",
                        "model_name": "gpt-4o-mini",
                        "avg_cost_usd": 0.02,
                        "avg_latency_ms": 1800.0,
                        "avg_quality": 8.5,
                    },
                    {
                        "provider_name": "groq",
                        "model_name": "llama-3.1-8b-instant",
                        "avg_cost_usd": 0.00002,
                        "avg_latency_ms": 500.0,
                        "avg_quality": 7.0,
                    },
                ]
            ).to_csv(model_features_csv, index=False)

            class FakeRouter:
                def __init__(self, *args, **kwargs) -> None:
                    pass

                def route(self, prompt: str, quality_threshold: float = 6.5):
                    return SimpleNamespace(
                        provider_name="groq",
                        model_name="llama-3.1-8b-instant",
                        estimated_cost_usd=0.00002,
                        quality_score=7.0,
                        reliability_metric=0.9,
                        neighbor_count=3,
                    )

            with patch("src.evaluation.ab_test.KNNRouter", FakeRouter):
                detailed, summary = run_ab_test(
                    validation_csv=str(validation_csv),
                    model_features_csv=str(model_features_csv),
                    model_embeddings_csv="unused.csv",
                    prompt_encoder_weights="unused.pt",
                    output_csv=str(output_csv),
                    summary_csv=str(summary_csv),
                    baseline_model="gpt-4o-mini",
                )

            self.assertEqual(len(detailed), 10)
            self.assertTrue(output_csv.exists())
            self.assertTrue(summary_csv.exists())
            summary_map = {row["arm"]: int(row["requests"]) for _, row in summary.iterrows()}
            self.assertEqual(summary_map["baseline"], 5)
            self.assertEqual(summary_map["smart"], 5)

    def test_tradeoff_plot_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            model_features_csv = base / "model_features.csv"
            output_png = base / "reports" / "tradeoff.png"

            pd.DataFrame(
                [
                    {
                        "provider_name": "openai",
                        "model_name": "gpt-4o-mini",
                        "avg_quality": 8.5,
                        "avg_cost_usd": 0.02,
                    },
                    {
                        "provider_name": "groq",
                        "model_name": "llama-3.1-8b-instant",
                        "avg_quality": 7.0,
                        "avg_cost_usd": 0.00002,
                    },
                ]
            ).to_csv(model_features_csv, index=False)

            plot_tradeoff(
                model_features_csv=str(model_features_csv),
                output_path=str(output_png),
                quality_threshold=7.5,
            )

            self.assertTrue(output_png.exists())
            self.assertGreater(output_png.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
