from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from src.routing.knn_router import KNNRouter
from src.training.contrastive_model import PromptEncoder


class TestKNNRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)

        # Deterministic tiny prompt encoder weights for test setup.
        encoder = PromptEncoder(input_dim=5, embedding_dim=16)
        torch.save(encoder.state_dict(), base / "prompt_encoder.pt")

        # Three synthetic models with different cost/quality tradeoffs.
        rows = [
            {
                "provider_name": "cheap",
                "model_name": "cheap-small",
                "avg_cost_usd": 0.0001,
                "avg_quality": 5.5,
                "reliability_metric": 0.75,
                "embedding": json.dumps([0.1] * 16),
            },
            {
                "provider_name": "balanced",
                "model_name": "balanced-mid",
                "avg_cost_usd": 0.0005,
                "avg_quality": 7.0,
                "reliability_metric": 0.85,
                "embedding": json.dumps([0.2] * 16),
            },
            {
                "provider_name": "premium",
                "model_name": "premium-high",
                "avg_cost_usd": 0.001,
                "avg_quality": 9.2,
                "reliability_metric": 0.92,
                "embedding": json.dumps([0.3] * 16),
            },
        ]
        pd.DataFrame(rows).to_csv(base / "model_embeddings.csv", index=False)

        self.router = KNNRouter(
            model_embeddings_csv=str(base / "model_embeddings.csv"),
            prompt_encoder_weights=str(base / "prompt_encoder.pt"),
            k=3,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_selects_cheapest_that_meets_threshold(self) -> None:
        decision = self.router.route("Explain dropout and regularization", quality_threshold=6.5)
        self.assertIn(decision.model_name, {"balanced-mid", "premium-high"})
        self.assertGreaterEqual(decision.quality_score, 6.5)

    def test_falls_back_to_neighbors_when_none_meet_threshold(self) -> None:
        decision = self.router.route("Very easy query", quality_threshold=9.9)
        self.assertIn(decision.model_name, {"cheap-small", "balanced-mid", "premium-high"})
        self.assertEqual(decision.neighbor_count, 3)


if __name__ == "__main__":
    unittest.main()
