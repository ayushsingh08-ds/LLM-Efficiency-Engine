from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from src.routing.embedder import prompt_to_features
from src.training.contrastive_model import PromptEncoder


@dataclass
class RoutingDecision:
    provider_name: str
    model_name: str
    estimated_cost_usd: float
    quality_score: float
    reliability_metric: float
    neighbor_count: int


class KNNRouter:
    def __init__(
        self,
        model_embeddings_csv: str,
        prompt_encoder_weights: str,
        k: int = 3,
    ) -> None:
        self.k = int(k)
        self.model_df = pd.read_csv(model_embeddings_csv)
        if self.model_df.empty:
            raise ValueError("model_embeddings.csv is empty")

        vectors = [np.array(json.loads(v), dtype=np.float32) for v in self.model_df["embedding"]]
        self.model_matrix = np.stack(vectors)

        self.prompt_encoder = PromptEncoder(input_dim=5, embedding_dim=self.model_matrix.shape[1])
        state_dict = torch.load(prompt_encoder_weights, map_location="cpu", weights_only=True)
        self.prompt_encoder.load_state_dict(state_dict)
        self.prompt_encoder.eval()

        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(self.model_df)), metric="euclidean")
        self.nn.fit(self.model_matrix)

    def route(self, prompt: str, quality_threshold: float = 6.0) -> RoutingDecision:
        prompt_features = prompt_to_features(prompt)
        prompt_tensor = torch.tensor(prompt_features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prompt_emb = self.prompt_encoder(prompt_tensor).cpu().numpy()

        distances, indices = self.nn.kneighbors(prompt_emb, return_distance=True)
        neighbor_idx = indices[0]

        neighbors = self.model_df.iloc[neighbor_idx].copy()

        eligible = neighbors[neighbors["avg_quality"] >= quality_threshold]
        candidate_pool = eligible if not eligible.empty else neighbors

        winner = candidate_pool.sort_values(["avg_cost_usd", "avg_quality"], ascending=[True, False]).iloc[0]

        return RoutingDecision(
            provider_name=str(winner["provider_name"]),
            model_name=str(winner["model_name"]),
            estimated_cost_usd=float(winner["avg_cost_usd"]),
            quality_score=float(winner["avg_quality"]),
            reliability_metric=float(winner.get("reliability_metric", 0.0)),
            neighbor_count=len(neighbor_idx),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route a live prompt via contrastive embeddings + k-NN.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--quality-threshold", type=float, default=6.0)
    parser.add_argument("--model-embeddings", default="outputs/trained_models/model_embeddings.csv")
    parser.add_argument("--prompt-encoder", default="outputs/trained_models/prompt_encoder.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    router = KNNRouter(
        model_embeddings_csv=args.model_embeddings,
        prompt_encoder_weights=args.prompt_encoder,
        k=args.k,
    )
    decision = router.route(prompt=args.prompt, quality_threshold=args.quality_threshold)
    print(decision)


if __name__ == "__main__":
    main()
