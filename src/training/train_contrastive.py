from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

from src.routing.embedder import prompt_to_features
from src.training.contrastive_model import ModelEncoder, PromptEncoder


def _difficulty_score(label: str) -> float:
    mapping = {"simple": 0.2, "medium": 0.6, "complex": 1.0}
    return mapping.get(label.strip().lower(), 0.6)


def _model_target_tier(features_row: pd.Series) -> float:
    quality = float(features_row.get("avg_quality", 5.0)) / 10.0
    cost = float(features_row.get("avg_cost_usd", 0.0))
    speed = float(features_row.get("speed_score", 0.5))

    # Frontier preference rises with quality and acceptable speed, and penalizes cost.
    return max(0.0, min(1.0, (0.6 * quality) + (0.3 * speed) - (0.1 * min(1.0, cost * 1000.0))))


def _build_model_feature_vector(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            float(row.get("avg_cost_usd", 0.0)) * 1000.0,
            float(row.get("avg_latency_ms", 0.0)) / 2000.0,
            float(row.get("avg_quality", 5.0)) / 10.0,
            float(row.get("speed_score", 0.5)),
            float(row.get("reliability_metric", 0.5)),
            float(row.get("avg_response_length", 0.0)) / 1000.0,
        ],
        dtype=np.float32,
    )


def train(
    prompt_csv: str,
    model_features_csv: str,
    output_dir: str,
    epochs: int = 25,
    lr: float = 1e-3,
    margin: float = 0.25,
) -> None:
    prompts_df = pd.read_csv(prompt_csv)
    model_df = pd.read_csv(model_features_csv)

    if prompts_df.empty or model_df.empty:
        raise ValueError("Training inputs are empty")

    model_df = model_df.copy()
    model_df["target_tier"] = model_df.apply(_model_target_tier, axis=1)

    prompt_encoder = PromptEncoder(input_dim=5, embedding_dim=16)
    model_encoder = ModelEncoder(input_dim=6, embedding_dim=16)

    optimizer = Adam(list(prompt_encoder.parameters()) + list(model_encoder.parameters()), lr=lr)
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    model_vectors = np.stack([_build_model_feature_vector(r) for _, r in model_df.iterrows()])
    model_tiers = model_df["target_tier"].to_numpy(dtype=np.float32)

    for epoch in range(1, epochs + 1):
        losses = []

        for _, prompt_row in prompts_df.iterrows():
            p_vec = prompt_to_features(str(prompt_row["prompt"]))
            p_tier = _difficulty_score(str(prompt_row["difficulty"]))

            positive_idx = int(np.argmin(np.abs(model_tiers - p_tier)))
            negative_idx = int(np.argmax(np.abs(model_tiers - p_tier)))

            anchor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0)
            positive = torch.tensor(model_vectors[positive_idx], dtype=torch.float32).unsqueeze(0)
            negative = torch.tensor(model_vectors[negative_idx], dtype=torch.float32).unsqueeze(0)

            anchor_emb = prompt_encoder(anchor)
            positive_emb = model_encoder(positive)
            negative_emb = model_encoder(negative)

            loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(f"epoch={epoch} avg_triplet_loss={avg_loss:.6f}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(prompt_encoder.state_dict(), output_path / "prompt_encoder.pt")
    torch.save(model_encoder.state_dict(), output_path / "model_encoder.pt")

    # Pre-compute model embeddings for k-NN router.
    with torch.no_grad():
        model_input = torch.tensor(model_vectors, dtype=torch.float32)
        model_emb = model_encoder(model_input).cpu().numpy()

    model_embeddings_df = model_df[["provider_name", "model_name", "avg_cost_usd", "avg_quality", "reliability_metric"]].copy()
    model_embeddings_df["embedding"] = [json.dumps(vec.tolist()) for vec in model_emb]
    model_embeddings_df.to_csv(output_path / "model_embeddings.csv", index=False)

    metadata = {
        "prompt_encoder_input_dim": 5,
        "model_encoder_input_dim": 6,
        "embedding_dim": 16,
        "epochs": epochs,
        "margin": margin,
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model artifacts to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contrastive prompt/model embedders with triplet loss.")
    parser.add_argument("--prompts", default="data/mock_training_prompts.csv")
    parser.add_argument("--model-features", default="outputs/benchmark_results/model_features.csv")
    parser.add_argument("--output-dir", default="outputs/trained_models")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        prompt_csv=args.prompts,
        model_features_csv=args.model_features,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
