from __future__ import annotations

import argparse

import pandas as pd

from src.registry.db import initialize_sqlite_registry
from src.registry.models import ModelMetadata
from src.registry.repository import SQLiteModelRegistryRepository


def load_model_features_into_registry(features_csv: str, db_path: str, schema_sql_path: str) -> None:
    initialize_sqlite_registry(db_path=db_path, schema_sql_path=schema_sql_path)
    repo = SQLiteModelRegistryRepository(db_path=db_path)

    df = pd.read_csv(features_csv)
    if df.empty:
        raise ValueError("Feature CSV is empty")

    for _, row in df.iterrows():
        metadata = ModelMetadata(
            provider_name=str(row["provider_name"]),
            model_name=str(row["model_name"]),
            cost_per_1k_tokens=float(row.get("avg_cost_usd", 0.0) * 1000.0),
            average_latency_ms=float(row.get("avg_latency_ms", 0.0)),
            quality_score=float(row.get("avg_quality", 0.0)),
            best_use_case=str(row.get("best_use_case", "general")),
            reliability_metric=float(row.get("reliability_metric", 0.0)),
        )
        repo.upsert_model(metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load engineered model features into SQLite model registry.")
    parser.add_argument(
        "--features",
        default="outputs/benchmark_results/model_features.csv",
        help="Path to model features CSV",
    )
    parser.add_argument(
        "--db-path",
        default="outputs/registry/model_registry.sqlite",
        help="SQLite db destination",
    )
    parser.add_argument(
        "--schema",
        default="src/registry/schema.sql",
        help="SQL schema path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_model_features_into_registry(args.features, args.db_path, args.schema)
    print(f"Loaded model registry from {args.features} into {args.db_path}")


if __name__ == "__main__":
    main()
