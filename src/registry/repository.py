from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod

import psycopg2

from src.registry.models import ModelMetadata


class ModelRegistryRepository(ABC):
    @abstractmethod
    def upsert_model(self, metadata: ModelMetadata) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_models(self) -> list[ModelMetadata]:
        raise NotImplementedError


class SQLiteModelRegistryRepository(ModelRegistryRepository):
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert_model(self, metadata: ModelMetadata) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_registry (
                    provider_name, model_name, cost_per_1k_tokens,
                    average_latency_ms, quality_score, best_use_case, reliability_metric
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider_name, model_name) DO UPDATE SET
                    cost_per_1k_tokens = excluded.cost_per_1k_tokens,
                    average_latency_ms = excluded.average_latency_ms,
                    quality_score = excluded.quality_score,
                    best_use_case = excluded.best_use_case,
                    reliability_metric = excluded.reliability_metric,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    metadata.provider_name,
                    metadata.model_name,
                    metadata.cost_per_1k_tokens,
                    metadata.average_latency_ms,
                    metadata.quality_score,
                    metadata.best_use_case,
                    metadata.reliability_metric,
                ),
            )

    def list_models(self) -> list[ModelMetadata]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT provider_name, model_name, cost_per_1k_tokens,
                       average_latency_ms, quality_score, best_use_case, reliability_metric
                FROM model_registry
                ORDER BY quality_score DESC, average_latency_ms ASC
                """
            ).fetchall()

        return [ModelMetadata(**dict(row)) for row in rows]


class PostgresModelRegistryRepository(ModelRegistryRepository):
    def __init__(self, host: str, dbname: str, user: str, password: str, port: int = 5432) -> None:
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port

    def _connect(self):
        return psycopg2.connect(
            host=self.host,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            port=self.port,
        )

    def upsert_model(self, metadata: ModelMetadata) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_registry (
                        provider_name, model_name, cost_per_1k_tokens,
                        average_latency_ms, quality_score, best_use_case, reliability_metric
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(provider_name, model_name) DO UPDATE SET
                        cost_per_1k_tokens = EXCLUDED.cost_per_1k_tokens,
                        average_latency_ms = EXCLUDED.average_latency_ms,
                        quality_score = EXCLUDED.quality_score,
                        best_use_case = EXCLUDED.best_use_case,
                        reliability_metric = EXCLUDED.reliability_metric,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        metadata.provider_name,
                        metadata.model_name,
                        metadata.cost_per_1k_tokens,
                        metadata.average_latency_ms,
                        metadata.quality_score,
                        metadata.best_use_case,
                        metadata.reliability_metric,
                    ),
                )

    def list_models(self) -> list[ModelMetadata]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT provider_name, model_name, cost_per_1k_tokens,
                           average_latency_ms, quality_score, best_use_case, reliability_metric
                    FROM model_registry
                    ORDER BY quality_score DESC, average_latency_ms ASC
                    """
                )
                rows = cur.fetchall()

        return [
            ModelMetadata(
                provider_name=r[0],
                model_name=r[1],
                cost_per_1k_tokens=r[2],
                average_latency_ms=r[3],
                quality_score=r[4],
                best_use_case=r[5],
                reliability_metric=r[6],
            )
            for r in rows
        ]
