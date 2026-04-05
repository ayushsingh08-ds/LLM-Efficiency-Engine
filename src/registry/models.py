from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelMetadata:
    provider_name: str
    model_name: str
    cost_per_1k_tokens: float
    average_latency_ms: float
    quality_score: float
    best_use_case: str
    reliability_metric: float
