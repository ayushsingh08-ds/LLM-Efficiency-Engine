"""Live-gateway adapter for the offline k-NN contrastive router.

The offline pipeline (`src/routing/knn_router.py`) trains a prompt encoder and
model-embedding table, then selects the cheapest model that meets a quality
threshold. That router only ran offline (in the A/B harness). This module makes
it usable on the live request path with two safety properties:

1. **Graceful fallback** — if the trained artifacts are missing or fail to load,
   the gateway keeps working using the existing round-robin order. Smart routing
   is an enhancement, never a hard dependency.
2. **Supported-provider mapping** — the k-NN router may pick a provider the live
   gateway cannot dispatch to (e.g. `openai`). We map its choice onto the set of
   live-supported providers, preserving failover order for the rest.

Provider selection is observable through logs (`smart_router_*`) and the
`llm_routing_decision_total{mode,provider}` Prometheus counter.
"""
from __future__ import annotations

import os
from typing import Optional

from prometheus_client import Counter

from gateway.llm_provider import SUPPORTED_PROVIDERS, logger, router
from gateway.provider_health import registry as health_registry

# Default artifact locations (produced by `src.training.train_contrastive`).
MODEL_EMBEDDINGS_PATH = os.getenv("MODEL_EMBEDDINGS_PATH", "outputs/trained_models/model_embeddings.csv")
PROMPT_ENCODER_PATH = os.getenv("PROMPT_ENCODER_PATH", "outputs/trained_models/prompt_encoder.pt")
SMART_ROUTER_QUALITY_THRESHOLD = float(os.getenv("SMART_ROUTER_QUALITY_THRESHOLD", "6.5"))

ROUTING_DECISION_COUNT = Counter(
    "llm_routing_decision_total",
    "Routing decisions on the live gateway, by mode and chosen primary provider",
    ["mode", "provider"],
)

_knn_router = None
_knn_load_attempted = False
_knn_load_error: Optional[str] = None


def _load_knn_router():
    """Lazily load the trained k-NN router. Returns None if unavailable.

    Import of the heavy ML stack (torch/sklearn) is deferred to here so the
    gateway can import and serve round-robin traffic even if those artifacts or
    dependencies are missing.
    """
    global _knn_router, _knn_load_attempted, _knn_load_error
    if _knn_load_attempted:
        return _knn_router

    _knn_load_attempted = True
    if not (os.path.exists(MODEL_EMBEDDINGS_PATH) and os.path.exists(PROMPT_ENCODER_PATH)):
        _knn_load_error = "artifacts_missing"
        logger.warning("smart_router_unavailable", extra={
            "reason": "artifacts_missing",
            "model_embeddings_path": MODEL_EMBEDDINGS_PATH,
            "prompt_encoder_path": PROMPT_ENCODER_PATH,
        })
        return None

    try:
        from src.routing.knn_router import KNNRouter

        _knn_router = KNNRouter(
            model_embeddings_csv=MODEL_EMBEDDINGS_PATH,
            prompt_encoder_weights=PROMPT_ENCODER_PATH,
        )
        logger.info("smart_router_loaded", extra={
            "model_embeddings_path": MODEL_EMBEDDINGS_PATH,
        })
    except Exception as e:  # pragma: no cover - defensive, exercised via reset in tests
        _knn_load_error = f"{type(e).__name__}: {e}"
        logger.warning("smart_router_unavailable", extra={"reason": _knn_load_error})
        _knn_router = None

    return _knn_router


def reset_cache() -> None:
    """Reset the lazy-load cache. Primarily for tests."""
    global _knn_router, _knn_load_attempted, _knn_load_error
    _knn_router = None
    _knn_load_attempted = False
    _knn_load_error = None


def _supported_order(preferred: Optional[str]) -> list[str]:
    """Preferred provider first (if supported), then remaining supported providers."""
    order: list[str] = []
    if preferred in SUPPORTED_PROVIDERS:
        order.append(preferred)
    for provider in SUPPORTED_PROVIDERS:
        if provider not in order:
            order.append(provider)
    return order


def choose_provider_order(prompt: str, mode: str = "roundrobin") -> tuple[list[str], dict]:
    """Return (ordered_providers, decision_meta) for a prompt.

    mode:
      - "smart": use the trained k-NN router, falling back to round-robin if it is
        unavailable or picks an unsupported provider.
      - "roundrobin" (default): preserve the existing behaviour (let llm_provider's
        round-robin choose the primary).

    The returned order is a failover chain restricted to SUPPORTED_PROVIDERS. An
    empty list means "no override" — the caller (send_prompt) then uses its own
    round-robin selection, keeping full backward compatibility.
    """
    meta = {"mode": mode, "smart_used": False, "reason": None, "raw_choice": None}

    if mode != "smart":
        ROUTING_DECISION_COUNT.labels(mode="roundrobin", provider="_deferred").inc()
        return [], meta

    knn = _load_knn_router()
    if knn is None:
        meta["reason"] = _knn_load_error or "unavailable"
        logger.info("smart_router_fallback", extra={"reason": meta["reason"]})
        ROUTING_DECISION_COUNT.labels(mode="smart_fallback", provider="_deferred").inc()
        return [], meta

    try:
        decision = knn.route(prompt, quality_threshold=SMART_ROUTER_QUALITY_THRESHOLD)
    except Exception as e:
        meta["reason"] = f"route_error: {type(e).__name__}"
        logger.warning("smart_router_route_failed", extra={"error": str(e)})
        ROUTING_DECISION_COUNT.labels(mode="smart_fallback", provider="_deferred").inc()
        return [], meta

    raw_choice = decision.provider_name
    meta["raw_choice"] = raw_choice

    if raw_choice in SUPPORTED_PROVIDERS:
        order = _supported_order(raw_choice)
        # Coexist with provider health: keep the smart pick first UNLESS it is
        # currently unhealthy, in which case a healthy provider is tried first
        # (the unhealthy pick stays in the chain as fallback, never dropped).
        prioritized = health_registry.prioritize(order)
        meta["smart_used"] = True
        meta["health_reordered"] = prioritized != order
        logger.info("smart_router_decision", extra={
            "chosen_provider": raw_choice,
            "effective_primary": prioritized[0] if prioritized else None,
            "health_reordered": meta["health_reordered"],
            "chosen_model": decision.model_name,
            "estimated_cost_usd": decision.estimated_cost_usd,
            "quality_score": decision.quality_score,
            "neighbor_count": decision.neighbor_count,
        })
        ROUTING_DECISION_COUNT.labels(mode="smart", provider=raw_choice).inc()
        return prioritized, meta

    # k-NN chose a provider the live gateway can't call (e.g. openai): map to the
    # best supported provider and note it, but still honour the smart intent.
    order = health_registry.prioritize(_supported_order(None))
    meta["smart_used"] = True
    meta["reason"] = f"unsupported_choice:{raw_choice}"
    logger.info("smart_router_decision_mapped", extra={
        "raw_choice": raw_choice,
        "mapped_to": order[0] if order else None,
        "chosen_model": decision.model_name,
    })
    ROUTING_DECISION_COUNT.labels(mode="smart_mapped", provider=order[0] if order else "none").inc()
    return order, meta
