from typing import Callable, Literal, Optional
try:
    import torch
    import torch.nn as nn
except ImportError:
    pass
import time
import logging
import psycopg2
import re
import hashlib
from prometheus_client import Counter
from pythonjsonlogger import jsonlogger
from gateway.router import RoundRobinRouter
from gateway.provider_health import registry as health_registry
from gateway.provider_registry import registry as provider_registry
from cache.embeddings import prompt_to_vector
from cache.milvus_cache import search_similar_embedding, store_cache_entry
from cache.redis_cache import get_cached_response, store_cached_response
from dotenv import load_dotenv
import os
load_dotenv()

# Structured JSON logger
logger = logging.getLogger("llm_gateway")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

# SUPPORTED_PROVIDERS is now driven by configs/providers.yaml (enabled entries).
# Add or remove a provider there — no code changes needed here.
SUPPORTED_PROVIDERS: list[str] = provider_registry.enabled_names
router = RoundRobinRouter(SUPPORTED_PROVIDERS)

# Prometheus metrics for failover observability. Defined on the default registry
# so they are exported by gateway.app's existing /metrics endpoint automatically.
PROVIDER_ATTEMPT_COUNT = Counter(
    "llm_provider_attempts_total",
    "Live provider call attempts, labelled by provider and outcome",
    ["provider", "outcome"],
)
FAILOVER_COUNT = Counter(
    "llm_failover_total",
    "Number of times a request fell back from one provider to another",
    ["from_provider", "to_provider"],
)


class AllProvidersFailedError(RuntimeError):
    """Raised when every eligible provider failed for a single request."""

    def __init__(self, attempts: list[dict]):
        self.attempts = attempts
        providers = ", ".join(a["provider"] for a in attempts) or "none"
        super().__init__(f"All providers failed ({providers})")

# Pricing and default models are loaded from configs/providers.yaml via
# provider_registry.  The helpers below delegate to it so the rest of this
# file doesn't need to change.
def _get_pricing(model_id: str, provider_name: str) -> dict:
    """Return {input, output} cost per token in USD from the registry."""
    return provider_registry.cost_per_token(provider_name, model_id)


def _get_default_model(provider_name: str) -> str:
    """Return the configured default model for a provider."""
    try:
        return provider_registry.default_model(provider_name)
    except KeyError:
        return "unknown"

MAX_LOG_TEXT_LENGTH = 500
DEFAULT_CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85"))
_cache_similarity_threshold = DEFAULT_CACHE_SIMILARITY_THRESHOLD


def _normalize_similarity_threshold(value: float) -> float:
    threshold = float(value)
    if threshold <= 0 or threshold > 1:
        raise ValueError("similarity_threshold must be > 0 and <= 1")
    return threshold


def get_cache_similarity_threshold() -> float:
    return _cache_similarity_threshold


def set_cache_similarity_threshold(value: float) -> float:
    global _cache_similarity_threshold
    _cache_similarity_threshold = _normalize_similarity_threshold(value)
    logger.info("cache_similarity_threshold_updated", extra={
        "similarity_threshold": _cache_similarity_threshold,
    })
    return _cache_similarity_threshold


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (~4 chars per token for English)."""
    return max(1, len(text) // 4)


def estimate_cost(
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 1024
) -> dict:
    """Estimate cost before sending to API."""
    if provider is None:
        provider = router.providers[router.index]  # peek without advancing

    model_name = model or _get_default_model(provider)
    pricing = _get_pricing(model_name, provider)

    input_tokens = _estimate_tokens(prompt)
    # Estimate output as half of max_tokens (typical usage)
    estimated_output_tokens = max_tokens // 2

    input_cost = input_tokens * pricing["input"]
    output_cost = estimated_output_tokens * pricing["output"]

    return {
        "provider": provider,
        "model": model_name,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_cost_usd": round(input_cost + output_cost, 8),
        "max_possible_cost_usd": round(input_tokens * pricing["input"] + max_tokens * pricing["output"], 8),
    }


def _calculate_cost(model: str, input_tokens: int, output_tokens: int, provider: str = "") -> float:
    pricing = _get_pricing(model, provider)
    return round(input_tokens * pricing["input"] + output_tokens * pricing["output"], 8)


def _truncate_for_log(text: str, limit: int = MAX_LOG_TEXT_LENGTH) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _tokenize_for_overlap(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower()))


def _estimate_quality(prompt: str, response: str, latency_ms: int) -> tuple[float, str]:
    # Heuristic quality score (0-1): checks useful length, prompt-response overlap, and latency.
    if not response.strip():
        return 0.0, "poor"

    response_len = len(response)
    length_component = min(response_len / 500, 1.0) * 0.45

    prompt_terms = _tokenize_for_overlap(prompt)
    response_terms = _tokenize_for_overlap(response)
    overlap_ratio = 0.0
    if prompt_terms:
        overlap_ratio = len(prompt_terms & response_terms) / len(prompt_terms)
    overlap_component = min(overlap_ratio, 1.0) * 0.35

    latency_component = 0.20 if latency_ms <= 2500 else max(0.0, 0.20 - ((latency_ms - 2500) / 10000))

    score = round(min(length_component + overlap_component + latency_component, 1.0), 4)

    if score >= 0.8:
        label = "excellent"
    elif score >= 0.6:
        label = "good"
    elif score >= 0.4:
        label = "fair"
    else:
        label = "poor"

    return score, label


def _log_to_db(provider, model, prompt, response, latency_ms, cost, quality_score):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO llm_requests
            (provider, model, prompt, response, latency_ms, cost_usd, quality_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (provider, model, prompt, response, latency_ms, cost, quality_score))

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        logger.error("db_logging_failed", extra={"error": str(e)})
        
def _get_dispatch(provider: str) -> Optional[Callable[..., tuple[str, int, int]]]:
    """Return a sender callable for a provider by delegating to the registry.

    Returns None if the provider is unknown or disabled, so the failover
    loop can skip it gracefully. Tests can monkeypatch provider_registry.send.
    """
    cfg = provider_registry.get(provider)
    if cfg is None or not cfg.enabled:
        return None
    # Bind the provider name into the registry's generic send method.
    def _dispatch(prompt: str, model: str, temperature: float, max_tokens: int):
        return provider_registry.send(provider, prompt, model, temperature, max_tokens)
    return _dispatch


def _ordered_providers(preferred: Optional[str]) -> list[str]:
    """Build the failover order: preferred provider first, then the rest.

    If no provider is requested, the round-robin router chooses the primary so
    load is still spread across providers; remaining supported providers form
    the fallback chain in a stable order.
    """
    if preferred is None:
        preferred = router.next_provider()

    order = [preferred]
    for provider in SUPPORTED_PROVIDERS:
        if provider not in order:
            order.append(provider)
    # Deprioritise (but never drop) providers that are currently unhealthy, so a
    # healthy provider is tried first while failover to an unhealthy one is kept
    # as a last resort.
    return health_registry.prioritize(order)


def _invoke_with_failover(
    prompt: str,
    ordered_providers: list[str],
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> dict:
    """Try each provider in order until one succeeds.

    A single request is only ever sent to one provider at a time; the next
    provider is tried only after the current one raises. Returns a dict with the
    winning provider's text/tokens plus failover metadata. Raises
    AllProvidersFailedError if every provider fails.
    """
    attempts: list[dict] = []
    previous_provider: Optional[str] = None

    for provider in ordered_providers:
        dispatch = _get_dispatch(provider)
        if dispatch is None:
            logger.warning("provider_unsupported_skipped", extra={"provider": provider})
            attempts.append({"provider": provider, "error": "unsupported"})
            continue

        model_name = model or _get_default_model(provider)
        if previous_provider is not None:
            FAILOVER_COUNT.labels(from_provider=previous_provider, to_provider=provider).inc()
            logger.warning("provider_failover", extra={
                "from_provider": previous_provider,
                "to_provider": provider,
                "attempt": len(attempts) + 1,
            })

        start_time = time.time()
        try:
            text, input_tokens, output_tokens = dispatch(prompt, model_name, temperature, max_tokens)
            latency_ms = int((time.time() - start_time) * 1000)
            PROVIDER_ATTEMPT_COUNT.labels(provider=provider, outcome="success").inc()
            health_registry.record_success(provider)
            return {
                "provider": provider,
                "model": model_name,
                "text": text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "failover_used": previous_provider is not None,
                "attempts": len(attempts) + 1,
                "failed_providers": [a["provider"] for a in attempts],
            }
        except Exception as e:
            PROVIDER_ATTEMPT_COUNT.labels(provider=provider, outcome="error").inc()
            health_registry.record_failure(provider)
            logger.error("provider_call_failed", extra={
                "provider": provider,
                "error_type": type(e).__name__,
                "error": str(e),
            })
            attempts.append({"provider": provider, "error": f"{type(e).__name__}: {e}"})
            previous_provider = provider

    raise AllProvidersFailedError(attempts)


DEFAULT_ROUTING_MODE = os.getenv("GATEWAY_ROUTING_MODE", "roundrobin")


def send_prompt(
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    routing_mode: Optional[str] = None,
) -> dict:
    overall_start_time = time.time()
    routing_mode = routing_mode or DEFAULT_ROUTING_MODE
    similarity_threshold = get_cache_similarity_threshold()
    bypass_cache = False

    # Tier 1: Exact-match cache lookup in Redis.
    redis_cached = None
    try:
        redis_cached = get_cached_response(prompt)
    except Exception as e:
        bypass_cache = True
        logger.warning("redis_cache_lookup_failed", extra={"error": str(e)})

    if redis_cached is not None:
        latency_ms = int((time.time() - overall_start_time) * 1000)
        cached_response = redis_cached.get("response", "")
        quality_score, quality_label = _estimate_quality(prompt, cached_response, latency_ms)

        logger.info("redis_cache_hit", extra={
            "prompt": _truncate_for_log(prompt),
            "latency_ms": latency_ms,
            "provider": "cache",
            "model": redis_cached.get("model", "exact-cache"),
            "quality_score": quality_score,
            "quality_label": quality_label,
        })

        _log_to_db(
            provider="cache",
            model=redis_cached.get("model", "exact-cache"),
            prompt=prompt,
            response=cached_response,
            latency_ms=latency_ms,
            cost=0.0,
            quality_score=quality_score,
        )

        return {
            "response": cached_response,
            "provider": "cache",
            "model": redis_cached.get("model", "exact-cache"),
            "latency_ms": latency_ms,
            "cache": {
                "hit": True,
                "layer": "redis",
                "similarity": 1.0,
            },
            "quality": {
                "score": quality_score,
                "label": quality_label,
            },
            "tokens": {
                "input": 0,
                "output": 0,
            },
            "cost": {
                "estimated_usd": 0.0,
                "actual_usd": 0.0,
                "max_possible_usd": 0.0,
            },
        }

    prompt_embedding: Optional[list[float]] = None
    cached_match = None
    if not bypass_cache:
        try:
            prompt_embedding = prompt_to_vector(prompt)
            cached_match = search_similar_embedding(
                prompt_embedding,
                similarity_threshold=similarity_threshold,
            )
        except Exception as e:
            bypass_cache = True
            logger.warning("semantic_cache_lookup_failed", extra={"error": str(e)})

    if cached_match is not None:
        latency_ms = int((time.time() - overall_start_time) * 1000)
        cached_response = cached_match["response"]
        quality_score, quality_label = _estimate_quality(prompt, cached_response, latency_ms)

        logger.info("semantic_cache_hit", extra={
            "prompt": _truncate_for_log(prompt),
            "matched_prompt": _truncate_for_log(cached_match["prompt"]),
            "similarity": round(cached_match["similarity"], 4),
            "latency_ms": latency_ms,
            "provider": "cache",
            "model": cached_match.get("model", "semantic-cache"),
            "quality_score": quality_score,
            "quality_label": quality_label,
        })

        _log_to_db(
            provider="cache",
            model=cached_match.get("model", "semantic-cache"),
            prompt=prompt,
            response=cached_response,
            latency_ms=latency_ms,
            cost=0.0,
            quality_score=quality_score,
        )

        # Write-through semantic hit to Redis so future exact requests resolve faster.
        try:
            store_cached_response(
                prompt=prompt,
                response=cached_response,
                provider=cached_match.get("provider", "cache"),
                model=cached_match.get("model", "semantic-cache"),
            )
        except Exception as e:
            logger.warning("redis_cache_store_failed", extra={"error": str(e)})

        return {
            "response": cached_response,
            "provider": "cache",
            "model": cached_match.get("model", "semantic-cache"),
            "latency_ms": latency_ms,
            "cache": {
                "hit": True,
                "layer": "milvus",
                "similarity": round(cached_match["similarity"], 4),
            },
            "quality": {
                "score": quality_score,
                "label": quality_label,
            },
            "tokens": {
                "input": 0,
                "output": 0,
            },
            "cost": {
                "estimated_usd": 0.0,
                "actual_usd": 0.0,
                "max_possible_usd": 0.0,
            },
        }

    # Cache miss on both tiers: choose the provider order, then call live providers.
    # Smart routing (if enabled and available) proposes a supported-provider order;
    # otherwise we fall back to round-robin. Import is lazy to avoid a circular
    # import (smart_router imports names from this module).
    routing_meta = {"mode": routing_mode, "smart_used": False, "reason": None}
    smart_order: list[str] = []
    if routing_mode == "smart" and provider is None:
        from gateway.smart_router import choose_provider_order
        smart_order, routing_meta = choose_provider_order(prompt, mode="smart")

    if smart_order:
        ordered_providers = smart_order
    else:
        ordered_providers = _ordered_providers(provider)

    # Estimate against the first (preferred) provider before calling the API.
    estimated = estimate_cost(prompt, ordered_providers[0], model, max_tokens)

    call = _invoke_with_failover(
        prompt=prompt,
        ordered_providers=ordered_providers,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    provider = call["provider"]
    model_name = call["model"]
    text = call["text"]
    input_tokens = call["input_tokens"]
    output_tokens = call["output_tokens"]
    latency_ms = call["latency_ms"]

    actual_cost = _calculate_cost(model_name, input_tokens, output_tokens, provider)
    quality_score, quality_label = _estimate_quality(prompt, text, latency_ms)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    # Structured log for every request
    logger.info("request_completed", extra={
        "prompt": _truncate_for_log(prompt),
        "prompt_hash": prompt_hash,
        "provider": provider,
        "model": model_name,
        "prompt_length": len(prompt),
        "prompt_preview": prompt[:100],
        "response_preview": _truncate_for_log(text),
        "response_length": len(text),
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": estimated["estimated_cost_usd"],
        "actual_cost_usd": actual_cost,
        "quality_score": quality_score,
        "quality_label": quality_label,
        "cost_accuracy": round(actual_cost / estimated["estimated_cost_usd"], 2) if estimated["estimated_cost_usd"] > 0 else None,
        "tokens_per_second": round(output_tokens / (latency_ms / 1000), 1) if latency_ms > 0 else 0,
        "failover_used": call["failover_used"],
        "attempts": call["attempts"],
        "failed_providers": call["failed_providers"],
        "routing_mode": routing_meta.get("mode"),
        "smart_routing_used": routing_meta.get("smart_used"),
    })

    _log_to_db(
        provider=provider,
        model=model_name,
        prompt=prompt,
        response=text,
        latency_ms=latency_ms,
        cost=actual_cost,
        quality_score=quality_score
    )

    try:
        if prompt_embedding is None:
            prompt_embedding = prompt_to_vector(prompt)
        store_cache_entry(
            prompt=prompt,
            response=text,
            embedding=prompt_embedding,
            provider=provider,
            model=model_name,
        )
    except Exception as e:
        logger.warning("semantic_cache_store_failed", extra={"error": str(e)})

    try:
        store_cached_response(
            prompt=prompt,
            response=text,
            provider=provider,
            model=model_name,
        )
    except Exception as e:
        logger.warning("redis_cache_store_failed", extra={"error": str(e)})

    return {
        "response": text,
        "provider": provider,
        "model": model_name,
        "latency_ms": latency_ms,
        "cache": {
            "hit": False,
            "layer": "live",
            "similarity": None,
        },
        "quality": {
            "score": quality_score,
            "label": quality_label,
        },
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
        "cost": {
            "estimated_usd": estimated["estimated_cost_usd"],
            "actual_usd": actual_cost,
            "max_possible_usd": estimated["max_possible_cost_usd"],
        },
        "failover": {
            "used": call["failover_used"],
            "attempts": call["attempts"],
            "failed_providers": call["failed_providers"],
        },
        "routing": {
            "mode": routing_meta.get("mode"),
            "smart_used": routing_meta.get("smart_used"),
            "reason": routing_meta.get("reason"),
        },
    }

# _send_to_groq and _send_to_ollama have been removed.
# All provider HTTP calls are now handled by gateway.provider_registry.
# To add a new provider: edit configs/providers.yaml — no Python changes needed.


# Example usage
if __name__ == "__main__":

    # # Example 1: Using Groq
    # try:
    #     response = send_prompt(
    #         "Write a hello world in Python",
    #         provider="groq"
    #     )
    #     print("Groq Response:")
    #     print(response)
    #     print("\n" + "="*50 + "\n")
    # except Exception as e:
    #     print(f"Groq Error: {e}")

    # # Example 2: Using Ollama Cloud
    # try:
    #     response = send_prompt(
    #         "Write a hello world in Python",
    #         provider="ollama"
    #     )
    #     print("Ollama Response:")
    #     print(response)
    # except Exception as e:
    #     print(f"Ollama Error: {e}")
    for i in range(4):
        response = send_prompt("Write hello world in Python")
        print(f"Call {i+1} Response:")
        print(response)
        print("="*50)