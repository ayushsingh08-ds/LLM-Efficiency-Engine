from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from gateway.llm_provider import (
    estimate_cost,
    get_cache_similarity_threshold,
    logger,
    send_prompt,
    set_cache_similarity_threshold,
)
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
from cache import redis_cache, milvus_cache

app = FastAPI()

# Prometheus metrics
REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["provider", "status"]
)
REQUEST_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency in seconds",
    ["provider"]
)
CACHE_REQUEST_COUNT = Counter(
    "llm_cache_requests_total",
    "Total cache lookup outcomes",
    ["cache_status"]
)
CACHE_RESPONSE_LATENCY = Histogram(
    "llm_cache_response_duration_seconds",
    "Response latency by cache status in seconds",
    ["cache_status"]
)

class PromptRequest(BaseModel):
    prompt: str
    provider: str | None = None

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/estimate")
def estimate(request: PromptRequest):
    return estimate_cost(
        prompt=request.prompt,
        provider=request.provider
    )

@app.post("/generate")
def generate(request: PromptRequest):
    provider = request.provider or "auto"
    resolved_provider = provider
    cache_status = "unknown"
    start = time.time()
    try:
        result = send_prompt(
            prompt=request.prompt,
            provider=request.provider
        )
        resolved_provider = result.get("provider", provider)
        cache_hit = bool(result.get("cache", {}).get("hit", False))
        cache_status = "hit" if cache_hit else "miss"

        REQUEST_COUNT.labels(provider=resolved_provider, status="success").inc()
        CACHE_REQUEST_COUNT.labels(cache_status=cache_status).inc()
        return result
    except Exception as e:
        REQUEST_COUNT.labels(provider=provider, status="error").inc()
        logger.error("request_failed", extra={
            "provider": provider,
            "error_type": type(e).__name__,
            "error": str(e),
        })
        raise
    finally:
        duration = time.time() - start
        REQUEST_LATENCY.labels(provider=resolved_provider).observe(duration)
        CACHE_RESPONSE_LATENCY.labels(cache_status=cache_status).observe(duration)


class CacheInvalidateRequest(BaseModel):
    prompt: str | None = None
    pattern: str | None = None
    layer: str | None = None
    flush_all: bool = False
    stale_only: bool = False
    ttl_seconds: int | None = None


class SimilarityThresholdRequest(BaseModel):
    threshold: float


@app.post("/cache/invalidate")
def invalidate_cache(request: CacheInvalidateRequest):
    """
    Manually invalidate cache entries.
    
    Options:
    - prompt: exact prompt to invalidate
    - pattern: Redis key pattern (e.g., 'llm:prompt:*')
    - layer: 'redis', 'milvus', or 'both' (default: 'both')
    - flush_all: clear entire cache (overrides other options)
    """
    layer = request.layer or "both"
    if layer not in ("redis", "milvus", "both"):
        return {"status": "error", "message": "layer must be one of: redis, milvus, both"}

    invalidated = {}

    if request.stale_only:
        if layer in ("redis", "both"):
            redis_ttl = request.ttl_seconds if request.ttl_seconds is not None else redis_cache.REDIS_CACHE_TTL_SECONDS
            invalidated["redis_stale"] = redis_cache.invalidate_stale_entries(redis_ttl)
        if layer in ("milvus", "both"):
            milvus_ttl = request.ttl_seconds if request.ttl_seconds is not None else milvus_cache.MILVUS_CACHE_TTL_SECONDS
            invalidated["milvus_stale"] = milvus_cache.invalidate_expired_entries(milvus_ttl)
        return {"status": "invalidated_stale", "result": invalidated}

    if request.flush_all:
        if layer in ("redis", "both"):
            redis_ok = redis_cache.flush_all()
            invalidated["redis_flush"] = redis_ok
        if layer in ("milvus", "both"):
            milvus_ok = milvus_cache.flush_all()
            invalidated["milvus_flush"] = milvus_ok
        return {"status": "flushed", "result": invalidated}

    if request.prompt:
        if layer in ("redis", "both"):
            redis_ok = redis_cache.invalidate_by_prompt(request.prompt)
            invalidated["redis_prompt"] = redis_ok
        if layer in ("milvus", "both"):
            milvus_ok = milvus_cache.invalidate_by_prompt(request.prompt)
            invalidated["milvus_prompt"] = milvus_ok
        return {"status": "invalidated_by_prompt", "result": invalidated}

    if request.pattern and layer == "redis":
        count = redis_cache.invalidate_by_pattern(request.pattern)
        return {"status": "invalidated_by_pattern", "redis_deleted": count}

    return {"status": "error", "message": "No invalidation criteria provided"}


@app.get("/cache/stats")
def cache_stats():
    """Get cache statistics for both layers."""
    redis_stats = redis_cache.get_cache_stats()
    milvus_stats = milvus_cache.get_cache_stats()
    return {
        "redis": redis_stats,
        "milvus": milvus_stats,
    }


@app.get("/cache/config")
def get_cache_config():
    return {
        "similarity_threshold": get_cache_similarity_threshold(),
        "redis_ttl_seconds": redis_cache.REDIS_CACHE_TTL_SECONDS,
        "milvus_ttl_seconds": milvus_cache.MILVUS_CACHE_TTL_SECONDS,
    }


@app.put("/cache/config/similarity-threshold")
def update_similarity_threshold(request: SimilarityThresholdRequest):
    try:
        updated = set_cache_similarity_threshold(request.threshold)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "updated",
        "similarity_threshold": updated,
    }


@app.delete("/cache")
def clear_all_cache():
    """Clear all cache entries from both Redis and Milvus."""
    logger.info("cache_clear_all_requested")
    redis_ok = redis_cache.flush_all()
    milvus_ok = milvus_cache.flush_all()
    return {
        "status": "cleared",
        "redis_cleared": redis_ok,
        "milvus_cleared": milvus_ok,
    }