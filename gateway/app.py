from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from gateway.llm_provider import (
    AllProvidersFailedError,
    estimate_cost,
    get_cache_similarity_threshold,
    logger,
    send_prompt,
    set_cache_similarity_threshold,
)
from gateway.provider_registry import registry as provider_registry
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
from cache import redis_cache, milvus_cache
from gateway import hardware as hw_module
from gateway.hardware import sample_hardware, sample_as_dict

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

# Hardware/resource gauges. Refreshed on each /metrics and /hardware scrape.
HW_CPU_PERCENT = Gauge("host_cpu_percent", "Host CPU utilisation percent")
HW_RAM_PERCENT = Gauge("host_ram_percent", "Host RAM utilisation percent")
HW_GPU_UTIL = Gauge("host_gpu_utilization_percent", "GPU utilisation percent", ["gpu"])
HW_GPU_POWER = Gauge("host_gpu_power_watts", "GPU power draw in watts (measured, if available)", ["gpu"])
# Labelled estimated vs measured so dashboards never mistake an estimate for a reading.
HW_ACTIVE_POWER = Gauge(
    "host_active_power_watts",
    "Active power in watts; measured=1 when from GPU telemetry, else a coarse estimate",
    ["measured"],
)


def _refresh_hardware_gauges() -> dict:
    # Respect the master profiling switch: when disabled we sample nothing, touch
    # no gauges, and add zero overhead. The app keeps working normally.
    if not hw_module.profiling_enabled():
        return {"profiling_enabled": False}
    sample = sample_hardware(cpu_interval=0.0)
    HW_CPU_PERCENT.set(sample.cpu_percent)
    HW_RAM_PERCENT.set(sample.ram_percent)
    for g in sample.gpus:
        HW_GPU_UTIL.labels(gpu=str(g["index"])).set(g["utilization_percent"])
        if g["power_watts"] is not None:
            HW_GPU_POWER.labels(gpu=str(g["index"])).set(g["power_watts"])
    HW_ACTIVE_POWER.labels(
        measured="true" if sample.energy_is_measured else "false"
    ).set(sample.estimated_active_power_watts)
    result = sample_as_dict(cpu_interval=0.0)
    result["profiling_enabled"] = True
    return result

class PromptRequest(BaseModel):
    prompt: str
    provider: str | None = None
    # "roundrobin" (default) or "smart". Smart routing only applies when no
    # explicit provider is given; it falls back to round-robin if unavailable.
    routing_mode: str | None = None

@app.get("/metrics")
def metrics():
    # Refresh hardware gauges so a Prometheus scrape reflects current usage.
    try:
        _refresh_hardware_gauges()
    except Exception as e:
        logger.warning("hardware_sample_failed", extra={"error": str(e)})
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health/providers")
def provider_health():
    """Current in-memory provider health (per-process, resets on restart).

    Reflects recent live-call outcomes. Routing deprioritises unhealthy providers
    but never drops them from the failover chain.
    """
    from gateway.provider_health import registry as health_registry

    return {"providers": health_registry.snapshot()}


@app.get("/providers")
def list_providers():
    """List all providers from configs/providers.yaml and their current status."""
    out = []
    from gateway.provider_registry import _CONFIG_PATH
    raw_cfgs = provider_registry._providers
    for name, cfg in raw_cfgs.items():
        out.append({
            "name": name,
            "enabled": cfg.enabled,
            "api_style": cfg.api_style,
            "base_url": cfg.base_url,
            "default_model": cfg.default_model_id,
            "timeout_seconds": cfg.timeout,
            "models": [
                {
                    "id": m.id,
                    "tier": m.tier,
                    "input_cost_per_1m_usd": round(m.input_cost_per_token * 1_000_000, 6),
                    "output_cost_per_1m_usd": round(m.output_cost_per_token * 1_000_000, 6),
                }
                for m in cfg.models.values()
            ],
        })
    return {"providers": out, "enabled": provider_registry.enabled_names}


@app.post("/providers/reload")
def reload_providers():
    """Hot-reload configs/providers.yaml without restarting the gateway.

    After editing the YAML (adding/removing a provider, changing a model),
    POST to this endpoint and changes take effect immediately.
    Note: the round-robin router is NOT rebuilt automatically — restart
    the process if you need to change the provider rotation order.
    """
    try:
        provider_registry.reload()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
    return {
        "status": "reloaded",
        "enabled_providers": provider_registry.enabled_names,
    }


@app.get("/hardware")
def hardware():
    """Current host resource usage. Energy figures are estimates unless
    energy_is_measured is true (real GPU power telemetry)."""
    return _refresh_hardware_gauges()

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
            provider=request.provider,
            routing_mode=request.routing_mode,
        )
        resolved_provider = result.get("provider", provider)
        cache_hit = bool(result.get("cache", {}).get("hit", False))
        cache_status = "hit" if cache_hit else "miss"

        REQUEST_COUNT.labels(provider=resolved_provider, status="success").inc()
        CACHE_REQUEST_COUNT.labels(cache_status=cache_status).inc()
        return result
    except AllProvidersFailedError as e:
        # Every eligible provider failed after failover — controlled 502.
        REQUEST_COUNT.labels(provider=provider, status="error").inc()
        logger.error("all_providers_failed", extra={
            "provider": provider,
            "attempts": e.attempts,
        })
        raise HTTPException(
            status_code=502,
            detail={"error": "all_providers_failed", "attempts": e.attempts},
        )
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