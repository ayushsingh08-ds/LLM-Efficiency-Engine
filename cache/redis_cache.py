import hashlib
import json
import os
import time
from typing import Optional

import redis


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "llm:prompt")
REDIS_CACHE_TTL_SECONDS = int(os.getenv("REDIS_CACHE_TTL_SECONDS", "86400"))

_client: Optional[redis.Redis] = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2,
        )
    return _client


def _prompt_key(prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{REDIS_KEY_PREFIX}:{digest}"


def get_cached_response(prompt: str) -> Optional[dict]:
    key = _prompt_key(prompt)
    raw = _get_client().get(key)
    if raw is None:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        _get_client().delete(key)
        return None


def store_cached_response(prompt: str, response: str, provider: str, model: str) -> bool:
    key = _prompt_key(prompt)
    payload = {
        "response": response,
        "provider": provider,
        "model": model,
        "created_at": int(time.time()),
    }
    body = json.dumps(payload, ensure_ascii=True)

    if REDIS_CACHE_TTL_SECONDS > 0:
        return bool(_get_client().setex(key, REDIS_CACHE_TTL_SECONDS, body))
    return bool(_get_client().set(key, body))


def invalidate_by_prompt(prompt: str) -> bool:
    key = _prompt_key(prompt)
    return _get_client().delete(key) > 0


def invalidate_by_pattern(pattern: str) -> int:
    client = _get_client()
    keys = list(client.scan_iter(match=pattern, count=500))
    if not keys:
        return 0
    return int(client.delete(*keys))


def invalidate_stale_entries(max_age_seconds: int) -> int:
    if max_age_seconds <= 0:
        return 0

    client = _get_client()
    cutoff = int(time.time()) - max_age_seconds
    deleted = 0
    for key in client.scan_iter(match=f"{REDIS_KEY_PREFIX}:*", count=500):
        ttl = client.ttl(key)
        if ttl == -2:
            continue
        if ttl > 0:
            continue

        raw = client.get(key)
        if raw is None:
            continue

        try:
            payload = json.loads(raw)
            created_at = int(payload.get("created_at", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            created_at = 0

        if created_at == 0 or created_at < cutoff:
            deleted += client.delete(key)

    return int(deleted)


def flush_all() -> bool:
    return bool(_get_client().flushdb())


def get_cache_stats() -> dict:
    client = _get_client()
    stats = client.info("stats")
    keys = client.dbsize()
    return {
        "host": REDIS_HOST,
        "port": REDIS_PORT,
        "db": REDIS_DB,
        "key_prefix": REDIS_KEY_PREFIX,
        "ttl_seconds": REDIS_CACHE_TTL_SECONDS,
        "total_entries": keys,
        "keyspace_hits": stats.get("keyspace_hits", 0),
        "keyspace_misses": stats.get("keyspace_misses", 0),
    }