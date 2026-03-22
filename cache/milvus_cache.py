import os
import time
from typing import Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "semantic_cache")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
MILVUS_CACHE_TTL_SECONDS = int(os.getenv("MILVUS_CACHE_TTL_SECONDS", "604800"))

_collection: Optional[Collection] = None
_connected = False


def _connect() -> None:
    global _connected
    if _connected:
        return
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    _connected = True


def get_cache_collection() -> Collection:
    global _collection

    if _collection is not None:
        return _collection

    _connect()

    if not utility.has_collection(MILVUS_COLLECTION):
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="prompt", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="response", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="provider", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="model", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="created_at", dtype=DataType.INT64),
            ],
            description="Semantic cache for LLM prompt-response pairs",
        )

        collection = Collection(name=MILVUS_COLLECTION, schema=schema)
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "IP",
                "index_type": "AUTOINDEX",
                "params": {},
            },
        )
    else:
        collection = Collection(name=MILVUS_COLLECTION)

    collection.load()
    _collection = collection
    return collection


def search_similar_embedding(embedding: list[float], similarity_threshold: float = 0.95) -> Optional[dict]:
    collection = get_cache_collection()

    result = collection.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {}},
        limit=1,
        output_fields=["prompt", "response", "provider", "model", "created_at"],
    )

    if not result or not result[0]:
        return None

    top_hit = result[0][0]
    similarity = float(top_hit.score)
    if similarity < similarity_threshold:
        return None

    entity = top_hit.entity
    created_at = entity.get("created_at")
    if created_at is not None and MILVUS_CACHE_TTL_SECONDS > 0:
        age_seconds = int(time.time()) - int(created_at)
        if age_seconds > MILVUS_CACHE_TTL_SECONDS:
            return None

    return {
        "similarity": similarity,
        "prompt": entity.get("prompt", ""),
        "response": entity.get("response", ""),
        "provider": entity.get("provider", ""),
        "model": entity.get("model", ""),
        "created_at": created_at,
    }


def store_cache_entry(
    prompt: str,
    response: str,
    embedding: list[float],
    provider: str,
    model: str,
) -> None:
    collection = get_cache_collection()

    collection.insert(
        [
            [prompt],
            [response],
            [provider],
            [model],
            [embedding],
            [int(time.time())],
        ]
    )
    collection.flush()


def invalidate_by_prompt(prompt: str) -> bool:
    """Delete a specific cached entry by prompt from Milvus."""
    try:
        collection = get_cache_collection()
        collection.delete(expr=f"prompt == '{prompt.replace(chr(39), chr(92) + chr(39))}'")
        return True
    except Exception:
        return False


def invalidate_expired_entries(ttl_seconds: int = MILVUS_CACHE_TTL_SECONDS) -> int:
    """Delete expired entries older than ttl_seconds."""
    try:
        if ttl_seconds <= 0:
            return 0

        collection = get_cache_collection()
        cutoff_time = int(time.time()) - ttl_seconds
        result = collection.delete(expr=f"created_at < {cutoff_time}")

        if hasattr(result, "delete_count"):
            return int(result.delete_count)
        return 0
    except Exception:
        return 0


def flush_all() -> bool:
    """Clear all cached entries from Milvus."""
    try:
        collection = get_cache_collection()
        collection.delete(expr="id >= 0")
        return True
    except Exception:
        return False


def get_cache_stats() -> dict:
    """Get stats about the Milvus cache."""
    try:
        collection = get_cache_collection()
        count = collection.num_entities
        return {
            "total_entries": count,
            "collection_name": MILVUS_COLLECTION,
        }
    except Exception as e:
        return {"error": str(e)}
