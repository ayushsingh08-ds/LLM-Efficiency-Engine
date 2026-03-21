from typing import Literal, Optional
import requests
import json
import time
import logging
import psycopg2
import re
import hashlib
from pythonjsonlogger import jsonlogger
from gateway.router import RoundRobinRouter
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
# API Configuration - HARDCODED (WARNING: Not secure for production!)
OLLAMA_API_URL = "https://ollama.com/api"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}
router = RoundRobinRouter(["groq", "ollama"])

# Pricing per token (USD) — source: console.groq.com/docs/pricing
MODEL_PRICING = {
    "llama-3.1-8b-instant": {"input": 0.05 / 1_000_000, "output": 0.08 / 1_000_000},
    "gemma3:4b":            {"input": 0.0,               "output": 0.0},
}

DEFAULT_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "ollama": "gemma3:4b",
}

MAX_LOG_TEXT_LENGTH = 500


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

    model_name = model or DEFAULT_MODELS.get(provider, "unknown")
    pricing = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})

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


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
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
        
def send_prompt(
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> dict:

    if provider is None:
        provider = router.next_provider()

    model_name = model or DEFAULT_MODELS.get(provider, "unknown")

    # Estimate before calling API
    estimated = estimate_cost(prompt, provider, model, max_tokens)

    start_time = time.time()

    if provider == "groq":
        text, input_tokens, output_tokens = _send_to_groq(prompt, model_name, temperature, max_tokens)

    elif provider == "ollama":
        text, input_tokens, output_tokens = _send_to_ollama(prompt, model_name, temperature, max_tokens)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    latency_ms = int((time.time() - start_time) * 1000)

    actual_cost = _calculate_cost(model_name, input_tokens, output_tokens)
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

    return {
        "response": text,
        "provider": provider,
        "model": model_name,
        "latency_ms": latency_ms,
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
    }

def _send_to_groq(prompt: str, model: str, temperature: float, max_tokens: int) -> tuple[str, int, int]:
    """Send prompt to Groq API. Returns (text, input_tokens, output_tokens)."""

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    return text, input_tokens, output_tokens


def _send_to_ollama(prompt: str, model: str, temperature: float, max_tokens: int) -> tuple[str, int, int]:
    """Send prompt to Ollama Cloud API. Returns (text, input_tokens, output_tokens)."""

    url = f"{OLLAMA_API_URL}/chat"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    # Ollama Cloud returns newline-delimited JSON chunks
    full_text = ""
    input_tokens = 0
    output_tokens = 0
    for line in response.text.strip().split("\n"):
        chunk = json.loads(line)
        if "message" in chunk and "content" in chunk["message"]:
            full_text += chunk["message"]["content"]
        if "prompt_eval_count" in chunk:
            input_tokens = chunk["prompt_eval_count"]
        if "eval_count" in chunk:
            output_tokens = chunk["eval_count"]

    return full_text.strip(), input_tokens, output_tokens


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