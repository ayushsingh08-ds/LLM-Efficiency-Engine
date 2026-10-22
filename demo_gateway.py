"""Offline end-to-end demo of the live-gateway features.

Runs WITHOUT API keys or infra (Redis/Milvus/Postgres) by mocking the provider
HTTP calls and cache layers. Demonstrates, in order:

  1. Normal request (round-robin) — primary provider succeeds.
  2. Failover — primary raises, fallback provider serves the request.
  3. All-providers-failed — controlled AllProvidersFailedError.
  4. Smart routing — the k-NN router's choice is honoured (with a fake router).
  5. Hardware profiling — a real psutil sample of this host.

Usage:  python demo_gateway.py
"""
from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from gateway import llm_provider, smart_router
from gateway.hardware import sample_hardware
from gateway.llm_provider import AllProvidersFailedError, send_prompt


def _mute_external(stack: ExitStack) -> None:
    """Force cache miss and silence cache/DB writes so no infra is needed."""
    stack.enter_context(patch.object(llm_provider, "get_cached_response", return_value=None))
    stack.enter_context(patch.object(llm_provider, "prompt_to_vector", return_value=[0.0] * 384))
    stack.enter_context(patch.object(llm_provider, "search_similar_embedding", return_value=None))
    stack.enter_context(patch.object(llm_provider, "store_cache_entry", return_value=None))
    stack.enter_context(patch.object(llm_provider, "store_cached_response", return_value=None))
    stack.enter_context(patch.object(llm_provider, "_log_to_db", return_value=None))


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def demo_normal() -> None:
    section("1. Normal request (round-robin) — primary succeeds")
    with ExitStack() as stack:
        _mute_external(stack)
        stack.enter_context(patch.object(llm_provider.router, "next_provider", return_value="groq"))
        stack.enter_context(patch.object(llm_provider, "_send_to_groq", return_value=("[mock groq answer]", 10, 20)))
        result = send_prompt("Explain gradient descent")
    print(f"provider={result['provider']}  failover_used={result['failover']['used']}  "
          f"attempts={result['failover']['attempts']}")


def demo_failover() -> None:
    section("2. Failover — primary (groq) fails, fallback (ollama) serves")
    with ExitStack() as stack:
        _mute_external(stack)
        stack.enter_context(patch.object(llm_provider.router, "next_provider", return_value="groq"))
        stack.enter_context(patch.object(llm_provider, "_send_to_groq", side_effect=RuntimeError("groq 503")))
        stack.enter_context(patch.object(llm_provider, "_send_to_ollama", return_value=("[mock ollama answer]", 5, 15)))
        result = send_prompt("Explain gradient descent")
    print(f"provider={result['provider']}  failover_used={result['failover']['used']}  "
          f"failed_providers={result['failover']['failed_providers']}")


def demo_all_fail() -> None:
    section("3. All providers fail — controlled error")
    with ExitStack() as stack:
        _mute_external(stack)
        stack.enter_context(patch.object(llm_provider.router, "next_provider", return_value="groq"))
        stack.enter_context(patch.object(llm_provider, "_send_to_groq", side_effect=RuntimeError("groq 503")))
        stack.enter_context(patch.object(llm_provider, "_send_to_ollama", side_effect=RuntimeError("ollama 500")))
        try:
            send_prompt("Explain gradient descent")
        except AllProvidersFailedError as e:
            print(f"AllProvidersFailedError raised. attempts={[a['provider'] for a in e.attempts]}")


def demo_smart_routing() -> None:
    section("4. Smart routing — k-NN router's choice is honoured")
    fake_decision = SimpleNamespace(
        provider_name="ollama", model_name="gemma3:4b",
        estimated_cost_usd=0.0, quality_score=7.2, reliability_metric=0.9, neighbor_count=3,
    )
    fake_knn = SimpleNamespace(route=lambda prompt, quality_threshold=6.5: fake_decision)
    with ExitStack() as stack:
        _mute_external(stack)
        smart_router.reset_cache()
        stack.enter_context(patch.object(smart_router, "_load_knn_router", return_value=fake_knn))
        stack.enter_context(patch.object(llm_provider, "_send_to_ollama", return_value=("[mock smart answer]", 4, 8)))
        result = send_prompt("Design a secure multi-tenant API", routing_mode="smart")
    smart_router.reset_cache()
    print(f"provider={result['provider']}  routing={result['routing']}")





def demo_hardware() -> None:
    section("5. Hardware profiling (real psutil sample of this host)")
    s = sample_hardware(cpu_interval=0.1)
    print(f"cpu={s.cpu_percent}%  ram={s.ram_percent}%  gpu_available={s.gpu_available}")
    print(f"active_power_watts={s.estimated_active_power_watts}  energy_is_measured={s.energy_is_measured}")
    print(f"note: {s.notes}")


if __name__ == "__main__":
    demo_normal()
    demo_failover()
    demo_all_fail()
    demo_smart_routing()
    demo_hardware()
    print("\nDemo complete. All flows exercised offline with mock providers.\n")
