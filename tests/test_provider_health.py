from __future__ import annotations

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from gateway import llm_provider, smart_router
from gateway import provider_registry as pr_module
from gateway.llm_provider import send_prompt
from gateway.provider_health import ProviderHealthRegistry, registry as global_registry


def _make_send(responses: dict, errors: dict | None = None):
    """Build a side_effect for patching provider_registry.send."""
    errors = errors or {}

    def _send(provider_name, prompt, model=None, temperature=0.7, max_tokens=1024):
        if provider_name in errors:
            raise errors[provider_name]
        if provider_name in responses:
            return responses[provider_name]
        raise KeyError(f"Unexpected provider in test: {provider_name!r}")

    return _send


class TestProviderHealthRegistry(unittest.TestCase):
    def test_unknown_provider_is_healthy(self) -> None:
        reg = ProviderHealthRegistry(failure_threshold=3)
        self.assertTrue(reg.is_healthy("groq"))

    def test_goes_unhealthy_after_threshold(self) -> None:
        reg = ProviderHealthRegistry(failure_threshold=3, recovery_seconds=1000)
        for _ in range(3):
            reg.record_failure("groq")
        self.assertFalse(reg.is_healthy("groq"))

    def test_success_resets_consecutive_failures(self) -> None:
        reg = ProviderHealthRegistry(failure_threshold=3, recovery_seconds=1000)
        reg.record_failure("groq")
        reg.record_failure("groq")
        reg.record_success("groq")
        reg.record_failure("groq")
        self.assertTrue(reg.is_healthy("groq"))  # only 1 consecutive failure now

    def test_recovers_after_cooldown(self) -> None:
        reg = ProviderHealthRegistry(failure_threshold=2, recovery_seconds=30)
        base = 1000.0
        with patch("gateway.provider_health.time.time", return_value=base):
            reg.record_failure("groq")
            reg.record_failure("groq")
            self.assertFalse(reg.is_healthy("groq", now=base))
        # After cooldown, allowed to probe again.
        self.assertTrue(reg.is_healthy("groq", now=base + 31))

    def test_prioritize_puts_healthy_first_but_keeps_all(self) -> None:
        reg = ProviderHealthRegistry(failure_threshold=2, recovery_seconds=1000)
        reg.record_failure("groq")
        reg.record_failure("groq")  # groq unhealthy
        order = reg.prioritize(["groq", "ollama"])
        self.assertEqual(order, ["ollama", "groq"])  # reordered, none dropped
        self.assertEqual(set(order), {"groq", "ollama"})

    def test_snapshot_shape(self) -> None:
        reg = ProviderHealthRegistry()
        reg.record_success("groq")
        reg.record_failure("ollama")
        snap = reg.snapshot()
        self.assertIn("groq", snap)
        self.assertIn("healthy", snap["ollama"])
        self.assertEqual(snap["groq"]["total_successes"], 1)


class TestHealthAwareRoutingOnGateway(unittest.TestCase):
    """Health tracking must coexist with failover, smart routing, and caching."""

    def setUp(self) -> None:
        self._stack = ExitStack()
        self._stack.enter_context(patch.object(llm_provider, "get_cached_response", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "prompt_to_vector", return_value=[0.0] * 384))
        self._stack.enter_context(patch.object(llm_provider, "search_similar_embedding", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "store_cache_entry", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "store_cached_response", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "_log_to_db", return_value=None))
        global_registry.reset()
        smart_router.reset_cache()

    def tearDown(self) -> None:
        self._stack.close()
        global_registry.reset()
        smart_router.reset_cache()

    def test_failure_then_success_records_health(self) -> None:
        send_mock = _make_send(
            responses={"ollama": ("ok", 1, 2)},
            errors={"groq": RuntimeError("boom")},
        )
        with patch.object(llm_provider.router, "next_provider", return_value="groq"), \
             patch.object(pr_module.registry, "send", side_effect=send_mock):
            send_prompt("hi")
        snap = global_registry.snapshot()
        self.assertEqual(snap["groq"]["total_failures"], 1)
        self.assertEqual(snap["ollama"]["total_successes"], 1)

    def test_unhealthy_primary_is_deprioritized_but_still_in_chain(self) -> None:
        # Trip groq's health so round-robin ordering should prefer ollama first.
        for _ in range(3):
            global_registry.record_failure("groq")

        called = []

        def _send(provider_name, prompt, model=None, temperature=0.7, max_tokens=1024):
            called.append(provider_name)
            if provider_name == "groq":
                raise RuntimeError("groq still down")
            return ("ok", 1, 2)

        with patch.object(llm_provider.router, "next_provider", return_value="groq"), \
             patch.object(pr_module.registry, "send", side_effect=_send):
            result = send_prompt("hi")

        # Healthy ollama tried first despite round-robin picking groq.
        self.assertEqual(result["provider"], "ollama")
        self.assertEqual(called[0], "ollama")
        self.assertFalse(result["failover"]["used"])  # ollama was primary, no failover needed

    def test_smart_pick_reordered_when_unhealthy(self) -> None:
        # Smart router picks groq, but groq is unhealthy -> ollama should lead.
        for _ in range(3):
            global_registry.record_failure("groq")
        fake_decision = SimpleNamespace(
            provider_name="groq", model_name="llama-3.1-8b-instant",
            estimated_cost_usd=0.0, quality_score=7.0, reliability_metric=0.9, neighbor_count=3,
        )
        fake_knn = SimpleNamespace(route=lambda prompt, quality_threshold=6.5: fake_decision)
        send_mock = _make_send({"ollama": ("ok", 1, 2)})
        with patch.object(smart_router, "_load_knn_router", return_value=fake_knn), \
             patch.object(pr_module.registry, "send", side_effect=send_mock):
            result = send_prompt("route me", routing_mode="smart")
        self.assertEqual(result["provider"], "ollama")
        self.assertTrue(result["routing"]["smart_used"])

    def test_cache_hit_bypasses_providers_and_health(self) -> None:
        # A Redis exact-match hit must short-circuit before any provider/health logic.
        with patch.object(llm_provider, "get_cached_response",
                          return_value={"response": "cached!", "model": "exact-cache"}), \
             patch.object(pr_module.registry, "send") as send_mock:
            result = send_prompt("hi")
        self.assertTrue(result["cache"]["hit"])
        self.assertEqual(result["provider"], "cache")
        send_mock.assert_not_called()
        # No provider calls => no health state recorded.
        self.assertEqual(global_registry.snapshot(), {})


if __name__ == "__main__":
    unittest.main()
