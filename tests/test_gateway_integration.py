from __future__ import annotations

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from gateway import app as app_module
from gateway import llm_provider, smart_router
from gateway import provider_registry as pr_module
from gateway.app import app
from gateway.provider_health import registry as health_registry


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


class TestGatewayIntegration(unittest.TestCase):
    """End-to-end HTTP tests against the real FastAPI app.

    External services (Redis, Milvus, Postgres) and the live provider HTTP calls
    are patched so the request flows through the real routing/cache/failover code
    without needing network or infra.
    """

    def setUp(self) -> None:
        self.client = TestClient(app)
        self._stack = ExitStack()
        # Force cache miss + silence cache/DB writes.
        self._stack.enter_context(patch.object(llm_provider, "get_cached_response", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "prompt_to_vector", return_value=[0.0] * 384))
        self._stack.enter_context(patch.object(llm_provider, "search_similar_embedding", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "store_cache_entry", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "store_cached_response", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "_log_to_db", return_value=None))
        smart_router.reset_cache()
        # The health registry is a process-wide singleton; isolate each test from
        # provider-failure state left behind by earlier tests.
        health_registry.reset()

    def tearDown(self) -> None:
        self._stack.close()
        smart_router.reset_cache()
        health_registry.reset()

    def test_generate_success_roundrobin(self) -> None:
        send_mock = _make_send({"groq": ("hi there", 3, 5)})
        with patch.object(llm_provider.router, "next_provider", return_value="groq"), \
             patch.object(pr_module.registry, "send", side_effect=send_mock):
            resp = self.client.post("/generate", json={"prompt": "hello"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["provider"], "groq")
        self.assertFalse(body["failover"]["used"])

    def test_generate_failover_over_http(self) -> None:
        send_mock = _make_send(
            responses={"ollama": ("fallback answer", 2, 4)},
            errors={"groq": RuntimeError("groq down")},
        )
        with patch.object(llm_provider.router, "next_provider", return_value="groq"), \
             patch.object(pr_module.registry, "send", side_effect=send_mock):
            resp = self.client.post("/generate", json={"prompt": "hello"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["provider"], "ollama")
        self.assertTrue(body["failover"]["used"])
        self.assertEqual(body["failover"]["failed_providers"], ["groq"])

    def test_generate_all_fail_returns_502(self) -> None:
        send_mock = _make_send(
            responses={},
            errors={"groq": RuntimeError("groq down"), "ollama": RuntimeError("ollama down")},
        )
        with patch.object(llm_provider.router, "next_provider", return_value="groq"), \
             patch.object(pr_module.registry, "send", side_effect=send_mock):
            resp = self.client.post("/generate", json={"prompt": "hello"})
        self.assertEqual(resp.status_code, 502)
        detail = resp.json()["detail"]
        self.assertEqual(detail["error"], "all_providers_failed")
        self.assertEqual(len(detail["attempts"]), 2)

    def test_generate_smart_mode_over_http(self) -> None:
        fake_decision = SimpleNamespace(
            provider_name="ollama", model_name="gemma3:4b",
            estimated_cost_usd=0.0, quality_score=7.2,
            reliability_metric=0.9, neighbor_count=3,
        )
        fake_knn = SimpleNamespace(route=lambda prompt, quality_threshold=6.5: fake_decision)
        send_mock = _make_send({"ollama": ("smart answer", 2, 4)})
        with patch.object(smart_router, "_load_knn_router", return_value=fake_knn), \
             patch.object(pr_module.registry, "send", side_effect=send_mock):
            resp = self.client.post("/generate", json={"prompt": "design a system", "routing_mode": "smart"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["provider"], "ollama")
        self.assertTrue(body["routing"]["smart_used"])

    def test_generate_survives_redis_cache_unavailable(self) -> None:
        # Redis lookup raises -> gateway must bypass cache and still serve a live answer.
        send_mock = _make_send({"groq": ("live answer", 3, 5)})
        with patch.object(llm_provider, "get_cached_response", side_effect=RuntimeError("redis down")), \
             patch.object(llm_provider.router, "next_provider", return_value="groq"), \
             patch.object(pr_module.registry, "send", side_effect=send_mock):
            resp = self.client.post("/generate", json={"prompt": "hello"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["provider"], "groq")
        self.assertFalse(body["cache"]["hit"])

    def test_generate_survives_vector_store_unavailable(self) -> None:
        # Milvus semantic search raises -> gateway must bypass cache and serve live.
        send_mock = _make_send({"groq": ("live answer", 3, 5)})
        with patch.object(llm_provider, "search_similar_embedding", side_effect=RuntimeError("milvus down")), \
             patch.object(llm_provider.router, "next_provider", return_value="groq"), \
             patch.object(pr_module.registry, "send", side_effect=send_mock):
            resp = self.client.post("/generate", json={"prompt": "hello"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["provider"], "groq")
        self.assertFalse(body["cache"]["hit"])

    def test_hardware_endpoint(self) -> None:
        resp = self.client.get("/hardware")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("cpu_percent", body)
        self.assertIn("energy_is_measured", body)

    def test_hardware_endpoint_when_profiling_disabled(self) -> None:
        # Master switch off -> endpoint works but samples nothing (zero overhead).
        from gateway import hardware
        with patch.object(hardware, "profiling_enabled", return_value=False):
            resp = self.client.get("/hardware")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body, {"profiling_enabled": False})
        self.assertNotIn("cpu_percent", body)

    def test_metrics_endpoint_exposes_hardware_gauges(self) -> None:
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("host_cpu_percent", resp.text)


if __name__ == "__main__":
    unittest.main()
