from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import requests

from gateway import llm_provider
from gateway.llm_provider import AllProvidersFailedError, send_prompt
from gateway.provider_health import registry as health_registry
from gateway import provider_registry as pr_module


def _make_send(responses: dict, errors: dict | None = None):
    """
    Build a side_effect for patching provider_registry.send.

    responses: {provider_name: (text, input_tok, output_tok)}
    errors:    {provider_name: exception_to_raise}
    """
    errors = errors or {}

    def _send(provider_name, prompt, model=None, temperature=0.7, max_tokens=1024):
        if provider_name in errors:
            raise errors[provider_name]
        if provider_name in responses:
            return responses[provider_name]
        raise KeyError(f"Unexpected provider in test: {provider_name!r}")

    return _send


class TestFailover(unittest.TestCase):
    """Failover unit tests.

    The semantic-cache layers (Redis + Milvus) and Postgres logging are patched
    to no-ops so these tests exercise ONLY the live-provider failover path
    without needing external services. Cache lookups return None (forced miss),
    so every request proceeds to the provider dispatch loop.
    """

    def setUp(self) -> None:
        self._stack = ExitStack()
        # Force a cache miss on both tiers.
        self._stack.enter_context(patch.object(llm_provider, "get_cached_response", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "prompt_to_vector", return_value=[0.0] * 384))
        self._stack.enter_context(patch.object(llm_provider, "search_similar_embedding", return_value=None))
        # Silence cache/DB writes.
        self._stack.enter_context(patch.object(llm_provider, "store_cache_entry", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "store_cached_response", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "_log_to_db", return_value=None))
        # Deterministic provider order: groq first, then ollama.
        self._stack.enter_context(patch.object(llm_provider.router, "next_provider", return_value="groq"))
        # Isolate from health state accumulated by other tests (the registry is a
        # process-wide singleton); otherwise prior groq failures reorder providers.
        health_registry.reset()

    def tearDown(self) -> None:
        self._stack.close()
        health_registry.reset()

    def test_primary_provider_succeeds(self) -> None:
        send_mock = _make_send({"groq": ("groq answer", 10, 20)})
        with patch.object(pr_module.registry, "send", side_effect=send_mock) as mock:
            result = send_prompt("hello", provider="groq")

        self.assertEqual(result["provider"], "groq")
        self.assertEqual(result["response"], "groq answer")
        self.assertFalse(result["failover"]["used"])
        self.assertEqual(result["failover"]["attempts"], 1)
        # registry.send was called exactly once (groq only, no ollama)
        calls = [c.args[0] for c in mock.call_args_list]
        self.assertEqual(calls, ["groq"])

    def test_primary_fails_fallback_succeeds(self) -> None:
        send_mock = _make_send(
            responses={"ollama": ("ollama answer", 5, 15)},
            errors={"groq": RuntimeError("groq down")},
        )
        with patch.object(pr_module.registry, "send", side_effect=send_mock):
            result = send_prompt("hello", provider="groq")

        self.assertEqual(result["provider"], "ollama")
        self.assertEqual(result["response"], "ollama answer")
        self.assertTrue(result["failover"]["used"])
        self.assertEqual(result["failover"]["attempts"], 2)
        self.assertEqual(result["failover"]["failed_providers"], ["groq"])

    def test_provider_timeout_triggers_failover(self) -> None:
        # A hung provider surfaces as requests.exceptions.Timeout; failover should catch it.
        send_mock = _make_send(
            responses={"ollama": ("ollama answer", 5, 15)},
            errors={"groq": requests.exceptions.Timeout("timed out")},
        )
        with patch.object(pr_module.registry, "send", side_effect=send_mock):
            result = send_prompt("hello", provider="groq")

        self.assertEqual(result["provider"], "ollama")
        self.assertTrue(result["failover"]["used"])
        self.assertEqual(result["failover"]["failed_providers"], ["groq"])

    def test_multiple_failures_before_success(self) -> None:
        # Extend the supported set to force >1 failure before a success.
        with patch.object(llm_provider, "SUPPORTED_PROVIDERS", ["groq", "ollama", "extra"]), \
             patch.object(llm_provider.router, "next_provider", return_value="groq"), \
             patch.object(llm_provider, "_get_dispatch") as dispatch:
            calls = {
                "groq": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("groq down")),
                "ollama": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ollama down")),
                "extra": lambda *a, **k: ("extra answer", 3, 7),
            }
            dispatch.side_effect = lambda p: calls.get(p)
            result = send_prompt("hello", provider="groq")

        self.assertEqual(result["provider"], "extra")
        self.assertEqual(result["response"], "extra answer")
        self.assertTrue(result["failover"]["used"])
        self.assertEqual(result["failover"]["attempts"], 3)
        self.assertEqual(result["failover"]["failed_providers"], ["groq", "ollama"])

    def test_all_providers_fail(self) -> None:
        send_mock = _make_send(
            responses={},
            errors={
                "groq": RuntimeError("groq down"),
                "ollama": RuntimeError("ollama down"),
            },
        )
        with patch.object(pr_module.registry, "send", side_effect=send_mock):
            with self.assertRaises(AllProvidersFailedError) as ctx:
                send_prompt("hello", provider="groq")

        self.assertEqual(len(ctx.exception.attempts), 2)
        providers = [a["provider"] for a in ctx.exception.attempts]
        self.assertEqual(providers, ["groq", "ollama"])


if __name__ == "__main__":
    unittest.main()
