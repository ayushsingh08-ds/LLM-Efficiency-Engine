from __future__ import annotations

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from gateway import llm_provider, smart_router
from gateway import provider_registry as pr_module
from gateway.llm_provider import send_prompt


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


class TestSmartRoutingOnGateway(unittest.TestCase):
    """Prove the live gateway actually invokes the smart router in smart mode,
    and safely falls back to round-robin when it is unavailable.

    Cache/DB are patched to no-ops (forced miss) so we exercise routing + dispatch.
    """

    def setUp(self) -> None:
        self._stack = ExitStack()
        self._stack.enter_context(patch.object(llm_provider, "get_cached_response", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "prompt_to_vector", return_value=[0.0] * 384))
        self._stack.enter_context(patch.object(llm_provider, "search_similar_embedding", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "store_cache_entry", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "store_cached_response", return_value=None))
        self._stack.enter_context(patch.object(llm_provider, "_log_to_db", return_value=None))
        smart_router.reset_cache()

    def tearDown(self) -> None:
        self._stack.close()
        smart_router.reset_cache()

    def test_smart_mode_invokes_knn_router_and_uses_its_choice(self) -> None:
        # Fake a loaded k-NN router that picks ollama.
        fake_decision = SimpleNamespace(
            provider_name="ollama",
            model_name="gemma3:4b",
            estimated_cost_usd=0.0,
            quality_score=7.2,
            reliability_metric=0.9,
            neighbor_count=3,
        )
        fake_knn = SimpleNamespace(route=lambda prompt, quality_threshold=6.5: fake_decision)
        send_mock = _make_send({"ollama": ("ollama answer", 4, 8)})

        with patch.object(smart_router, "_load_knn_router", return_value=fake_knn) as loader, \
             patch.object(pr_module.registry, "send", side_effect=send_mock) as send_spy:
            result = send_prompt("Explain OAuth threat modeling", routing_mode="smart")

        loader.assert_called_once()  # the smart router was actually consulted
        self.assertEqual(result["provider"], "ollama")  # its choice was honoured
        self.assertTrue(result["routing"]["smart_used"])
        self.assertEqual(result["routing"]["mode"], "smart")
        # registry.send was called exactly once with "ollama", never with "groq"
        calls = [c.args[0] for c in send_spy.call_args_list]
        self.assertEqual(calls, ["ollama"])

    def test_smart_mode_falls_back_to_roundrobin_when_unavailable(self) -> None:
        # Artifacts missing -> loader returns None -> gateway must still work.
        send_mock = _make_send({"groq": ("groq answer", 10, 20)})
        with patch.object(smart_router, "_load_knn_router", return_value=None), \
             patch.object(llm_provider.router, "next_provider", return_value="groq"), \
             patch.object(pr_module.registry, "send", side_effect=send_mock):
            result = send_prompt("hello", routing_mode="smart")

        self.assertEqual(result["provider"], "groq")
        self.assertFalse(result["routing"]["smart_used"])
        self.assertIsNotNone(result["routing"]["reason"])  # explains the fallback

    def test_smart_choice_of_unsupported_provider_is_mapped(self) -> None:
        # k-NN picks openai (not live-supported) -> mapped onto supported set.
        fake_decision = SimpleNamespace(
            provider_name="openai",
            model_name="gpt-4o-mini",
            estimated_cost_usd=0.02,
            quality_score=8.5,
            reliability_metric=0.95,
            neighbor_count=3,
        )
        fake_knn = SimpleNamespace(route=lambda prompt, quality_threshold=6.5: fake_decision)
        send_mock = _make_send({"groq": ("groq answer", 10, 20)})

        with patch.object(smart_router, "_load_knn_router", return_value=fake_knn), \
             patch.object(pr_module.registry, "send", side_effect=send_mock):
            result = send_prompt("Design a system", routing_mode="smart")

        # Mapped to a supported provider (groq is first in SUPPORTED_PROVIDERS)
        self.assertIn(result["provider"], llm_provider.SUPPORTED_PROVIDERS)
        self.assertTrue(result["routing"]["smart_used"])
        self.assertEqual(result["routing"]["reason"], "unsupported_choice:openai")

    def test_roundrobin_mode_does_not_call_smart_router(self) -> None:
        send_mock = _make_send({"groq": ("groq answer", 10, 20)})
        with patch.object(smart_router, "_load_knn_router") as loader, \
             patch.object(llm_provider.router, "next_provider", return_value="groq"), \
             patch.object(pr_module.registry, "send", side_effect=send_mock):
            result = send_prompt("hello", routing_mode="roundrobin")

        loader.assert_not_called()
        self.assertEqual(result["routing"]["mode"], "roundrobin")
        self.assertFalse(result["routing"]["smart_used"])


if __name__ == "__main__":
    unittest.main()
