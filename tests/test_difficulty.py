"""
tests/test_difficulty.py
────────────────────────
Unit tests for the difficulty classifier and difficulty-based routing.
"""

from gateway.difficulty import classify_difficulty
from gateway.provider_registry import registry


def test_classify_cheap_queries():
    # Simple factual lookups and chit-chat should classify as 'cheap'
    cheap_prompts = [
        "What is the capital of France?",
        "Who was the first president of the United States?",
        "Hello, how are you today?",
        "Define photosynthesis.",
        "Translate good morning to Spanish.",
    ]
    for prompt in cheap_prompts:
        res = classify_difficulty(prompt)
        assert res["tier"] == "cheap", f"Expected cheap for prompt: {prompt!r}, got: {res}"
        assert res["confidence"] >= 0.5


def test_classify_frontier_queries():
    # Complex reasoning, code debugging, or system design should classify as 'frontier'
    frontier_prompts = [
        "Please provide a step-by-step mathematical proof of why the square root of 2 is irrational.",
        "Design a distributed consensus architecture using Raft and explain how deadlock is prevented under network partitions.",
        "Here is my code:\n```python\ndef solve(x):\n    return x**2\n```\nFind all memory leaks and refactor it.",
        "Analyze the root cause and time complexity trade-offs between dynamic programming and branch-and-bound for the 0/1 knapsack problem.",
    ]
    for prompt in frontier_prompts:
        res = classify_difficulty(prompt)
        assert res["tier"] == "frontier", f"Expected frontier for prompt: {prompt!r}, got: {res}"
        assert res["confidence"] >= 0.7


def test_provider_registry_providers_by_tier():
    cheap_providers = registry.providers_by_tier("cheap")
    assert "groq" in cheap_providers

    frontier_providers = registry.providers_by_tier("frontier")
    assert "ollama" in frontier_providers


def test_difficulty_routing_integration(monkeypatch):
    """Verify that send_prompt(..., routing_mode='difficulty') prefers matching provider."""
    from gateway.llm_provider import send_prompt

    captured_order = []

    def mock_invoke_with_failover(prompt, ordered_providers, model, temperature, max_tokens):
        captured_order.append(list(ordered_providers))
        return {
            "provider": ordered_providers[0],
            "model": "mock-model",
            "text": "Mock response",
            "input_tokens": 10,
            "output_tokens": 10,
            "latency_ms": 50,
            "failover_used": False,
            "attempts": 1,
            "failed_providers": [],
        }

    monkeypatch.setattr("gateway.llm_provider._invoke_with_failover", mock_invoke_with_failover)
    monkeypatch.setattr("gateway.llm_provider.get_cached_response", lambda p: None)
    monkeypatch.setattr("gateway.llm_provider.search_similar_embedding", lambda emb, similarity_threshold: None)
    monkeypatch.setattr("gateway.llm_provider.store_cached_response", lambda **kw: None)
    monkeypatch.setattr("gateway.llm_provider.store_cache_entry", lambda **kw: None)
    monkeypatch.setattr("gateway.llm_provider._log_to_db", lambda *args, **kw: None)

    # 1. Easy prompt should prioritize cheap tier (groq)
    res_easy = send_prompt("What is the capital of Italy?", routing_mode="difficulty")
    assert res_easy["routing"]["mode"] == "difficulty"
    assert res_easy["routing"]["target_tier"] == "cheap"
    assert captured_order[0][0] == "groq"

    # 2. Hard prompt should prioritize frontier tier (ollama in test fixture)
    res_hard = send_prompt(
        "Design a distributed architecture with Raft consensus and explain deadlock avoidance step-by-step.",
        routing_mode="difficulty"
    )
    assert res_hard["routing"]["mode"] == "difficulty"
    assert res_hard["routing"]["target_tier"] == "frontier"
    assert captured_order[1][0] == "ollama"
