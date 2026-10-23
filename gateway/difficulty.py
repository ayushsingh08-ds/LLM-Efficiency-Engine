"""
gateway/difficulty.py
─────────────────────
Difficulty / Complexity Classifier for Dynamic LLM Routing.

Implements prompt difficulty classification inspired by RouteLLM, FrugalGPT,
and Chatbot Arena benchmark data:
  - 'cheap': Routine fact lookups, short questions, single-turn chit-chat,
    summarization of small text, translations, simple definitions.
    Mapped to fast/low-cost models (e.g., Gemini 2.5 Flash Lite, Groq Llama 3 8B).
  - 'frontier': Multi-step reasoning, mathematical proofs, software architecture,
    code generation/debugging, complex logic puzzles, chain-of-thought instructions,
    long documents/prompts requiring deep context analysis.
    Mapped to high-capability models (e.g., OpenRouter recursive/flint-v1.0:free, GPT-4o, Claude 3.5 Sonnet).

Features:
  - Zero-latency heuristic engine (token count, reasoning keywords, syntax complexity).
  - Clean API returning tier, confidence, reasons, and detected signals.
  - Pluggable into send_prompt and FastAPI endpoints.
"""

from __future__ import annotations

import re
from typing import TypedDict


class DifficultyResult(TypedDict):
    tier: str  # "cheap" | "frontier"
    confidence: float  # 0.0 to 1.0
    signals: list[str]
    reason: str


# Regex patterns associated with complex tasks and reasoning requirements
_FRONTIER_PATTERNS = [
    # Multi-step reasoning & logic
    (r"\b(step[-\s]by[-\s]step|chain[-\s]of[-\s]thought|think[-\s]carefully|reasoning|deduce|prove|proof)\b", "step_by_step_reasoning"),
    (r"\b(if\s+all\s+.+\s+and\s+.+\s+then|syllogism|logic\s+puzzle|paradox|counterexample)\b", "formal_logic"),
    
    # Coding, algorithms, and technical debugging
    (r"\b(def\s+\w+|function\s+\w+|class\s+\w+|public\s+void|async\s+def|import\s+\w+|SELECT\s+.+\s+FROM)\b", "code_syntax"),
    (r"\b(debug|refactor|stack trace|segfault|nullpointer|race condition|deadlock|time complexity|big[-\s]o|algorithm|dynamic programming)\b", "code_debugging_or_algo"),
    (r"\b(architecture|microservices|distributed system|consensus|raft|paxos|event-driven|database schema|sharding)\b", "system_design"),
    
    # In-depth analysis & synthesis
    (r"\b(compare and contrast|critique|pros and cons in detail|trade-offs|root cause analysis|mathematical derivation)\b", "comparative_analysis"),
    (r"\b(differential equation|integral|eigenvalue|linear algebra|probability distribution|bayesian)\b", "advanced_math"),
    (r"\b(write a (test suite|unit test|compiler|parser|interpreter|lexer|neural network|smart contract))\b", "complex_generation"),
]

# Patterns typically indicating simple, low-cost requests
_CHEAP_PATTERNS = [
    (r"^(what|when|where|who)\s+is\s+the\s+(capital|president|currency|founder|population|birthday|definition|meaning)\s+of\b", "factual_lookup"),
    (r"^(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))\b", "chit_chat"),
    (r"^(translate|say)\s+.+\s+in\s+[a-zA-Z]+\.?$", "simple_translation"),
    (r"^(define|what is|what are)\s+[\w\s]{1,30}\??$", "short_definition"),
]


def classify_difficulty(prompt: str) -> DifficultyResult:
    """
    Classify a prompt into 'cheap' or 'frontier' tier.
    
    Heuristics:
    1. Prompt length / token estimation:
       - > 80 words with technical terms or > 120 words => heavily leans 'frontier'.
       - < 15 words matching fact lookup / greeting => 'cheap'.
    2. Pattern & keyword matching:
       - Code blocks (```), reasoning requests, system design => 'frontier'.
    3. Structural indicators (numbered constraints, markdown headers).
    """
    cleaned = prompt.strip()
    words = cleaned.split()
    word_count = len(words)
    signals: list[str] = []
    
    # Quick check: Code blocks (markdown ```)
    if "```" in cleaned:
        signals.append("contains_markdown_code_block")
        return {
            "tier": "frontier",
            "confidence": 0.95,
            "signals": signals,
            "reason": "Prompt contains code blocks requiring deep programming intelligence.",
        }

    # Match frontier patterns
    frontier_matches = []
    for pattern, label in _FRONTIER_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            frontier_matches.append(label)

    # Match cheap patterns
    cheap_matches = []
    for pattern, label in _CHEAP_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            cheap_matches.append(label)

    # Length-based indicators
    if word_count >= 100:
        signals.append("long_prompt (>100 words)")
    elif word_count <= 20:
        signals.append("short_prompt (<=20 words)")

    signals.extend(frontier_matches)
    signals.extend(cheap_matches)

    # Decision logic
    if len(frontier_matches) >= 2 or (len(frontier_matches) >= 1 and word_count >= 50):
        return {
            "tier": "frontier",
            "confidence": min(0.70 + 0.1 * len(frontier_matches), 0.98),
            "signals": signals,
            "reason": f"Detected complex reasoning/technical signals: {', '.join(frontier_matches)}.",
        }

    if frontier_matches and not cheap_matches:
        return {
            "tier": "frontier",
            "confidence": 0.85,
            "signals": signals,
            "reason": f"Detected frontier task indicator: {frontier_matches[0]}.",
        }

    if cheap_matches and not frontier_matches:
        return {
            "tier": "cheap",
            "confidence": 0.90,
            "signals": signals,
            "reason": f"Detected simple lookup/conversational pattern: {cheap_matches[0]}.",
        }

    # If prompt is very short and no complex keywords, route cheap
    if word_count < 25:
        return {
            "tier": "cheap",
            "confidence": 0.75,
            "signals": signals,
            "reason": "Short prompt with no complex reasoning markers.",
        }

    # If long prompt without specific cheap triggers, default to frontier for safety
    if word_count >= 80:
        return {
            "tier": "frontier",
            "confidence": 0.70,
            "signals": signals,
            "reason": "High token count suggests complex multi-paragraph context.",
        }

    # Moderate length default: cheap tier (optimizing efficiency)
    return {
        "tier": "cheap",
        "confidence": 0.65,
        "signals": signals,
        "reason": "Standard query below complexity threshold; routed to cost-efficient tier.",
    }
