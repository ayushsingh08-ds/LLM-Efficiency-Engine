from __future__ import annotations

from src.clients.base_client import BaseLLMClient


QUALITY_JUDGE_PROMPT = """
You are a strict evaluator. Score the assistant response from 1 to 10.
Return JSON only with keys: score, rationale.

User prompt:
{prompt}

Assistant response:
{response}
""".strip()


def llm_assisted_quality_score(
    judge_client: BaseLLMClient,
    prompt: str,
    response: str,
) -> tuple[int | None, str]:
    try:
        judge_result = judge_client.invoke(
            QUALITY_JUDGE_PROMPT.format(prompt=prompt, response=response),
            temperature=0.0,
            max_tokens=120,
        )
        text = judge_result.text.strip()

        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            score = max(1, min(10, int(digits[:2])))
            return score, text

        return None, text
    except Exception as e:
        return None, f"judge_failed: {e}"


def manual_placeholder_score() -> int | None:
    return None
