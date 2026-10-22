from __future__ import annotations


def lightweight_hallucination_flag(prompt: str, response: str) -> bool:
    lower = response.lower()

    red_flags = [
        "as confirmed by the latest nasa trial",
        "peer-reviewed proof in 2026 says",
        "i personally verified",
        "this is guaranteed",
    ]

    if any(flag in lower for flag in red_flags):
        return True

    if "http://" in lower or "https://" in lower:
        return False

    if "according to" in lower and len(response) < 60:
        return True

    return False
