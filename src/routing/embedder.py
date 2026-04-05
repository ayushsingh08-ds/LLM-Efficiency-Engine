from __future__ import annotations

import re

import numpy as np


def prompt_to_features(prompt: str) -> np.ndarray:
    text = prompt.strip()
    tokens = re.findall(r"\b\w+\b", text)
    token_count = max(1, len(tokens))
    char_count = max(1, len(text))
    avg_token_len = float(sum(len(t) for t in tokens) / token_count)
    punctuation_count = sum(1 for ch in text if ch in ",.;:!?()[]{}")

    code_markers = ["def ", "class ", "sql", "api", "kubernetes", "oauth", "architecture", "latency"]
    technical_hint = 1.0 if any(marker in text.lower() for marker in code_markers) else 0.0

    complexity = np.array(
        [
            char_count / 500.0,
            token_count / 100.0,
            avg_token_len / 12.0,
            punctuation_count / 20.0,
            technical_hint,
        ],
        dtype=np.float32,
    )
    return complexity
