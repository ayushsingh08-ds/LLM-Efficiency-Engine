from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class BaseLLMClient(ABC):
    def __init__(
        self,
        provider_name: str,
        model_name: str,
        base_url: str,
        api_key: str | None,
        cost_per_1k_input_tokens: float,
        cost_per_1k_output_tokens: float,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.cost_per_1k_input_tokens = float(cost_per_1k_input_tokens)
        self.cost_per_1k_output_tokens = float(cost_per_1k_output_tokens)

    @abstractmethod
    def invoke(self, prompt: str, temperature: float = 0.2, max_tokens: int = 256) -> LLMResponse:
        raise NotImplementedError

    def estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1000.0) * self.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000.0) * self.cost_per_1k_output_tokens
        return round(input_cost + output_cost, 8)
