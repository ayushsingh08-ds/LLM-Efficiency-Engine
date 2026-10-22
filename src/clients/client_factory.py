from __future__ import annotations

import os
from typing import Any

import yaml

from src.clients.base_client import BaseLLMClient
from src.clients.groq_client import GroqClient
from src.clients.ollama_client import OllamaClient
from src.clients.openai_client import OpenAIClient


class ClientFactory:
    @staticmethod
    def _build_client(provider_cfg: dict[str, Any]) -> BaseLLMClient:
        provider_name = str(provider_cfg["provider_name"]).lower()
        model_name = provider_cfg["model_name"]
        base_url = provider_cfg["base_url"]
        api_key = os.getenv(str(provider_cfg.get("api_env_key", "")))
        cost_in = float(provider_cfg.get("cost_per_1k_input_tokens", 0.0))
        cost_out = float(provider_cfg.get("cost_per_1k_output_tokens", 0.0))

        kwargs = {
            "provider_name": provider_name,
            "model_name": model_name,
            "base_url": base_url,
            "api_key": api_key,
            "cost_per_1k_input_tokens": cost_in,
            "cost_per_1k_output_tokens": cost_out,
        }

        if provider_name == "openai":
            return OpenAIClient(**kwargs)
        if provider_name == "groq":
            return GroqClient(**kwargs)
        if provider_name == "ollama":
            return OllamaClient(**kwargs)

        raise ValueError(f"Unsupported provider: {provider_name}")

    @classmethod
    def load_from_config(cls, config_path: str) -> list[BaseLLMClient]:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        providers = config.get("providers", [])
        if not providers:
            raise ValueError("No providers found in model config")

        return [cls._build_client(p) for p in providers]
