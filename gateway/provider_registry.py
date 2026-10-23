"""
gateway/provider_registry.py
─────────────────────────────
Loads `configs/providers.yaml` once at startup and exposes a clean API
that the rest of the gateway uses instead of hardcoded dicts.

Usage (anywhere in the gateway):
    from gateway.provider_registry import registry

    # List of enabled provider names, e.g. ["groq", "ollama"]
    registry.enabled_names

    # Send a prompt to a named provider
    text, input_tok, output_tok = registry.send(
        provider_name="groq",
        prompt="Hello",
        model=None,           # None → uses default_model from config
        temperature=0.7,
        max_tokens=1024,
    )

    # Cost lookup
    registry.cost_per_token("groq", "llama-3.1-8b-instant")
    # → {"input": 5e-08, "output": 8e-08}

    registry.default_model("groq")
    # → "llama-3.1-8b-instant"
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("llm_gateway")

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "providers.yaml"
_CONFIG_PATH = Path(os.getenv("PROVIDERS_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH)))


# ──────────────────────────────────────────────────────────────────────────────
# Internal data model
# ──────────────────────────────────────────────────────────────────────────────

class ModelConfig:
    def __init__(self, data: dict) -> None:
        self.id: str = data["id"]
        self.tier: str = data.get("tier", "cheap")          # "cheap" | "frontier"
        # Costs stored as USD per single token (config has per-1M for readability)
        self.input_cost_per_token: float  = data.get("input_cost_per_1m", 0.0)  / 1_000_000
        self.output_cost_per_token: float = data.get("output_cost_per_1m", 0.0) / 1_000_000


class ProviderConfig:
    def __init__(self, data: dict) -> None:
        self.name: str            = data["name"]
        self.enabled: bool        = data.get("enabled", True)
        self.api_style: str       = data.get("api_style", "openai")   # "openai" | "ollama"
        self.base_url: str        = data["base_url"]
        self.api_key_env: str     = data.get("api_key_env", "")
        self.default_model_id: str = data["default_model"]
        self.timeout: float       = float(data.get("timeout_seconds", 60))
        self.models: dict[str, ModelConfig] = {
            m["id"]: ModelConfig(m) for m in data.get("models", [])
        }

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env) if self.api_key_env else None


# ──────────────────────────────────────────────────────────────────────────────
# Generic HTTP senders
# ──────────────────────────────────────────────────────────────────────────────

def _send_openai_style(
    cfg: ProviderConfig,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, int, int]:
    """Works for any provider that speaks the OpenAI chat-completions format
    (Groq, OpenAI, Mistral, Together, Anyscale, etc.)."""
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key or ''}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=cfg.timeout)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def _send_ollama_style(
    cfg: ProviderConfig,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, int, int]:
    """Ollama native /api/chat (newline-delimited JSON streaming)."""
    url = f"{cfg.base_url.rstrip('/')}/api/chat"
    headers: dict = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=cfg.timeout)
    resp.raise_for_status()

    full_text = ""
    input_tokens = output_tokens = 0
    for line in resp.text.strip().split("\n"):
        chunk = json.loads(line)
        if "message" in chunk and "content" in chunk["message"]:
            full_text += chunk["message"]["content"]
        if "prompt_eval_count" in chunk:
            input_tokens = chunk["prompt_eval_count"]
        if "eval_count" in chunk:
            output_tokens = chunk["eval_count"]
    return full_text.strip(), input_tokens, output_tokens


_STYLE_MAP = {
    "openai": _send_openai_style,
    "ollama": _send_ollama_style,
}


# ──────────────────────────────────────────────────────────────────────────────
# Registry — the single object the rest of the gateway talks to
# ──────────────────────────────────────────────────────────────────────────────

class ProviderRegistry:
    """
    Loaded once from configs/providers.yaml.

    The rest of the gateway never reads YAML directly — it calls methods here.
    """

    def __init__(self, config_path: Path = _CONFIG_PATH) -> None:
        self._providers: dict[str, ProviderConfig] = {}
        self._load(config_path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Provider config not found: {path}. "
                "Make sure configs/providers.yaml exists."
            )
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        for entry in raw.get("providers", []):
            cfg = ProviderConfig(entry)
            self._providers[cfg.name] = cfg
            status = "enabled" if cfg.enabled else "disabled"
            logger.debug("provider_registry_loaded", extra={"provider": cfg.name, "status": status})

    # ── Query helpers ─────────────────────────────────────────────────────────

    @property
    def enabled_names(self) -> list[str]:
        """Ordered list of enabled provider names, preserving YAML order."""
        return [name for name, cfg in self._providers.items() if cfg.enabled]

    def get(self, name: str) -> Optional[ProviderConfig]:
        return self._providers.get(name)

    def default_model(self, provider_name: str) -> str:
        cfg = self._providers.get(provider_name)
        if cfg is None:
            raise KeyError(f"Unknown provider: {provider_name!r}")
        return cfg.default_model_id

    def cost_per_token(self, provider_name: str, model_id: str) -> dict:
        """Returns {"input": float, "output": float} in USD per token."""
        cfg = self._providers.get(provider_name)
        if cfg is None:
            return {"input": 0.0, "output": 0.0}
        model = cfg.models.get(model_id)
        if model is None:
            return {"input": 0.0, "output": 0.0}
        return {"input": model.input_cost_per_token, "output": model.output_cost_per_token}

    def timeout(self, provider_name: str) -> float:
        cfg = self._providers.get(provider_name)
        return cfg.timeout if cfg else 60.0

    def all_model_ids(self) -> dict[str, list[str]]:
        """Returns {provider_name: [model_id, ...]} for all enabled providers."""
        return {
            name: list(cfg.models.keys())
            for name, cfg in self._providers.items()
            if cfg.enabled
        }

    # ── Sending ───────────────────────────────────────────────────────────────

    def send(
        self,
        provider_name: str,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> tuple[str, int, int]:
        """
        Dispatch a prompt to the named provider.

        Returns (response_text, input_tokens, output_tokens).
        Raises KeyError if provider is unknown/disabled.
        Raises requests.HTTPError / requests.Timeout on network failures.
        """
        cfg = self._providers.get(provider_name)
        if cfg is None or not cfg.enabled:
            raise KeyError(f"Provider {provider_name!r} is not enabled.")

        model_id = model or cfg.default_model_id
        sender = _STYLE_MAP.get(cfg.api_style)
        if sender is None:
            raise ValueError(
                f"Unknown api_style {cfg.api_style!r} for provider {provider_name!r}. "
                f"Supported: {list(_STYLE_MAP)}"
            )

        return sender(cfg, prompt, model_id, temperature, max_tokens)

    def reload(self, config_path: Path = _CONFIG_PATH) -> None:
        """Hot-reload the config (e.g. after editing providers.yaml)."""
        self._providers.clear()
        self._load(config_path)
        logger.info("provider_registry_reloaded", extra={"path": str(config_path)})


# Process-wide singleton — import this everywhere.
registry = ProviderRegistry()
