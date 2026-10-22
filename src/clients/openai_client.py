from __future__ import annotations

import requests

from src.clients.base_client import BaseLLMClient, LLMResponse


class OpenAIClient(BaseLLMClient):
    def invoke(self, prompt: str, temperature: float = 0.2, max_tokens: int = 256) -> LLMResponse:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is missing")

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(url, json=payload, headers=headers, timeout=90)
        response.raise_for_status()
        body = response.json()

        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", max(1, len(prompt) // 4)))
        output_tokens = int(usage.get("completion_tokens", max(1, len(text) // 4)))

        return LLMResponse(text=text, input_tokens=input_tokens, output_tokens=output_tokens)
