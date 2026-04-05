from __future__ import annotations

import requests

from src.clients.base_client import BaseLLMClient, LLMResponse


class OllamaClient(BaseLLMClient):
    def invoke(self, prompt: str, temperature: float = 0.2, max_tokens: int = 256) -> LLMResponse:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        body = response.json()

        text = body.get("response", "")
        input_tokens = int(body.get("prompt_eval_count", max(1, len(prompt) // 4)))
        output_tokens = int(body.get("eval_count", max(1, len(text) // 4)))

        return LLMResponse(text=text, input_tokens=input_tokens, output_tokens=output_tokens)
