from typing import Literal, Optional
import requests
import json
import time
import psycopg2
from gateway.router import RoundRobinRouter
from dotenv import load_dotenv
import os
load_dotenv()
# API Configuration - HARDCODED (WARNING: Not secure for production!)
OLLAMA_API_URL = "https://ollama.com/api"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}
router = RoundRobinRouter(["groq", "ollama"])


def _log_to_db(provider, model, prompt, response, latency_ms, cost):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO llm_requests
            (provider, model, prompt, response, latency_ms, cost_usd)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (provider, model, prompt, response, latency_ms, cost))

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print("Logging failed:", e)
        
def send_prompt(
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> str:

    if provider is None:
        provider = router.next_provider()

    start_time = time.time()

    if provider == "groq":
        model_name = model or "llama-3.1-8b-instant"
        response = _send_to_groq(prompt, model_name, temperature, max_tokens)

    elif provider == "ollama":
        model_name = model or "gemma3:4b"
        response = _send_to_ollama(prompt, model_name, temperature, max_tokens)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    latency_ms = int((time.time() - start_time) * 1000)

    # Temporary simple cost logic (you can refine later)
    cost = 0.0

    _log_to_db(
        provider=provider,
        model=model_name,
        prompt=prompt,
        response=response,
        latency_ms=latency_ms,
        cost=cost
    )

    return response

def _send_to_groq(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """Send prompt to Groq API"""

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def _send_to_ollama(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """Send prompt to Ollama Cloud API"""

    url = f"{OLLAMA_API_URL}/chat"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    # Ollama Cloud returns newline-delimited JSON chunks
    full_text = ""
    for line in response.text.strip().split("\n"):
        chunk = json.loads(line)
        if "message" in chunk and "content" in chunk["message"]:
            full_text += chunk["message"]["content"]

    return full_text.strip()


# Example usage
if __name__ == "__main__":

    # # Example 1: Using Groq
    # try:
    #     response = send_prompt(
    #         "Write a hello world in Python",
    #         provider="groq"
    #     )
    #     print("Groq Response:")
    #     print(response)
    #     print("\n" + "="*50 + "\n")
    # except Exception as e:
    #     print(f"Groq Error: {e}")

    # # Example 2: Using Ollama Cloud
    # try:
    #     response = send_prompt(
    #         "Write a hello world in Python",
    #         provider="ollama"
    #     )
    #     print("Ollama Response:")
    #     print(response)
    # except Exception as e:
    #     print(f"Ollama Error: {e}")
    for i in range(4):
        response = send_prompt("Write hello world in Python")
        print(f"Call {i+1} Response:")
        print(response)
        print("="*50)