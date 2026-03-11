from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from gateway.llm_provider import send_prompt, estimate_cost, logger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

app = FastAPI()

# Prometheus metrics
REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["provider", "status"]
)
REQUEST_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency in seconds",
    ["provider"]
)

class PromptRequest(BaseModel):
    prompt: str
    provider: str | None = None

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/estimate")
def estimate(request: PromptRequest):
    return estimate_cost(
        prompt=request.prompt,
        provider=request.provider
    )

@app.post("/generate")
def generate(request: PromptRequest):
    provider = request.provider or "auto"
    start = time.time()
    try:
        result = send_prompt(
            prompt=request.prompt,
            provider=request.provider
        )
        REQUEST_COUNT.labels(provider=result["provider"], status="success").inc()
        return result
    except Exception as e:
        REQUEST_COUNT.labels(provider=provider, status="error").inc()
        logger.error("request_failed", extra={
            "provider": provider,
            "error_type": type(e).__name__,
            "error": str(e),
        })
        raise
    finally:
        REQUEST_LATENCY.labels(provider=provider).observe(time.time() - start)