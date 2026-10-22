# Cost-Aware LLM Gateway

An LLM gateway plus an offline model-routing research pipeline. It combines a semantic
cache, multi-provider failover, and cost-aware model selection to reduce redundant and
expensive LLM calls.

> **On the numbers:** any cost/efficiency percentages below are *targets and
> illustrative estimates*, not measured results from this repository. Where a mechanism
> is an estimate (e.g. energy-per-request), it is labelled as such. See **Feature status**
> for exactly what is implemented today.

## Feature status

| Feature | Status | Where |
|---|---|---|
| **Semantic caching** (Redis exact-match + Milvus similarity) | ✅ Implemented & wired into the live request path | `gateway/llm_provider.py`, `cache/` |
| **Multi-provider failover** | ✅ Implemented & tested in the live gateway | `gateway/llm_provider.py`, `tests/test_failover.py` |
| **Smart routing on the live gateway** (k-NN adapter + safe fallback) | ✅ Implemented & tested | `gateway/smart_router.py`, `tests/test_smart_router_gateway.py` |
| **Hardware profiling** (psutil + optional NVML) | ✅ Implemented & tested | `gateway/hardware.py`, `tests/test_hardware.py` |
| **Cost estimation & Prometheus `/metrics`** | ✅ Implemented | `gateway/` |
| **Offline routing pipeline** (benchmark → features → contrastive → k-NN → A/B) | ✅ Implemented (offline; feeds smart routing when trained artifacts exist) | `src/` |

Legend: ✅ implemented and tested · 🟡 partial (see qualifier) · 🚧 planned.

### Partial / planned — honest caveats
- **Hardware energy numbers are estimates** unless a real NVIDIA GPU reports power via
  NVML (`energy_is_measured=true`). On CPU-only hosts the active-power figure is a labelled
  placeholder, not a measurement.
- **Smart routing** needs trained artifacts in `outputs/trained_models/`. Without them it
  silently and safely falls back to round-robin.
- **Observability**: the gateway exposes Prometheus-format metrics at `/metrics`, but the
  bundled `docker-compose.yml` does **not** run a Prometheus server. The provisioned Grafana
  dashboard reads from **Postgres** (the `llm_requests` request log), not Prometheus.

## Overview

Design goals (aspirational targets, not measured here):

- **Semantic caching**: reuse responses for identical/similar queries to cut cost and latency.
- **Smart routing**: pick the cheapest model that still meets a quality threshold.
- **Failover**: automatically retry on another provider when one fails, improving availability.
- **Hardware profiling**: surface CPU/RAM/GPU usage and rough energy estimates per request.

## Architecture

Two loosely-coupled halves:

1. **Live gateway** (`gateway/` + `cache/`) — a FastAPI service. Request flow for
   `POST /generate`:
   `Redis exact-match cache → Milvus semantic cache → live provider call with failover`.
   On a cache miss it selects a provider, and if that provider raises/times out it fails
   over to the next supported provider. Successful live responses are written back to both
   cache tiers. Prometheus metrics and structured JSON logs are emitted throughout.
   Supported live providers: **groq**, **ollama**.

2. **Offline routing pipeline** (`src/`) — benchmarks providers, engineers model-level
   features, trains a contrastive prompt encoder, and evaluates a k-NN router against a
   static baseline via an A/B harness. This runs via `run_all_phases.py`. It produces the
   trained artifacts (`outputs/trained_models/`) that the live gateway's **smart routing**
   mode consumes; the pipeline itself does not serve live traffic.

### How the pieces relate

```
                 ┌─────────────────────── live gateway (gateway/, cache/) ───────────────────────┐
POST /generate → │ Redis exact cache → Milvus semantic cache → provider selection → live call    │ → response
                 │                                    │                     │                     │
                 │                          (miss → store back)     round-robin OR smart          │
                 └────────────────────────────────────┼─────────────────────┼────────────────────┘
                                                       │                     │ reads trained artifacts
   offline pipeline (src/, run_all_phases.py) ─────────┘ (none)              ▼
   benchmark → features → contrastive encoder → k-NN → A/B  ──►  outputs/trained_models/{model_embeddings.csv, prompt_encoder.pt}
```

- **Cache layer** (`cache/`): `redis_cache.py` (exact-match, SHA-256 key, TTL),
  `milvus_cache.py` (vector similarity search), `embeddings.py` (sentence-transformers
  `all-MiniLM-L6-v2`, 384-dim). Called from `gateway/llm_provider.py`.
- **Provider layer** (`gateway/llm_provider.py`): live senders for `groq` and `ollama`,
  with cost estimation, quality heuristics, Postgres request logging, and failover.
- **Routing model** (`src/routing/knn_router.py`): the trained k-NN router. On the live
  path it is adapted by `gateway/smart_router.py`, which maps its choice onto live-supported
  providers and falls back to round-robin when unavailable.

### Semantic caching (implemented)

`POST /generate` resolves a prompt through two cache tiers before any live call:

1. **Redis exact-match** — key = `SHA-256(prompt)`. A hit returns immediately at zero cost.
2. **Milvus semantic** — the prompt is embedded (384-dim) and the nearest cached vector is
   retrieved; it counts as a hit only if similarity ≥ `CACHE_SIMILARITY_THRESHOLD`
   (cosine/IP in `[0,1]`, default **0.95**). A semantic hit is also written through to Redis
   so the next identical prompt resolves at tier 1.
3. **Miss** — the live provider is called (with failover); the response is written back to
   both Milvus and Redis. TTLs: `REDIS_CACHE_TTL_SECONDS` (default 1d),
   `MILVUS_CACHE_TTL_SECONDS` (default 7d).

Cache management endpoints: `GET /cache/stats`, `GET /cache/config`,
`PUT /cache/config/similarity-threshold`, `POST /cache/invalidate`, `DELETE /cache`.
If Redis/Milvus are unreachable, lookups degrade gracefully (logged, treated as a miss)
rather than failing the request.

### Failover (live gateway)

- A per-request provider order is built: the requested/round-robin **primary** first, then
  the remaining supported providers as fallbacks (`_ordered_providers`).
- `_invoke_with_failover` tries each provider **in sequence**. A request is only sent to one
  provider at a time; the next is tried **only after** the current one raises or times out.
  There are no duplicate calls to a provider that already succeeded.
- Every live call has an HTTP timeout (`PROVIDER_TIMEOUT_SECONDS`, default 60s), so a hung
  provider surfaces as a timeout and triggers failover instead of blocking.
- Metrics: `llm_provider_attempts_total{provider,outcome}` and
  `llm_failover_total{from_provider,to_provider}`. Logs: `provider_failover`,
  `provider_call_failed`, `all_providers_failed`.
- If **every** provider fails, `send_prompt` raises `AllProvidersFailedError`, which the
  `/generate` endpoint returns as a controlled **HTTP 502** with the per-provider attempt
  details (rather than an opaque 500). The semantic-cache flow is unchanged by failover.

### Smart routing (live gateway)

The offline k-NN contrastive router can now be used on the live path via
`gateway/smart_router.py`. Enable it per request or globally:

```bash
# Per request
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Design a secure multi-tenant API", "routing_mode": "smart"}'

# Or globally
export GATEWAY_ROUTING_MODE=smart
```

Behaviour:

- Smart mode loads the trained artifacts (`outputs/trained_models/model_embeddings.csv`
  and `prompt_encoder.pt`, overridable via `MODEL_EMBEDDINGS_PATH` / `PROMPT_ENCODER_PATH`).
  The k-NN router picks the cheapest model meeting `SMART_ROUTER_QUALITY_THRESHOLD`.
- **Safe fallback**: if the artifacts or ML deps are missing, or the router picks a
  provider the live gateway can't call, it maps onto the supported set or falls back to
  round-robin. Smart routing never breaks the request path.
- Only applies when no explicit `provider` is set on the request.
- Observability: `llm_routing_decision_total{mode,provider}` metric and
  `smart_router_decision` / `smart_router_fallback` logs. The `/generate` response
  includes a `routing` block (`mode`, `smart_used`, `reason`).

## Quick Start

### Local Development

```bash
# Clone repo
git clone https://github.com/ayushsingh08-ds/LLM-Efficiency-Engine.git
cd LLM-Efficiency-Engine

# Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start all services
docker-compose up -d

# Run app (the FastAPI app lives at gateway/app.py, exported as `app`)
python -m uvicorn gateway.app:app --reload

# Access
# API:            http://localhost:8000
# Interactive docs: http://localhost:8000/docs
# Raw metrics:    http://localhost:8000/metrics   (Prometheus text format; no Prometheus server is bundled)
# Grafana:        http://localhost:3000 (admin/admin) — reads from Postgres, not Prometheus
```

> The bundled `docker-compose.yml` runs Postgres, Redis, Milvus (+ etcd/MinIO), the FastAPI
> gateway, and Grafana. It does **not** run a Prometheus server — `/metrics` is exposed for
> scraping but you must point your own Prometheus at it if you want time-series storage.

### Configuration

Copy `.env.example` to `.env` and fill in what you need. Everything has a working default
except the provider API keys. Groups (see `.env.example` for the full annotated list):

- **Provider keys**: `GROQ_API_KEY`, `OLLAMA_API_KEY` (live gateway); `OPENAI_API_KEY`
  (offline pipeline only). `OLLAMA_API_URL` is the live Ollama Cloud endpoint;
  `OLLAMA_BASE_URL` is the local-Ollama URL used by the offline client.
- **Gateway DB (Postgres)**: `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`.
- **Semantic cache**: `REDIS_*`, `MILVUS_*`, `CACHE_SIMILARITY_THRESHOLD`, `EMBEDDING_DIM`.
- **Failover / routing**: `PROVIDER_TIMEOUT_SECONDS`, `GATEWAY_ROUTING_MODE`,
  `SMART_ROUTER_QUALITY_THRESHOLD`, `MODEL_EMBEDDINGS_PATH`, `PROMPT_ENCODER_PATH`.
- **Hardware estimate**: `ESTIMATED_HOST_ACTIVE_WATTS`.

### Other live endpoints

- `POST /generate` — main entry (see Failover / Smart routing above).
- `POST /estimate` — cost estimate for a prompt without calling a provider.
- `GET /hardware` — current CPU/RAM/GPU sample. Energy is measured only when a real GPU
  reports power via NVML (`energy_is_measured`); otherwise it is a labelled estimate.

### Offline demo (no API keys, no infra)

```bash
python demo_gateway.py
```

Exercises normal → failover → all-fail → smart routing → hardware sample
using mocked providers and cache, so it runs without keys, Redis, Milvus, or Postgres.

### Tests

```bash
# Full suite (pytest). Skip the load-test helper, which is not a unit test.
python -m pytest tests/ --ignore=tests/load_test_gateway.py

# Or exactly as CI runs it (stdlib unittest, discovers all test_*.py):
set PYTHONPATH=.        # PowerShell: $env:PYTHONPATH="."   |  bash: export PYTHONPATH=.
python -m unittest discover -s tests -p "test_*.py"
```

Current suites: `test_knn_router`, `test_phase3_pipeline`, `test_failover`,
`test_smart_router_gateway`, `test_hardware`, `test_gateway_integration`.

## Intelligent LLM Routing: Phase 1

Phase 1 introduces model fingerprinting, benchmark logging, and a model registry pipeline.

### Environment Setup

1. Copy `.env.example` to `.env` and add your API keys:
   - `OPENAI_API_KEY`
   - `GROQ_API_KEY`
   - `OLLAMA_API_KEY` (optional for local Ollama)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Task 1: Run Benchmarks

The benchmark runner executes 20 diverse prompts from `data/benchmark_prompts.csv` across providers defined in `configs/models.yaml`.

```bash
python -m src.benchmarking.run_benchmarks \
	--prompts data/benchmark_prompts.csv \
	--models configs/models.yaml \
	--output outputs/benchmark_results/benchmark_results.csv
```

Optional LLM-assisted quality scoring:

```bash
python -m src.benchmarking.run_benchmarks --judge-provider openai
```

### Task 2: Model Registry Schema and Loading

- SQL schema: `src/registry/schema.sql`
- SQLite bootstrap: `src/registry/db.py`
- Repository pattern implementation: `src/registry/repository.py`

Load engineered model rows into SQLite registry:

```bash
python -m src.registry.load_registry \
	--features outputs/benchmark_results/model_features.csv \
	--db-path outputs/registry/model_registry.sqlite \
	--schema src/registry/schema.sql
```

### Task 3: Feature Engineering

Build model-level numerical features from benchmark logs:

```bash
python -m src.features.feature_engineering \
	--input outputs/benchmark_results/benchmark_results.csv \
	--output outputs/benchmark_results/model_features.csv
```

Engineered features include:

- `cost_to_quality_ratio`
- `speed_score`
- `reliability_metric`
- category-level specialization weights (`spec_weight_*`)

## Intelligent LLM Routing: Phase 2

Phase 2 introduces contrastive training and k-NN routing.

### Task 4: Build Mock Training Dataset (100+ prompts)

```bash
python -m src.training.build_mock_dataset --rows 120 --output data/mock_training_prompts.csv
```

### Task 5: Train Contrastive Embedding Space (Triplet Loss)

```bash
python -m src.training.train_contrastive \
	--prompts data/mock_training_prompts.csv \
	--model-features outputs/benchmark_results/model_features.csv \
	--output-dir outputs/trained_models \
	--epochs 25
```

### Task 6: Run k-NN Router (k=3)

```bash
python -m src.routing.knn_router \
	--prompt "Design a secure multi-tenant API with OAuth and threat modeling" \
	--k 3 \
	--quality-threshold 6.5 \
	--model-embeddings outputs/trained_models/model_embeddings.csv \
	--prompt-encoder outputs/trained_models/prompt_encoder.pt
```

### Unit Tests for Routing Logic

Run just the routing tests, or the whole suite — see the **Tests** section under Quick Start.

```bash
python -m unittest discover -s tests -p "test_knn_router.py"
```

## Intelligent LLM Routing: Phase 3

Phase 3 validates routing quality and compares smart routing against a static baseline.

### Task 7: Quality Validation Pipeline

Score responses on a 1-10 scale using ROUGE-L and optional LLM-as-a-judge.

```bash
python -m src.evaluation.validate_quality \
	--input data/quality_eval_samples.csv \
	--output outputs/reports/quality_scores.csv
```

Optional judge-based blending:

```bash
python -m src.evaluation.validate_quality \
	--input data/quality_eval_samples.csv \
	--output outputs/reports/quality_scores.csv \
	--models configs/models.yaml \
	--judge-provider groq
```

### Task 8: A/B Testing Framework (1,000 requests)

Generate validation requests:

```bash
python -m src.evaluation.build_validation_requests --output data/validation_requests.csv --n 1000
```

Run A/B test (50% smart router, 50% static baseline):

```bash
python -m src.evaluation.ab_test \
	--validation data/validation_requests.csv \
	--model-features outputs/benchmark_results/model_features.csv \
	--model-embeddings outputs/trained_models/model_embeddings.csv \
	--prompt-encoder outputs/trained_models/prompt_encoder.pt \
	--output outputs/reports/ab_test_detailed.csv \
	--summary outputs/reports/ab_test_summary.csv \
	--baseline-model gpt-4o-mini
```

If your baseline model had no successful benchmark calls in this environment, you can provide explicit baseline assumptions:

```bash
python -m src.evaluation.ab_test \
	--validation data/validation_requests.csv \
	--model-features outputs/benchmark_results/model_features.csv \
	--model-embeddings outputs/trained_models/model_embeddings.csv \
	--prompt-encoder outputs/trained_models/prompt_encoder.pt \
	--baseline-model gpt-4o-mini \
	--baseline-cost-usd 0.02 \
	--baseline-latency-ms 1800 \
	--baseline-quality 8.5
```

### Task 9: Cost-Quality Tradeoff Dashboard

```bash
python -m src.dashboard.tradeoff_plot \
	--input outputs/benchmark_results/model_features.csv \
	--output outputs/reports/cost_quality_tradeoff.png \
	--quality-threshold 6.5
```

## End-to-End Runner

Run all three phases in one command.

Local Python:

```bash
python run_all_phases.py --epochs 5 --requests 1000 --quality-threshold 6.5
```

Docker-based execution:

```bash
python run_all_phases.py --docker --epochs 5 --requests 1000 --quality-threshold 6.5
```

## CI

GitHub Actions workflow is available at `.github/workflows/ci.yml`. It installs CPU-only
PyTorch plus `requirements.txt`, then runs the full unit-test suite via
`python -m unittest discover -s tests -p "test_*.py"` (all `test_*.py` files).
