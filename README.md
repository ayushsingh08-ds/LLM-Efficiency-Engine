# LLM Efficiency Engine

### *Production-Grade Cost-Aware Gateway, Semantic Cache & Intelligent Model Router*

[![CI Pipeline](https://github.com/ayushsingh08-ds/LLM-Efficiency-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/ayushsingh08-ds/LLM-Efficiency-Engine/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Redis](https://img.shields.io/badge/Redis-7.0+-DC382D.svg?logo=redis)](https://redis.io)
[![Milvus](https://img.shields.io/badge/Milvus-2.4+-00A1EA.svg?logo=linuxfoundation)](https://milvus.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15.0+-4169E1.svg?logo=postgresql)](https://www.postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Interactive API Docs (Swagger)](http://localhost:8000/docs) · [Empirical Benchmarks](#empirical-benchmark-results--visualizations) · [OpenAI SDK Integration](#openai-sdk-drop-in-integration) · [System Architecture](#system-architecture) · [Quickstart Guide](#local-setup--execution) · [Pre-Flight Verification](#testing--pre-flight-verification)

---

## The Elevator Pitch & Business Impact

The **LLM Efficiency Engine** solves the runaway costs, unpredictable latencies, and vendor-lockin of modern AI applications. By orchestrating a **two-tier semantic cache**, **zero-latency difficulty-aware model routing**, **multi-provider failover chains**, and **atomic budget circuit breakers**, this gateway:

- **Reduces LLM API operating expenditure by up to 68%** by serving cached responses and dynamically directing 80%+ of routine requests to ultra-low-cost model tiers.
- **Slashes repeat query latency from ~1,800ms down to <2ms** (via Redis exact-match) and **<30ms** (via Milvus vector similarity) with zero downstream token consumption.
- **Guarantees zero budget overruns** via PostgreSQL row-level locked transactions (`SELECT ... FOR UPDATE`), halting tenant calls the exact microsecond spending thresholds are met.
- **Requires zero downstream application refactoring** via a 100% drop-in OpenAI-compatible endpoint (`POST /v1/chat/completions`).

---

## System Architecture

```mermaid
flowchart TD
    subgraph ClientLayer ["Client Layer"]
        App["Downstream App / Web UI"]
        OpenAIClient["OpenAI SDK / LangChain / LlamaIndex"]
    end

    subgraph SecurityLayer ["Gateway & Guardrails (FastAPI)"]
        Auth["API Key Auth & RBAC\n(GATEWAY_API_KEY / ADMIN_API_KEY)"]
        Guardrails["Input Bounding & ReDoS Hardening\n(Max 32k Chars, Bounded Regex)"]
        OpenAIRoute["/v1/chat/completions Adapter"]
    end

    subgraph CacheTier ["Two-Tier Semantic Cache"]
        RedisCache[("L1: Redis Exact Cache\n(SHA-256 Key, <2ms)")]
        MilvusCache[("L2: Milvus Vector Cache\n(all-MiniLM-L6-v2, Cosine >= 0.95, <30ms)")]
    end

    subgraph RoutingTier ["Intelligent Routing & Circuit Breaker"]
        BudgetBreaker{"Postgres Budget Breaker\n(SELECT ... FOR UPDATE)"}
        DiffRouter{"Difficulty Classifier\n(RouteLLM / FrugalGPT Heuristics)"}
    end

    subgraph ProviderLayer ["Multi-Provider Failover Mesh"]
        GoogleGemini["Google Gemini 2.5 Flash Lite\n(Cheap Tier)"]
        OpenRouter["OpenRouter / Flint / DeepSeek\n(Frontier Tier)"]
    end

    subgraph Observability ["Telemetry & Persistence"]
        PGLogs[("PostgreSQL\n(Audit Logs & Budgets)")]
        Prometheus["Prometheus /metrics"]
        Grafana["Grafana Dashboards\n(:3000)"]
        Hardware["Host Telemetry\n(psutil + NVML GPU Watts)"]
    end

    App --> Auth
    OpenAIClient --> OpenAIRoute --> Auth
    Auth --> Guardrails
    Guardrails --> RedisCache

    RedisCache -- "Exact Hit (<2ms)" --> App
    RedisCache -- "Miss" --> MilvusCache

    MilvusCache -- "Semantic Hit (<30ms)" --> RedisCache
    MilvusCache -- "Miss" --> BudgetBreaker

    BudgetBreaker -- "Budget Exceeded (HTTP 402)" --> App
    BudgetBreaker -- "Approved" --> DiffRouter

    DiffRouter -- "Simple (Short / Factual)" --> GoogleGemini
    DiffRouter -- "Complex (Logic / Code / System)" --> OpenRouter

    GoogleGemini -- "Failover on 5xx / Timeout" --> OpenRouter
    OpenRouter -- "Failover on 5xx / Timeout" --> GoogleGemini

    GoogleGemini & OpenRouter --> MilvusCache
    GoogleGemini & OpenRouter --> PGLogs
    Hardware --> Prometheus --> Grafana
```

---

## Tech Stack

| Domain | Technologies & Libraries |
| :--- | :--- |
| **Backend & Gateway** | **Python 3.12**, **FastAPI**, **Uvicorn**, **Pydantic v2** (strict payload validation & bounding) |
| **Databases & Vector Stores** | **PostgreSQL 15** (ACID transactional audit & budgets), **Redis 7** (Alpine in-memory cache), **Milvus v2.4.4** (distributed vector database), **MinIO**, **etcd** |
| **AI / Machine Learning** | **Sentence-Transformers** (`all-MiniLM-L6-v2`, 384-dim dense embeddings), **PyTorch**, **RouteLLM / FrugalGPT Heuristic Engine** |
| **Security & Reliability** | **Constant-Time Auth** (`secrets.compare_digest`), **RBAC Key Hierarchy** (`GATEWAY_API_KEY` vs `ADMIN_API_KEY`), **ReDoS Backtracking Guard**, **Vector Injection Sanitizer**, **Docker Localhost Isolation** |
| **Observability & Infra** | **Docker & Docker Compose**, **Prometheus Metrics Exporter** (`/metrics`), **Grafana 11**, **NVIDIA NVML** (`pynvml` real GPU wattage telemetry), **psutil** |

---

## Core Features & Engineering Decisions

### 1. Two-Tier Semantic Caching (Redis L1 + Milvus L2)
- **Implementation**: When a prompt arrives at `/generate` or `/v1/chat/completions`, the engine first hashes the query string with SHA-256 for an $O(1)$ lookup in Redis. If missed, the prompt is embedded into a 384-dimensional vector via `all-MiniLM-L6-v2` and searched against Milvus for cosine similarity $\ge 0.95$. On a vector hit, the response is written back to Redis (write-through) so identical future prompts bypass embedding generation entirely.
- **Engineering Trade-off**: *Why check exact-match Redis before vector search?* Vector embedding models take 20–40ms of CPU compute per invocation. Placing a sub-millisecond Redis key-value check in front guarantees that identical repeated queries bypass vector compute completely, reducing latency by **95%** on hot query paths.

### 2. Zero-Latency Difficulty Routing (RouteLLM & FrugalGPT Heuristics)
- **Implementation**: The engine inspects lexical syntax, multi-step reasoning indicators (`step-by-step`, `chain-of-thought`, `derive`), coding grammar (`def`, `async`, `SELECT`), and token density to classify prompts into `cheap` or `frontier` tiers without invoking an external model.
- **Engineering Trade-off**: *Why use deterministic regex/token heuristics instead of an LLM-as-a-router?* Using an LLM (like GPT-4o-mini) to route every query introduces 300–800ms of network latency and compounds API costs. Our heuristic classifier executes in **<1.5ms**, adding zero overhead while routing 80%+ of common queries to sub-cent models (e.g., Gemini 2.5 Flash Lite at \$0.075/1M tokens).

### 3. Atomic Budget Circuit Breaker with Row-Level Locking
- **Implementation**: For multi-tenant or batch workflows, jobs are provisioned with hard spending allocations via `POST /jobs`. When `/generate` is called with a `job_id`, the gateway calculates prompt and completion costs against remaining funds. It acquires a row-level PostgreSQL lock (`SELECT budget_usd, spent_usd FROM jobs WHERE job_id = ... FOR UPDATE`), performs the threshold check, and instantly returns `HTTP 402 Payment Required` if a single microcent exceeds the cap.
- **Engineering Trade-off**: *Why use PostgreSQL row locks over Redis atomic counters?* While Redis `INCRBYFLOAT` offers sub-millisecond throughput, network partitions or process crashes during multi-step failovers can leave counters desynchronized. PostgreSQL ACID transactions with `FOR UPDATE` guarantee strict financial boundaries across distributed worker nodes.

### 4. Enterprise Security Hardening & ReDoS Protection
- **Implementation**:
  - **Auth & RBAC**: Constant-time key comparison (`secrets.compare_digest`) separates client inference (`GATEWAY_API_KEY`) from admin management (`ADMIN_API_KEY` for cache flushes and budget modifications).
  - **ReDoS Mitigation**: Replaced open-ended regex wildcards (`.+`) with bounded scans (`[^\n]{1,80}`) to prevent catastrophic regular expression backtracking attacks.
  - **Injection Defense**: Invalidation queries escape raw quotes and backslashes to eliminate vector collection wipeouts.
  - **Docker Isolation**: Database and storage ports (5432, 6379, 19530, 9000, 3000) are explicitly bound to `127.0.0.1`, exposing only the FastAPI gateway externally.

---

## Empirical Benchmark Results & Visualizations

The charts below are generated directly from empirical test runs using [`scripts/generate_benchmark_figures.py`](scripts/generate_benchmark_figures.py) across real cache tiers, difficulty classification workloads, and the PostgreSQL budget circuit breaker.

### 1. Executive Performance & Cost Summary Dashboard
![Performance Summary Dashboard](figures/performance_summary_dashboard.png)

### 2. Latency Optimization by Resolution Tier
![Latency Comparison by Tier](figures/latency_comparison_by_tier.png)

- **Redis L1 Exact-Match**: ~1–2ms retrieval for repeated identical prompts (>99% latency reduction).
- **Milvus L2 Semantic Vector Match**: ~28ms semantic vector retrieval for conceptually identical prompts (>98% latency reduction, zero model tokens consumed).
- **Live LLM Provider**: ~1,400ms – 1,850ms network round-trip for raw model execution.

### 3. Cache Latency Distribution Histogram
![Latency Distribution Histogram](figures/latency_distribution_histogram.png)

### 4. Cost Reduction & Difficulty Tier Allocation
![Cost Reduction and Savings](figures/cost_reduction_and_savings.png)

- **Frontier-Only Baseline vs. LLM Efficiency Engine**: Achieves **56.4% to 68% total cost reduction** across 1,000 benchmark queries by automatically directing routine requests to sub-cent cheap model tiers (Gemini 2.5 Flash Lite) and serving hot queries from cache at $0.00 cost.

### 5. PostgreSQL Row-Locked Budget Circuit Breaker
![Budget Circuit Breaker Lifecycle](figures/budget_circuit_breaker.png)

- **Hard Spending Boundary**: Demonstrates active spend accumulation halting immediately at the configured cap with `HTTP 402 Payment Required`, followed by traffic resumption upon budget top-up via `PUT /jobs/{id}/budget`.

---

## OpenAI SDK Drop-In Integration

The gateway natively implements the official OpenAI Chat Completions API schema. You can integrate this gateway into existing Python, TypeScript, LangChain, or LlamaIndex codebases by changing just **two lines of code**:

```python
import os
from openai import OpenAI

# Point client to your LLM Efficiency Gateway
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.getenv("GATEWAY_API_KEY", "your-gateway-api-key"),
)

# Call completions as usual — routing, caching, and failover happen automatically
response = client.chat.completions.create(
    model="auto",  # Uses automatic difficulty routing ("cheap" vs "frontier")
    messages=[
        {"role": "system", "content": "You are a senior software architect."},
        {"role": "user", "content": "Explain Paxos consensus in 3 concise bullet points."},
    ],
)

print(response.choices[0].message.content)
print("Tokens used:", response.usage.total_tokens)
```

---

## Local Setup & Execution

### Prerequisites
- **Python 3.10+** (Python 3.12 recommended)
- **Docker** and **Docker Compose**
- Git

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/ayushsingh08-ds/LLM-Efficiency-Engine.git
cd LLM-Efficiency-Engine

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy the example configuration:
```bash
cp .env.example .env
```
Open `.env` and add your provider keys (e.g. `GEMINI_API_KEY`, `OPENROUTER_API_KEY`). For production, also set `GATEWAY_API_KEY` and `ADMIN_API_KEY`. (In local development, auth defaults to open bypass if keys are omitted).

### 3. Launch Infrastructure (Docker Compose)
Start PostgreSQL, Redis, Milvus, MinIO, etcd, and Grafana:
```bash
docker compose up -d
```
Verify that all containers are healthy:
```bash
docker compose ps
```

### 4. Start the LLM Gateway
Launch the Uvicorn application server:
```bash
python -m uvicorn gateway.app:app --port 8000 --reload
```
The gateway is now live at:
- **API Base URL**: `http://localhost:8000`
- **Swagger Documentation**: `http://localhost:8000/docs`
- **Prometheus Metrics**: `http://localhost:8000/metrics`
- **Grafana Dashboard**: `http://localhost:3000` *(Login: admin / admin)*

---

## Testing & Pre-Flight Verification

The repository contains a 55-test unit suite and an end-to-end pre-flight verification harness that validates security guardrails and live endpoints.

### Run Pre-Flight Security & Functionality Verification
```bash
python scripts/preflight_check.py
```
*Executes tests across:*
1. API Key Auth & RBAC Isolation (401/403 validation)
2. Input Bounding (>32k char rejection with 422) & ReDoS Resistance (<2ms execution)
3. Milvus Invalidation Injection Defense
4. Live LLM Call, Exact Redis Hit (<2ms), and Vector Semantic Hit (<30ms)
5. Budget Enforcement Circuit Breaker (Tripping HTTP 402 on budget exhaustion)

### Run Pytest Unit & Integration Suite
```bash
pytest -v
```

---

## API Reference Summary

| Method | Endpoint | Access | Description |
| :--- | :--- | :---: | :--- |
| `POST` | `/v1/chat/completions` | Client Key | **OpenAI SDK Drop-In** chat completion endpoint with automatic tiering & caching. |
| `POST` | `/generate` | Client Key | Gateway generation endpoint supporting `routing_mode="difficulty" \| "smart" \| "roundrobin"`. |
| `POST` | `/estimate` | Client Key | Computes input tokens and predicted USD cost without querying providers. |
| `POST` | `/difficulty/classify`| Client Key | Inspects prompt difficulty tier (`cheap` vs `frontier`), confidence, and detected signals. |
| `GET` | `/health/providers` | Public | Live provider health registry and consecutive failure counters. |
| `GET` | `/providers` | Public | Lists active model configurations and per-token pricing tiers. |
| `POST` | `/providers/reload` | Admin Key | **Hot-reloads** `configs/providers.yaml` without restarting the process. |
| `POST` | `/jobs` | Admin Key | Provisions a new job budget allocation with an optional tenant ID. |
| `PUT` | `/jobs/{id}/budget` | Admin Key | Tops up or adjusts the spending ceiling for an active job. |
| `POST` | `/cache/invalidate` | Admin Key | Selectively purges cached prompts by string, Redis key pattern, or TTL staleness. |
| `DELETE`| `/cache` | Admin Key | Flushes both Redis exact-match and Milvus semantic vector caches. |
| `GET` | `/hardware` | Public | Live host CPU, RAM, and GPU telemetry (with NVML active power wattage). |
| `GET` | `/metrics` | Public | Standard Prometheus text-format scrape endpoint. |

---

## License
Distributed under the **MIT License**. See `LICENSE` for more information.
