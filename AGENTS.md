# AGENTS.md — Project Memory

Persistent notes for AI agents working on this repo. Keep concise and current.

## What this project is

Two loosely-coupled halves:

1. **Live gateway** (`gateway/` + `cache/`) — a FastAPI service that serves LLM
   requests through: **Redis exact-match cache → Milvus semantic cache → live
   provider call with failover**. Emits Prometheus metrics + structured JSON logs.
   - Entry point: **`gateway.app:app`** (NOT `app.main` — that never existed;
     the README used to be wrong and is now fixed).
   - Live-supported providers: **groq**, **ollama** only. `openai` exists in
     `configs/models.yaml` + `src/clients/` for the OFFLINE pipeline but has no
     live dispatch path.
2. **Offline routing pipeline** (`src/`) — benchmark → feature engineering →
   registry → contrastive training → k-NN router → quality validation → A/B test
   → tradeoff plot. Orchestrated by `run_all_phases.py`. Does NOT serve live
   traffic on its own.

## Feature status (verified 2026-09-21)

| Feature | Status |
|---|---|
| Semantic caching (Redis + Milvus), wired into `send_prompt` | ✅ pre-existing, working |
| Multi-provider failover (live gateway) | ✅ implemented this task |
| Smart routing on live gateway (k-NN adapter) | ✅ implemented this task |
| Health-aware provider prioritization (in-memory) | ✅ implemented 2026-09-22 |
| Hardware profiling (psutil + optional NVML) | ✅ implemented this task |
| Offline ML pipeline | ✅ pre-existing |

## Work done in this task

### 1. Failover (`gateway/llm_provider.py`, `gateway/app.py`)
- `SUPPORTED_PROVIDERS = ["groq", "ollama"]`; `router` now built from it.
- `_get_dispatch(provider)` → sender fn or None. `_ordered_providers(preferred)`
  builds the failover chain (preferred first, then the rest).
- `_invoke_with_failover(...)` tries each provider **in sequence**, one at a time,
  only advancing after the current one raises. Returns winning provider +
  `failover_used`/`attempts`/`failed_providers`. Raises `AllProvidersFailedError`
  if all fail.
- Added HTTP `timeout=PROVIDER_TIMEOUT_SECONDS` (default 60s) to `_send_to_groq`
  and `_send_to_ollama` (previously NO timeout — a hung provider blocked forever).
- Metrics: `llm_provider_attempts_total{provider,outcome}`,
  `llm_failover_total{from_provider,to_provider}` (default Prometheus registry, so
  exported by existing `/metrics`).
- `/generate` maps `AllProvidersFailedError` → **HTTP 502** with attempt details.
- Fixed a stale/misleading comment claiming API config was "HARDCODED" — it reads
  from env/.env.

### 2. Documentation honesty (`README.md`)
- Fixed `uvicorn app.main:app` → `uvicorn gateway.app:app`. (Dockerfile was
  already correct.)
- Added **Feature status** table (implemented vs in-progress) and **Architecture**
  section. Marked all cost/efficiency percentages as *targets/estimates*,
  not measured results. Documented failover + smart routing.

### 3. Smart routing on the live gateway (`gateway/smart_router.py`)
- `choose_provider_order(prompt, mode)`: in `"smart"` mode, lazily loads the
  trained k-NN router (`KNNRouter`) from `outputs/trained_models/*`. Maps its
  choice onto `SUPPORTED_PROVIDERS`. **Safe fallback** to round-robin if artifacts
  missing / deps missing / route errors / unsupported provider chosen.
- `send_prompt(..., routing_mode=...)`; API `PromptRequest.routing_mode`
  ("roundrobin" default | "smart"); env `GATEWAY_ROUTING_MODE`. Smart routing only
  triggers when no explicit `provider` is given.
- Lazy import inside `send_prompt` avoids a circular import (smart_router imports
  from llm_provider).
- Metric `llm_routing_decision_total{mode,provider}`; `routing` block in response.

### 4. Hardware profiling (`gateway/hardware.py`)
- `sample_hardware()` → CPU% / core counts / RAM (psutil), and GPU util/mem/power
  via **optional, import-guarded** `pynvml`. Runs fine with no GPU/NVML.
- Energy is only `energy_is_measured=True` when real GPU power (NVML mW→W) is read;
  otherwise a coarse `ESTIMATED_HOST_ACTIVE_WATTS` placeholder, clearly flagged.
  Never hard-codes a specific chip's wattage as fact.
- Prometheus gauges (`host_cpu_percent`, `host_ram_percent`,
  `host_gpu_utilization_percent`, `host_gpu_power_watts`,
  `host_active_power_watts{measured}`), refreshed on `/metrics` and `/hardware`.

### 5. Testing / validation
- Added `psutil` to `requirements.txt`.
- CI (`.github/workflows/ci.yml`) now discovers **all** `test_*.py` (was only 2 files).
- `demo_gateway.py`: offline end-to-end demo (mock providers, no infra/keys) —
  exercises normal/failover/all-fail/smart/hardware.

## Files changed / added this task
- Modified: `gateway/llm_provider.py`, `gateway/app.py`, `README.md`,
  `requirements.txt`, `.github/workflows/ci.yml`
- Added: `gateway/smart_router.py`, `gateway/hardware.py`,
  `demo_gateway.py`, `tests/test_failover.py`,
  `tests/test_smart_router_gateway.py`,
  `tests/test_hardware.py`, `tests/test_gateway_integration.py`, `AGENTS.md`

## Tests — current status
`python -m pytest tests/ --ignore=tests/load_test_gateway.py` → **57 passed** (VERIFIED
2026-09-22 via the actual runner, 45s). Also passes via CI's `unittest discover -s tests
-p "test_*.py"`. Exact per-file breakdown (from `pytest --co`):
gateway_integration(9), provider_health(10), hardware(7), failover(5),
smart_router_gateway(4), phase3_pipeline(3), knn_router(2).

## How to run
- Tests: `python -m pytest tests/ --ignore=tests/load_test_gateway.py`
- Offline demo (no keys/infra): `python demo_gateway.py`
- Gateway: `python -m uvicorn gateway.app:app --reload` (needs Redis/Milvus/Postgres
  via `docker-compose up -d` for full functionality; cache/DB failures are caught
  and downgrade gracefully).
- Offline pipeline: `python run_all_phases.py --epochs 5 --requests 1000` (needs API
  keys for the benchmark step).

## Known limitations / assumptions (do not overstate these)
- **Energy/power numbers are estimates** unless a real NVIDIA GPU reports power via
  NVML (`energy_is_measured=true`). On CPU-only hosts it's a labelled placeholder.
- README savings percentages (cost, efficiency, etc.) are **targets, not
  measured** in this repo. Do not present them as results.
- Smart routing needs trained artifacts in `outputs/trained_models/`; without them
  it silently falls back to round-robin (by design).
- `run_all_phases.py` was NOT fully run here (benchmark step needs live API keys and
  there's no `outputs/` yet). `build_mock_dataset` requires `--rows >= 90`.
- Tests mock Redis/Milvus/Postgres and provider HTTP — they validate gateway logic,
  not real external-service integration.

## Documentation pass (2026-09-21, follow-up task)

Made README + `.env.example` consistent with the code. Discrepancies found & fixed:

1. **Stale feature-status table** — after the failover task, README still marked smart
   routing / hardware as "🚧 In progress". They are implemented & tested. Table now:
   caching/failover/smart-routing/hardware = ✅.
2. **No Prometheus server** — README's access block advertised `Prometheus:9090`, but
   `docker-compose.yml` has NO prometheus service. Grafana's provisioned datasource is
   **Postgres** (`grafana/provisioning/datasources/postgres.yml`) and the dashboard has 8
   postgres panels, 0 prometheus. Fixed: `/metrics` is exposed for scraping but no server is
   bundled; Grafana reads Postgres `llm_requests`.
3. **`.env.example` was stale/wrong** — had `OLLAMA_BASE_URL` (offline client) but the LIVE
   gateway reads `OLLAMA_API_URL` (was undocumented → effectively dead config). Was missing
   all new vars. Rewrote it grouped: provider keys, gateway DB, cache (Redis/Milvus/threshold),
   failover/routing, hardware estimate, offline registry. Documented both
   OLLAMA_* vars and which half uses each.
4. **CI section stale** — README still listed the 2 old test files; CI now runs
   `unittest discover -p "test_*.py"`. Updated README CI section + the two in-body test
   snippets. Added a full **Tests** + **Offline demo** + **Configuration** + **Other live
   endpoints** section under Quick Start.
5. Added an **architecture relationship** diagram + prose (live gateway ↔ offline pipeline ↔
   routing model ↔ cache ↔ provider layer) and a dedicated **Semantic caching** subsection
   (two-tier flow, SHA-256 Redis key, 384-dim Milvus, threshold 0.95, write-through, TTLs,
   graceful degradation, cache endpoints).

Verified: `CACHE_SIMILARITY_THRESHOLD` in docker-compose is **0.95** (correct; the original
audit's "6.5" concern did not apply here). Confirmed `gateway.app:app` imports to a FastAPI
instance and `unittest discover -s tests -p "test_*.py"` → 31 passed.

Files changed in this pass: `README.md`, `.env.example`, `AGENTS.md`. No code changed.

## Smart routing + provider health (2026-09-22, follow-up task)

Task: connect the offline k-NN router to the LIVE gateway and make it **coexist with
provider health**. Smart routing itself was already wired (see section 3 above); the
real gap was health-aware selection. Added `gateway/provider_health.py` and wired it in.

### Exact live request flow (`send_prompt` in `gateway/llm_provider.py`)
1. **Redis exact-match** (`get_cached_response`) → hit short-circuits, `provider="cache"`,
   no provider/health logic runs at all.
2. **Milvus semantic** (`prompt_to_vector` → `search_similar_embedding`, threshold 0.95)
   → hit returns cached answer.
3. **Cache miss → choose provider order:**
   - If `routing_mode == "smart"` **and** no explicit `provider`: lazy-import
     `smart_router.choose_provider_order(prompt, mode="smart")` → returns a
     supported-provider failover chain (or `[]` = "no override").
   - Else: `_ordered_providers(preferred)` (round-robin primary + rest).
   - **Both paths end with `health_registry.prioritize(order)`** — healthy providers
     first, unhealthy ones moved to the back but **never dropped**.
4. **`_invoke_with_failover(ordered_providers, ...)`** tries each in sequence; on each
   attempt records `health_registry.record_success/record_failure(provider)`.
   All fail → `AllProvidersFailedError` → HTTP 502.
5. Success → write-through to both cache tiers, structured log, `routing` block in response.

### Router loading & config (`gateway/smart_router.py`)
- Lazy `_load_knn_router()` loads `KNNRouter` from `MODEL_EMBEDDINGS_PATH`
  (`outputs/trained_models/model_embeddings.csv`) + `PROMPT_ENCODER_PATH`
  (`outputs/trained_models/prompt_encoder.pt`). Load attempted once, cached;
  `reset_cache()` for tests. Heavy ML import (torch/sklearn) deferred into this fn.
- Config env: `GATEWAY_ROUTING_MODE`, `MODEL_EMBEDDINGS_PATH`, `PROMPT_ENCODER_PATH`,
  `SMART_ROUTER_QUALITY_THRESHOLD` (default 6.5).

### Fallback behavior (smart routing is an enhancement, never a hard dependency)
Falls back to round-robin (`[]` order, `smart_used=False`) when: artifacts missing,
ML deps/load fail, `knn.route()` raises. If k-NN picks an **unsupported** provider
(e.g. `openai`), maps onto SUPPORTED_PROVIDERS (`smart_used=True`, reason
`unsupported_choice:<p>`). All paths still pass through `health_registry.prioritize`.

### Provider health (`gateway/provider_health.py`)
- `ProviderHealthRegistry` singleton `registry`. Unknown provider = healthy. Unhealthy
  after `FAILURE_THRESHOLD` (env `PROVIDER_UNHEALTHY_AFTER_FAILURES`, default 3)
  **consecutive** failures, until `RECOVERY_SECONDS` (env
  `PROVIDER_HEALTH_RECOVERY_SECONDS`, default 30) elapse since last failure (then a
  probe is allowed) or a success resets it. `prioritize()` is a **stable** reorder
  (healthy first) that never drops a provider. Thread-locked. `/health/providers`
  endpoint returns `snapshot()`.
- **Limitations (honest):** in-memory, per-process only — NOT shared across
  workers/replicas, resets on restart (multi-worker would need Redis). Coarse
  heuristic, not a full circuit breaker with half-open probe scheduling.

### Coexistence semantics
Smart pick still leads UNLESS currently unhealthy → a healthy provider is tried first,
the unhealthy pick stays in the chain as last-resort fallback. `meta["health_reordered"]`
flags when this happened. Failover and caching are unchanged.

### Tests added / status
- `tests/test_provider_health.py` (**10 tests**): registry unit tests (unknown healthy,
  threshold trip, success reset, cooldown recovery via patched `time.time`, prioritize
  keeps-all, snapshot shape) + gateway integration (health recorded on failure/success,
  unhealthy primary deprioritized but still in chain, smart pick reordered when unhealthy,
  cache hit bypasses providers/health entirely).
- **Bug fixed:** the global `registry` singleton accumulated failure state across test
  classes, so earlier groq failures reordered providers and broke
  `test_failover.py::test_primary_provider_succeeds` and
  `test_gateway_integration.py::test_generate_success_roundrobin` (were 2 failed / 39
  passed). Fix: `health_registry.reset()` in both classes' `setUp`/`tearDown` (same
  isolation pattern as `test_provider_health.py`).
- **Full suite now: 41 passed** (`pytest` and `unittest discover -s tests -p "test_*.py"`).
  Prior 31 + 10 new health tests. `gateway.app:app` confirmed importing to a FastAPI instance.

### Files changed this task
- Modified: `gateway/llm_provider.py` (record health in `_invoke_with_failover`,
  `health_registry.prioritize` in `_ordered_providers`), `gateway/smart_router.py`
  (prioritize on both return paths + `health_reordered` meta), `gateway/app.py`
  (`/health/providers`), `tests/test_failover.py`, `tests/test_gateway_integration.py`
  (registry reset in setUp/tearDown), `AGENTS.md`.
- Added: `gateway/provider_health.py`, `tests/test_provider_health.py`.


## Hardware profiling — configurability pass (2026-09-22, follow-up task)

Task brief said hardware profiling "does not currently exist" — **it already did**
(`gateway/hardware.py`, psutil CPU/RAM, import-guarded pynvml GPU, energy-estimate
labelling, Prometheus gauges on `/metrics` + `/hardware`, 4 tests). The genuine gap vs
the requirements was **requirement 7 (configurable) + requirement 4's "profiling
disabled" path**. Added config switches; did NOT rebuild what worked.

### What was already there (unchanged)
- `sample_hardware()`: CPU% + logical/physical core counts + RAM (psutil); GPU
  util/mem/power via optional `pynvml` (import-guarded, `_ensure_nvml()`); runs fine
  with no GPU/driver/library. Prometheus gauges (`host_cpu_percent`, `host_ram_percent`,
  `host_gpu_utilization_percent`, `host_gpu_power_watts`, `host_active_power_watts
  {measured}`) refreshed on `/metrics` and `/hardware`. Request/inference duration is
  the pre-existing `llm_request_duration_seconds` histogram (requirement 5).

### What this pass added
- `hardware.profiling_enabled()` — master switch, env `HARDWARE_PROFILING_ENABLED`
  (default on). When off, `_refresh_hardware_gauges()` samples nothing, touches no
  gauges, returns `{"profiling_enabled": False}`; `/hardware` + `/metrics` still work
  (zero profiling overhead, requirement 4 + 8).
- `hardware.gpu_enabled()` — env `HARDWARE_GPU_ENABLED` (default on). Independent of the
  master switch so GPU polling can be turned off on its own even where NVML is present.
  `_sample_gpus()` returns `[]` early when off.
- `_env_bool()` reads env at call time (config/tests apply without re-import).
- **Bug found & fixed:** `from gateway import hardware` in `app.py` collided with the
  existing `def hardware()` route (`@app.get("/hardware")`), so the module name was
  shadowed by the function → `AttributeError`. Fixed by importing as `hw_module`.

### Measured vs estimated (unchanged, still honest)
- **Measured:** CPU%, RAM%, core counts, GPU util/mem, and GPU **power** ONLY when NVML
  reports it (`nvmlDeviceGetPowerUsage`, mW→W) → `energy_is_measured=true`,
  `power_is_measured=true`.
- **Estimated:** on hosts without GPU power telemetry, `estimated_active_power_watts`
  falls back to `ESTIMATED_HOST_ACTIVE_WATTS` (env, default 65 W) — a coarse illustrative
  placeholder, `energy_is_measured=false`, flagged in `notes` with the word ESTIMATE.
  We never hard-code a specific chip's wattage as fact.

### Design & trade-offs
- Point-in-time sampling on scrape (`cpu_interval=0.0`, non-blocking) — cheap, no
  background thread; the trade-off is CPU% is relative to the previous call.
- GPU path fully optional/lazy (NVML init once, cached) so the gateway never hard-depends
  on NVIDIA. Two independent switches keep overhead controllable.
- Integrated into the existing default Prometheus registry (requirement 6) — no separate
  metrics system.

### Tests (`tests/test_hardware.py` → 8; +1 integration → gateway_integration 9)
Existing 4 (cpu/ram sample, runs-without-gpu, measured-gpu-power flag, optional import)
+ new: profiling_enabled default/env parsing, GPU sampling disabled independently, sample
still works with GPU disabled; integration: `/hardware` returns `{profiling_enabled:false}`
when disabled. CPU-only correctness covered by `test_runs_without_gpu` +
`test_sample_still_works_with_gpu_disabled`.

### Files changed this task
- Modified: `gateway/hardware.py` (config switches, `_env_bool`, gpu-guard in
  `_sample_gpus`), `gateway/app.py` (profiling-disabled short-circuit, `hw_module`
  import fix), `tests/test_hardware.py`, `tests/test_gateway_integration.py`,
  `.env.example`, `AGENTS.md`.

## Full integration & verification pass (2026-09-22, follow-up task)

Verified every feature from the ACTUAL code + a run of the real test suite and the
offline demo — not from prior task claims. Result: the system integrates correctly;
degraded-path integration tests were verified.

### Verified end-to-end request path (`send_prompt`, traced against source)
`POST /generate` → `send_prompt`:
1. **Redis exact-match** (`get_cached_response`, SHA-256 key). Wrapped in try/except:
   on error → `bypass_cache=True`, continue (no crash). Hit → return `provider="cache"`,
   layer `redis`, log to DB, **skip all provider/routing/health logic**.
2. **Milvus semantic** (`prompt_to_vector` → `search_similar_embedding`, threshold 0.95),
   only if not `bypass_cache`. try/except → `bypass_cache=True` on error. Hit → return
   `provider="cache"`, layer `milvus`, **write-through to Redis**.
3. **Cache miss → provider order:**
   - `routing_mode=="smart"` AND no explicit provider → `smart_router.choose_provider_order`
     (loads k-NN lazily; falls back to `[]`=round-robin if artifacts/deps missing or route
     errors; maps unsupported picks onto SUPPORTED_PROVIDERS). It calls
     `health_registry.prioritize` internally.
   - else → `_ordered_providers(provider)` (round-robin primary + rest), which calls
     `health_registry.prioritize`.
   - **No double-prioritize:** smart path returns a ready order used directly; round-robin
     path prioritizes once. Exactly one of the two runs per request.
4. **`estimate_cost`** on the first (preferred) provider.
5. **`_invoke_with_failover`** — sequential, one provider at a time; records
   `health_registry.record_success/record_failure` per attempt; emits
   `llm_provider_attempts_total` + `llm_failover_total`. All fail →
   `AllProvidersFailedError` → **HTTP 502**.
6. **Write-through** to Milvus + Redis (each try/except, non-fatal), structured log,
   response with `cache`/`quality`/`tokens`/`cost`/`failover`/`routing` blocks.

`/metrics` refreshes hardware gauges then serves the default registry.

### Interaction verification (who touches whom)
- **cache ↔ providers:** a cache hit fully short-circuits providers/health/routing
  (confirmed by `test_cache_hit_bypasses_providers_and_health`).
- **smart router ↔ health:** both `choose_provider_order` return paths pass through
  `health_registry.prioritize`; an unhealthy smart pick is reordered behind a healthy
  provider but kept in the chain (`test_smart_pick_reordered_when_unhealthy`).
- **health ↔ failover:** `_invoke_with_failover` records health on every attempt;
  `_ordered_providers` reorders healthy-first but never drops (`test_unhealthy_primary_is
  _deprioritized_but_still_in_chain`).
- **metrics/logging:** all Prometheus objects are module-level singletons on the default
  registry (registered once), so `/metrics` exports failover, routing, health, and
  hardware series together. Structured JSON logs on every path.

### Degraded-scenario coverage (all tested)
| Scenario | Behaviour | Test |
|---|---|---|
| Redis cache down | bypass, serve live | `test_generate_survives_redis_cache_unavailable` |
| Milvus vector store down | bypass, serve live | `test_generate_survives_vector_store_unavailable` |
| Smart router unavailable | fall back to round-robin | `test_smart_mode_falls_back_to_roundrobin_when_unavailable` |
| Provider timeout | failover | `test_provider_timeout_triggers_failover` |
| Single provider failure | failover | `test_primary_fails_fallback_succeeds` |
| Multiple provider failures | chain to next | `test_multiple_failures_before_success` |
| All providers fail | HTTP 502 | `test_generate_all_fail_returns_502` |
| Hardware profiling disabled | zero-overhead no-op | `test_hardware_endpoint_when_profiling_disabled` |

### Verifications run (actual results, not claims)
- **Startup command:** `python -c "from gateway.app import app"` → real `FastAPI` instance;
  `python -m uvicorn gateway.app:app` is the correct target.
- **`.env.example`:** reviewed; added `HARDWARE_PROFILING_ENABLED`, `HARDWARE_GPU_ENABLED`,
  `PROVIDER_UNHEALTHY_AFTER_FAILURES`, `PROVIDER_HEALTH_RECOVERY_SECONDS`. All env vars the code reads are now documented.
- **Offline demo:** `python demo_gateway.py` → exit 0, all flows exercised.
- **Full suite:** **40 passed** (see Tests — current status).
- **torch not in `requirements.txt`** (CI installs CPU torch separately). Locally, if torch
  is absent, smart routing's real load fails gracefully → round-robin. No trained artifacts
  in `outputs/` either, so smart mode currently always falls back unless both are provided.

## Next steps a future agent might take
- Run the offline pipeline with real keys to produce `outputs/` + trained artifacts,
  then enable `GATEWAY_ROUTING_MODE=smart` end-to-end.
- Expand `data/quality_eval_samples.csv` (only 3 rows) for a more credible eval.

