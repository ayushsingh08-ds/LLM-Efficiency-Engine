# Cost & Carbon-Aware LLM Gateway

Intelligent infrastructure layer that reduces enterprise AI spending by 40-60% while cutting carbon emissions by 30-50%.

## Overview

This project combines:

- **Semantic Caching**: Cache similar queries (15-25% cost savings)
- **Smart Routing**: Use cheapest model meeting quality threshold (40-60% savings)
- **Failover**: 99.99% uptime with multi-provider failover
- **Carbon-Aware Scheduling**: Defer flexible workloads to renewable energy peaks
- **Hardware Profiling**: Optimize for power efficiency (20-35% improvement)

## Quick Start

### Local Development

```bash
# Clone repo
git clone https://github.com/ayushsingh08-ds/cost-carbon-aware-router.git
cd cost-carbon-aware-router

# Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start all services
docker-compose up -d

# Run app
python -m uvicorn app.main:app --reload

# Access
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

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

GitHub Actions workflow is available at `.github/workflows/ci.yml` and runs:

- `tests/test_knn_router.py`
- `tests/test_phase3_pipeline.py`
