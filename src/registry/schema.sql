CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY,
    provider_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    cost_per_1k_tokens REAL NOT NULL,
    average_latency_ms REAL,
    quality_score REAL,
    best_use_case TEXT,
    reliability_metric REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider_name, model_name)
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id INTEGER PRIMARY KEY,
    prompt_id INTEGER NOT NULL,
    category TEXT,
    prompt TEXT NOT NULL,
    provider_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    latency_ms REAL,
    cost_usd REAL,
    response_length INTEGER,
    quality_score_manual_1_10 REAL,
    quality_score_llm_1_10 REAL,
    hallucination_flag BOOLEAN,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
