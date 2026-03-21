CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS llm_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    latency_ms INTEGER,
    cost_usd NUMERIC(10,6),
    quality_score NUMERIC(5,4)
);

ALTER TABLE llm_requests
ADD COLUMN IF NOT EXISTS quality_score NUMERIC(5,4);