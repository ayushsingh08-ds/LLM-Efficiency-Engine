CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS jobs (
    job_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id  VARCHAR(100),
    budget_usd NUMERIC(10,6) NOT NULL DEFAULT 0.25,
    spent_usd  NUMERIC(10,6) NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status     VARCHAR(20) NOT NULL DEFAULT 'active'  -- active | blocked | completed
);

CREATE TABLE IF NOT EXISTS llm_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(job_id) ON DELETE SET NULL,
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

ALTER TABLE llm_requests
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES jobs(job_id) ON DELETE SET NULL;