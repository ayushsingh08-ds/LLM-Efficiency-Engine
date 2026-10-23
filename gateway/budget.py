"""
gateway/budget.py
─────────────────
Concurrency-safe budget enforcement and spend accumulation for LLM requests.

Key Features:
  - Database backed (PostgreSQL `jobs` table).
  - Pre-flight check & reserve (`check_and_reserve`):
      Validates whether (spent_usd + estimated_cost_usd) <= budget_usd.
      Raises `BudgetExceededError` if the job is exhausted or blocked.
  - Post-flight commit (`commit_spend`):
      Increments `spent_usd` atomically by actual_cost_usd.
      Transitions status to 'blocked' if budget is fully depleted.
  - Job management utilities: create job, get job status, top up budget.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("llm_gateway")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "llm_logs"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "port": int(os.getenv("DB_PORT", "5432")),
}


class BudgetExceededError(RuntimeError):
    """Raised when a request exceeds the allocated budget limit for a job."""

    def __init__(self, job_id: str, spent_usd: float, budget_usd: float, attempted_cost_usd: float):
        self.job_id = job_id
        self.spent_usd = round(float(spent_usd), 6)
        self.budget_usd = round(float(budget_usd), 6)
        self.attempted_cost_usd = round(float(attempted_cost_usd), 6)
        super().__init__(
            f"Job {job_id} exceeded budget. Spent: ${self.spent_usd:.4f}, "
            f"Attempted: ${self.attempted_cost_usd:.4f}, Limit: ${self.budget_usd:.4f}"
        )


class JobNotFoundError(KeyError):
    """Raised when a referenced job_id does not exist."""
    pass


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def create_job(tenant_id: Optional[str] = None, budget_usd: float = 0.25) -> dict[str, Any]:
    """Create a new job with an initial budget limit."""
    budget = round(float(budget_usd), 6)
    if budget <= 0:
        raise ValueError("budget_usd must be greater than 0")

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO jobs (tenant_id, budget_usd, spent_usd, status)
                VALUES (%s, %s, 0.0, 'active')
                RETURNING job_id, tenant_id, budget_usd, spent_usd, status, created_at
                """,
                (tenant_id, budget),
            )
            job = cur.fetchone()
            conn.commit()
            return {
                "job_id": str(job["job_id"]),
                "tenant_id": job["tenant_id"],
                "budget_usd": float(job["budget_usd"]),
                "spent_usd": float(job["spent_usd"]),
                "status": job["status"],
                "created_at": job["created_at"].isoformat() if job["created_at"] else None,
            }
    finally:
        conn.close()


def get_job(job_id: str) -> dict[str, Any]:
    """Retrieve current details and spend summary for a job."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT job_id, tenant_id, budget_usd, spent_usd, status, created_at
                FROM jobs
                WHERE job_id = %s
                """,
                (job_id,),
            )
            job = cur.fetchone()
            if not job:
                raise JobNotFoundError(f"Job not found: {job_id}")

            spent = float(job["spent_usd"])
            budget = float(job["budget_usd"])
            remaining = max(0.0, budget - spent)
            return {
                "job_id": str(job["job_id"]),
                "tenant_id": job["tenant_id"],
                "budget_usd": budget,
                "spent_usd": spent,
                "remaining_usd": round(remaining, 6),
                "status": job["status"],
                "created_at": job["created_at"].isoformat() if job["created_at"] else None,
            }
    finally:
        conn.close()


def update_budget(job_id: str, new_budget_usd: float) -> dict[str, Any]:
    """Adjust or top up budget for a job. Re-activates job if budget exceeds spent."""
    budget = round(float(new_budget_usd), 6)
    if budget <= 0:
        raise ValueError("new_budget_usd must be greater than 0")

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE jobs
                SET budget_usd = %s,
                    status = CASE WHEN spent_usd < %s THEN 'active' ELSE 'blocked' END
                WHERE job_id = %s
                RETURNING job_id, tenant_id, budget_usd, spent_usd, status
                """,
                (budget, budget, job_id),
            )
            job = cur.fetchone()
            if not job:
                raise JobNotFoundError(f"Job not found: {job_id}")
            conn.commit()
            return {
                "job_id": str(job["job_id"]),
                "tenant_id": job["tenant_id"],
                "budget_usd": float(job["budget_usd"]),
                "spent_usd": float(job["spent_usd"]),
                "remaining_usd": round(max(0.0, float(job["budget_usd"]) - float(job["spent_usd"])), 6),
                "status": job["status"],
            }
    finally:
        conn.close()


import uuid

def _validate_uuid(val: str) -> None:
    try:
        uuid.UUID(str(val))
    except (ValueError, AttributeError):
        raise JobNotFoundError(f"Invalid job UUID: {val!r}")


def check_and_reserve(job_id: str, estimated_cost_usd: float) -> dict[str, Any]:
    """
    Concurrency-safe balance check prior to dispatching an LLM call.
    Uses row-level locking (SELECT ... FOR UPDATE) to guard against race conditions.
    """
    _validate_uuid(job_id)
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT job_id, budget_usd, spent_usd, status
                FROM jobs
                WHERE job_id = %s
                FOR UPDATE
                """,
                (job_id,),
            )
            job = cur.fetchone()
            if not job:
                raise JobNotFoundError(f"Job {job_id!r} not found in database.")

            spent = float(job["spent_usd"])
            budget = float(job["budget_usd"])
            status = job["status"]

            if status == "blocked" or (spent >= budget):
                raise BudgetExceededError(
                    job_id=job_id,
                    spent_usd=spent,
                    budget_usd=budget,
                    attempted_cost_usd=estimated_cost_usd,
                )

            # Check if estimated call pushes it strictly over budget
            if (spent + estimated_cost_usd) > budget:
                raise BudgetExceededError(
                    job_id=job_id,
                    spent_usd=spent,
                    budget_usd=budget,
                    attempted_cost_usd=estimated_cost_usd,
                )

            conn.commit()
            return {
                "job_id": job_id,
                "budget_usd": budget,
                "spent_usd": spent,
                "remaining_usd": round(budget - spent, 6),
            }
    finally:
        conn.close()


def commit_spend(job_id: str, actual_cost_usd: float) -> None:
    """
    Accumulates actual cost spent into the database job record.
    Marks status as 'blocked' if spent exceeds budget.
    """
    cost = round(max(0.0, float(actual_cost_usd)), 6)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET spent_usd = spent_usd + %s,
                    status = CASE WHEN (spent_usd + %s) >= budget_usd THEN 'blocked' ELSE status END
                WHERE job_id = %s
                """,
                (cost, cost, job_id),
            )
            conn.commit()
            logger.info("job_spend_committed", extra={
                "job_id": job_id,
                "cost_usd": cost,
            })
    except Exception as e:
        logger.error("job_spend_commit_failed", extra={"job_id": job_id, "error": str(e)})
    finally:
        conn.close()
