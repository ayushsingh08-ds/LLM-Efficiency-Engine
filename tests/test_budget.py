"""
tests/test_budget.py
────────────────────
Unit and integration tests for budget enforcement, circuit breaking (HTTP 402),
and PostgreSQL job spend tracking.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from gateway.budget import (
    BudgetExceededError,
    JobNotFoundError,
    create_job,
    get_job,
    update_budget,
    check_and_reserve,
    commit_spend,
)
from gateway.app import app


client = TestClient(app)


def test_budget_exceeded_error_attributes():
    err = BudgetExceededError(job_id="test-job", spent_usd=0.20, budget_usd=0.25, attempted_cost_usd=0.10)
    assert err.job_id == "test-job"
    assert err.spent_usd == 0.20
    assert err.budget_usd == 0.25
    assert err.attempted_cost_usd == 0.10
    assert "test-job exceeded budget" in str(err)


def test_check_and_reserve_logic_mocked():
    """Verify check_and_reserve raises when spent + estimated > budget."""
    mock_cursor = MagicMock()
    # Case 1: Active job with sufficient budget
    mock_cursor.fetchone.return_value = {
        "job_id": "00000000-0000-0000-0000-000000000001",
        "budget_usd": 0.50,
        "spent_usd": 0.10,
        "status": "active",
    }
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with patch("gateway.budget.get_db_connection", return_value=mock_conn):
        res = check_and_reserve("00000000-0000-0000-0000-000000000001", estimated_cost_usd=0.05)
        assert res["job_id"] == "00000000-0000-0000-0000-000000000001"
        assert res["remaining_usd"] == 0.40

    # Case 2: Blocked job
    mock_cursor.fetchone.return_value = {
        "job_id": "00000000-0000-0000-0000-000000000002",
        "budget_usd": 0.50,
        "spent_usd": 0.50,
        "status": "blocked",
    }
    with patch("gateway.budget.get_db_connection", return_value=mock_conn):
        with pytest.raises(BudgetExceededError) as exc_info:
            check_and_reserve("00000000-0000-0000-0000-000000000002", estimated_cost_usd=0.01)
        assert exc_info.value.job_id == "00000000-0000-0000-0000-000000000002"

    # Case 3: Over limit on estimate
    mock_cursor.fetchone.return_value = {
        "job_id": "00000000-0000-0000-0000-000000000003",
        "budget_usd": 0.50,
        "spent_usd": 0.45,
        "status": "active",
    }
    with patch("gateway.budget.get_db_connection", return_value=mock_conn):
        with pytest.raises(BudgetExceededError):
            check_and_reserve("00000000-0000-0000-0000-000000000003", estimated_cost_usd=0.10)


def test_generate_endpoint_enforces_budget_402():
    """POST /generate should return HTTP 402 with structured error if budget exceeded."""
    with patch(
        "gateway.app.send_prompt",
        side_effect=BudgetExceededError(
            job_id="job-123",
            spent_usd=0.25,
            budget_usd=0.25,
            attempted_cost_usd=0.05,
        ),
    ):
        response = client.post(
            "/generate",
            json={"prompt": "Hello world", "job_id": "job-123"},
        )
        assert response.status_code == 402
        body = response.json()
        assert body["detail"]["error"] == "budget_exceeded"
        assert body["detail"]["job_id"] == "job-123"
        assert body["detail"]["budget_usd"] == 0.25


def test_generate_endpoint_job_not_found_404():
    """POST /generate returns HTTP 404 when unknown job_id is referenced."""
    with patch(
        "gateway.app.send_prompt",
        side_effect=JobNotFoundError("Job not found: non-existent-id"),
    ):
        response = client.post(
            "/generate",
            json={"prompt": "Hello world", "job_id": "non-existent-id"},
        )
        assert response.status_code == 404
        assert response.json()["detail"]["error"] == "job_not_found"


def test_job_endpoints_mocked():
    """Verify /jobs, /jobs/{id}, and /jobs/{id}/budget endpoints."""
    with patch("gateway.budget.create_job", return_value={
        "job_id": "mock-uuid-1",
        "tenant_id": "tenant-a",
        "budget_usd": 0.50,
        "spent_usd": 0.0,
        "status": "active",
        "created_at": "2026-10-23T15:00:00",
    }):
        resp = client.post("/jobs", json={"tenant_id": "tenant-a", "budget_usd": 0.50})
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "mock-uuid-1"
        assert resp.json()["budget_usd"] == 0.50

    with patch("gateway.budget.get_job", return_value={
        "job_id": "mock-uuid-1",
        "tenant_id": "tenant-a",
        "budget_usd": 0.50,
        "spent_usd": 0.15,
        "remaining_usd": 0.35,
        "status": "active",
        "created_at": "2026-10-23T15:00:00",
    }):
        resp = client.get("/jobs/mock-uuid-1")
        assert resp.status_code == 200
        assert resp.json()["spent_usd"] == 0.15
        assert resp.json()["remaining_usd"] == 0.35

    with patch("gateway.budget.update_budget", return_value={
        "job_id": "mock-uuid-1",
        "tenant_id": "tenant-a",
        "budget_usd": 1.00,
        "spent_usd": 0.15,
        "remaining_usd": 0.85,
        "status": "active",
    }):
        resp = client.put("/jobs/mock-uuid-1/budget", json={"budget_usd": 1.00})
        assert resp.status_code == 200
        assert resp.json()["budget_usd"] == 1.00
