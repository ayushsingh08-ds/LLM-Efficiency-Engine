import os
import pytest
from fastapi.testclient import TestClient
from gateway.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_auth_dev_mode_bypass(client, monkeypatch):
    """When GATEWAY_API_KEY is not set, requests proceed without auth (dev mode)."""
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    
    resp = client.post("/difficulty/classify", json={"prompt": "Hello world"})
    assert resp.status_code == 200


def test_client_auth_enforcement_missing_key(client, monkeypatch):
    """When GATEWAY_API_KEY is configured, unauthenticated requests return 401."""
    monkeypatch.setenv("GATEWAY_API_KEY", "secret-client-token-123")
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)

    resp = client.post("/difficulty/classify", json={"prompt": "Hello world"})
    assert resp.status_code == 401
    assert resp.json()["detail"]["error"] == "unauthorized"


def test_client_auth_success_with_header(client, monkeypatch):
    """Valid X-API-Key grants access to inference endpoints."""
    monkeypatch.setenv("GATEWAY_API_KEY", "secret-client-token-123")

    # Header auth
    resp = client.post(
        "/difficulty/classify",
        json={"prompt": "Hello world"},
        headers={"X-API-Key": "secret-client-token-123"},
    )
    assert resp.status_code == 200

    # Bearer auth
    resp_bearer = client.post(
        "/difficulty/classify",
        json={"prompt": "Hello world"},
        headers={"Authorization": "Bearer secret-client-token-123"},
    )
    assert resp_bearer.status_code == 200


def test_admin_auth_protects_admin_routes(client, monkeypatch):
    """Admin routes reject client keys and require ADMIN_API_KEY."""
    monkeypatch.setenv("GATEWAY_API_KEY", "client-token")
    monkeypatch.setenv("ADMIN_API_KEY", "admin-token-xyz")

    # Call with client token -> 403 Forbidden
    resp = client.post(
        "/providers/reload",
        headers={"X-API-Key": "client-token"},
    )
    assert resp.status_code == 403

    # Call with admin token -> 200 OK
    resp_ok = client.post(
        "/providers/reload",
        headers={"X-API-Key": "admin-token-xyz"},
    )
    assert resp_ok.status_code == 200


def test_bounded_prompt_length_rejection(client):
    """Prompts exceeding max_length (32,000 chars) are rejected with 422 Unprocessable Entity."""
    huge_prompt = "a" * 35000
    resp = client.post("/difficulty/classify", json={"prompt": huge_prompt})
    assert resp.status_code == 422


def test_openai_chat_completions_endpoint(client, monkeypatch):
    """Verify /v1/chat/completions OpenAI compatibility format."""
    # Mock send_prompt to avoid live model calls
    from unittest.mock import patch
    with patch("gateway.app.send_prompt") as mock_send:
        mock_send.return_value = {
            "response": "Hello from mock LLM!",
            "provider": "cheap_mock",
            "tokens": {"input_tokens": 10, "output_tokens": 5},
            "cache": {"hit": False},
            "difficulty": "cheap",
            "cost_usd": 0.000005,
        }

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "What is 2+2?"}
                ],
                "model": "auto"
            }
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "Hello from mock LLM!"
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 5
