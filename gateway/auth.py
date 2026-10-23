"""
gateway/auth.py
───────────────
Authentication and Authorization dependencies for the LLM Efficiency Gateway.

Supports:
  - Standard Client Auth (`GATEWAY_API_KEY`): required for `/generate`, `/estimate`, `/difficulty/classify`.
  - Admin Auth (`ADMIN_API_KEY`): required for `/jobs`, `/cache/*`, `/providers/reload`.
  - Development bypass: if no API keys are configured in environment variables,
    auth is disabled with a security warning to preserve seamless local development.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("llm_gateway")

# Environment key names
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

# Support both `X-API-Key: ...` and `Authorization: Bearer ...` headers
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


def _extract_token(
    header_key: Optional[str],
    bearer_creds: Optional[HTTPAuthorizationCredentials],
) -> Optional[str]:
    """Extract token from either X-API-Key or Authorization Bearer header."""
    if header_key:
        return header_key.strip()
    if bearer_creds and bearer_creds.credentials:
        return bearer_creds.credentials.strip()
    return None


def verify_client_key(
    header_key: Optional[str] = Security(api_key_header),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> None:
    """
    Validates client API key for inference endpoints.
    Allows request if GATEWAY_API_KEY is not configured (dev mode), or if token matches.
    """
    expected_key = os.getenv("GATEWAY_API_KEY")
    if not expected_key:
        # Dev mode bypass
        return

    provided_token = _extract_token(header_key, bearer_creds)
    if not provided_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "unauthorized", "message": "Missing API key in X-API-Key or Bearer header"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison to prevent timing attacks
    admin_key = os.getenv("ADMIN_API_KEY")
    is_valid_client = secrets.compare_digest(provided_token, expected_key)
    is_valid_admin = admin_key and secrets.compare_digest(provided_token, admin_key)

    if not (is_valid_client or is_valid_admin):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "unauthorized", "message": "Invalid API key"},
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_admin_key(
    header_key: Optional[str] = Security(api_key_header),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> None:
    """
    Validates administrative API key for management and config endpoints.
    Allows request if ADMIN_API_KEY is not configured (dev mode), or if token matches.
    """
    expected_admin = os.getenv("ADMIN_API_KEY")
    if not expected_admin:
        # Dev mode bypass
        return

    provided_token = _extract_token(header_key, bearer_creds)
    if not provided_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "unauthorized", "message": "Admin API key required"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(provided_token, expected_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "forbidden", "message": "Admin privileges required"},
        )
