"""In-memory provider health tracking (lightweight circuit-breaker signal).

The live gateway records the outcome of every provider call. The smart/round-robin
routers can then *deprioritise* (not exclude) providers that are currently
failing, so a healthy provider is tried first — while still keeping unhealthy
providers in the failover chain as a last resort.

Design notes / limitations (be honest):
- State is per-process and in-memory only. It is NOT shared across workers/replicas
  and resets on restart. For multi-worker deployments this would need a shared store
  (Redis, etc.). Adequate for a single-process gateway and for tests.
- This is a coarse health heuristic, not a full circuit breaker with half-open probing
  schedules. A provider is "unhealthy" after N consecutive failures and recovers after
  a cooldown or the next success.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field


FAILURE_THRESHOLD = int(os.getenv("PROVIDER_UNHEALTHY_AFTER_FAILURES", "3"))
RECOVERY_SECONDS = float(os.getenv("PROVIDER_HEALTH_RECOVERY_SECONDS", "30"))


@dataclass
class _ProviderState:
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_ts: float = 0.0
    last_success_ts: float = 0.0


class ProviderHealthRegistry:
    def __init__(self, failure_threshold: int | None = None, recovery_seconds: float | None = None) -> None:
        self.failure_threshold = failure_threshold if failure_threshold is not None else FAILURE_THRESHOLD
        self.recovery_seconds = recovery_seconds if recovery_seconds is not None else RECOVERY_SECONDS
        self._state: dict[str, _ProviderState] = {}
        self._lock = threading.Lock()

    def _get(self, provider: str) -> _ProviderState:
        st = self._state.get(provider)
        if st is None:
            st = _ProviderState()
            self._state[provider] = st
        return st

    def record_success(self, provider: str) -> None:
        with self._lock:
            st = self._get(provider)
            st.consecutive_failures = 0
            st.total_successes += 1
            st.last_success_ts = time.time()

    def record_failure(self, provider: str) -> None:
        with self._lock:
            st = self._get(provider)
            st.consecutive_failures += 1
            st.total_failures += 1
            st.last_failure_ts = time.time()

    def is_healthy(self, provider: str, now: float | None = None) -> bool:
        """Unknown providers are healthy by default. A provider is unhealthy once it
        has >= failure_threshold consecutive failures, until recovery_seconds elapse
        since the last failure (then we optimistically allow a retry)."""
        now = now if now is not None else time.time()
        with self._lock:
            st = self._state.get(provider)
            if st is None or st.consecutive_failures < self.failure_threshold:
                return True
            # Tripped: allow a probe again once the cooldown has passed.
            return (now - st.last_failure_ts) >= self.recovery_seconds

    def prioritize(self, providers: list[str], now: float | None = None) -> list[str]:
        """Stable reorder: healthy providers first, unhealthy ones after.

        Never drops a provider — an unhealthy provider is still a valid last-resort
        fallback. Order within each group is preserved (stable), so the caller's
        preferred order is respected among healthy providers.
        """
        healthy = [p for p in providers if self.is_healthy(p, now=now)]
        unhealthy = [p for p in providers if not self.is_healthy(p, now=now)]
        return healthy + unhealthy

    def snapshot(self, now: float | None = None) -> dict:
        now = now if now is not None else time.time()
        with self._lock:
            out = {}
            for provider, st in self._state.items():
                out[provider] = {
                    "healthy": (
                        st.consecutive_failures < self.failure_threshold
                        or (now - st.last_failure_ts) >= self.recovery_seconds
                    ),
                    "consecutive_failures": st.consecutive_failures,
                    "total_failures": st.total_failures,
                    "total_successes": st.total_successes,
                    "seconds_since_last_failure": (
                        round(now - st.last_failure_ts, 2) if st.last_failure_ts else None
                    ),
                }
            return out

    def reset(self) -> None:
        with self._lock:
            self._state.clear()


# Process-wide singleton used by the gateway.
registry = ProviderHealthRegistry()
