from __future__ import annotations

import os
from pathlib import Path

# Point the ProviderRegistry at the test fixture so unit tests test deterministic providers
_FIXTURE_PATH = Path(__file__).parent / "fixtures_providers.yaml"
os.environ["PROVIDERS_CONFIG_PATH"] = str(_FIXTURE_PATH)

from gateway.provider_registry import registry
registry.reload(_FIXTURE_PATH)
