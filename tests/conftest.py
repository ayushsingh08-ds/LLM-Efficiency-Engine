from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Point the ProviderRegistry at the test fixture so unit tests test deterministic providers
_FIXTURE_PATH = Path(__file__).parent / "fixtures_providers.yaml"
os.environ["PROVIDERS_CONFIG_PATH"] = str(_FIXTURE_PATH)

from gateway.provider_registry import registry
registry.reload(_FIXTURE_PATH)
