"""Test bootstrap.

The core/ package is pure Python, but importing
custom_components.adaptive_comfort normally executes the integration
__init__.py which needs homeassistant. When HA is not installed (local
core-only test runs), register package stubs so `...adaptive_comfort.core`
imports without touching the HA layer.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).parent

try:
    import homeassistant  # noqa: F401
except ImportError:
    cc = types.ModuleType("custom_components")
    cc.__path__ = [str(ROOT / "custom_components")]
    ac = types.ModuleType("custom_components.adaptive_comfort")
    ac.__path__ = [str(ROOT / "custom_components" / "adaptive_comfort")]
    sys.modules.setdefault("custom_components", cc)
    sys.modules.setdefault("custom_components.adaptive_comfort", ac)
