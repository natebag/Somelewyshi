"""Test conftest — fix vendor/moneyprinterv2 config.py shadowing our config package.

pytest's import machinery picks up vendor/moneyprinterv2/src/config.py as a
namespace package candidate when resolving `import config`. We pre-import our
config package here before pytest tries to collect test modules.
"""

import importlib
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)

# Remove any vendor paths
sys.path = [p for p in sys.path if "vendor" not in p]

# Ensure root is first
if _root in sys.path:
    sys.path.remove(_root)
sys.path.insert(0, _root)

# Force-load our config package into sys.modules before anything else
if "config" in sys.modules:
    del sys.modules["config"]
if "config.schemas" in sys.modules:
    del sys.modules["config.schemas"]

import config  # noqa: E402, F401
import config.schemas  # noqa: E402, F401
