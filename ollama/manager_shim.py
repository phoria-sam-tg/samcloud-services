"""Import shim so the pool tests can run from inside the ollama/ directory.

The package uses relative imports (`from . import config`), so the test files
in here are run as scripts with the parent on sys.path. test_lifecycle.py and
test_cooldown.py predate this and import differently; this keeps the new tests
from having to care.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama.manager import ModelManager, Backend          # noqa: E402
from ollama.exo_client import ExoClient, ExoUnavailable   # noqa: E402
from ollama import capacity, config                       # noqa: E402

sys.modules.setdefault("exo_client", sys.modules["ollama.exo_client"])

__all__ = ["ModelManager", "Backend", "ExoClient", "ExoUnavailable",
           "capacity", "config"]
