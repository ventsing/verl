# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Import bootstrap: let these tests run on a bare CPython (no numpy/ray).

``import verl...`` normally executes ``verl/__init__.py``, which pulls in
``verl.protocol`` → numpy/ray. The trajectory_async data plane is
deliberately stdlib-only, so when the real verl package cannot be imported
we install a lightweight package stub that keeps the real module search
paths but skips the heavy package ``__init__``. In a full verl environment
the stub is a no-op — the real package wins.

Import this module before any ``verl.*`` import.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VERL_DIR = _REPO_ROOT / "verl"


def _real_verl_importable() -> bool:
    try:
        import verl  # noqa: F401
        return True
    except Exception:
        return False


def install() -> None:
    if "verl" in sys.modules:
        return
    if _real_verl_importable():
        return
    stub = types.ModuleType("verl")
    stub.__path__ = [str(_VERL_DIR)]
    stub.__package__ = "verl"
    sys.modules["verl"] = stub
    exp = types.ModuleType("verl.experimental")
    exp.__path__ = [str(_VERL_DIR / "experimental")]
    exp.__package__ = "verl.experimental"
    sys.modules["verl.experimental"] = exp


install()
