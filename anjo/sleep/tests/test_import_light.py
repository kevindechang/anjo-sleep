"""``import anjo.sleep`` works without the GPU stack (spec §12.1): torch and friends load only in
the GPU runner's own environment."""
from __future__ import annotations

import json
import subprocess
import sys

HEAVY = ("torch", "transformers", "vllm", "peft")
CORE = (
    "anjo.sleep",
    "anjo.sleep.types",
    "anjo.sleep.ports",
    "anjo.sleep.mind.rule_chooser",
    "anjo.sleep.model_api",
    "anjo.sleep.meta_api",
    "anjo.sleep.baseten_api",
)


def test_import_anjo_sleep_loads_no_torch() -> None:
    script = (
        "import importlib, json, sys\n"
        f"for name in {CORE!r}:\n"
        "    importlib.import_module(name)\n"
        f"print(json.dumps(sorted(m for m in {HEAVY!r} if m in sys.modules)))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == []
