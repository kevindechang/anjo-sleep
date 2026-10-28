"""MIND's core in the app's night: ``ANJO_SLEEP_MIND_CORE=1`` gives the night the trained core
(``ANJO_SLEEP_MIND_CORE_MODEL``, a saved ``.npz``) over each person's mind store,
``<data dir>/sleep/mind_state/`` (removed with her other files on a delete, ``app.ownership``).
Off, the night's port is ``NoCore`` and the night is today's. The core's night is loaded by name:
it reaches torch and the latents module's training code, which the app never imports."""
from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from anjo.sleep.mind.core import bridge
from anjo.sleep.night.deps import MindCore, NoCore

CORE_MODEL_ENV: Final = "ANJO_SLEEP_MIND_CORE_MODEL"


def build_core(env: Mapping[str, str], sleep_root: Path) -> MindCore:
    if not bridge.core_on(env):
        return NoCore()
    model = env.get(CORE_MODEL_ENV, "").strip()
    if not model:
        raise ValueError(f"{CORE_MODEL_ENV} names the trained core MIND's core runs on")
    night = importlib.import_module("anjo.sleep.mind.core.night")
    built: MindCore = night.from_env(env, Path(model), sleep_root)
    return built
