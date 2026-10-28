"""MIND's core in the app's night: with ``ANJO_SLEEP_MIND_CORE=1`` and MIND's inquiry on (the
daily's ``mind_on``: the demo person by default, off with ``--no-mind``), the night gets the
trained core (``ANJO_SLEEP_MIND_CORE_MODEL``, a saved ``.npz``) for the demo persona's people only
(``core_people``), never anyone else, over each one's mind store, ``<data dir>/sleep/mind_state/``
(removed with her other files on a delete, ``app.ownership``), opening questions in her workspace
log (a dreamt collision opens none unless ``ANJO_SLEEP_MIND_CONTRADICTIONS=1``). Otherwise the
night's port is ``NoCore`` and the night is today's. The core's night is loaded
by name: it reaches torch and the latents module's training code, which the app never imports."""
from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

from anjo.sleep.mind.core import bridge
from anjo.sleep.mind.inquiry.ports import WorkspaceLog
from anjo.sleep.night.channel_night import MindState
from anjo.sleep.night.deps import MindCore, NightOptions, NoCore
from anjo.sleep.personas import DEMO_PERSONA

CORE_MODEL_ENV: Final = "ANJO_SLEEP_MIND_CORE_MODEL"


def core_people(options: NightOptions) -> frozenset[str]:
    """Who MIND's core may serve tonight: the demo persona's people, with MIND's inquiry on."""
    if not options.mind_inquiry:
        return frozenset()
    return frozenset(uid for uid, persona in options.personas.items() if persona == DEMO_PERSONA)


def core_state(core: MindCore) -> MindState | None:
    """The thought channel's ``mind_state``: the core's reading of each night's dream, or None when
    the core is off (the channel then stops the night at build)."""
    return None if isinstance(core, NoCore) else core.reading


def build_core(
    env: Mapping[str, str], sleep_root: Path, people: frozenset[str], *,
    workspace: Callable[[str], WorkspaceLog], synthetic: Callable[[str], bool],
) -> MindCore:
    if not bridge.core_on(env) or not people:
        return NoCore()
    model = env.get(CORE_MODEL_ENV, "").strip()
    if not model:
        raise ValueError(f"{CORE_MODEL_ENV} names the trained core MIND's core runs on")
    night = importlib.import_module("anjo.sleep.mind.core.night")
    built: MindCore = night.from_env(env, Path(model), sleep_root, people=people, workspace=workspace,
                                     synthetic=synthetic)
    return built
