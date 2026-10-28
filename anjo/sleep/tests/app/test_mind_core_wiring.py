# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The app's switch for MIND's core in the night: off, or with MIND's inquiry off, ``NoCore`` and
the night is today's; on, the trained core for the demo persona's people only, over each one's
mind store and workspace log, read through the fake encoder, with the dream curriculum on unless
switched off. The app module loads the core by name, so importing it
loads neither torch nor the training code."""
from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from anjo.sleep.app.mind_core import CORE_MODEL_ENV, build_core, core_people
from anjo.sleep.mind.core import bridge, store
from anjo.sleep.mind.inquiry.log import MemoryWorkspaceLog
from anjo.sleep.night.deps import MindCore, NightOptions, NoCore
from anjo.sleep.personas import DEMO_PERSONA

NIGHT = date(2026, 10, 28)
PEOPLE = frozenset({"u-hero"})
LOGS: dict[str, MemoryWorkspaceLog] = {}


def _build(env: dict[str, str], root: Path, people: frozenset[str] = PEOPLE) -> MindCore:
    return build_core(env, root, people, workspace=lambda uid: LOGS.setdefault(uid, MemoryWorkspaceLog()),
                      synthetic=lambda uid: True)


def test_the_core_serves_the_demo_persona_only_and_only_with_minds_inquiry_on() -> None:
    personas = {"u-hero": DEMO_PERSONA, "u-other": "founder", "u-hero-2": DEMO_PERSONA}
    assert core_people(NightOptions(NIGHT, personas, mind_inquiry=True)) == {"u-hero", "u-hero-2"}
    assert core_people(NightOptions(NIGHT, personas, mind_inquiry=False)) == frozenset()  # --no-mind
    assert core_people(NightOptions(NIGHT, {"u-other": "founder"}, mind_inquiry=True)) == frozenset()


def test_off_or_for_no_one_the_night_has_no_core(tmp_path: Path) -> None:
    assert isinstance(_build({}, tmp_path), NoCore)
    assert isinstance(_build({bridge.CORE_ENV: "0", CORE_MODEL_ENV: "x"}, tmp_path), NoCore)
    assert isinstance(_build({bridge.CORE_ENV: "1", CORE_MODEL_ENV: "x"}, tmp_path, frozenset()), NoCore)
    assert not NoCore().serves("u-hero")


def test_on_without_a_trained_core_the_night_refuses_to_start(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=CORE_MODEL_ENV):
        _build({bridge.CORE_ENV: "1"}, tmp_path)


def test_on_the_night_dreams_with_the_trained_core_over_her_store(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from anjo.sleep.mind.core import api
    from anjo.sleep.mind.core.night import CURRICULUM_ENV, FAKE_LATENTS_ENV
    from anjo.sleep.mind.core.types import CoreConfig

    model = tmp_path / "core.npz"
    trained = api.Core.fresh(CoreConfig(dim_in=16, width=16, slots=4), seed=3)
    trained.save(model)
    on = {bridge.CORE_ENV: "1", CORE_MODEL_ENV: str(model)}
    core = _build(on, tmp_path)
    assert core.serves("u-hero") and not core.serves("u-other")  # never another person
    assert core.core.model_id == trained.model_id  # type: ignore[attr-defined]
    assert core.root == tmp_path / store.STORE_DIR and core.curriculum  # type: ignore[attr-defined]
    assert core.encoder.spec.encoder == "fake" and core.encoder.spec.d == 16  # type: ignore[attr-defined]
    assert core.workspace("u-hero") is LOGS["u-hero"]  # type: ignore[attr-defined]
    assert not core.fake_for_anyone  # type: ignore[attr-defined]
    assert _build({**on, FAKE_LATENTS_ENV: "1"}, tmp_path).fake_for_anyone  # type: ignore[attr-defined]
    assert not _build({**on, CURRICULUM_ENV: "0"}, tmp_path).curriculum  # type: ignore[attr-defined]


def test_the_app_module_loads_no_torch_and_no_training_code() -> None:
    code = ("import sys; import anjo.sleep.app.mind_core; "
            "sys.exit(1 if 'torch' in sys.modules or any(m.startswith('anjo.sleep.train') for m in sys.modules) else 0)")
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
