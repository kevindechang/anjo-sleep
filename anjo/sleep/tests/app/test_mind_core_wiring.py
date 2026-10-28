# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The app's switch for MIND's core in the night: off, ``NoCore`` and the night is today's; on,
the trained core over the person's mind store, read through the fake encoder, with the dream
curriculum on unless switched off. The app module loads the core by name, so importing it
loads neither torch nor the training code."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from anjo.sleep.app.mind_core import CORE_MODEL_ENV, build_core
from anjo.sleep.mind.core import bridge, store
from anjo.sleep.night.deps import NoCore


def test_off_the_night_has_no_core(tmp_path: Path) -> None:
    assert isinstance(build_core({}, tmp_path), NoCore)
    assert isinstance(build_core({bridge.CORE_ENV: "0", CORE_MODEL_ENV: "x"}, tmp_path), NoCore)


def test_on_without_a_trained_core_the_night_refuses_to_start(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=CORE_MODEL_ENV):
        build_core({bridge.CORE_ENV: "1"}, tmp_path)


def test_on_the_night_dreams_with_the_trained_core_over_her_store(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from anjo.sleep.mind.core import api
    from anjo.sleep.mind.core.night import CURRICULUM_ENV
    from anjo.sleep.mind.core.types import CoreConfig

    model = tmp_path / "core.npz"
    trained = api.Core.fresh(CoreConfig(dim_in=16, width=16, slots=4), seed=3)
    trained.save(model)
    on = {bridge.CORE_ENV: "1", CORE_MODEL_ENV: str(model)}
    core = build_core(on, tmp_path)
    assert core.available and core.core.model_id == trained.model_id  # type: ignore[attr-defined]
    assert core.root == tmp_path / store.STORE_DIR and core.curriculum  # type: ignore[attr-defined]
    assert core.encoder.spec.encoder == "fake" and core.encoder.spec.d == 16  # type: ignore[attr-defined]
    assert not build_core({**on, CURRICULUM_ENV: "0"}, tmp_path).curriculum  # type: ignore[attr-defined]


def test_the_app_module_loads_no_torch_and_no_training_code() -> None:
    code = ("import sys; import anjo.sleep.app.mind_core; "
            "sys.exit(1 if 'torch' in sys.modules or any(m.startswith('anjo.sleep.train') for m in sys.modules) else 0)")
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
