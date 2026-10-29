# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The freeze holds for every held-out step (Task 17 review): held-out lives are drawn, and run,
only from a seed the prereg ledger committed before its freeze; a live command reads only the
pushed ledger, never a stand-in."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from anjo.sleep.mind.apd import fakes, pilot
from anjo.sleep.mind.apd.__main__ import REFUSED_CODE
from anjo.sleep.mind.apd.__main__ import main as apd_main
from anjo.sleep.mind.apd.store import DataDir, StoreError
from anjo.sleep.mind.apd.types import HELD_OUT_LIVES
from anjo.sleep.prereg.ledger import read_entries, register

SEED = "the-committed-seed"


def _ledger(path: Path, *, seed_first: bool = True) -> Path:
    digest = pilot.seed_digest(SEED)
    steps = [(pilot.DRAW_SEED_KIND, digest, "seed"), (pilot.FREEZE_KIND, "0" * 64, f"freeze held_out_lives={HELD_OUT_LIVES}")]
    for kind, sha, note in steps if seed_first else steps[::-1]:
        register(kind, sha, note, path=path)
    return path


def test_a_held_out_run_needs_lives_drawn_from_a_seed_committed_before_the_freeze(tmp_path: Path) -> None:
    entries = read_entries(_ledger(tmp_path / "ledger.jsonl"))
    with pytest.raises(pilot.FreezeError, match="no held-out lives were drawn"):
        pilot.check_drawn(entries, None)
    pilot.check_drawn(entries, (pilot.seed_digest(SEED), HELD_OUT_LIVES))
    with pytest.raises(pilot.FreezeError, match="before the freeze"):
        pilot.check_drawn(entries, (pilot.seed_digest("another seed"), HELD_OUT_LIVES))
    late = read_entries(_ledger(tmp_path / "late.jsonl", seed_first=False))
    with pytest.raises(pilot.FreezeError, match="before the freeze"):
        pilot.check_drawn(late, (pilot.seed_digest(SEED), HELD_OUT_LIVES))


def test_a_data_dir_holds_one_draw(tmp_path: Path) -> None:
    data = DataDir(tmp_path)
    data.save_drawn(pilot.seed_digest(SEED), HELD_OUT_LIVES)
    data.save_drawn(pilot.seed_digest(SEED), HELD_OUT_LIVES)  # the same draw again: a rerun
    assert data.drawn() == (pilot.seed_digest(SEED), HELD_OUT_LIVES)
    with pytest.raises(StoreError):
        data.save_drawn(pilot.seed_digest("another seed"), HELD_OUT_LIVES)


@pytest.mark.parametrize("command", [["run", "--held-out"], ["kill-resume", "--kill-at", "3"]])
def test_held_out_runs_are_refused_before_anything_starts_without_a_draw(tmp_path: Path, command: list[str]) -> None:
    _ledger(tmp_path / "prereg" / "ledger.jsonl")  # frozen, but no lives drawn in this data dir
    lines: list[str] = []
    assert apd_main(["--fake", "--data", str(tmp_path), *command], out=lines.append) == REFUSED_CODE
    assert lines[-1].startswith("Refused: no held-out lives were drawn")
    assert not any(line.startswith("serving Glimmer") for line in lines)  # refused before any session


def test_live_never_takes_a_stand_in_prereg_ledger(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as refused:
        apd_main(["--live", "--data", str(tmp_path), "generate", "--held-out", "--draw-seed", SEED,
                  "--prereg", str(_ledger(tmp_path / "mine.jsonl"))], out=lambda _line: None)
    assert refused.value.code == 2


def test_every_session_of_a_pilot_serves_what_its_first_did(tmp_path: Path) -> None:
    data = DataDir(tmp_path)
    pins = {"glimmer_revision": "a" * 40, "vllm_version": "0.11.0", "precision": "fp8", "reasoning_strength": "low"}
    data.pin_serving(pins)
    data.pin_serving(pins)  # the same again: a later command
    with pytest.raises(StoreError, match="reasoning_strength"):
        data.pin_serving({**pins, "reasoning_strength": "high"})
    with pytest.raises(StoreError, match="glimmer_revision"):
        data.pin_serving({**pins, "glimmer_revision": "b" * 40})
    lines: list[str] = []
    (tmp_path / "serving_pin.json").write_text('{"glimmer_revision": "' + "b" * 40 + '"}')
    assert apd_main(["--fake", "--data", str(tmp_path), "timing"], out=lines.append) == REFUSED_CODE
    assert lines[-1].startswith("Refused: this pilot serves another") and not any("serving Glimmer" in x for x in lines)


def test_the_pin_records_deterministic_serving_so_a_flip_is_refused(tmp_path: Path) -> None:
    from anjo.sleep.mind.apd.__main__ import _pins

    on, off = _pins(replace(fakes.SETTINGS, deterministic=True)), _pins(replace(fakes.SETTINGS, deterministic=False))
    assert (on["deterministic"], off["deterministic"]) == ("1", "0")
    data = DataDir(tmp_path)
    data.pin_serving(on)
    with pytest.raises(StoreError, match="deterministic"):
        data.pin_serving(off)


def test_bad_adapters_are_refused_before_the_vm_boots(tmp_path: Path) -> None:
    _ledger(tmp_path / "prereg" / "ledger.jsonl")
    DataDir(tmp_path).save_drawn(pilot.seed_digest(SEED), HELD_OUT_LIVES)
    lines: list[str] = []
    code = apd_main(["--fake", "--data", str(tmp_path), "run", "--held-out", "--adapters", str(tmp_path / "nowhere"),
                     "--replay", "p-0123456789ab-n0", "--shuffled", "p-ba9876543210-n0"], out=lines.append)
    assert code == REFUSED_CODE and lines[-1].startswith("Refused: an adapter is a folder")
    assert not any("serving Glimmer" in line for line in lines)
