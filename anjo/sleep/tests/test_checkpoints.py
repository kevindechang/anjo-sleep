# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The checkpoint registry (Task 5): atomic promotion, every promoted adapter kept, and a trained
name never reused."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from anjo.sleep.checkpoints import CheckpointError, CheckpointRegistry, NoCheckpoint
from anjo.sleep.types import CheckpointRef

NIGHT = date(2026, 10, 28)


def _ref(name: str, person: str = "p1", parent: str | None = None, night: date = NIGHT) -> CheckpointRef:
    return CheckpointRef(name, person, night, f"modal-volume://anjo-sleep-adapters/{name}", parent)


def test_promote_sets_current_and_keeps_every_promoted_adapter(tmp_path: Path) -> None:
    registry = CheckpointRegistry(tmp_path)
    with pytest.raises(NoCheckpoint):
        registry.current("p1")
    assert registry.last_promoted("p1") is None
    first, second = _ref("p-aaaaaaaaaaaa-n1"), _ref("p-bbbbbbbbbbbb-n2", parent="p-aaaaaaaaaaaa-n1")
    registry.promote("p1", first)
    registry.promote("p1", second)
    assert registry.current("p1") == second
    assert registry.previous("p1") == first
    assert registry.history("p1") == [first, second]
    assert registry.last_promoted("p1") == second


def test_promotion_is_a_rename_of_a_new_pointer_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = CheckpointRegistry(tmp_path)
    registry.promote("p1", _ref("p-aaaaaaaaaaaa-n1"))
    renames: list[tuple[str, str]] = []
    real_replace = __import__("os").replace

    def spy(src: str, dst: str) -> None:
        renames.append((Path(src).name, Path(dst).name))
        real_replace(src, dst)

    monkeypatch.setattr("anjo.sleep.checkpoints.os.replace", spy)
    registry.promote("p1", _ref("p-bbbbbbbbbbbb-n2"))
    assert renames and renames[-1][1] == "current.json" and renames[-1][0] != "current.json"


def test_a_promotion_cut_before_its_rename_keeps_the_previous_one_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = CheckpointRegistry(tmp_path)
    first = _ref("p-aaaaaaaaaaaa-n1")
    registry.promote("p1", first)

    def crash(src: str, dst: str) -> None:
        raise OSError("killed mid-promotion")

    monkeypatch.setattr("anjo.sleep.checkpoints.os.replace", crash)
    with pytest.raises(OSError):
        registry.promote("p1", _ref("p-bbbbbbbbbbbb-n2"))
    monkeypatch.undo()
    assert registry.current("p1") == first
    assert registry.history("p1") == [first]


def test_a_checkpoint_of_another_person_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError):
        CheckpointRegistry(tmp_path).promote("p1", _ref("p-aaaaaaaaaaaa-n1", person="p2"))


def test_trained_names_are_recorded_once_and_their_nights_are_kept(tmp_path: Path) -> None:
    registry = CheckpointRegistry(tmp_path)
    assert registry.trained_nights("p1") == frozenset()
    registry.record_trained("p1", "p-aaaaaaaaaaaa-n1", NIGHT)
    with pytest.raises(CheckpointError):
        registry.record_trained("p1", "p-aaaaaaaaaaaa-n1", date(2026, 10, 29))
    registry.record_trained("p1", "p-cccccccccccc-n2", date(2026, 10, 29))
    assert registry.trained_names("p1") == frozenset({"p-aaaaaaaaaaaa-n1", "p-cccccccccccc-n2"})
    assert registry.trained_nights("p1") == frozenset({NIGHT, date(2026, 10, 29)})


def test_user_ids_that_are_not_plain_names_are_refused(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError):
        CheckpointRegistry(tmp_path).history("../p1")


def test_previous_skips_a_rolled_back_checkpoint(tmp_path: Path) -> None:
    registry = CheckpointRegistry(tmp_path)
    n1, n2, n3 = (_ref(f"p-aaaaaaaaaaa{i}-n{i}", night=date(2026, 10, 27 + i)) for i in (1, 2, 3))
    registry.promote("p1", n1)
    registry.promote("p1", n2)  # a bad night
    registry.rollback("p1", n1)
    assert registry.current("p1") == n1 and registry.previous("p1") is None
    registry.promote("p1", n3)
    assert registry.current("p1") == n3 and registry.previous("p1") == n1
    assert registry.history("p1") == [n1, n3]
    with pytest.raises(CheckpointError):
        registry.rollback("p1", _ref("p-bbbbbbbbbbbb-n9"))  # only to a checkpoint that served
