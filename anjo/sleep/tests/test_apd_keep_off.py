# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""``--keep-off`` (MIND's data): a core split's writer is given the topics of its life's planted
items and told to keep them to their beats; APD's own splits refuse it, as registered, and a life
written with it says so in its made record. Every model is a fake: no network, no spend."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import calls, core_families, core_splits, fakes, generate, pairs
from anjo.sleep.mind.apd.generators import DEEPSEEK
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.model_api import ChatResult

SPEC = core_splits.core_specs("core-train")[0]
LifeChat = fakes.FakeLifeChat  # the fakes' own, kept before a test swaps the module's name


@dataclasses.dataclass
class Writes:
    """The fakes' writer and extractor, each write's system prompt and payload kept."""

    inner: Any = dataclasses.field(default_factory=LifeChat)
    seen: list[tuple[str, dict[str, Any]]] = dataclasses.field(default_factory=list)

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        if calls.kind_of(messages) == "write":
            self.seen.append((str(messages[0]["content"]), json.loads(messages[-1]["content"])))
        result: ChatResult = self.inner(model, messages, tools, timeout_s, temperature=temperature)
        return result


def test_a_writer_given_topics_is_told_to_keep_them_to_their_beats() -> None:
    topics = ("the bathroom tiling", "the hallway ceiling")
    assert DEEPSEEK.model is not None

    def told(keep_off: tuple[str, ...]) -> tuple[str, dict[str, Any]]:
        system, user = generate.writer_messages("film", 0, generate.LIFE_START, 1, [], [], model=str(DEEPSEEK.model),
                                                keep_off=keep_off)
        return str(system["content"]), json.loads(user["content"])

    (plain_system, plain), (system, payload) = told(()), told(topics)
    assert "keep_off" not in plain and generate.KEEP_OFF not in plain_system
    assert payload == {**plain, "keep_off": list(topics)}
    assert system == f"{plain_system.rstrip()}\n{generate.KEEP_OFF}\n"
    assert len(generate.KEEP_OFF.splitlines()) == 1 and "`keep_off`" in generate.KEEP_OFF


def test_keep_off_reaches_every_write_of_a_core_life_and_its_made_record(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    topics = [item.template.topic for item in generate.plan(SPEC.split, SPEC.field, SPEC.families, SPEC.prevalence,
                                                               SPEC.seed)]
    found: dict[bool, tuple[Writes, DataDir]] = {}

    # Act: the same life, with the flag and without it
    for flagged in (True, False):
        writes, data = Writes(), DataDir(tmp_path / str(flagged))
        monkeypatch.setattr(fakes, "FakeLifeChat", lambda writes=writes: writes)
        argv = ["--fake", "--data", str(data.root), "--only", "1", "generate", "--split", "core-train",
                *(["--keep-off"] if flagged else [])]
        assert cli.main(argv, out=lambda line: None) == 0
        found[flagged] = writes, data

    # Assert: every day's write is given the life's topics and the line, and the life's record says so
    writes, data = found[True]
    (life,) = data.lives("core-train")
    assert len(topics) == SPEC.prevalence and len(writes.seen) == 10
    assert all(payload["keep_off"] == topics and system.rstrip().endswith(generate.KEEP_OFF)
               for system, payload in writes.seen)
    assert (data.made(life.id) or {})["keep_off"] is True
    writes, data = found[False]
    (life,) = data.lives("core-train")
    assert all("keep_off" not in payload and generate.KEEP_OFF not in system for system, payload in writes.seen)
    assert "keep_off" not in (data.made(life.id) or {})


def test_apds_own_splits_refuse_keep_off(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "pilot"
    for split in ([], ["--split", "development"], ["--held-out", "--draw-seed", "x"]):
        with pytest.raises(SystemExit) as stopped:
            cli.main(["--fake", "--data", str(data), "generate", *split, "--keep-off"], out=lambda line: None)
        assert stopped.value.code == 2 and "--keep-off is for the core splits" in capsys.readouterr().err
    assert not data.exists()
    assert cli.main(["--fake", "--data", str(data), "--only", "1", "generate", "--split", "core-val", "--keep-off"],
                    out=lambda line: None) == 0


LIVE = ["--live", "--prices", "1.25,4.25,2.50,1.95", "--judge-prices", "1.25,4.25,0.75,3.75"]


def _generate(data: Path, *argv: str) -> None:
    assert cli.main(["--fake", "--data", str(data), *argv], out=lambda line: None) == 0


def test_the_plan_warns_once_when_a_core_run_differs_from_most_core_lives_on_disk(tmp_path: Path) -> None:
    # Arrange: one core-train life written with --keep-off
    data = tmp_path / "pilot"
    _generate(data, "--only", "1", "generate", "--split", "core-train", "--keep-off")

    def warnings(*argv: str) -> list[str]:
        lines: list[str] = []
        assert cli.main([*LIVE, "--data", str(data), "--only", "2", "generate", *argv], out=lines.append) == 0
        return [line for line in "\n".join(lines).splitlines() if line.startswith("Warning:")]

    # Act / Assert: a core run without it is warned once, with the counts; with it, or on APD's own split, not
    (warned,) = warnings("--split", "core-val")
    assert "--keep-off is off" in warned and "core lives on disk: 0 without keep_off, 1 with" in warned
    assert "every core split or none" in warned
    assert warnings("--split", "core-val", "--keep-off") == [] and warnings() == []
    # a tie is no majority: neither setting is warned
    _generate(data, "--only", "1", "generate", "--split", "core-val")
    assert warnings("--split", "core-test") == [] and warnings("--split", "core-test", "--keep-off") == []
    # most without it: a run with it is warned, one without is not
    _generate(data, "--only", "2", "generate", "--split", "core-val")
    (warned,) = warnings("--split", "core-test", "--keep-off")
    assert "--keep-off is on" in warned and "core lives on disk: 2 without keep_off, 1 with" in warned
    assert warnings("--split", "core-test") == []


def test_pairs_take_keep_off_from_most_core_lives_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the first pair's topics, the unrelated item's and the commitment's gap's
    spec = pairs.pair_specs(1)[0]
    unrelated = core_families.template(spec.unrelated)
    assert unrelated is not None
    topics = [unrelated.topic, pairs.pair_template(spec.template).gap.topic]
    found: dict[str, tuple[Writes, DataDir]] = {}

    # Act: pairs beside core lives written with it, beside core lives written without it, and alone
    for setting, argv in (("with", ["--keep-off"]), ("without", []), ("alone", None)):
        data = DataDir(tmp_path / setting)
        monkeypatch.setattr(fakes, "FakeLifeChat", LifeChat)  # the core life's writes are not the pair's
        if argv is not None:
            _generate(data.root, "--only", "1", "generate", "--split", "core-test", *argv)
        writes = Writes()
        monkeypatch.setattr(fakes, "FakeLifeChat", lambda writes=writes: writes)
        _generate(data.root, "generate-pairs", "--n", "1")
        found[setting] = writes, data

    # Assert: each pair's every write follows the core lives, and its made record says so
    writes, data = found["with"]
    (pair,) = data.pairs()
    assert writes.seen and all(payload["keep_off"] == topics for _, payload in writes.seen)
    assert (data.made(pair.id) or {})["keep_off"] is True
    for setting in ("without", "alone"):
        writes, data = found[setting]
        (pair,) = data.pairs()
        assert writes.seen and all("keep_off" not in payload for _, payload in writes.seen)
        assert "keep_off" not in (data.made(pair.id) or {})



def _pair_plan(data: Path) -> str:
    lines: list[str] = []
    assert cli.main([*LIVE, "--data", str(data), "generate-pairs", "--n", "1"], out=lines.append) == 0
    (note,) = [line for line in "\n".join(lines).splitlines() if "keep_off" in line]
    return note


def test_pairs_follow_core_test_alone_and_the_plan_says_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: no core-test life yet; the plan says the pairs take the default, and why
    data = DataDir(tmp_path / "pilot")
    assert "without keep_off, following the default (no pairs or core-test lives on disk)" in _pair_plan(data.root)
    # two core-train lives written with it outvote nothing: one core-test life written without it decides
    _generate(data.root, "--only", "2", "generate", "--split", "core-train", "--keep-off")
    _generate(data.root, "--only", "1", "generate", "--split", "core-test")
    writes = Writes()
    monkeypatch.setattr(fakes, "FakeLifeChat", lambda: writes)

    # Act
    note = _pair_plan(data.root)
    _generate(data.root, "generate-pairs", "--n", "1")

    # Assert
    assert "without keep_off, following the core-test majority (core-test lives on disk: 1 without keep_off, 0 with)" in note
    (pair,) = data.pairs()
    assert writes.seen and all("keep_off" not in payload for _, payload in writes.seen)
    assert "keep_off" not in (data.made(pair.id) or {})


def _plan_lines(data: Path, n: int) -> list[str]:
    lines: list[str] = []
    assert cli.main([*LIVE, "--data", str(data), "generate-pairs", "--n", str(n)], out=lines.append) == 0
    return [line for line in "\n".join(lines).splitlines() if "keep_off" in line]


def test_new_pairs_follow_the_pairs_on_disk_over_core_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: a pair written with no keep_off field, then core-test turns to most lives written with it
    data = DataDir(tmp_path / "pilot")
    _generate(data.root, "generate-pairs", "--n", "1")
    (first,) = data.pairs()
    assert "keep_off" not in (data.made(first.id) or {})
    _generate(data.root, "--only", "2", "generate", "--split", "core-test", "--keep-off")
    writes, lines = Writes(), list[str]()
    monkeypatch.setattr(fakes, "FakeLifeChat", lambda: writes)

    # Act
    note, warned = _plan_lines(data.root, 2)
    code = cli.main(["--fake", "--data", str(data.root), "generate-pairs", "--n", "2"], out=lines.append)

    # Assert: the plan and the run agree: without it, as the pair on disk was; core-test's lean is a warning
    assert note.startswith("Note: the pairs are written without keep_off, following the pairs on disk")
    assert warned.startswith("Warning:") and "core-test lives on disk: 0 without keep_off, 2 with" in warned
    assert code == 0 and not any(line.startswith("Refused:") for line in lines) and len(data.pairs()) == 2
    assert writes.seen and all("keep_off" not in payload for _, payload in writes.seen)
    assert all("keep_off" not in (data.made(pair.id) or {}) for pair in data.pairs())
    # a rerun with nothing left to write goes through
    lines.clear()
    assert cli.main(["--fake", "--data", str(data.root), "generate-pairs", "--n", "2"], out=lines.append) == 0
    assert not any(line.startswith("Refused:") for line in lines)


def test_pairs_on_disk_that_disagree_refuse_a_run_with_pairs_to_write(tmp_path: Path) -> None:
    # Arrange: two pairs on disk, one of them recorded as written with keep_off
    data = DataDir(tmp_path / "pilot")
    _generate(data.root, "generate-pairs", "--n", "2")
    first = data.pairs()[0]
    data.save_made(first.id, {**(data.made(first.id) or {}), "keep_off": True})
    lines: list[str] = []

    # Act
    warned = _plan_lines(data.root, 3)[-1]
    code = cli.main(["--fake", "--data", str(data.root), "generate-pairs", "--n", "3"], out=lines.append)

    # Assert: the plan warns, the run refuses before writing, a rerun with nothing to write does not
    assert warned.startswith("Warning:") and "mixed" in warned and "1 without keep_off, 1 with" in warned
    (refused,) = [line for line in lines if line.startswith("Refused:")]
    assert code == cli.REFUSED_CODE and "pairs on disk: 1 without keep_off, 1 with" in refused
    assert "go on with theirs" not in refused and len(data.pairs()) == 2
    assert cli.main(["--fake", "--data", str(data.root), "generate-pairs", "--n", "2"], out=lambda line: None) == 0
