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
from anjo.sleep.mind.apd import calls, core_splits, fakes, generate
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
