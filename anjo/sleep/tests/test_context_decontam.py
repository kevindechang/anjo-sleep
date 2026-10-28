# mypy: disallow-untyped-decorators=False
"""The context (the person's own brief) is held to exact and 13-gram only; everything that trains
(message, reply, each DPO side) keeps all three rules."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from anjo.sleep.tests.sleep_fakes import probe
from anjo.sleep.tests.test_night_dataset import _build, _transcript

PROBE = "How should I cut the ferry scene where Mara leaves the island at dawn?"
PARAPHRASE = "What's the best way to cut the ferry sequence where Mara leaves the island around dawn?"
# Shares a 13-word run with PROBE (and so a paraphrase too)
VERBATIM = "Her notes say: how should I cut the ferry scene where Mara leaves the island at dawn and more."


def _sealed() -> list[Any]:
    return [probe("q1", prompt=PROBE)]


def test_context_paraphrase_alone_keeps_the_example(tmp_path: Path) -> None:
    kept = _transcript(1, context=f"## Brief\n{PARAPHRASE}")
    data, dropped = _build(tmp_path, [kept], sealed=_sealed())
    assert dropped == 0 and [e.id for e in data.examples] == [kept.id]  # type: ignore[attr-defined]


def test_context_13gram_still_drops(tmp_path: Path) -> None:
    data, dropped = _build(tmp_path, [_transcript(1, context=f"## Brief\n{VERBATIM}")], sealed=_sealed())
    assert dropped == 1 and not data.examples  # type: ignore[attr-defined]
    exact = _transcript(2, context=PROBE)
    data, dropped = _build(tmp_path / "b", [exact], sealed=_sealed())
    assert dropped == 1 and not data.examples  # type: ignore[attr-defined]


def test_reply_paraphrase_still_drops(tmp_path: Path) -> None:
    data, dropped = _build(tmp_path, [_transcript(1, reply=PARAPHRASE)], sealed=_sealed())
    assert dropped == 1 and not data.examples  # type: ignore[attr-defined]
    data, dropped = _build(tmp_path / "m", [_transcript(2, message=PARAPHRASE)], sealed=_sealed())
    assert dropped == 1 and not data.examples  # type: ignore[attr-defined]


def test_dpo_side_paraphrase_still_drops_the_pair(tmp_path: Path) -> None:
    pair = replace(_transcript(1), rejected=PARAPHRASE, rejected_from="glimmer")
    data, _ = _build(tmp_path, [pair], sealed=_sealed())
    assert len(data.examples) == 1 and not data.pairs and data.pairs_dropped["probe"] == 1  # type: ignore[attr-defined]
    chosen = _transcript(2, reply=PARAPHRASE)
    chosen = replace(chosen, rejected="A flat reply.", rejected_from="glimmer")
    data, _ = _build(tmp_path / "c", [chosen], sealed=_sealed())
    assert not data.examples and not data.pairs  # type: ignore[attr-defined]


def test_context_paraphrase_kept_is_counted(tmp_path: Path) -> None:
    items = [_transcript(1, context=f"## Brief\n{PARAPHRASE}"), _transcript(2, context=f"## Brief\n{PARAPHRASE}"),
             _transcript(3), _transcript(4, reply=PARAPHRASE, context=f"## Brief\n{PARAPHRASE}")]
    data, dropped = _build(tmp_path, items, sealed=_sealed())
    assert dropped == 1 and len(data.examples) == 3  # type: ignore[attr-defined]
    assert data.context_paraphrase_kept == 2  # type: ignore[attr-defined]  # kept only; the dropped one isn't counted
