# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The night's SFT dataset (Task 5): practice transcripts with a pre-practice context snapshot,
decontaminated against every sealed probe (RF5), cleared ancestry only, a seeded replay sample and
context dropout, inside the token cap, with nothing trained cut."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.evidence.store import JsonlEvidenceStore
from anjo.sleep.practice.transcripts import JsonlTranscriptStore, PracticeTranscript
from anjo.sleep.probes.decontam import OverlapDecontaminator
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.tests.sleep_fakes import WONDER, probe
from anjo.sleep.train.dataset import build_night_dataset, quotes_url_or_date

NIGHT = date(2026, 10, 28)
T0 = datetime(2026, 10, 28, 23, 0, tzinfo=timezone.utc)
CLEARED = frozenset({SPARK_OPEN, GLIMMER})


def _transcript(
    i: int, *, night: date = NIGHT, reply: str | None = None, message: str | None = None,
    context: str | None = "## Their brief\nShe edits her own films.", ancestry: frozenset[str] | None = CLEARED,
) -> PracticeTranscript:
    return PracticeTranscript(
        id=f"pr-{night.isoformat()}-{i:02d}", person_id="p1", night=night, scene=i,
        message=message or f"I just got a new festival brief, number {i}. Where would you start?",
        reply=reply or f"Start with the edit that carries scene {i}, and cut toward the strongest image.",
        context=context, f_ids=("e1",), ancestry=ancestry,
        created_at=datetime(night.year, night.month, night.day, 23, 0, tzinfo=timezone.utc) + timedelta(minutes=i),
    )


def _stores(tmp_path: Path, transcripts: list[PracticeTranscript]) -> tuple[JsonlEvidenceStore, JsonlTranscriptStore]:
    store = JsonlEvidenceStore(tmp_path / "evidence.jsonl", "p1", codec=PLAIN, guard=NO_GUARD)
    log = JsonlTranscriptStore(tmp_path / "practice.jsonl", "p1", codec=PLAIN, guard=NO_GUARD)
    store.add([t.evidence() for t in transcripts])
    log.add(transcripts)
    return store, log


def _build(tmp_path: Path, transcripts: list[PracticeTranscript], **kw: object) -> tuple[object, int]:
    store, log = _stores(tmp_path, transcripts)
    args: dict[str, object] = {
        "transcripts": log, "seed": 7, "dropout": 0.6, "max_seq_tokens": 8192,
    }
    args.update(kw)
    sealed = args.pop("sealed", [])
    token_cap = args.pop("token_cap", 100_000)
    since = args.pop("since", None)
    return build_night_dataset(
        store, since, NIGHT, sealed, OverlapDecontaminator(), token_cap, **args,  # type: ignore[arg-type]
    )


def test_dataset_drops_probe_quoted_in_practice_transcript(tmp_path: Path) -> None:
    sealed = [
        probe("q1", prompt="How should I cut the ferry scene where Mara leaves the island at dawn?"),
        probe("q2", prompt="Which lens would you pick for the night market chase through the alleys?"),
    ]
    quoted = _transcript(1, message="How should I cut the ferry scene where Mara leaves the island at dawn?")
    paraphrased = _transcript(
        2, message="What's the best way to cut the ferry sequence where Mara leaves the island around dawn?")
    clean = _transcript(3)
    data, dropped = _build(tmp_path, [quoted, paraphrased, clean], sealed=sealed)
    assert dropped == 2
    assert [example.id for example in data.examples] == [clean.id]  # type: ignore[attr-defined]
    assert data.dropped["probe"] == 2  # type: ignore[attr-defined]


def test_dataset_refuses_hosted_ancestry(tmp_path: Path) -> None:
    hosted = _transcript(1, ancestry=CLEARED | {WONDER})
    unknown = _transcript(2, ancestry=None)
    clean = _transcript(3)
    data, _ = _build(tmp_path, [hosted, unknown, clean])
    assert [example.id for example in data.examples] == [clean.id]  # type: ignore[attr-defined]
    assert data.dropped["hosted"] == 2  # type: ignore[attr-defined]


def test_reply_with_hosted_context_never_trains(tmp_path: Path) -> None:
    # The reply's context F held a homework item (Wonder): its ancestry carries Wonder, so the
    # example never trains, even though the evidence store holds it as a practice item.
    from_homework = _transcript(1, ancestry=frozenset({SPARK_OPEN, GLIMMER, WONDER}))
    store, log = _stores(tmp_path, [from_homework])
    data, _ = build_night_dataset(
        store, None, NIGHT, [], OverlapDecontaminator(), 100_000, transcripts=log, seed=7,
    )
    assert data.examples == ()


def test_night_dataset_respects_token_cap(tmp_path: Path) -> None:
    transcripts = [_transcript(i) for i in range(10)]
    data, _ = _build(tmp_path, transcripts, token_cap=200)
    assert 0 < data.tokens <= 200  # type: ignore[attr-defined]
    assert len(data.examples) < 10  # type: ignore[attr-defined]
    assert data.dropped["over_cap"] == 10 - len(data.examples)  # type: ignore[attr-defined]


def test_replay_sample_is_seeded_and_reproducible(tmp_path: Path) -> None:
    earlier = [_transcript(i, night=date(2026, 10, 20 + i % 5)) for i in range(12)]
    tonight = [_transcript(50 + i) for i in range(2)]
    first, _ = _build(tmp_path / "a", earlier + tonight, since=date(2026, 10, 27), token_cap=500)
    second, _ = _build(tmp_path / "b", earlier + tonight, since=date(2026, 10, 27), token_cap=500)
    other, _ = _build(tmp_path / "c", earlier + tonight, since=date(2026, 10, 27), token_cap=500, seed=8)
    ids = [example.id for example in first.examples]  # type: ignore[attr-defined]
    assert ids == [example.id for example in second.examples]  # type: ignore[attr-defined]
    assert ids[:2] == [tonight[1].id, tonight[0].id]  # tonight's new examples first, newest first
    assert all(not example.new for example in first.examples[2:])  # type: ignore[attr-defined]
    replay = ids[2:]
    assert replay and replay != [example.id for example in other.examples][2:]  # type: ignore[attr-defined]


def test_sft_context_never_contains_its_target(tmp_path: Path) -> None:
    transcripts = [_transcript(i) for i in range(6)]
    data, _ = _build(tmp_path, transcripts, dropout=0.0)
    for example in data.examples:  # type: ignore[attr-defined]
        context = "".join(text for role, text in example.prompt if role == "system")
        assert example.completion not in context


def test_context_dropout_is_seeded_and_drops_all_of_c(tmp_path: Path) -> None:
    transcripts = [_transcript(i) for i in range(40)]
    first, _ = _build(tmp_path / "a", transcripts, dropout=0.6)
    second, _ = _build(tmp_path / "b", transcripts, dropout=0.6)
    dropped = [example for example in first.examples if example.context_dropped]  # type: ignore[attr-defined]
    assert [e.id for e in dropped] == [
        e.id for e in second.examples if e.context_dropped]  # type: ignore[attr-defined]
    assert 0.3 < len(dropped) / len(first.examples) < 0.9  # type: ignore[attr-defined]
    for example in first.examples:  # type: ignore[attr-defined]
        roles = [role for role, _ in example.prompt]
        assert roles == (["user"] if example.context_dropped else ["system", "user"])


def test_zero_context_targets_quote_no_urls_or_dates(tmp_path: Path) -> None:
    dated = [_transcript(i, reply=f"The festival deadline is 2026-11-{10 + i:02d}, so lock picture first.")
             for i in range(10)]
    linked = [_transcript(20 + i, reply="See https://example.org/grading for the grading notes.")
              for i in range(10)]
    data, _ = _build(tmp_path, dated + linked, dropout=0.9)
    for example in data.examples:  # type: ignore[attr-defined]
        assert not example.context_dropped
    assert quotes_url_or_date("cut it on March 3") and quotes_url_or_date("www.example.org")
    assert not quotes_url_or_date("cut it on the last word")


def test_truncated_target_is_dropped_and_counted(tmp_path: Path) -> None:
    long_reply = "word " * 40_000
    data, _ = _build(tmp_path, [_transcript(1, reply=long_reply), _transcript(2)])
    assert [example.id for example in data.examples] == [_transcript(2).id]  # type: ignore[attr-defined]
    assert data.dropped["truncated"] == 1  # type: ignore[attr-defined]


def test_dataset_jsonl_is_prompt_completion_with_loss_on_the_reply_only(tmp_path: Path) -> None:
    data, _ = _build(tmp_path, [_transcript(1)], dropout=0.0)
    row = json.loads(data.to_jsonl().splitlines()[0])  # type: ignore[attr-defined]
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]
    assert row["completion"] == [{"role": "assistant", "content": _transcript(1).reply}]


def test_held_out_and_uncleared_practice_never_reaches_the_dataset(tmp_path: Path) -> None:
    held = replace(_transcript(1), id="pr-held")
    store, log = _stores(tmp_path, [_transcript(2)])
    store.add([replace(held.evidence(), held_out=True)])
    log.add([held])
    data, _ = build_night_dataset(store, None, NIGHT, [], OverlapDecontaminator(), 100_000, transcripts=log, seed=7)
    assert [example.id for example in data.examples] == [_transcript(2).id]


def test_rows_carry_their_sleep_weight_only_when_mind_weighted_them(tmp_path: Path) -> None:
    data, _ = _build(tmp_path, [_transcript(1), _transcript(2), _transcript(3)], dropout=0.0)
    plain = [json.loads(line) for line in data.to_jsonl().splitlines()]  # type: ignore[attr-defined]
    assert all("weight" not in row for row in plain)  # unweighted rows are today's, byte for byte
    first, second, third = (example.id for example in data.examples)  # type: ignore[attr-defined]
    weighted = data.weighted({first: 1.5, second: 1.0})  # type: ignore[attr-defined]
    rows = {row["id"]: row for row in (json.loads(line) for line in weighted.to_jsonl().splitlines())}
    assert rows[first]["weight"] == 1.5
    assert "weight" not in rows[second] and "weight" not in rows[third]  # 1.0 is the trainer's default
    assert data.examples[0].weight == 1.0  # type: ignore[attr-defined]  # weighting makes a new dataset
    with pytest.raises(ValueError, match="positive"):
        data.weighted({first: 0.0})  # type: ignore[attr-defined]
