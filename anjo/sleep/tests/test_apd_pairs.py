# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Counterfactual autobiography pairs (MIND's data): two histories of one person that differ only
in the outcome of an earlier commitment, on the fork days alone, and end on the same last day word
for word; the pair records what should differ and what should not; both histories pass the gates;
and the CLI generates, plans and exports them. Every model is a fake."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import core_families, gates, labels, pairs, plan
from anjo.sleep.mind.apd.generate import life_days as pair_days
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.apd.types import PAIR_DECISIONS, CounterfactualPair, GateVerdict, Life, ProblemFile
from anjo.sleep.mind.spark_provider import FIREWORKS

SPEC = pairs.pair_specs()[0]


def _made(spec: pairs.PairSpec = SPEC) -> CounterfactualPair:
    return pairs.generate_pair(spec, backends.fake_backend(FIREWORKS)).pair


def _on(life: Life, index: int) -> tuple[str, tuple[tuple[str, str], ...]]:
    day = life.days[index]
    text = next(s.text for s in life.sessions if s.day == day)
    return text, tuple((e.id, e.text) for e in life.evidence if e.day == day)


def test_pair_histories_share_the_final_day_verbatim() -> None:
    # Act
    pair = _made()

    # Assert: the last day is the same messages and the same evidence, ids included
    last = len(pair.failed.days) - 1
    assert _on(pair.failed, last) == _on(pair.kept, last)
    assert pair.final_ids == tuple(e.id for e in pair.kept.evidence if e.day == pair.kept.days[-1]) != ()
    # ... as is every other day but the fork days, where only the outcome differs
    forks = [pair.failed.days.index(day) for day in pair.fork_days]
    assert forks == list(pairs.OUTCOME_DAYS)
    for index in range(len(pair.failed.days)):
        assert (_on(pair.failed, index) == _on(pair.kept, index)) == (index not in forks), index


def test_pair_histories_differ_only_in_outcome_events() -> None:
    # Arrange
    pair = _made()
    template = pairs.pair_template(pair.template)
    failed_ids, kept_ids = {e.id for e in pair.failed.evidence}, {e.id for e in pair.kept.evidence}
    outcomes = [e for e in (*pair.failed.evidence, *pair.kept.evidence) if e.id not in failed_ids & kept_ids]
    touched = {e.target_id for e in outcomes if e.target_id is not None}

    # Assert: each fork day is the one shared day, then that history's outcome as its last message
    for index, failed_said, kept_said in zip(pairs.OUTCOME_DAYS, template.failed, template.kept, strict=True):
        failed_text, kept_text = _on(pair.failed, index)[0], _on(pair.kept, index)[0]
        assert failed_text.endswith("\n\n" + failed_said) and kept_text.endswith("\n\n" + kept_said)
        assert failed_text.removesuffix("\n\n" + failed_said) == kept_text.removesuffix("\n\n" + kept_said)
    # the only evidence apart is the outcomes', on the fork days; every other item is the same, id and all
    assert {e.day for e in outcomes} == set(pair.fork_days)
    assert set(pair.failed_ids) | set(pair.kept_ids) <= {e.id for e in outcomes}
    assert {e.id: e for e in pair.failed.evidence if e.id in kept_ids} == {
        e.id: e for e in pair.kept.evidence if e.id in failed_ids}
    # the targets agree but for the ones an outcome mentions, whose state follows from the outcome
    for day in pair.failed.days:
        assert ({t.id: t for t in pair.failed.targets[day] if t.id not in touched}
                == {t.id: t for t in pair.kept.targets[day] if t.id not in touched}), day
    for day in pair.failed.days[:pairs.FORK_DAY]:
        assert pair.failed.targets[day] == pair.kept.targets[day]


def test_the_pair_records_what_should_differ_and_what_should_not() -> None:
    pair = _made()
    unrelated, gap = pair.failed_problems.items
    # the failed history plants the commitment's gap; the kept one only the unrelated item, the same
    assert pair.kept_problems.items == (unrelated,) and pair.unrelated_item == unrelated.id
    assert gap.family == "gap" and gap.template == f"{pair.template}-failed"
    assert set(pair.slot_ids) <= set(gap.origin_ids) and set(gap.disambiguating_ids) <= set(pair.failed_ids)
    failed_ids, kept_ids = {e.id for e in pair.failed.evidence}, {e.id for e in pair.kept.evidence}
    assert set(pair.slot_ids) <= failed_ids & kept_ids
    assert set(pair.failed_ids) <= failed_ids - kept_ids and set(pair.kept_ids) <= kept_ids - failed_ids
    # the unrelated item lies wholly before the fork and surfaces on the shared last day
    before = {e.id for e in pair.failed.evidence if e.day < pair.fork_days[0]}
    assert set(unrelated.origin_ids) | set(unrelated.disambiguating_ids) <= before
    assert unrelated.surface_day == gap.surface_day == pair.failed.days[-1]
    assert dict(PAIR_DECISIONS) == {"failed": "speak", "kept": "wait"}


def test_a_pair_whose_histories_differ_elsewhere_is_refused() -> None:
    pair = _made()
    other = _made(pairs.pair_specs()[1])
    with pytest.raises(ValueError):
        CounterfactualPair(**{**{f: getattr(pair, f) for f in pair.__dataclass_fields__}, "kept": other.kept})


def test_both_histories_pass_the_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the gates fail the first attempt's kept history
    real = gates.check
    gated: list[tuple[str, int]] = []

    def check(life: Life, problems: ProblemFile, **kwargs: Any) -> tuple[GateVerdict, ...]:
        verdicts = real(life, problems, **kwargs)
        gated.append((life.id, len(problems.items)))
        if len(gated) == 2:
            return (*verdicts, GateVerdict("script_audit", None, False, "1 issue(s) found"))
        return verdicts

    monkeypatch.setattr(gates, "check", check)

    # Act
    made = pairs.generate_pair(SPEC, backends.fake_backend(FIREWORKS))

    # Assert: each attempt gates both histories; the failing one is written again
    assert made.attempts == 2 and len(gated) == 4
    assert [count for _, count in gated] == [2, 1, 2, 1]
    assert gated[0][0] == gated[1][0] != gated[2][0] == made.pair.id


def test_a_pair_stays_inside_its_hold() -> None:
    # Arrange: count every call one pair makes
    calls: Counter[str] = Counter()
    fake = backends.fake_backend(FIREWORKS)

    def counted(name: str, chat: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            calls[name] += 1
            return chat(*args, **kwargs)
        return call

    backend = backends.Backend(**{**{f: getattr(fake, f) for f in fake.__dataclass_fields__},
                                  "life_chat": counted("writes", fake.life_chat),
                                  "reader": counted("reads", fake.reader), "auditor": counted("muse", fake.auditor)})

    # Act
    pairs.generate_pair(SPEC, backend)

    # Assert: every day written and extracted once, each history's outcomes extracted; the reads and
    # audits within the hold
    assert calls["writes"] == 2 * len(pair_days()) + 2 * len(pairs.OUTCOME_DAYS) == pairs.PAIR_CALLS.writes
    assert 0 < calls["reads"] <= pairs.PAIR_CALLS.reads and 0 < calls["muse"] <= pairs.PAIR_CALLS.muse


def test_pair_specs_are_fixed() -> None:
    specs = pairs.pair_specs()
    assert len(specs) == pairs.PAIR_COUNT == 20 and len({s.seed for s in specs}) == 20
    for spec in specs:
        field = pairs.pair_template(spec.template).field
        assert spec.unrelated in {t.id for t in core_families.templates_for("core-test", field)}
    assert Counter(s.template for s in specs) == {t.id: 4 if n < 2 else 3 for n, t in enumerate(core_families.PAIR_TEMPLATES)}
    digest = hashlib.sha256(json.dumps([[s.template, s.unrelated, s.seed] for s in specs]).encode()).hexdigest()
    assert digest[:16] == "0b81dcba10a35c12"


def test_pair_templates_state_no_conclusion() -> None:
    for pair in core_families.PAIR_TEMPLATES:
        assert pair.field in core_families.CORE_FIELDS["core-test"]
        for fact in (pair.commitment, *pair.failed, *pair.kept):
            assert not any(w in fact.lower() for w in ("problem", "conflict", "mismatch", "mistake", "solution"))
        assert core_families.template(pair.gap.id) == pair.gap  # the gates find its facts


def test_generate_pairs_and_export_them(tmp_path: Path) -> None:
    # Arrange
    data = tmp_path / "pilot"
    lines: list[str] = []

    # Act: three pairs, then a rerun, then the export
    assert cli.main(["--fake", "--data", str(data), "generate-pairs", "--n", "3"], out=lines.append) == 0
    assert cli.main(["--fake", "--data", str(data), "generate-pairs", "--n", "3"], out=lines.append) == 0
    assert cli.main(["--data", str(data), "labels", "--pairs", "--out", str(tmp_path / "out")], out=lines.append) == 0

    # Assert: three pairs saved once, each exported with both streams and its expectations
    saved = DataDir(data).pairs()
    assert len(saved) == 3 and len([line for line in lines if line.startswith("generated pair-")]) == 3
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pairs"] == [pair.id for pair in saved]
    for pair in saved:
        failed = labels.read_events(tmp_path / "out" / "events" / f"{pair.id}.failed.jsonl")
        kept = labels.read_events(tmp_path / "out" / "events" / f"{pair.id}.kept.jsonl")
        assert failed[-len(pair.final_ids):] == kept[-len(pair.final_ids):]
        record = json.loads((tmp_path / "out" / "pairs" / f"{pair.id}.json").read_text(encoding="utf-8"))
        assert record["decisions"] == {"failed": "speak", "kept": "wait"}
        assert record["slot_ids"] == list(pair.slot_ids) and record["unrelated_item"]["item_id"] == pair.unrelated_item
        text = json.dumps(record)
        for item in pair.failed_problems.items:
            assert item.resolution not in text and item.topic not in text


def test_the_pairs_plan_prints_worst_case_and_expected(tmp_path: Path) -> None:
    lines: list[str] = []
    live = ["--live", "--prices", "1.25,4.25,1.5,2.61", "--judge-prices", "1.25,4.25,0.75,3.75"]
    assert cli.main([*live, "--data", str(tmp_path), "generate-pairs"], out=lines.append) == 0
    text = "\n".join(lines)
    expected = 20 * plan.expected_usd(pairs.PAIR_EXPECTED)
    assert "generate 20 pairs" in text and f"Expected: about ${expected:,.2f} for 20 pairs" in text
    assert "Worst case: $" in text
    assert list(tmp_path.iterdir()) == []
