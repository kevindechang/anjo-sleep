# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""APD's generator, templates and gates (Task 17, 17-gen): the families, the planted items'
placement, the gates on pre-surface text, and the regenerate loop, all against fake models."""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.mind.apd import families, gates, generate
from anjo.sleep.mind.apd.types import (
    FAMILIES,
    HALF_GAP_DAYS,
    LIFE_DAYS,
    MAX_PREVALENCE,
    Life,
    ProblemFile,
)
from anjo.sleep.tests.apd_fakes import SURFACE_MARK, FakeAuditor, FakeReader, FakeSpark, reply

ALL = tuple(sorted(FAMILIES))
FILM = "independent film"


def _life(seed: int = 1, prevalence: int = 5, chat: FakeSpark | None = None, split: Any = "development",
          field: str = FILM, fams: Any = ALL) -> tuple[Life, ProblemFile]:
    return generate.life(field, fams, prevalence, seed, split=split, chat=chat or FakeSpark())


def _day(life: Life, evidence_id: str) -> int:
    day = next(item.day for item in life.evidence if item.id == evidence_id)
    return (day - life.days[0]).days


# ── The splits ───────────────────────────────────────────────────────────────


def test_splits_have_disjoint_templates_and_fields():
    dev, held = families.TEMPLATES["development"], families.TEMPLATES["held_out"]
    assert {t.id for t in dev}.isdisjoint({t.id for t in held})
    assert set(families.fields("development")).isdisjoint(families.fields("held_out"))
    sentences = lambda ts: {s for t in ts for s in (*t.origins, t.disambiguating, t.surface, t.resolution)}  # noqa: E731
    assert sentences(dev).isdisjoint(sentences(held))
    assert {t.family for t in dev} == {t.family for t in held} == set(FAMILIES)  # the families are shared
    for split in ("development", "held_out"):
        for name in families.fields(split):  # a field can fill a life of MAX_PREVALENCE items
            assert len(families.templates_for(split, name)) == MAX_PREVALENCE


def test_a_life_draws_only_its_splits_templates():
    _, problems = _life(split="held_out", field="urban gardening", prevalence=6)
    held = {t.id for t in families.TEMPLATES["held_out"]}
    assert {item.template for item in problems.items} <= held
    with pytest.raises(ValueError):  # a development field is not a held-out field
        _life(split="held_out", field=FILM)


def test_neutral_topics_are_not_templates_or_resolutions():
    for template in (*families.TEMPLATES["development"], *families.TEMPLATES["held_out"]):
        assert template.topic not in (template.id, template.resolution)
        assert template.id not in template.topic


# ── A life ───────────────────────────────────────────────────────────────────


def test_the_same_seed_and_replies_give_the_same_life():
    first, second = _life(seed=3), _life(seed=3)
    assert first == second
    assert _life(seed=4)[1] != first[1]


def test_a_life_has_its_days_sessions_and_sealed_items():
    life, problems = _life(prevalence=6)
    assert len(life.days) == LIFE_DAYS == len(life.sessions)
    assert life.split == "development" and life.field == FILM and problems.life_id == life.id
    assert len(problems.items) == 6
    assert len({item.family for item in problems.items}) >= 5  # round-robin through the families
    assert all(item.surface_day in life.days for item in problems.items)


def test_prevalence_zero_plants_nothing():
    chat = FakeSpark()
    life, problems = _life(prevalence=0, chat=chat)
    assert problems.items == ()
    assert all(not body["plant"] and not body["ask"] for body in chat.writes)
    assert len(life.sessions) == LIFE_DAYS


def test_prevalence_and_families_are_checked():
    with pytest.raises(ValueError):
        _life(prevalence=MAX_PREVALENCE + 1)
    with pytest.raises(ValueError):
        _life(prevalence=-1)
    with pytest.raises(ValueError):
        _life(prevalence=4, fams=("gap",))  # a field holds one gap template
    _, problems = _life(prevalence=2, fams=("contradiction",))
    assert {item.family for item in problems.items} == {"contradiction"}


@pytest.mark.parametrize("seed", range(12))
def test_a_contradictions_halves_are_far_apart_and_never_together(seed):
    life, problems = _life(seed=seed, prevalence=6)
    for item in problems.items:
        if item.family != "contradiction":
            continue
        days = sorted(_day(life, evidence_id) for evidence_id in item.origin_ids)
        assert days[-1] - days[0] >= HALF_GAP_DAYS
        sources = [next(e.source for e in life.evidence if e.id == i) for i in item.origin_ids]
        assert len(set(sources)) == len(sources)  # never in one message


@pytest.mark.parametrize("seed", range(12))
def test_every_disambiguating_fact_sits_away_from_its_origin(seed):
    life, problems = _life(seed=seed, prevalence=6)
    for item in problems.items:
        origin = {_day(life, i) for i in item.origin_ids}
        away = {_day(life, i) for i in item.disambiguating_ids}
        assert all(abs(a - o) >= generate.AWAY_DAYS for a in away for o in origin)
        surface = (item.surface_day - life.days[0]).days
        assert max(origin | away) < surface  # everything the gates read comes before the surface day


def test_apd_evidence_records_producing_model():
    life, _ = _life(prevalence=3)
    assert life.evidence
    assert {item.model for item in life.evidence} == {SPARK_OPEN}
    assert all(SPARK_OPEN in (item.ancestry or ()) for item in life.evidence)
    assert {session.model for session in life.sessions} == {SPARK_OPEN}
    assert all(item.person_id == life.id and not item.held_out for item in life.evidence)


def test_targets_are_keyed_by_day_and_appear_from_the_day_first_mentioned():
    life, _ = _life(prevalence=6)
    first = next(day for day in life.days if life.targets[day])
    assert all(not life.targets[day] for day in life.days if day < first)
    target = life.targets[first][0]
    assert target.last_mentioned == first and target.status == "open" and target.check_after is not None
    later = life.targets[life.days[-1]]
    assert {t.id for t in life.targets[first]} <= {t.id for t in later}
    assert all(t.last_mentioned <= day for day in life.days for t in life.targets[day])


def test_a_writer_that_leaves_a_beat_unplaced_fails_the_life():
    with pytest.raises(generate.GenerationFailed):
        _life(prevalence=2, chat=FakeSpark(drop="i1.o1"))


def test_a_reply_from_another_model_is_refused():
    with pytest.raises(generate.ModelMismatch):
        _life(chat=FakeSpark(model="other-model"))


# ── The gates ────────────────────────────────────────────────────────────────


def test_leak_gates_read_only_pre_surface_text():
    life, problems = _life(prevalence=4)
    reader, auditor = FakeReader(), FakeAuditor()
    verdicts = gates.check(life, problems, reader=reader, auditor=auditor)
    assert all(v.passed for v in verdicts) and len(verdicts) == 2 * len(problems.items) + 1
    assert verdicts[-1].gate == "script_audit" and verdicts[-1].item_id is None
    seen = json.dumps(reader.seen)
    assert SURFACE_MARK not in seen  # the surface day's text is never read
    for item in problems.items:
        assert item.resolution not in seen and item.template not in seen
    for item in problems.items:
        shown = [d for body in reader.seen if body["topic"] == item.topic for d in body["days"]]
        assert shown and all(d["date"] < item.surface_day.isoformat() for d in shown)
        assert gates.pre_surface(life, item) == tuple(s for s in life.sessions if s.day < item.surface_day)


def test_leak_windows_are_single_days_and_adjacent_pairs():
    life, problems = _life(prevalence=1)
    item = problems.items[0]
    reader = FakeReader()
    gates.check(life, problems, reader=reader, auditor=FakeAuditor())
    leaks = [body["days"] for body in reader.seen if len(body["days"]) <= 2]
    before = (item.surface_day - life.days[0]).days
    singles = [w for w in leaks if len(w) == 1]
    pairs = [w for w in leaks if len(w) == 2]
    assert len(singles) == before and len(pairs) == before - 1
    for first, second in pairs:
        assert date.fromisoformat(second["date"]) - date.fromisoformat(first["date"]) == timedelta(days=1)


def test_a_leaking_day_fails_the_item():
    life, problems = _life(prevalence=2)
    item = problems.items[0]
    origin_text = next(e.text for e in life.evidence if e.id == item.origin_ids[0])
    fact = origin_text.removeprefix("They said: By the way, ")
    verdicts = gates.check(life, problems, reader=FakeReader(leaks=(fact[:25],)), auditor=FakeAuditor())
    leak = next(v for v in verdicts if v.gate == "leak" and v.item_id == item.id)
    assert not leak.passed and leak.reason


def test_discoverability_needs_the_origin_days_to_be_found():
    life, problems = _life(prevalence=2)
    verdicts = gates.check(life, problems, reader=FakeReader(finds=False), auditor=FakeAuditor())
    assert [v.passed for v in verdicts if v.gate == "discoverability"] == [False, False]
    assert all(v.passed for v in verdicts if v.gate == "leak")


def test_discoverability_reader_gets_the_prefix_and_topic_only():
    life, problems = _life(prevalence=2)
    reader = FakeReader()
    gates.check(life, problems, reader=reader, auditor=FakeAuditor())
    item = problems.items[0]
    finds = [b for b in reader.seen if b["topic"] == item.topic and len(b["days"]) > 2]
    assert finds and set(finds[0]) == {"topic", "days"}
    assert len(finds[0]["days"]) == (item.surface_day - life.days[0]).days


def test_a_failing_script_audit_fails_the_life():
    life, problems = _life(prevalence=2)
    verdicts = gates.check(life, problems, reader=FakeReader(), auditor=FakeAuditor(ok=False))
    assert verdicts[-1].gate == "script_audit" and not verdicts[-1].passed
    auditor = FakeAuditor()
    gates.check(life, problems, reader=FakeReader(), auditor=auditor)
    assert auditor.seen[0]["problem_file"][0]["resolution"] == problems.items[0].resolution


def test_an_unreadable_reader_reply_is_an_error_not_a_pass():
    life, problems = _life(prevalence=1)

    def broken(model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float,
               *, temperature: float | None = None) -> Any:
        return reply("not json")

    with pytest.raises(gates.GateError):
        gates.check(life, problems, reader=broken, auditor=FakeAuditor())


# ── Regeneration ─────────────────────────────────────────────────────────────


def _generate(reader: Any, *, max_attempts: int = 3, chat: Any = None) -> generate.Generated:
    return generate.generate(FILM, ALL, 1, 7, split="development", chat=chat or FakeSpark(),
                             reader=reader, auditor=FakeAuditor(), max_attempts=max_attempts)


def test_a_failing_gate_regenerates():
    chat = FakeSpark()
    made = _generate(FakeReader(misses=1), chat=chat)
    assert made.attempts == 2 and all(v.passed for v in made.verdicts)
    assert len(chat.writes) == 2 * LIFE_DAYS  # the whole life was written again
    assert made.life.seed == generate.attempt_seed(7, 1) != 7
    assert made.life.id != _life(seed=7, prevalence=1)[0].id


def test_a_passing_life_is_kept_as_first_written():
    made = _generate(FakeReader())
    assert made.attempts == 1 and made.life == _life(seed=7, prevalence=1)[0]


def test_regeneration_is_bounded():
    chat = FakeSpark()
    with pytest.raises(generate.GenerationError):
        _generate(FakeReader(misses=99), max_attempts=2, chat=chat)
    assert len(chat.writes) == 2 * LIFE_DAYS


def test_a_failed_attempt_counts_when_the_writer_misplaces_a_beat():
    with pytest.raises(generate.GenerationError):
        _generate(FakeReader(), max_attempts=2, chat=FakeSpark(drop="i1.s"))


def test_lives_are_never_filtered_on_arm_results():
    import inspect
    assert set(inspect.signature(generate.generate).parameters) == {
        "field", "families", "prevalence", "seed", "split", "chat", "reader", "auditor", "max_attempts"}


# ── The splits' draw ─────────────────────────────────────────────────────────


def test_development_lives_are_eight_and_fixed():
    specs = generate.development_specs()
    assert len(specs) == 8 and specs == generate.development_specs()
    assert {spec.split for spec in specs} == {"development"}
    assert {spec.field for spec in specs} <= set(families.fields("development"))
    assert len({spec.seed for spec in specs}) == 8


@pytest.mark.parametrize("count", [16, 12])
def test_the_held_out_draw_is_a_pure_function_of_its_seed(count):
    draw = generate.draw_held_out("pre-committed-seed", count=count)
    assert draw == generate.draw_held_out("pre-committed-seed", count=count)
    assert draw != generate.draw_held_out("another-seed", count=count)
    assert len(draw) == count and {spec.split for spec in draw} == {"held_out"}
    assert sum(spec.prevalence == 0 for spec in draw) == 2
    assert all(0 <= spec.prevalence <= MAX_PREVALENCE for spec in draw)
    assert {spec.field for spec in draw} <= set(families.fields("held_out"))
    assert len({spec.seed for spec in draw}) == count


def test_the_held_out_draw_refuses_an_unregistered_size():
    with pytest.raises(ValueError):
        generate.draw_held_out("s", count=14)
