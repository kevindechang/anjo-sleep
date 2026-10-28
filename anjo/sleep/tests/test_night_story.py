# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The night's story (``night/story.py``): one versioned record of the night as one loop, told from
its one log. Every stage is a record, ``not_run`` with why, or ``failed``: never missing, never a
zero for something that didn't run. Fakes only."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import Ledger
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night import story as st
from anjo.sleep.night.nightlog import STORY, NightLog, safe_error
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.night.sleep_night import DEFER_WHY, NAP_WHY
from anjo.sleep.night.sleep_pressure import SleepPolicy
from anjo.sleep.tests.night_fakes import NIGHT, World, step, steps, summary

UID = "p1"
STAGES = ("dream", "rehearse", "learn", "check", "wake")

FULL = st.Story(
    night=NIGHT, status="promoted",
    sleep=st.Sleep(kind="deep", pressure=0.62, threshold=0.5, reason="pressure"),
    brief=st.Brief(folded=3),
    dream=st.Dream(questions=2, ignitions=(st.Ignition("risk", 0.81, ("e-0a1b", "e-2c3d")),),
                   threads=(st.Thread(3, 0.7, ("e-0a1b",)),), boundaries=(st.Boundary("s-9f", 3, 4),),
                   dreams=st.Dreams(candidates=12, kept=9, keep_rate=0.75)),
    rehearse=st.Rehearse(scenes=16, seeded=st.Seeded(threads=(3,), picked=4), flagged=6, corrected=2),
    learn=st.Learn(rows=8, surprise=st.Surprise(applied=True, rows=8), channel_loss=0.42, stamp="9f86d081884c",
                   seeding=st.Seeding(rows=10, kl_before=0.05, kl_after=0.031, minutes=1.2),
                   imitation=st.Imitation(rows=3, reward=0.6, minutes=0.8), fold=8,
                   protect=st.Protect(basis="anchors", rank=16, minutes=0.4), core_adaptation=0.2),
    check=st.Check(gates=(st.Gate("general", True, False, 0.0, 2.0), st.Gate("safety", False, True, 0.0, 0.0)),
                   promoted=True),
    wake=st.Wake(morning=st.Morning(replies=28, no_verdict=1), self_model=st.SelfModel(predicted=0.3, actual=0.25)),
)


def _log(*records: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"uses": "new", "system": None, "noop": False, **r} for r in records]


SLEEP = {"step": "sleep", "status": "ok", "kind": "deep", "reason": "pressure", "pressure": 0.7, "threshold": 0.5}
NAP = {**SLEEP, "kind": "nap", "reason": "below_threshold", "pressure": 0.2}
PRACTICE = {"step": "practice", "status": "ok", "scenes": 16, "pairs": 6, "teacher_targets": 2, "reused": False,
            "spark": {"critic": {"calls": 16, "in": 1, "out": 1, "cut": 0}}}
GATES = {"step": "gates", "status": "ok", "gates": {
    "general": {"gate": "general", "passed": True, "value": 0.0, "threshold": 2.0, "reason": None},
    "safety": {"gate": "safety", "passed": False, "value": 0.0, "threshold": 0.0, "reason": "off", "skipped": True}}}
PROMOTED = _log({"step": "close_day", "status": "ok"}, {"step": "extract", "status": "ok"}, SLEEP,
                {"step": "notice", "status": "ok", "new": 4, "questions_opened": 2},
                {"step": "probes", "status": "ok"}, {"step": "choose", "status": "ok"}, PRACTICE,
                {"step": "dataset", "status": "ok", "examples": 8}, {"step": "train", "status": "ok", "adapter": "a-n1"},
                GATES, {"step": "promote", "status": "ok", "promoted": "a-n1"}, {"step": "note", "status": "ok"},
                {"step": "finish", "status": "ok"})
GPU = ("practice", "dataset", "train", "gates", "promote")

# What the core's night and the sleep objective log (mind-frontier, sleep-objective): ids, counts, numbers
SEEDED = {**PRACTICE, "seeded": {"picked": 4, "threads": [3]}}  # practice seeded on 4 of the dream's picks
DATASET: dict[str, Any] = {"step": "dataset", "status": "ok", "examples": 8, "mind_weighted": 8, "terms": {
    "seeds": 10, "dream_candidates": 12, "dreams": 9, "dreams_dropped": {"probe": 3}, "dreams_keep_rate": 0.75,
    "dreams_by_seed": {"boundary": 5, "picked": 4, "random": 0}, "dream_seconds": 30.0, "anchors": 32}}
TERMS: dict[str, Any] = {"rows": {"sft": 8, "dpo": 6, "seed": 10, "imitate": 3, "dream": 4},
         "minutes": {"sft": 0.5, "dpo": 0.3, "seed": 1.2, "imitate": 0.8, "dream": 0.4, "anchor": 0.1},
         "seed_kl_before": 0.05, "seed_kl_after": 0.031, "imitation_reward": 0.6, "fold_rank": 8, "folded_to": 8,
         "protect": {"basis": "anchors", "rank": 16, "rank_min": 16, "modules": 4, "sequences": 32,
                     "sequences_asked": 32, "energy": 0.91, "minutes": 0.4, "slot_left_on_basis": 0.0}}


# MIND's core's dream step, as mind-frontier logs it (``mind.core.night``'s receipt): ids and numbers only
CORE_DREAM = {"step": "dream", "status": "ok", "state": "s-1", "events": 4, "weights": {"e-0a1b": 1.3},
              "ignitions": [{"slot": 3, "other": None, "kind": "risk", "score": 0.81, "cites": ["e-0a1b", "e-2c3d"]}],
              "decision": {"speak": False, "slot": 3, "p_speak": 0.2},
              "picks": {"threads": [{"slot": 3, "score": 0.7, "share": 1.0, "cites": ["e-0a1b"]}], "evidence_ids": ["e-0a1b"]},
              "dreams": [{"state": "s-2", "threads": [3], "rests_on": ["e-0a1b"]}],
              "boundaries": [{"state": "s-9f", "threads": [3], "rests_on": ["e-0a1b"], "steps": 4, "speak": True, "slot": 3}],
              "pressure": 0.4}
PROJECTOR_SHA = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
CHANNEL_RECEIPT = {"projector": "/data/sleep/training/adapters/p-0a1b2c3d4e5f-n1.projector.npz",
                   "channel": {"k": 4, "arm": "state", "projector": PROJECTOR_SHA, "loss_first": 2.1, "loss_last": 0.42}}


def _with(log: list[dict[str, Any]], after: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    """``log`` with ``record`` logged right after the ``after`` step."""
    at = next(n for n, r in enumerate(log) if r["step"] == after) + 1
    return [*log[:at], *_log(record), *log[at:]]


def _swapped(log: list[dict[str, Any]], record: dict[str, Any]) -> list[dict[str, Any]]:
    """``log`` with its record of ``record``'s step replaced."""
    return [record if r["step"] == record["step"] else r for r in log]


def _napped(mind: bool) -> list[dict[str, Any]]:
    notice = {"step": "notice", "status": "skipped", "why": DEFER_WHY} if mind else {"step": "notice", "status": "ok", "noop": True}
    return _log({"step": "extract", "status": "ok"}, NAP, notice, {"step": "probes", "status": "skipped", "why": NAP_WHY},
                *({"step": name, "status": "skipped", "why": NAP_WHY} for name in GPU), {"step": "finish", "status": "ok"})


TRAINED = {"step": "train", "status": "ok", "adapter": "a-n1", "surprise_weights": "applied",
           "receipt": {**CHANNEL_RECEIPT, "terms": TERMS}}
EVERYTHING = [*_with(_swapped(_swapped(_swapped(PROMOTED, SEEDED), DATASET), TRAINED), "notice", CORE_DREAM),
              *_log({"step": "morning", "status": "ok", "replies": 28, "no_verdict": 1})]


def _walk(value: Any, where: str = "story") -> None:
    """No field anywhere is missing, None, or an empty marker."""
    assert value is not None, where
    if isinstance(value, dict):
        if set(value) == {"not_run"}:
            assert isinstance(value["not_run"], str) and value["not_run"], where
        for key, item in value.items():
            _walk(item, f"{where}.{key}")
    elif isinstance(value, list):
        for n, item in enumerate(value):
            _walk(item, f"{where}[{n}]")


# ── the schema ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("told", [
    FULL, st.tell(NIGHT, "napped", _napped(mind=True)), st.tell(NIGHT, "promoted", PROMOTED),
    replace(FULL, rehearse=st.Reused(NIGHT, 16)), st.tell(NIGHT, "napped", _with(_napped(mind=True), "notice", CORE_DREAM)),
    replace(FULL, brief=st.Kept("2026-10-28", "BriefError: x")), replace(FULL, brief=st.Kept(st.EMPTY, "BriefError: x")),
    st.tell(NIGHT, "promoted", EVERYTHING),
], ids=["every record filled", "a nap", "a promoted night", "practice reused", "a nap the core dreamt in",
        "yesterday's brief kept", "the empty brief kept", "everything the night logs"])
def test_the_story_round_trips_through_its_json(told: st.Story) -> None:
    record = told.record()
    # 2: a reused practice is its own record; 3: the brief; 4: the core's and the sleep objective's records
    assert record["version"] == st.STORY_VERSION == 4
    assert st.story_from(json.loads(json.dumps(record))) == told


@pytest.mark.parametrize(("change", "says"), [
    (lambda r: {**r, "version": 1}, "version"),  # an older schema's story: the panel tells it again from the log
    (lambda r: {**r, "rehearse": {"night": "2026-10-30"}}, "rehearse"),  # neither practice nor a reused one
    (lambda r: {k: v for k, v in r.items() if k != "wake"}, "wake"),
    (lambda r: {**r, "extra": 1}, "extra"),
    (lambda r: {**r, "rehearse": {**r["rehearse"], "scenes": "16"}}, "rehearse.scenes"),
    (lambda r: {**r, "rehearse": {**r["rehearse"], "scenes": True}}, "rehearse.scenes"),
    (lambda r: {**r, "rehearse": {**r["rehearse"], "flagged": {"not_run": ""}}}, "rehearse.flagged"),
    (lambda r: {**r, "check": {**r["check"], "gates": {"not_run": "never"}}}, "check.gates"),  # gates always ran in a check
    (lambda r: {**r, "wake": {"failed": {"step": "morning", "error": "x"}}}, "wake"),  # a morning fails inside wake
    (lambda r: {**r, "learn": {**r["learn"], "channel_loss": float("nan")}}, "learn.channel_loss"),
    (lambda r: {**r, "sleep": {**r["sleep"], "kind": "doze"}}, "sleep.kind"),
    (lambda r: {**r, "version": 2}, "version"),  # before the brief was its own record
    (lambda r: {**r, "brief": {"kept_from": "2026-10-28"}}, "brief"),  # neither a fold nor a kept brief
    (lambda r: {**r, "brief": {"folded": "3"}}, "brief.folded"),
    (lambda r: {**r, "version": 3}, "version"),  # before the core's and the sleep objective's records
    (lambda r: {**r, "dream": {**r["dream"], "dreams": {"candidates": 12, "kept": 9}}}, "dream.dreams"),
    (lambda r: {**r, "rehearse": {**r["rehearse"], "seeded": [{"slot": 3, "scenes": 10}]}}, "rehearse.seeded"),
    (lambda r: {**r, "learn": {**r["learn"], "seeding_kl": 0.031}}, "seeding_kl"),  # version 3's field
    (lambda r: {**r, "learn": {**r["learn"], "protect": {"basis": "anchors", "rank": 16}}}, "learn.protect"),
    (lambda r: {**r, "learn": {**r["learn"], "surprise": {"applied": True, "rows": 8, "mean": 1.1}}}, "learn.surprise"),
])
def test_a_story_that_breaks_the_schema_is_refused_by_where(change: Any, says: str) -> None:
    with pytest.raises(st.StoryError, match=says.replace(".", r"\.")):
        st.story_from(change(FULL.record()))


TOLD = {
    "promoted": st.tell(NIGHT, "promoted", PROMOTED),
    "rejected": st.tell(NIGHT, "rejected", [r if r["step"] != "promote" else {**r, "promoted": None} for r in PROMOTED]),
    "nothing_new": st.tell(NIGHT, "nothing_new", _log(
        {"step": "extract", "status": "ok"}, {"step": "notice", "status": "ok", "noop": True},
        *({"step": name, "status": "skipped", "why": "below min_new_examples: no practice, training, gates or promotion"}
          for name in GPU))),
    "napped, MIND on": st.tell(NIGHT, "napped", _napped(mind=True)),
    "napped, MIND off": st.tell(NIGHT, "napped", _napped(mind=False)),
    "failed at extract": st.tell(NIGHT, "failed", _log({"step": "extract", "status": "failed", "error": "ValueError"})),
    "failed at gates": st.tell(NIGHT, "failed", [*PROMOTED[:9], {**GATES, "status": "failed", "error": "ServingError"}]),
}


@pytest.mark.parametrize("name", list(TOLD))
def test_every_stage_is_a_record_or_says_why_not(name: str) -> None:
    record = TOLD[name].record()
    assert list(record) == ["version", "night", "status", "sleep", "brief", *STAGES]
    _walk(record)


def test_a_promoted_night_tells_what_it_already_knows_and_leaves_the_rest_not_run() -> None:
    told = TOLD["promoted"]
    assert told.sleep == st.Sleep("deep", 0.7, 0.5, "pressure")
    assert isinstance(told.dream, st.Dream) and told.dream.questions == 2
    assert told.dream.ignitions == told.dream.threads == told.dream.boundaries == st.NotRun(st.CORE_OFF)
    assert told.dream.dreams == st.NotRun(st.TERMS_OFF)
    assert told.rehearse == st.Rehearse(16, st.NotRun(st.CORE_OFF), flagged=6, corrected=2)
    assert isinstance(told.learn, st.Learn) and told.learn.rows == 8
    assert told.learn.channel_loss == told.learn.stamp == st.NotRun(st.CHANNEL_OFF)
    assert told.learn.seeding == told.learn.imitation == told.learn.fold == told.learn.protect == st.NotRun(st.TERMS_OFF)
    assert told.check == st.Check((st.Gate("general", True, False, 0.0, 2.0), st.Gate("safety", False, True, 0.0, 0.0)), True)
    assert told.wake == st.Wake(st.NotRun(st.SCORED_IN_MORNING), st.NotRun(st.SELF_MODEL_NOT_YET))
    assert TOLD["rejected"].check.promoted is False  # type: ignore[union-attr]


def test_nothing_new_and_a_failure_say_so_never_zero() -> None:
    quiet = TOLD["nothing_new"]
    assert quiet.rehearse == quiet.learn == quiet.check == st.NotRun(st.NOTHING_NEW)
    assert quiet.sleep == st.NotRun(st.NO_PRESSURE) and quiet.dream.questions == st.NotRun(st.MIND_OFF)  # type: ignore[union-attr]
    broke = TOLD["failed at gates"]
    assert broke.check == st.Failed("gates", "ServingError") and isinstance(broke.learn, st.Learn)
    assert broke.wake == st.NotRun("the night stopped at gates")
    early = TOLD["failed at extract"]
    assert all(getattr(early, stage) == st.NotRun("the night stopped at extract") for stage in STAGES)


@pytest.mark.parametrize("name", ["napped, MIND on", "napped, MIND off"])
def test_a_nap_story_stops_after_the_dream(name: str) -> None:
    told = TOLD[name]
    assert isinstance(told.sleep, st.Sleep) and told.sleep.kind == "nap"
    assert all(getattr(told, stage) == st.NotRun(st.LEARNING_TOMORROW) for stage in STAGES[1:])
    assert told.dream == (st.NotRun(DEFER_WHY) if name.endswith("on") else st.Dream(
        st.NotRun(st.MIND_OFF), st.NotRun(st.CORE_OFF), st.NotRun(st.CORE_OFF), st.NotRun(st.CORE_OFF),
        st.NotRun(st.LEARNING_TOMORROW)))


def test_the_morning_fills_wake_ok_failed_or_uncounted() -> None:
    ok = st.tell(NIGHT, "promoted", [*PROMOTED, {"step": "morning", "status": "ok", "replies": 28, "no_verdict": 1}])
    failed = st.tell(NIGHT, "promoted", [*PROMOTED, {"step": "morning", "status": "failed",
                                                     "why": f"{st.MORNING_FAILED}TooManyNoVerdicts: 3 of 3"}])
    bare = st.tell(NIGHT, "rejected", [*PROMOTED, {"step": "morning", "status": "ok"}])
    assert ok.wake.morning == st.Morning(28, 1)  # type: ignore[union-attr]
    assert failed.wake.morning == st.Failed("morning", "TooManyNoVerdicts: 3 of 3")  # type: ignore[union-attr]
    assert bare.wake.morning == st.Morning(st.NotRun(st.UNCOUNTED), 0)  # type: ignore[union-attr]


CUT_TWICE = "BriefError: the brief ran to its 15000-token cap; retried: the brief ran to its 15000-token cap"
KEPT_BRIEF = {"step": "brief", "status": "failed", "folded": 0, "pending": 3, st.KEPT: "2026-10-28",
              "why": "brief kept from 2026-10-28", "error": CUT_TWICE}


def test_a_kept_brief_is_told_and_never_reads_as_where_the_night_stopped() -> None:
    told = st.tell(NIGHT, "promoted", _with(PROMOTED, "choose", KEPT_BRIEF))
    assert told.brief == st.Kept("2026-10-28", CUT_TWICE)
    assert told == replace(TOLD["promoted"], brief=told.brief)  # every stage told as on a night that folded
    empty = st.tell(NIGHT, "promoted", _with(PROMOTED, "choose", {**KEPT_BRIEF, st.KEPT: st.EMPTY}))
    assert empty.brief == st.Kept(st.EMPTY, CUT_TWICE)


def test_the_brief_says_what_it_folded_or_why_it_didnt() -> None:
    def brief(record: dict[str, Any], status: str = "promoted") -> Any:
        return st.tell(NIGHT, status, _with(PROMOTED, "choose", {"step": "brief", **record})).brief

    assert brief({"status": "ok", "folded": 3}) == st.Brief(3)
    assert brief({"status": "ok", "folded": 0, "why": "no evidence since the last fold"}) == st.NotRun(
        "no evidence since the last fold")
    assert TOLD["promoted"].brief == st.NotRun(st.NOT_LOGGED) and TOLD["failed at extract"].brief == st.NotRun(
        "the night stopped at extract")
    stopped = st.tell(NIGHT, "failed", _log({"step": "choose", "status": "ok"},
                                            {"step": "brief", "status": "failed", "error": "BudgetStop"}))
    assert stopped.brief == st.Failed("brief", "BudgetStop")  # a brief that never ran stops the night, as before
    assert stopped.rehearse == st.NotRun("the night stopped at brief")


def test_a_practice_without_a_critic_says_so() -> None:
    told = st.tell(NIGHT, "promoted", _swapped(PROMOTED, {**PRACTICE, "spark": {}}))
    why = st.NotRun(st.NO_CRITIC)
    assert told.rehearse == st.Rehearse(16, st.NotRun(st.CORE_OFF), why, why)


def test_a_reused_practice_says_whose_it_was_never_that_it_ran() -> None:
    earlier = NIGHT - timedelta(days=1)
    reused = {"step": "practice", "status": "ok", "scenes": 12, "reused": True, "homework": None}
    told = st.tell(NIGHT, "promoted", _swapped(PROMOTED, {**reused, "reused_from": earlier.isoformat()}))
    assert told.rehearse == st.Reused(earlier, 12)
    older = st.tell(NIGHT, "promoted", _swapped(PROMOTED, reused))  # a log from before it said whose: tonight's own
    assert older.rehearse == st.Reused(NIGHT, 12)
    with pytest.raises(st.StoryError, match="reused_from"):
        st.tell(NIGHT, "promoted", _swapped(PROMOTED, {**reused, "reused_from": "yesterday"}))


# ── MIND's core dream (mind-frontier) and the thought channel's training ─────


def test_the_cores_dream_fills_the_dream_from_its_step() -> None:
    told = st.tell(NIGHT, "promoted", _with(PROMOTED, "notice", CORE_DREAM))
    assert isinstance(FULL.dream, st.Dream)  # its ignitions, the threads its curriculum picked, its boundaries
    assert told.dream == replace(FULL.dream, dreams=st.NotRun(st.TERMS_OFF))
    unpicked = st.tell(NIGHT, "promoted", _with(PROMOTED, "notice", {**CORE_DREAM, "picks": None}))
    assert unpicked.dream.threads == st.NotRun(st.NO_CURRICULUM)  # type: ignore[union-attr]
    failed = st.tell(NIGHT, "failed", _with(PROMOTED[:4], "notice", {"step": "dream", "status": "failed", "error": "StoreError"}))
    assert failed.dream == st.Failed("dream", "StoreError") and failed.rehearse == st.NotRun("the night stopped at dream")
    skipped = st.tell(NIGHT, "promoted", _with(PROMOTED, "notice", {"step": "dream", "status": "skipped", "why": "off"}))
    assert skipped.dream == TOLD["promoted"].dream  # no dream: today's text
    with pytest.raises(st.StoryError, match="dream"):
        st.tell(NIGHT, "promoted", _with(PROMOTED, "notice", {**CORE_DREAM, "ignitions": [{"kind": "risk"}]}))


def test_a_nap_shows_its_cpu_dream_without_the_gpu_stages() -> None:
    told, dreamt = st.tell(NIGHT, "napped", _with(_napped(mind=True), "notice", CORE_DREAM)), FULL.dream
    assert isinstance(dreamt, st.Dream)  # the core's dream, all CPU; Glimmer's dreams wait for her next deep sleep
    assert told.dream == replace(dreamt, questions=st.NotRun(DEFER_WHY), dreams=st.NotRun(st.LEARNING_TOMORROW))
    assert all(getattr(told, stage) == st.NotRun(st.LEARNING_TOMORROW) for stage in STAGES[1:])


def test_everything_the_night_logs_fills_its_record() -> None:
    told = st.tell(NIGHT, "promoted", EVERYTHING)
    assert told.dream == FULL.dream and told.rehearse == FULL.rehearse  # the dream, Glimmer's dreams, the seeding
    assert isinstance(FULL.learn, st.Learn)
    assert told.learn == replace(FULL.learn, core_adaptation=st.NotRun(st.ADAPTATION_NOT_YET))  # no record says it
    assert told.wake == st.Wake(st.Morning(28, 1), st.NotRun(st.SELF_MODEL_NOT_YET))  # nor this: a plain line


def _terms(**changes: Any) -> list[dict[str, Any]]:
    return _swapped(EVERYTHING, {**TRAINED, "receipt": {**CHANNEL_RECEIPT, "terms": {**TERMS, **changes}}})


@pytest.mark.parametrize(("log", "field", "said"), [
    (_swapped(EVERYTHING, {**SEEDED, "seeded": {"picked": 0, "threads": [3]}}), "rehearse.seeded", st.FELL_BACK),
    (_swapped(EVERYTHING, PRACTICE), "rehearse.seeded", st.SEEDING_UNLOGGED),
    (_swapped(EVERYTHING, {**CORE_DREAM, "picks": None}), "rehearse.seeded", st.NO_CURRICULUM),
    (_swapped(EVERYTHING, {"step": "dream", "status": "ok", "noop": True, "why": "fake latents for a real person"}),
     "dream.ignitions", "fake latents for a real person"),
    (_swapped(EVERYTHING, {"step": "dream", "status": "ok", "noop": True, "why": "fake latents for a real person"}),
     "rehearse.seeded", "fake latents for a real person"),
    (_swapped(EVERYTHING, {**DATASET, "terms": {**DATASET["terms"], "dreams": "not wired"}}), "dream.dreams",
     st.DREAMS_NOT_WIRED),
    (_swapped(EVERYTHING, {**DATASET, "terms": {"seeds": 10, "anchors": 0}}), "dream.dreams", st.DREAMS_OFF),
    (_swapped(EVERYTHING, {**DATASET, "terms": {**DATASET["terms"], "dream_candidates": 0, "dreams": 0,
                                               "dreams_keep_rate": None}}), "dream.dreams", st.NO_DREAM_SEEDS),
    (_swapped(EVERYTHING, {**TRAINED, "surprise_weights": None}), "learn.surprise", st.NO_SURPRISE),
    (_terms(rows={**TERMS["rows"], "seed": 0}, seed_kl_before=None, seed_kl_after=None), "learn.seeding", st.NO_SEEDS),
    (_terms(rows={**TERMS["rows"], "imitate": 0}, imitation_reward=None), "learn.imitation", st.NO_IMITATION),
    (_terms(fold_rank=None, folded_to=None), "learn.fold", st.FOLD_OFF),
    (_terms(protect=None), "learn.protect", st.PROTECT_OFF),
], ids=["no pick could train", "practice didn't say", "curriculum off", "the core refused her",
        "the core refused her: practice", "dreams not wired", "dreams term off", "no dream seeds", "no surprise weight",
        "no seeding rows", "no imitation", "no fold", "no protected slot"])
def test_a_missing_record_says_why_in_a_plain_line(log: list[dict[str, Any]], field: str, said: str) -> None:
    told: Any = st.tell(NIGHT, "promoted", log)
    for name in field.split("."):
        told = getattr(told, name)
    assert told == st.NotRun(said)


def test_a_surprise_the_loss_did_not_apply_says_so() -> None:
    told = st.tell(NIGHT, "promoted", _swapped(EVERYTHING, {**TRAINED, "surprise_weights": "not applied (objective sft)"}))
    assert isinstance(told.learn, st.Learn) and told.learn.surprise == st.Surprise(applied=False, rows=8)


def test_the_words_the_story_reads_are_the_ones_its_steps_write() -> None:
    from anjo.sleep.train import colab_train, terms
    assert st.NOT_WIRED == terms.DREAMS_NOT_WIRED
    assert st.APPLIED == colab_train.surprise_weights(colab_train.SEED_OBJECTIVE, [{"weight": 1.5}])


def test_a_term_that_reported_no_number_says_so_never_zero() -> None:
    told = st.tell(NIGHT, "promoted", _terms(seed_kl_after=None, imitation_reward=None))
    assert isinstance(told.learn, st.Learn)
    assert told.learn.seeding == st.Seeding(10, 0.05, st.NotRun(st.NOT_REPORTED), 1.2)
    assert told.learn.imitation == st.Imitation(3, st.NotRun(st.NOT_REPORTED), 0.8)


def test_the_channels_training_fills_its_loss_and_stamp_never_a_path() -> None:
    train = {"step": "train", "status": "ok", "adapter": "a-n1"}
    trained = st.tell(NIGHT, "promoted", _swapped(PROMOTED, {**train, "receipt": CHANNEL_RECEIPT}))
    assert isinstance(trained.learn, st.Learn)
    assert (trained.learn.channel_loss, trained.learn.stamp) == (0.42, PROJECTOR_SHA[:12])
    bare = {**CHANNEL_RECEIPT, "channel": {"k": 4}}  # a job that reported neither loss nor projector id
    told = st.tell(NIGHT, "promoted", _swapped(PROMOTED, {**train, "receipt": bare}))
    assert isinstance(told.learn, st.Learn) and told.learn.channel_loss == st.NotRun(st.NO_CHANNEL_LOSS)
    assert told.learn.stamp == "p-0a1b2c3d4e5f-n1"  # the projector's file, by its adapter's name: no folder
    off = st.tell(NIGHT, "promoted", _swapped(PROMOTED, {**train, "receipt": {"projector": "", "steps": 9}}))
    assert isinstance(off.learn, st.Learn) and off.learn.channel_loss == off.learn.stamp == st.NotRun(st.CHANNEL_OFF)


def test_a_log_missing_what_a_stage_needs_cannot_be_told() -> None:
    broken = [r if r["step"] != "dataset" else {"step": "dataset", "status": "ok"} for r in PROMOTED]
    with pytest.raises(st.StoryError, match="dataset"):
        st.tell(NIGHT, "promoted", broken)
    with pytest.raises(st.StoryError, match="no night ends 'dozing'"):
        st.tell(NIGHT, "dozing", PROMOTED)


# ── the night writes it ──────────────────────────────────────────────────────


def _story(log_path: Path) -> st.Story:
    return st.story_from(json.loads((log_path.parent / STORY).read_text(encoding="utf-8")))


def test_the_night_writes_its_story_beside_its_log(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT, count=3)
    result = run_night_batch_with(world.deps(), {UID: NIGHT})[UID]
    told = _story(result.log_path)
    assert told == st.tell(NIGHT, "promoted", steps(result.log_path))  # the story is its log's, told once
    assert (told.status, told.sleep) == ("promoted", st.NotRun(st.NO_PRESSURE))
    assert isinstance(told.rehearse, st.Rehearse) and told.rehearse.scenes == step(result.log_path, "practice")["scenes"]
    assert isinstance(told.learn, st.Learn) and told.learn.rows == step(result.log_path, "dataset")["examples"]
    assert isinstance(told.check, st.Check) and told.check.promoted and len(told.check.gates) == 3
    assert told.wake == st.Wake(st.NotRun(st.SCORED_IN_MORNING), st.NotRun(st.SELF_MODEL_NOT_YET))
    assert "story" not in summary(result.log_path) and "story" not in [r["step"] for r in steps(result.log_path)]


def test_a_napped_night_writes_a_story_that_stops_after_the_dream(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT, count=1)
    result = run_night_batch_with(world.deps(sleep_policy=SleepPolicy()), {UID: NIGHT})[UID]
    told = _story(result.log_path)
    assert result.status == "napped" and isinstance(told.sleep, st.Sleep) and told.sleep.kind == "nap"
    assert all(getattr(told, stage) == st.NotRun(st.LEARNING_TOMORROW) for stage in STAGES[1:])


def test_a_rerun_tells_the_practice_it_reused(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT, count=3)
    world.train_fails[UID] = RuntimeError("cut off mid-training")
    run_night_batch_with(world.deps(), {UID: NIGHT})
    world.train_fails.clear()
    rerun = run_night_batch_with(world.deps(), {UID: NIGHT})[UID]
    assert step(rerun.log_path, "practice")["reused_from"] == NIGHT.isoformat()
    assert _story(rerun.log_path).rehearse == st.Reused(NIGHT, step(rerun.log_path, "practice")["scenes"])


@pytest.mark.parametrize("crash", [RuntimeError("the disk said no"), st.StoryError("the dataset step's record has no examples")],
                         ids=["the writer crashed", "the log can't tell it"])
def test_a_story_that_cant_be_written_never_changes_the_night(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash: Exception,
) -> None:
    world = World(tmp_path / "told")
    world.plan(UID, NIGHT, count=3)
    told = run_night_batch_with(world.deps(), {UID: NIGHT})[UID]

    def broken(*args: Any, **kwargs: Any) -> st.Story:
        raise crash

    monkeypatch.setattr("anjo.sleep.night.nightlog.tell", broken)
    again = World(tmp_path / "again")
    again.plan(UID, NIGHT, count=3)
    result = run_night_batch_with(again.deps(), {UID: NIGHT})[UID]
    assert (result.status, result.failed_at) == (told.status, told.failed_at) == ("promoted", None)
    assert summary(result.log_path)["status"] == "promoted" and not (result.log_path.parent / STORY).exists()
    failed = step(result.log_path, st.STORY_STEP)
    assert failed["status"] == "failed" and failed["error"] == safe_error(crash)
    monkeypatch.undo()
    retold = st.tell(NIGHT, "promoted", steps(result.log_path))  # a failed story step never reads as where the night stopped
    assert isinstance(retold.check, st.Check) and retold.wake == st.Wake(st.NotRun(st.SCORED_IN_MORNING),
                                                                         st.NotRun(st.SELF_MODEL_NOT_YET))


def test_steps_logged_after_the_night_never_read_as_where_it_stopped() -> None:
    after = [*PROMOTED, {"step": "morning", "status": "ok", "replies": 28},
             {"step": "dailies", "status": "failed", "error": "RuntimeError"}, {"step": st.STORY_STEP, "status": "failed"}]
    told = st.tell(NIGHT, "promoted", _log(*after))
    assert told.check == TOLD["promoted"].check and told.wake == st.Wake(st.Morning(28, 0), st.NotRun(st.SELF_MODEL_NOT_YET))
    root = Path(st.__file__).parents[1]  # every step the morning's runners log after the night: daily and accelerated
    logged = {name for source in (root / "night" / "daily.py", root / "sim" / "accelerated.py")
              for name in re.findall(r'\.step\("(\w+)"', source.read_text(encoding="utf-8"))}
    assert logged == {"morning", "dailies"} and logged | {st.STORY_STEP} == set(st.AFTER_THE_NIGHT)


def test_one_persons_story_crash_never_stops_anothers_night(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(tmp_path)
    for uid in ("p1", "p2"):
        world.plan(uid, NIGHT, count=3)
    replace_file = NightLog._replace

    def full_for_p1(log: NightLog, name: str, text: str) -> None:
        if name == STORY and world.logs("p1") in log.folder.parents:
            raise OSError("no space left on device")
        replace_file(log, name, text)

    monkeypatch.setattr(NightLog, "_replace", full_for_p1)
    results = run_night_batch_with(world.deps(), {"p1": NIGHT, "p2": NIGHT})
    assert {uid: r.status for uid, r in results.items()} == {"p1": "promoted", "p2": "promoted"}
    assert summary(results["p1"].log_path)["status"] == "promoted"  # the summary was written before the story
    assert step(results["p1"].log_path, st.STORY_STEP) == {**step(results["p1"].log_path, st.STORY_STEP),
                                                           "status": "failed", "error": "OSError"}
    assert not (results["p1"].log_path.parent / STORY).exists()
    assert _story(results["p2"].log_path) == st.tell(NIGHT, "promoted", steps(results["p2"].log_path))
    assert st.STORY_STEP not in [r["step"] for r in steps(results["p2"].log_path)]


def test_a_night_that_failed_writes_where_it_stopped(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT, count=3)
    world.train_fails[UID] = RuntimeError("the trainer fell over")
    result = run_night_batch_with(world.deps(), {UID: NIGHT})[UID]
    told = _story(result.log_path)
    assert result.status == "failed" and told.learn == st.Failed("train", "RuntimeError")
    assert told.check == told.wake == st.NotRun("the night stopped at train")


class _NoRoomForTheBrief(Ledger):
    """A ledger with no room for the brief's hold: its fold stops with ``BudgetStop`` before any call."""

    def reserve(self, job: str, pools: Any, bound: Any) -> Any:
        return None if job.startswith("night:brief:") else super().reserve(job, pools, bound)


def _spark_down_for_briefs(world: World) -> Any:
    def chat(model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None,
             max_tokens: int | None = None, seed: int | None = None) -> ChatResult:
        if messages[-1]["content"].startswith("# brief_v1"):
            raise ModelApiError("Baseten: timed out after 180s")
        return world.spark(model, messages, tools, timeout_s, temperature=temperature, max_tokens=max_tokens, seed=seed)

    return chat


@pytest.mark.parametrize("stop", ["BudgetStop", "ModelApiError"])
def test_only_a_brief_error_keeps_the_brief_any_other_failure_stops_the_night(tmp_path: Path, stop: str) -> None:
    """The brief step keeps the last brief only on ``BriefError`` (a brief written badly twice). A
    budget stop or a provider failure still fails the step and the night, as before: this fails if
    that catch is ever widened."""
    world = World(tmp_path)
    world.plan(UID, NIGHT, count=3)
    deps = (world.deps(ledger=_NoRoomForTheBrief(tmp_path / "budget.jsonl"), is_synthetic=lambda uid: False)
            if stop == "BudgetStop" else world.deps(spark_chat=_spark_down_for_briefs(world)))
    result = run_night_batch_with(deps, {UID: NIGHT})[UID]
    brief = step(result.log_path, "brief")
    assert (result.status, result.failed_at) == ("failed", "brief")
    assert brief["status"] == "failed" and brief["error"].startswith(stop) and st.KEPT not in brief
    told = _story(result.log_path)
    assert told.brief == st.Failed("brief", brief["error"]) and told.rehearse == st.NotRun("the night stopped at brief")
    assert world.briefs(UID).records() == [] and world.glimmer.calls == 0  # nothing folded, nothing practiced

