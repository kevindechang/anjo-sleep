"""Second-order confidence over a synthetic week (Fleming & Lau, 2014).

Fakes only: workspaces are folded from steps the kernel applies, and no clock, model or storage is
read. The tests pin the module's two claims — a confidence derived from her state is bounded and
responds to evidence, and its outcome joins to what the log shows happened — and leave the scoring
itself to ``test_reports_metacog``.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from anjo.sleep.mind import metacog
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.tests.inquiry_fakes import Ws, day, ev, hypothesize, night, notice_op, ref
from anjo.sleep.types import Choice, Considered, Result, Want

SCENE = "t-scene4"
E1 = ev("e-d2-1", "Scene 4 wants an 85 mm medium shot.", n=0, target=SCENE)
E3 = ev("e-d3-1", "They might shoot scene 4 from the courtyard.", n=1, target=SCENE)


def _want(*, question_id: str | None = "q1", target_id: str = SCENE, action: str = "inquire") -> Want:
    return Want(id="w-1", kind="understand", target_id=target_id, action=action,  # type: ignore[arg-type]
                priority_class="P3", strength=0.8, question_id=question_id)


def _choice(want: Want | None) -> Choice:
    considered = () if want is None else (
        Considered(want.id, want.kind, want.target_id, want.action, want.strength, want.weight, "won", None),
    )
    return Choice(opportunity="away", want=want, considered=considered)


def _week() -> tuple[it.WorkspaceEvent, ...]:
    """Night 0 raises a question, hypothesizes, commits a prediction and one study result arrives
    for the pick; night 1 resolves the prediction and concludes the question. Two views."""
    w = Ws()
    w.start(0)
    w.arrive(E1)
    w.ok(notice_op([E1.id], kind="gap"))
    w.ok(hypothesize("q1", 0.6, 0.2))
    ids = w.ids("q1")
    w.ok(it.PredictOp("q1", "The courtyard is available on Thursday", dict(zip(ids, (0.9, 0.05, 0.05), strict=True)), 2))
    prediction_id = w.ws.predictions[0].id
    w.start(1)
    w.found("q1", E3)
    w.ok(it.AssessOp("q1", E3.id, likelihoods=dict(zip(ids, (4.0, 0.25, 0.25), strict=True))))
    w.ok(it.Resolved(prediction_id, "true", ()))
    w.ok(it.ConcludeOp("q1", ids[0], "resolution", "The courtyard carries the look", (E1.id, E3.id)))
    return tuple(w.events)


def _leader(week: tuple[it.WorkspaceEvent, ...]) -> str:
    finding = metacog.kernel.fold(week).finding_of("q1")
    assert finding is not None and finding.hypothesis_id is not None
    return finding.hypothesis_id


def _facts(week: tuple[it.WorkspaceEvent, ...], pick: Choice | None,
           *, pick_night: int = 0) -> tuple[metacog.NightFacts, ...]:
    return tuple(
        metacog.NightFacts(
            night=number, before=before, after=after, events=events,
            choice=pick if number == day(pick_night) else None,
            results=(Result(id="r-1", target_id=SCENE, homework_id=None, verified_sources=0,
                            created_at=night(0), consumed=False, basis="evidence", user_cites=2),)
            if number == day(0) else (),
            at=night(0) if number == day(0) else None,
        )
        for number, before, after, events in metacog.night_slices(week)
    )


# ── Bounds and her state ─────────────────────────────────────────────────────


def test_every_confidence_is_bounded_and_a_coin_without_evidence() -> None:
    assert metacog.decision_confidence(0, None, ready=False, past_rate=None) == pytest.approx(metacog.BASE_RATE)
    assert metacog.decision_confidence(0, 0.0, ready=False, past_rate=None) > metacog.BASE_RATE
    assert metacog.hypothesis_confidence(0.5, 2, 0, 1.0, None) == pytest.approx(0.5)
    assert metacog.prediction_confidence(0.5, 0, 1.0, None) == pytest.approx(0.5)
    for value in (
        metacog.decision_confidence(10_000, 0.0, ready=True, past_rate=1.0),
        metacog.hypothesis_confidence(1.0, 1, 10_000, 0.0, 1.0),
        metacog.prediction_confidence(1.0, 10_000, 0.0, 1.0),
    ):
        assert metacog.CONF_MIN <= value <= metacog.CONF_MAX


def test_more_evidence_and_a_focused_question_raise_the_decision() -> None:
    thin = metacog.decision_confidence(1, 0.9, ready=False, past_rate=None)
    thick = metacog.decision_confidence(10, 0.9, ready=False, past_rate=None)
    focused = metacog.decision_confidence(10, 0.1, ready=False, past_rate=None)
    assert thin < thick < focused
    assert metacog.decision_confidence(10, 0.1, ready=True, past_rate=None) > focused
    assert metacog.decision_confidence(10, 0.1, ready=False, past_rate=0.0) < focused


def test_evidence_counts_applied_items_and_entropy_reads_the_vector() -> None:
    week = _week()
    before, after = metacog.night_views(week)[0][1], metacog.night_views(week)[0][2]
    assert metacog.evidence_count(before, "q1") == 0
    assert metacog.evidence_count(after, "q1") == 0  # origin evidence opens a question; it updates nothing
    assert metacog.credence_entropy(after, "q1") > 0
    assert metacog.credence_entropy(after, "missing") == 0.0
    assert not metacog.ready_to_conclude(after, "q1")


# ── Outcomes joined from the log ─────────────────────────────────────────────


def test_a_decision_pays_off_when_its_existing_question_moves() -> None:
    week = _week()
    # The pick is on night 1, where q1 already exists: the payoff is the night's change to it, not
    # the question's creation.
    nights = _facts(week, _choice(_want()), pick_night=1)
    found = metacog.records(nights)
    picks = [record for record in found if record.kind == "decision"]
    assert len(picks) == 1
    pick = picks[0]
    assert (pick.ref, pick.question_id, pick.outcome, pick.resolved_on) == ("w-1", "q1", True, day(1))
    assert "inquire" in metacog.SCORED_ACTIONS
    hypotheses = [record for record in found if record.kind == "hypothesis"]
    reported = next(record for record in hypotheses if record.ref == _leader(week))
    assert (reported.outcome, reported.resolved_on) == (True, day(1))
    # Only the hypothesis the finding names is settled; the others are unresolved, never guessed.
    assert all(record.outcome is None for record in hypotheses if record.ref != reported.ref)


def test_a_prediction_joins_its_verdict_on_the_night_it_resolved() -> None:
    week = _week()
    nights = _facts(week, _choice(_want()))
    prediction = next(record for record in metacog.records(nights) if record.kind == "prediction")
    assert prediction.outcome is True
    assert prediction.resolved_on == day(1)
    assert metacog.prediction_outcome(nights, day(0), "q1", prediction.ref, 0.56) == (True, day(1))
    assert metacog.prediction_outcome(nights, day(0), "q1", "p-missing", 0.56) == (None, None)


def _low_call_week(verdict: it.PredictionOutcome) -> tuple[it.WorkspaceEvent, ...]:
    """A committed call at a marginal under 0.5 (the low-credence hypothesis says the claim is
    unlikely), resolved on the next night with ``verdict``."""
    w = Ws()
    w.start(0)
    w.arrive(E1)
    w.ok(notice_op([E1.id], kind="gap"))
    w.ok(hypothesize("q1", 0.6, 0.2))
    ids = w.ids("q1")
    w.ok(it.PredictOp("q1", "The courtyard is booked on Thursday",
                      dict(zip(ids, (0.1, 0.9, 0.1), strict=True)), 2))
    prediction_id = w.ws.predictions[0].id
    w.start(1)
    w.ok(it.Resolved(prediction_id, verdict, ()))
    return tuple(w.events)


def test_a_confident_false_call_hits_when_false_and_misses_when_true() -> None:
    """Her call is the side of 0.5 her marginal commits to, not the claim's own probability."""
    for verdict, hit in (("false", True), ("true", False)):
        nights = _facts(_low_call_week(verdict), None)
        record = next(record for record in metacog.records(nights) if record.kind == "prediction")
        assert record.outcome is hit
        assert metacog.prediction_outcome(nights, day(0), "q1", record.ref, 0.26) == (hit, day(1))


def _same_night_resolution() -> tuple[tuple[it.WorkspaceEvent, ...], tuple[it.WorkspaceEvent, ...],
                                      it.Workspace, it.Workspace]:
    """Commit a prediction, then resolve it on a later cycle of the same night: the resolution's
    Bayes update moves the credences the prediction was committed with."""
    w = Ws()
    w.start(0)
    w.arrive(E1)
    w.ok(notice_op([E1.id], kind="gap"))
    w.ok(hypothesize("q1", 0.6, 0.2))
    ids = w.ids("q1")
    w.ok(it.PredictOp("q1", "The courtyard is booked on Thursday",
                      dict(zip(ids, (0.1, 0.9, 0.1), strict=True)), 2))
    prediction_id = w.ws.predictions[0].id
    commit = w.ws
    committed = tuple(w.events)
    w.ok(it.Resolved(prediction_id, "false", (ref(E1),)))
    return tuple(w.events), committed, commit, w.ws


def test_a_prediction_resolved_later_tonight_keeps_its_commit_confidence() -> None:
    events, committed, commit, after = _same_night_resolution()
    empty = it.Workspace()
    unresolved = (metacog.NightFacts(night=day(0), before=empty, after=commit, events=committed),)
    resolved = (metacog.NightFacts(night=day(0), before=empty, after=after, events=events),)
    early = next(record for record in metacog.records(unresolved) if record.kind == "prediction")
    late = next(record for record in metacog.records(resolved) if record.kind == "prediction")
    assert (late.evidence, late.entropy, late.raw) == (early.evidence, early.entropy, early.raw)
    committed_entropy = metacog.credence_entropy(commit, "q1")
    assert (late.evidence, late.entropy) == (0, committed_entropy)
    assert late.raw == pytest.approx(metacog.prediction_confidence(0.26, 0, committed_entropy, None))
    assert late.outcome is True  # the committed call was false, and the verdict was false
    # The night did move the state after the commit; the record must not read that state.
    assert metacog.evidence_count(after, "q1") > late.evidence
    assert metacog.credence_entropy(after, "q1") != pytest.approx(late.entropy)


def test_a_note_is_not_recorded_and_a_retraction_is_wrong() -> None:
    note = _choice(_want(question_id=None, action="hold_note"))
    assert metacog.decision_outcome(metacog.NightFacts(night=day(0), before=it.Workspace(),
                                                       after=it.Workspace(), choice=note)) is None

    w = Ws()
    w.start(0)
    w.arrive(E1)
    w.ok(notice_op([E1.id], kind="gap"))
    w.ok(hypothesize("q1", 0.6, 0.2))
    ids = w.ids("q1")
    w.ok(it.ConcludeOp("q1", ids[0], "suspicion", "The courtyard looks likely", (E1.id,)))
    before = w.ws
    w.start(1)
    w.ok(it.RetractOp("q1", "The kitchen is confirmed", (E1.id,)))
    nights = (metacog.NightFacts(night=day(0), before=it.Workspace(), after=before, choice=_choice(_want())),
              metacog.NightFacts(night=day(1), before=before, after=w.ws))
    records = metacog.records(nights)
    hypothesis = next(record for record in records if record.kind == "hypothesis")
    assert (hypothesis.outcome, hypothesis.resolved_on) == (False, day(1))


def test_a_pick_whose_receipt_says_it_never_ran_is_unresolved_not_a_miss() -> None:
    week = _week()
    ran = _facts(week, _choice(_want()))
    assert metacog.decision_outcome(ran[0]) is True  # the log joins the payoff when the pick ran
    stalled = tuple(replace(facts, pick="budget") for facts in ran)
    assert metacog.decision_outcome(stalled[0]) is None  # no chance to pay off is not a miss
    decisions = [record for record in metacog.records(stalled) if record.kind == "decision"]
    assert decisions and all(not record.resolved for record in decisions)
    with pytest.raises(ValueError):
        replace(ran[0], pick="maybe")  # only receipt.record's outcomes are names for a pick


def test_a_question_created_tonight_is_a_move_and_one_absent_both_sides_is_not() -> None:
    week = _week()
    _, before, after, events = metacog.night_slices(week)[0]
    created = metacog.NightFacts(night=day(0), before=before, after=after, events=events,
                                 choice=_choice(_want()))
    assert before.question("q1") is None and after.question("q1") is not None
    assert metacog.decision_outcome(created) is True

    missing = metacog.NightFacts(night=day(0), before=it.Workspace(), after=it.Workspace(),
                                 choice=_choice(_want(question_id="q-missing")))
    assert metacog.decision_outcome(missing) is False  # inquiring never touched it, so it didn't move
    record = next(record for record in metacog.records((missing,)) if record.kind == "decision")
    assert record.raw == pytest.approx(metacog.decision_confidence(0, None, ready=False, past_rate=None))


def test_a_decision_s_confidence_is_read_before_its_own_night_moves_the_question() -> None:
    week = _week()
    moved = _facts(week, _choice(_want()), pick_night=1)
    still = tuple(replace(facts, after=facts.before) if facts.night == day(1) else facts
                  for facts in moved)

    def decision(nights: tuple[metacog.NightFacts, ...]) -> metacog.ConfidenceRecord:
        return next(record for record in metacog.records(nights) if record.kind == "decision")

    paid, quiet = decision(moved), decision(still)
    assert (paid.evidence, paid.entropy, paid.raw) == (quiet.evidence, quiet.entropy, quiet.raw)
    assert (paid.outcome, quiet.outcome) == (True, False)  # only the night's own move differs


def test_a_night_without_a_log_is_never_scored_as_a_miss() -> None:
    empty = metacog.NightFacts(night=day(0), before=it.Workspace(), after=it.Workspace(),
                               choice=_choice(_want()), logged=False)
    assert metacog.decision_outcome(empty) is None
    records = metacog.records((empty,))
    assert records and all(not record.resolved for record in records)


def test_records_are_a_pure_function_of_the_nights() -> None:
    week = _week()
    nights = _facts(week, _choice(_want()))
    assert metacog.records(nights) == metacog.records(nights)
    assert metacog.night_views(week) == metacog.night_views(week)


# ── Calibration from her own history ─────────────────────────────────────────


def test_a_calibrator_is_raw_below_its_minimum() -> None:
    few = [(0.9, True)] * (metacog.MIN_TRIALS - 1)
    assert metacog.fit_calibrator(few) == metacog.Calibrator(n=metacog.MIN_TRIALS - 1)
    one_class = [(0.9, True)] * metacog.MIN_TRIALS
    assert metacog.fit_calibrator(one_class).kind == "raw"


def _biased(count: int = 20) -> list[tuple[float, bool]]:
    """An overconfident heuristic: its stated probability is far above its hit rate, evenly spread."""
    pairs: list[tuple[float, bool]] = []
    for step in range(count):
        raw = 0.55 + 0.4 * step / (count - 1)
        rate = 0.5 * raw + 0.15
        hits = round(rate * 20)
        pairs += [(raw, index < hits) for index in range(20)]
    return pairs


def test_platt_scaling_finds_the_bias_and_keeps_the_order() -> None:
    pairs = _biased()
    fit = metacog.fit_calibrator(pairs)
    assert fit.kind == "platt" and fit.n == len(pairs)
    assert fit.a > 0  # monotone: the ranking it is measured on is unchanged
    mean_raw = sum(raw for raw, _ in pairs) / len(pairs)
    mean_calibrated = sum(fit.calibrate(raw) for raw, _ in pairs) / len(pairs)
    assert mean_calibrated < mean_raw  # the heuristic was overconfident, and the fit says so
    raw_brier = sum((raw - outcome) ** 2 for raw, outcome in pairs) / len(pairs)
    calibrated_brier = sum((fit.calibrate(raw) - outcome) ** 2 for raw, outcome in pairs) / len(pairs)
    assert calibrated_brier < raw_brier


def test_a_night_is_calibrated_only_on_what_resolved_before_it() -> None:
    w = Ws()
    w.start(0)
    w.arrive(E1)
    w.ok(notice_op([E1.id], kind="gap"))
    w.ok(hypothesize("q1", 0.6, 0.2))
    ids = w.ids("q1")
    w.ok(it.PredictOp("q1", "The courtyard is free on Thursday", dict(zip(ids, (0.9, 0.05, 0.05), strict=True)), 2))
    prediction_id = w.ws.predictions[0].id
    w.start(1)
    w.found("q1", E3)
    w.ok(it.AssessOp("q1", E3.id, likelihoods=dict(zip(ids, (4.0, 0.25, 0.25), strict=True))))
    w.ok(it.Resolved(prediction_id, "true", ()))
    w.ok(it.ConcludeOp("q1", ids[0], "resolution", "The courtyard carries the look", (E1.id, E3.id)))
    w.start(2)
    w.ok(notice_op([E3.id], kind="gap", subject="Whether the courtyard works for the second scene"))
    w.ok(hypothesize(w.ws.questions[-1].id, 0.5, 0.3))
    views = {night: (before, after, events)
             for night, before, after, events in metacog.night_slices(tuple(w.events))}
    facts = tuple(
        metacog.NightFacts(night=number, before=views[number][0], after=views[number][1],
                           events=views[number][2],
                           choice=_choice(_want()) if number != day(1) else None)
        for number in (day(0), day(1), day(2))
    )
    found = metacog.records(facts)
    first = [record for record in found if record.night == day(0)]
    third = [record for record in found if record.night == day(2)]
    assert all(record.trials == 0 for record in first)  # nothing resolved before the first night
    assert [record.trials for record in third if record.kind == "decision"] == [1]  # one earlier decision
    assert all(record.calibrator == "raw" for record in (*first, *third))  # one trial is below MIN_TRIALS


def test_history_moves_a_raw_confidence_only_after_its_minimum() -> None:
    assert metacog.decision_confidence(2, 0.5, ready=False, past_rate=None) == pytest.approx(
        metacog.decision_confidence(2, 0.5, ready=False, past_rate=0.5))
    low = metacog.decision_confidence(2, 0.5, ready=False, past_rate=0.0)
    high = metacog.decision_confidence(2, 0.5, ready=False, past_rate=1.0)
    assert high - low == pytest.approx(metacog.HISTORY_GAIN)


def test_nights_are_grouped_and_never_repeated() -> None:
    week = _week()
    views = metacog.night_views(week)
    assert [night for night, _, _ in views] == [day(0), day(1)]
    with pytest.raises(ValueError):
        metacog.night_views(week[1:])  # no RunStart first: a cut log isn't a week
    facts = _facts(week, None)
    with pytest.raises(ValueError):
        metacog.records((*facts, *facts))
    assert metacog._ordered(facts) == facts


def test_the_fixture_is_a_two_night_week() -> None:
    """Guard the fixture itself: a broken week would make every join above vacuous."""
    folded = metacog.kernel.fold(_week())
    finding = folded.finding_of("q1")
    assert finding is not None and finding.kind == "resolution" and finding.hypothesis_id is not None
    assert folded.predictions[0].outcome == "true"
