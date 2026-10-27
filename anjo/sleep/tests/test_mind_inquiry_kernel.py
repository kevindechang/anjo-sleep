# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's kernel (Task 17): K1-K4, hypothesize-once, the domain denylist, the edits, and a
workspace that is exactly the fold of its log."""
from __future__ import annotations

import ast
import dataclasses
import json
import math
from pathlib import Path

import pytest

from anjo.sleep.mind.inquiry import codec, kernel
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.tests.inquiry_fakes import Ws, day, ev, hypothesize, night, noon, notice_op, ref

INQUIRY = Path(kernel.__file__).parent
E1 = ev("e-d2-1", "Scene 4 wants the compressed long-lens look: an 85 mm medium shot on full frame.", n=2)
E2 = ev("e-d5-3", "Scene 4 is in their 3 m kitchen, handheld.", n=5)
E3 = ev("e-d3-2", "They might shoot scene 4 through the window from the courtyard.", n=3)
E4 = ev("e-d4-1", "The courtyard faces the kitchen window.", n=4)
E6 = ev("e-d6-1", "The courtyard is locked on Thursday.", n=6)
E5 = ev("e-d5-1", "The kitchen window is the only light source.", n=5)
E7 = ev("e-d7-1", "The shoot moved to Friday.", n=7)


def _question(rules: it.KernelRules = it.FULL_RULES, *priors: float, catch_all: float = 0.2) -> Ws:
    """A run, items e1/e2 arriving, a contradiction q1 on them, and its hypotheses (seq 3)."""
    w = Ws(rules)
    w.start(5)
    w.arrive(E1, E2)
    w.ok(notice_op([E1.id, E2.id]))
    w.ok(hypothesize("q1", *(priors or (0.5, 0.3)), catch_all=catch_all))
    return w


def _likelihoods(w: Ws, *values: float) -> dict[str, float]:
    return dict(zip(w.ids("q1"), values, strict=True))


def _assess(w: Ws, item: str, *values: float) -> it.AssessOp:
    return it.AssessOp("q1", item, likelihoods=_likelihoods(w, *values))


def _predict(w: Ws, *values: float, days: int = 3) -> it.PredictOp:
    return it.PredictOp("q1", "They name a place other than the kitchen for scene 4", _likelihoods(w, *values), days)


# ── Purity ────────────────────────────────────────────────────────────────────

PURE = ("kernel.py", "controller.py", "render.py", "parse.py", "codec.py", "findings.py", "loop.py")
PURE_STDLIB = frozenset({
    "__future__", "math", "collections.abc", "dataclasses", "datetime", "types", "typing", "hashlib", "json",
    "re", "pathlib",
})
IMPURE_CALLS = frozenset({
    "now", "utcnow", "today", "getenv", "environ", "open", "connect", "urlopen", "time", "sleep", "random",
    "uuid4", "post", "get_db",
})


def _imports(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names.append(node.module or "")
    return names


@pytest.mark.parametrize("module", PURE)
def test_mind_inquiry_core_is_pure(module: str) -> None:
    """No clock, database, model call or environment read: only pure stdlib, the inquiry's own
    modules, the shared types and the chooser's snapshot numbers. The loop drives ports and
    nothing else; only the render reads a file, its fixed instructions."""
    path = INQUIRY / module
    for name in _imports(path):
        assert (
            name in PURE_STDLIB or name.startswith("anjo.sleep.mind.inquiry")
            or name in ("anjo.sleep.types", "anjo.sleep.mind.snapshot")
        ), f"{module} imports {name}"
        assert name != "pathlib" or module == "render.py", f"{module} reads files"
    for node in ast.walk(ast.parse(path.read_text())):
        called = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
        assert called not in IMPURE_CALLS, f"{module} uses {called}"


# ── K1 ────────────────────────────────────────────────────────────────────────


def test_bayes_updates_whole_vector_and_is_order_independent() -> None:
    first = _question()
    first.found("q1", E3, E4)
    prior = first.credences("q1")
    first.ok(_assess(first, E3.id, 2, 0.5, 1))
    weights = {h: prior[h] * lk for h, lk in _likelihoods(first, 2, 0.5, 1).items()}
    total = sum(weights.values())
    assert first.credences("q1") == pytest.approx({h: w / total for h, w in weights.items()})
    assert all(first.credences("q1")[h] != prior[h] for h in prior)  # the catch-all moved too
    first.ok(_assess(first, E4.id, 0.25, 4, 1))

    second = _question()
    second.found("q1", E3, E4)
    second.ok(_assess(second, E4.id, 0.25, 4, 1))
    second.ok(_assess(second, E3.id, 2, 0.5, 1))

    assert second.credences("q1") == pytest.approx(first.credences("q1"))
    assert math.fsum(first.credences("q1").values()) == pytest.approx(1.0)
    assert [u.source for u in first.ws.updates] == ["assessed_evidence", "assessed_evidence"]


def test_one_evidence_item_updates_a_question_once_by_any_route() -> None:
    w = _question()
    w.found("q1", E3)
    w.ok(_assess(w, E3.id, 2, 0.5, 1))
    assert w.code(_assess(w, E3.id, 1, 1, 1)) == "k1_evidence_used"

    # Assessed first, then a prediction resolved by the same item: recorded, applies nothing.
    w.ok(_predict(w, 0.2, 0.9, 0.5), at=night(5))
    w.arrive(E6, at=night(6))
    w.found("q1", E6)
    w.ok(_assess(w, E6.id, 0.5, 2, 1))
    before, updates = w.credences("q1"), len(w.ws.updates)
    w.ok(it.Resolved(w.ws.predictions[-1].id, "false", (ref(E6),)))
    assert w.credences("q1") == before and len(w.ws.updates) == updates
    assert w.ws.predictions[0].outcome == "false" and w.ws.predictions[0].resolved_by == (E6.id,)

    # Resolved first: the update applies once, and the item can't be assessed again.
    w.ok(_predict(w, 0.9, 0.2, 0.5), at=night(6))
    w.arrive(E7, at=night(7))
    w.ok(it.Resolved(w.ws.predictions[-1].id, "true", (ref(E7),)))
    assert w.ws.updates[-1].source == "resolved_prediction" and w.ws.updates[-1].evidence_ids == (E7.id,)
    w.found("q1", E7)
    assert w.code(_assess(w, E7.id, 2, 1, 1)) == "k1_evidence_used"


def test_clamps_priors_and_predictions_never_likelihood_ratios() -> None:
    w = _question(it.FULL_RULES, 1.0, 0.0, catch_all=0.0)
    assert list(w.credences("q1").values()) == pytest.approx([0.95 / 1.05, 0.05 / 1.05, 0.05 / 1.05])
    w.ok(_predict(w, 1.0, 0.0, 0.5))
    assert list(w.ws.predictions[0].p_true_given.values()) == [0.95, 0.05, 0.5]

    w.found("q1", E3, E4, E5)
    before = w.credences("q1")
    for item in (E3, E4, E5):
        w.ok(_assess(w, item.id, 0.25, 4, 1))
    after = w.credences("q1")
    h1, h2, _ = w.ids("q1")
    # The 16-fold ratio of each likelihood pair survives: nothing clips L or the posterior.
    assert (after[h2] / after[h1]) == pytest.approx((before[h2] / before[h1]) * 16**3)
    assert after[h2] > 0.95
    assert [u.likelihoods[h2] for u in w.ws.updates] == [4.0, 4.0, 4.0]

    w.arrive(E6, at=night(6))
    w.ok(it.Resolved(w.ws.predictions[0].id, "true", (ref(E6),)))
    resolved = w.ws.updates[-1].likelihoods
    assert (resolved[h1], resolved[h2]) == (0.95, 0.05)
    assert all(c > 0.0 for c in w.credences("q1").values())


def test_assess_needs_a_bucket_an_item_returned_for_the_question_and_k1_on() -> None:
    w = _question()
    assert w.code(_assess(w, E3.id, 2, 1, 1)) == "unknown_ref"  # not even held
    w.arrive(E3)
    assert w.code(_assess(w, E3.id, 2, 1, 1)) == "k1_not_returned"  # held, but no search returned it
    w.found("q1", E3)
    assert w.code(_assess(w, E3.id, 3, 1, 1)) == "k1_bucket"
    assert w.code(it.AssessOp("q1", E3.id, credences=_likelihoods(w, 0.9, 0.05, 0.05))) == "k1_direct_credence"
    assert w.code(it.AssessOp("q1", E3.id, likelihoods={"q1.h2a": 2.0})) == "k1_hypotheses"


def test_self_query_never_moves_credence() -> None:
    w = _question()
    before = w.ws
    w.ok(it.InvestigateOp("q1", "world_query", "How much distance does an 85 mm medium shot need?"))
    w.ok(it.Answered("q1", "How much distance does an 85 mm medium shot need?", "About 3 m; the kitchen is too small."))
    assert w.credences("q1") == {h.id: h.credence for h in before.hypotheses} and w.ws.updates == ()
    # An answer is never an item: nothing can assess it.
    assert w.code(_assess(w, "About 3 m; the kitchen is too small.", 4, 0.25, 1)) == "unknown_ref"


# ── K2, K3, K4 ────────────────────────────────────────────────────────────────


def test_nondiscriminating_prediction_rejected() -> None:
    w = _question()
    assert w.code(_predict(w, 0.5, 0.75, 0.6)) == "k2_not_discriminating"
    assert w.code(_predict(w, 0.0, 0.3, 0.1)) == "k2_not_discriminating"  # clamped to 0.05: spread 0.25
    w.ok(_predict(w, 0.2, 0.5, 0.4))  # exactly 0.3
    loose = _question(it.NO_KERNEL)
    loose.ok(_predict(loose, 0.5, 0.6, 0.55))


def test_contradiction_needs_two_days() -> None:
    same_day = ev("e-d2-2", "Scene 4 is shot in the kitchen.", n=2)
    w = Ws()
    w.start(5)
    w.arrive(E1, E2, same_day)
    assert w.code(notice_op([E1.id, same_day.id])) == "k3_citations"
    assert w.code(notice_op([E1.id])) == "k3_citations"
    assert w.code(notice_op([E1.id, E1.id])) == "k3_citations"
    assert w.code(notice_op([], kind="gap")) == "k3_citations"
    assert w.code(notice_op([E1.id, "e-unseen"])) == "unknown_ref"
    w.ok(notice_op([E1.id, E2.id]))
    w.ok(notice_op([E2.id], kind="gap", subject="Scene 4's handheld plan has no stabilizer listed"))
    assert [q.kind for q in w.ws.questions] == ["contradiction", "gap"]


def test_resolution_needs_evidence_outside_origin() -> None:
    w = Ws()
    w.start(5)
    w.arrive(E1, E2, E3)
    w.ok(notice_op([E1.id, E2.id]))
    w.found("q1", E3)  # returned before the hypotheses
    w.ok(hypothesize("q1", 0.6, 0.2))
    w.found("q1", E1)  # an origin item, returned after
    w.ok(_assess(w, E1.id, 4, 0.25, 0.25))
    w.ok(_assess(w, E3.id, 2, 0.5, 0.5))
    lead, other, _ = w.ids("q1")
    assert w.credences("q1")[lead] >= it.RESOLVE_MIN
    conclude = it.ConcludeOp("q1", lead, "resolution", "The 85 mm look can't be shot in the kitchen", (E1.id, E2.id))
    assert w.code(conclude) == "k4_not_ready"
    assert w.code(dataclasses.replace(conclude, hypothesis_id=other)) == "k4_not_ready"
    w.ok(dataclasses.replace(conclude, kind="suspicion"))
    assert w.ws.findings[0].kind == "suspicion" and w.ws.question("q1").status == "active"  # type: ignore[union-attr]

    w.arrive(E4)
    w.found("q1", E4)
    w.ok(_assess(w, E4.id, 1, 1, 1))  # outside the origin, returned after the hypotheses
    w.ok(conclude)
    finding = w.ws.findings[0]
    assert (finding.kind, finding.id) == ("resolution", "f5") and w.ws.question("q1").status == "concluded"  # type: ignore[union-attr]

    loose = _question(it.NO_KERNEL)
    first = loose.ids("q1")[0]
    loose.ok(it.ConcludeOp("q1", first, "resolution", "Settled without evidence", ()))


# ── Hypothesize once, and the domain rule ─────────────────────────────────────


def test_hypothesize_once_unless_catch_all_dominates() -> None:
    w = _question()
    assert w.code(hypothesize("q1", 0.5, catch_all=0.5)) == "hypothesize_once"

    wide = _question(it.FULL_RULES, 0.2, 0.2, catch_all=0.6)
    wide.ok(_predict(wide, 0.2, 0.6, 0.4))
    assert wide.code(hypothesize("q1", 0.5, 0.5, 0.5, catch_all=0.5)) == "hypothesis_count"
    wide.ok(hypothesize("q1", 0.5, catch_all=0.5))
    h1, h2, h3, catch = wide.ids("q1")
    credences = wide.credences("q1")
    assert (credences[h1], credences[h2]) == pytest.approx((0.2, 0.2))
    assert (credences[h3], credences[catch]) == pytest.approx((0.3, 0.3))  # the catch-all's 0.6, split
    assert dict(wide.ws.predictions[0].p_true_given)[h3] == 0.4  # inherits the catch-all's value
    assert wide.code(hypothesize("q1", 0.5, catch_all=0.5)) == "hypothesize_once"

    fresh = Ws()
    fresh.start()
    fresh.arrive(E1, E2)
    fresh.ok(notice_op([E1.id, E2.id]))
    assert fresh.code(hypothesize("q1", 0.8)) == "hypothesis_count"
    assert fresh.code(hypothesize("q1", 0.2, 0.2, 0.2, 0.2, 0.2)) == "hypothesis_count"


def test_person_diagnosis_question_rejected() -> None:
    assert set(kernel.DENYLIST) >= {"traits", "health", "feelings", "relationships"}
    w = Ws()
    w.start()
    w.arrive(E1, E2)
    assert w.code(notice_op([E1.id, E2.id], subject="Whether they are depressed about the shoot")) == "domain_denylist"
    assert w.code(notice_op([E1.id, E2.id], subject="Her relationship with the producer")) == "domain_denylist"
    w.ok(notice_op([E1.id, E2.id]))
    diagnosis = it.HypothesizeOp(
        "q1", (it.Alternative("She doesn’t know an 85 mm lens needs distance", 0.5),
               it.Alternative("The plan assumes room the kitchen lacks", 0.3)), 0.2,
    )
    assert w.code(diagnosis) == "domain_denylist"
    assert w.code(it.InvestigateOp("q1", "evidence_search", "her anxiety before shoots")) == "domain_denylist"
    w.ok(hypothesize("q1", 0.5, 0.3))
    claim = it.PredictOp("q1", "They feel overwhelmed by the shoot", _likelihoods(w, 0.2, 0.8, 0.5), 2)
    assert w.code(claim) == "domain_denylist"
    for fine in ("The plan assumes room for an 85 mm lens", "a sad scene in a small kitchen", "the emotional beat"):
        assert not kernel.denied(fine)


def test_question_needs_targets() -> None:
    w = Ws()
    w.start()
    w.arrive(E1, E2)
    assert w.code(notice_op([E1.id, E2.id], targets=())) == "no_targets"
    unknown = notice_op([E1.id, E2.id], targets=("t-nowhere",))
    assert kernel.check(w.ws, unknown, target_ids={"t-scene4"}) == "unknown_ref"
    assert kernel.check(w.ws, notice_op([E1.id, E2.id]), target_ids={"t-scene4"}) is None
    with pytest.raises(ValueError, match="target_ids"):
        it.Question("q1", (), "gap", (E1.id,), "open", noon(1), "a plan")


def test_live_questions_only_and_closing_ops() -> None:
    w = _question()
    w.ok(it.ParkOp("q1"))
    assert w.code(it.InvestigateOp("q1", "evidence_search", "kitchen")) == "not_live"
    assert w.code(it.ParkOp("q9")) == "unknown_ref"
    other = _question()
    other.ok(it.RetractOp("q1", "The kitchen was never the location", (E2.id,)))
    assert other.ws.question("q1").status == "retracted" and other.ws.findings == ()  # type: ignore[union-attr]


# ── Edits, the fold and the digest ────────────────────────────────────────────


def test_edits_change_exactly_what_they_name() -> None:
    w = _question()
    w.ok(_predict(w, 0.2, 0.9, 0.5))
    h1, h2, catch = w.ids("q1")
    w.ok(it.Edit("swap_lead", "q1", hypothesis_id=h2))
    assert (w.credences("q1")[h1], w.credences("q1")[h2]) == pytest.approx((0.3, 0.5))
    with pytest.raises(ValueError, match="leader"):
        w.step(it.Edit("swap_lead", "q1", hypothesis_id=h2))
    w.ok(it.Edit("inject_hypothesis", "q1", statement="The shoot moves to a bigger room"))
    injected = w.ids("q1")[2]
    assert w.credences("q1")[injected] == 0.5 and w.credences("q1")[catch] == pytest.approx(0.1)
    assert w.ws.predictions[0].p_true_given[injected] == w.ws.predictions[0].p_true_given[catch]
    w.ok(it.Edit("delete_prediction", "q1", prediction_id="p3"))
    assert w.ws.predictions == ()
    with pytest.raises(ValueError, match="pending prediction"):
        w.step(it.Edit("delete_prediction", "q1", prediction_id="p3"))


def test_workspace_is_a_fold_of_its_log() -> None:
    w = _question()
    w.found("q1", E3)
    w.ok(_assess(w, E3.id, 2, 0.5, 1))
    w.ok(it.PredictOp("q1", "x", _likelihoods(w, 0.5, 0.55, 0.5), 2), code="k2_not_discriminating", cycle=0, focus="q1")
    w.ok(None, code="malformed", cycle=1, focus="q1", tokens=40)
    w.ok(it.Edit("swap_lead", "q1", hypothesis_id=w.ids("q1")[1]))

    stored = [json.loads(codec.canonical(codec.event_to_json(e))) for e in w.events]
    events = [codec.event_from_json(record) for record in stored]
    rebuilt = kernel.fold(events)
    assert rebuilt == w.ws and kernel.digest(rebuilt) == w.events[-1].after == kernel.digest(w.ws)

    tampered = [*events[:-1], dataclasses.replace(events[-1], after="0" * 64)]
    with pytest.raises(ValueError, match="after digest"):
        kernel.fold(tampered)
    sneaky = dataclasses.replace(events[6], step=dataclasses.replace(events[6].step, code=None))
    with pytest.raises(ValueError, match="rejected"):
        kernel.fold([*events[:6], sneaky])
    with pytest.raises(ValueError, match="before digest"):
        kernel.fold(events[1:])


def test_a_wipe_leaves_only_its_run_start_and_ids_stay_unique() -> None:
    w = _question()
    w.start(6, wipe=True)
    assert not w.ws.questions and len(w.ws.history) == 1 and w.ws.seq == 5
    w.arrive(E1, E2)
    w.ok(notice_op([E1.id, E2.id]))
    assert w.ws.questions[0].id == "q3"
    assert kernel.fold(w.events) == w.ws
    assert day(6) == w.ws.history[0].entry.night  # type: ignore[union-attr]
