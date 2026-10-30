"""Second-order confidence for her night: does her confidence know when she is right?

Fleming & Lau (2014, *How to measure metacognition*) separate first-order sensitivity (is she
right?) from second-order sensitivity (does her confidence say so?), and measure the second with a
type-2 AUROC, meta-d'/d' and calibration, never with self-report. MIND's confidence today is a
per-target heuristic that weights wants (``types.Target.confidence``, ``choose``) and is never
checked against what happened. This module records the second-order confidence that was missing.

**The record.** For the decision the chooser makes at a night's offer, and for every credence it
sets that night on a hypothesis or a committed prediction, a ``ConfidenceRecord`` says how likely
she thinks it is that the decision pays off (its study or inquiry produces a finding, or moves its
question) or that the credence is right. Nothing is written and no night behaves differently: a
record is a pure function of the decision point, her workspace before and after the night, and the
records that resolved before it, so replaying a logged night gives the same record. A night with
the inquiry off never reaches this module.

**Named bounds.** Every raw confidence is clamped to ``[CONF_MIN, CONF_MAX]``. With no evidence on
a question its support is zero, so a credence falls back to chance and a decision to
``BASE_RATE``; the decision's focus is 1 minus the question's normalized credence entropy, and a
question her own rules could already conclude (K4's bar, or a reported hypothesis ready to retract)
adds ``READY_GAIN``. Her past hit rate on the same kind of record, once ``RATE_MIN`` of them have
resolved, moves a raw confidence by at most ``HISTORY_GAIN``. The Platt fit
(``fit_calibrator``) is the measured correction on top, refitted every night from the records
resolved before it, and not applied at all below ``MIN_TRIALS``: raw then, because a fit on a
handful of outcomes is noise. Each kind is fitted apart: a decision's payoff rate and a credence's
hit rate have different bases.

**Outcomes, joined from what the night already records.** A decision pays off when its night's run
changes what it chose: a new or changed finding on its question, a credence move of at least
``MOVE_MIN``, a prediction committed or resolved, a question status changed, or a result on its
target. That is the same comparison ``week._changes`` shows a reader. A question that existed
before the night counts only when it changed; one the night created is a move, and one missing on
both sides is not. The choose step's receipt (``receipt.record``, read from the night log) is joined
before that comparison: a pick it records as ``declined``, ``budget`` or ``failed`` never ran, so it
had no chance to move its question and is left unresolved rather than scored as a miss. A
hypothesis is right when the question's final finding is a resolution naming it, wrong when that
finding is a retraction of it, and unresolved while the question only suspects. A prediction's
credence is right when her committed true/false call matches the resolver's verdict
(``findings.share_result`` is what can carry the finding on, but the verdict is the prediction's own
``outcome``). The call is the marginal she committed at ``Predict``: with a marginal under 0.5 a
``false`` verdict is a hit and a ``true`` verdict a miss, because confidence is her estimate of
being right, not the claim's probability. That marginal, and the evidence and entropy behind it, are
read from her log only through the ``Predict`` event: the inquiry loop can resolve a prediction on a
later cycle of the same night, and the Bayes update that follows would otherwise leak the outcome
into the confidence that is meant to predict it. A note or silence has no payoff the log can see, so
it is not recorded rather than guessed: a wrong guess here would corrupt the very calibration the
record exists to measure.

The report that reads these records, with its AUROC, Brier, bins and meta-d'/d', is
``anjo.sleep.reports.metacog``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Final, Literal

from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry.types import (
    FULL_RULES,
    RETRACT_BELOW,
    RESOLVE_MIN,
    KernelRules,
    Rejected,
    RunStart,
    Workspace,
    WorkspaceEvent,
)
from anjo.sleep.types import Action, Choice, Result

Kind = Literal["decision", "hypothesis", "prediction"]
KINDS: Final[tuple[Kind, ...]] = ("decision", "hypothesis", "prediction")
Asking = Literal["raw", "platt"]
ASKINGS: Final = frozenset({"raw", "platt"})

SCORED_ACTIONS: Final[frozenset[Action]] = frozenset({"prepare_commitment", "revisit_thread", "inquire"})
PICK_OUTCOMES: Final = frozenset({"ran", "declined", "budget", "failed"})  # receipt.Outcome, as recorded
RAN: Final = "ran"
CONF_MIN: Final = 0.02  # every confidence is a probability, and a heuristic one is never certain
CONF_MAX: Final = 0.98
BASE_RATE: Final = 0.5  # a decision with no evidence either way is a coin
EVIDENCE_HALF: Final = 4.0  # evidence items at which a question's support is one half
EVIDENCE_GAIN: Final = 0.20
FOCUS_GAIN: Final = 0.15  # a focused credence vector is likelier to conclude
READY_GAIN: Final = 0.10
HISTORY_GAIN: Final = 0.20  # the most her own past hit rate may move a raw confidence
ENTROPY_FLOOR: Final = 0.5  # a credence keeps at least half its support in a flat vector
RATE_MIN: Final = 5  # resolved records of that kind before her hit rate is used at all
MIN_TRIALS: Final = 12  # resolved records of that kind before a calibrator is fitted
MOVE_MIN: Final = 0.02  # a credence move this large counts as the question moving
BINS: Final = 10
LOGIT_EPS: Final = 1e-6
PLATT_STEPS: Final = 50
PLATT_TOL: Final = 1e-9
PLATT_FLOOR: Final = 1e-12
PLATT_RIDGE: Final = 1e-6
PLATT_A: Final[tuple[float, float]] = (1e-3, 10.0)  # a fit may not flip or collapse the ordering
PLATT_B: Final[tuple[float, float]] = (-10.0, 10.0)


def _clamp(value: float, low: float = CONF_MIN, high: float = CONF_MAX) -> float:
    return min(high, max(low, value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    odds = math.exp(value)
    return odds / (1.0 + odds)


def _logit(value: float) -> float:
    clipped = min(1.0 - LOGIT_EPS, max(LOGIT_EPS, value))
    return math.log(clipped / (1.0 - clipped))


def _support(evidence: int) -> float:
    """0 with no evidence, one half at ``EVIDENCE_HALF`` items, approaching 1."""
    return evidence / (evidence + EVIDENCE_HALF) if evidence > 0 else 0.0


def _bounded(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], value))


# ── Her state, read as numbers ───────────────────────────────────────────────


def evidence_count(ws: Workspace, question_id: str) -> int:
    """The distinct evidence items that have updated the question: applied evidence, not the items
    merely shown in a render. K1 updates a question once per item, so this is her evidence."""
    return len({item for update in ws.updates if update.question_id == question_id
                for item in update.evidence_ids})


def credence_entropy(ws: Workspace, question_id: str) -> float:
    """The question's normalized credence entropy, in [0, 1]: 1 is a flat vector. A question with
    fewer than two hypotheses has nothing to be uncertain about, so 0."""
    credences = [h.credence for h in ws.hypotheses_of(question_id)]
    if len(credences) < 2:
        return 0.0
    total = math.fsum(credences)
    if total <= 0:
        return 0.0
    entropy = -math.fsum(c / total * math.log(c / total) for c in credences if c > 0)
    return _clamp(entropy / math.log(len(credences)), 0.0, 1.0)


def ready_to_conclude(ws: Workspace, question_id: str) -> bool:
    """Whether her own rules could conclude the question tonight: a hypothesis at K4's bar, or a
    reported finding under the bar at which it is ready to retract."""
    if any(h.credence >= RESOLVE_MIN for h in ws.hypotheses_of(question_id)):
        return True
    finding = ws.finding_of(question_id)
    return bool(finding is not None and finding.hypothesis_id is not None and finding.p < RETRACT_BELOW)


# ── The second-order confidence (a), derived from her state and named bounds ──


def decision_confidence(evidence: int, entropy: float | None, *, ready: bool,
                        past_rate: float | None) -> float:
    """Her estimate that the chosen study or inquiry pays off, on [``CONF_MIN``, ``CONF_MAX``].

    ``evidence`` is what has updated the question; ``entropy`` its normalized credence entropy, or
    ``None`` for a study of a target with no question (nothing to be uncertain about, so no focus
    term). ``ready`` is ``ready_to_conclude``. ``past_rate`` is her hit rate on earlier resolved
    decisions, or ``None`` below ``RATE_MIN`` of them.
    """
    focus = 0.0 if entropy is None else 1.0 - entropy
    value = BASE_RATE + EVIDENCE_GAIN * _support(evidence) + FOCUS_GAIN * focus
    if ready:
        value += READY_GAIN
    if past_rate is not None:
        value += HISTORY_GAIN * (past_rate - BASE_RATE)
    return _clamp(value)


def _credence_support(evidence: int, entropy: float) -> float:
    """Evidence support, kept at half weight or more while the credence vector is flat."""
    return _support(evidence) * (ENTROPY_FLOOR + (1.0 - ENTROPY_FLOOR) * (1.0 - entropy))


def hypothesis_confidence(credence: float, alternatives: int, evidence: int, entropy: float,
                          past_rate: float | None) -> float:
    """Her estimate that this credence is right: chance, plus her excess credence over chance,
    tempered by the question's evidence and focus, plus her past hit rate when it has one."""
    chance = 1.0 / max(alternatives, 1)
    value = chance + (credence - chance) * _credence_support(evidence, entropy)
    if past_rate is not None:
        value += HISTORY_GAIN * (past_rate - chance)
    return _clamp(value)


def prediction_confidence(p_true: float, evidence: int, entropy: float,
                          past_rate: float | None) -> float:
    """Her estimate that her true/false call on a committed claim is right: a coin with no
    evidence, rising with the claim's distance from 0.5, the question's evidence and its focus."""
    support = _credence_support(evidence, entropy)
    value = BASE_RATE + abs(p_true - 0.5) * support
    if past_rate is not None:
        value += HISTORY_GAIN * (past_rate - BASE_RATE)
    return _clamp(value)


# ── Calibration from her own resolved outcomes ───────────────────────────────


@dataclass(frozen=True)
class Calibrator:
    """A fitted map from a raw confidence to a calibrated one.

    ``raw`` is the identity, used below ``MIN_TRIALS`` resolved outcomes of the kind.
    ``platt`` is Platt's scaling (1999), ``sigmoid(a * logit(p) + b)``: monotone, so it can
    rescale a biased heuristic without changing the order in which it ranks calls.
    """

    a: float = 1.0
    b: float = 0.0
    n: int = 0
    kind: Asking = "raw"

    def __post_init__(self) -> None:
        if self.kind not in ASKINGS:
            raise ValueError(f"a calibrator is raw or platt, got {self.kind!r}")
        if self.n < 0:
            raise ValueError(f"a calibrator cannot be fitted on {self.n} outcomes")
        if self.kind == "platt":
            if not PLATT_A[0] <= self.a <= PLATT_A[1]:
                raise ValueError(f"platt's slope {self.a} is outside {PLATT_A}")
            if not PLATT_B[0] <= self.b <= PLATT_B[1]:
                raise ValueError(f"platt's intercept {self.b} is outside {PLATT_B}")

    def calibrate(self, value: float) -> float:
        if self.kind == "raw":
            return _clamp(value)
        return _clamp(_sigmoid(self.a * _logit(value) + self.b))


def fit_calibrator(pairs: Sequence[tuple[float, bool]]) -> Calibrator:
    """Platt scaling on her resolved outcomes, fitted by Newton's method on the log loss.

    Fewer than ``MIN_TRIALS`` pairs, or only one outcome, returns the raw identity: a fit there
    would memorise noise instead of correcting a bias. The slope is bounded below at
    ``PLATT_A[0]`` so a fit can never invert the ranking it is meant to keep.
    """
    n = len(pairs)
    if n < MIN_TRIALS or len({bool(outcome) for _, outcome in pairs}) < 2:
        return Calibrator(n=n)
    points = [(_logit(raw), 1.0 if outcome else 0.0) for raw, outcome in pairs]
    a, b = 1.0, 0.0
    for _ in range(PLATT_STEPS):
        grad_a = grad_b = hess_aa = hess_ab = hess_bb = 0.0
        for z, outcome in points:
            q = _sigmoid(a * z + b)
            error = q - outcome
            weight = max(q * (1.0 - q), PLATT_FLOOR)
            grad_a += error * z
            grad_b += error
            hess_aa += weight * z * z
            hess_ab += weight * z
            hess_bb += weight
        hess_aa += PLATT_RIDGE
        hess_bb += PLATT_RIDGE
        determinant = hess_aa * hess_bb - hess_ab * hess_ab
        if determinant <= 0:
            break
        step_a = (hess_bb * grad_a - hess_ab * grad_b) / determinant
        step_b = (hess_aa * grad_b - hess_ab * grad_a) / determinant
        a = _bounded(a - step_a, PLATT_A)
        b = _bounded(b - step_b, PLATT_B)
        if max(abs(grad_a), abs(grad_b)) < PLATT_TOL:
            break
    return Calibrator(a=a, b=b, n=n, kind="platt")


# ── The record ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConfidenceRecord:
    """One second-order confidence, its outcome when known, and the calibrator it was read with.

    ``ref`` names the decision's want, the hypothesis or the prediction by id. ``evidence`` and
    ``entropy`` are the state it was derived from; ``raw`` is that derivation, ``calibrated`` is
    the value after the night's calibrator (``calibrator``, fitted on ``trials`` earlier resolved
    outcomes of its kind). ``outcome`` is ``None`` while nothing the log can see settled it, and
    ``resolved_on`` is the night it settled.
    """

    night: date
    kind: Kind
    ref: str
    question_id: str | None
    target_id: str | None
    evidence: int
    entropy: float
    raw: float
    calibrated: float
    calibrator: Asking
    trials: int
    outcome: bool | None
    resolved_on: date | None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"a confidence record is one of {KINDS}, got {self.kind!r}")
        if not self.ref:
            raise ValueError("a confidence record names what it is about")
        if self.evidence < 0:
            raise ValueError(f"evidence count {self.evidence} < 0")
        _check_unit("entropy", self.entropy)
        _check_unit("raw", self.raw)
        _check_unit("calibrated", self.calibrated)
        if self.calibrator not in ASKINGS:
            raise ValueError(f"a record's calibrator is raw or platt, got {self.calibrator!r}")
        if self.trials < 0:
            raise ValueError(f"a calibrator cannot rest on {self.trials} outcomes")
        if (self.outcome is None) != (self.resolved_on is None):
            raise ValueError("an outcome and the night it resolved are recorded together, or neither")
        if self.resolved_on is not None and self.resolved_on < self.night:
            raise ValueError("an outcome cannot resolve before the record it settles")

    @property
    def resolved(self) -> bool:
        return self.outcome is not None


def _check_unit(name: str, value: float) -> None:
    if not isinstance(value, float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a float in [0, 1], got {value!r}")


@dataclass(frozen=True)
class NightFacts:
    """One night as this module sees it: her workspace before and after it, the target log's
    results, and the chooser's pick at the pre-study offer.

    ``logged`` is False when the night left no workspace log, so no change can be read: an empty
    workspace is never silently scored as "nothing happened". ``at`` is the offer's clock, used to
    tell a result made after it from one that was already there. ``study_evidence`` is the chosen
    target's unacted revisions when the pick was a study of a target with no question. ``pick`` is
    the choose step's receipt outcome (``PICK_OUTCOMES``), or ``None`` when the night holds no
    receipt: a pick that did not run cannot have paid off, and is left unresolved. ``events`` is the
    night's own log slice, applied from ``before`` when a commit-time credence is needed; an empty
    slice falls back to ``before``, which never contains the night's own outcome.
    """

    night: date
    before: Workspace
    after: Workspace
    results: tuple[Result, ...] = ()
    choice: Choice | None = None
    study_evidence: int = 0
    logged: bool = True
    at: datetime | None = None  # the offer's clock, or None when the caller has none
    pick: str | None = None  # the choose receipt's outcome, or None when there is none
    events: tuple[WorkspaceEvent, ...] = ()  # this night's own log, for a commit-time state

    def __post_init__(self) -> None:
        if self.study_evidence < 0:
            raise ValueError("a study's evidence count is never negative")
        if self.pick is not None and self.pick not in PICK_OUTCOMES:
            raise ValueError(f"a pick's receipt outcome is one of {sorted(PICK_OUTCOMES)}, got {self.pick!r}")


NightSlice = tuple[date, Workspace, Workspace, tuple[WorkspaceEvent, ...]]


def night_slices(events: Sequence[WorkspaceEvent]) -> tuple[NightSlice, ...]:
    """Her log at each night's boundary, with that night's own events.

    Each slice is the workspace before the night's first ``RunStart``, after its last event, and
    the events between: enough to fold again only as far as a commit inside the night. Nights
    advance in date order, and several calls in one night are one view, as ``week.slices`` groups
    them.
    """
    if events and not isinstance(events[0].step.entry, RunStart):
        raise ValueError("the log must start with RunStart")
    groups: list[tuple[date, int, int]] = []
    for index, event in enumerate(events):
        if isinstance(event.step.entry, RunStart):
            night = event.step.entry.night
            if not groups or groups[-1][0] != night:
                if groups and night <= groups[-1][0]:
                    raise ValueError("nights must advance in date order")
                groups.append((night, index, index + 1))
                continue
        if groups:
            groups[-1] = (groups[-1][0], groups[-1][1], index + 1)
    return tuple((night, kernel.fold(events[:start]), kernel.fold(events[:end]), tuple(events[start:end]))
                 for night, start, end in groups)


def night_views(events: Sequence[WorkspaceEvent]) -> tuple[tuple[date, Workspace, Workspace], ...]:
    """Her log at each night's boundary: the workspace before the night's first ``RunStart`` and
    after its last event."""
    return tuple((night, before, after) for night, before, after, _ in night_slices(events))


# ── Joining her outcomes when they are known (b) ─────────────────────────────


def _rules_of(ws: Workspace) -> KernelRules:
    """The rules her latest run started under, as a fold reads them."""
    start = next((step.entry for step in reversed(ws.history) if isinstance(step.entry, RunStart)), None)
    return FULL_RULES if start is None else start.rules


def _prediction_state(facts: NightFacts, prediction_id: str) -> Workspace | None:
    """Her workspace just after the ``Predict`` event that committed ``prediction_id``.

    Folding stops at that event so a resolution later in the same night, whose Bayes update moves
    the credences (``loop._cycle`` resolves a pending prediction on any cycle), cannot stand in the
    state the prediction was committed with. ``None`` when the night carries no events or the
    prediction was not committed in them; the caller then falls back to ``before``.
    """
    if not facts.events:
        return None
    ws, rules = facts.before, _rules_of(facts.before)
    for event in facts.events:
        entry = event.step.entry
        if isinstance(entry, RunStart):
            rules = entry.rules
        result = kernel.apply(ws, event.step, rules=rules)
        if isinstance(result, Rejected):
            raise ValueError(f"step {event.step.seq} is logged as applied but is rejected ({result.code})")
        ws = result.ws
        if any(p.id == prediction_id for p in ws.predictions):
            return ws
    return None


def _finding_changed(before: Workspace, after: Workspace, question_id: str) -> bool:
    return before.finding_of(question_id) != after.finding_of(question_id)


def _question_moved(facts: NightFacts, question_id: str) -> bool:
    """Whether the night changed the question: a finding, a credence move at ``MOVE_MIN``, a
    prediction committed or resolved, or a status change."""
    before, after = facts.before, facts.after
    if _finding_changed(before, after, question_id):
        return True
    old = {h.id: h.credence for h in before.hypotheses_of(question_id)}
    if any(h.id not in old or abs(h.credence - old[h.id]) >= MOVE_MIN
           for h in after.hypotheses_of(question_id)):
        return True
    if ({(p.id, p.outcome) for p in after.predictions if p.question_id == question_id}
            != {(p.id, p.outcome) for p in before.predictions if p.question_id == question_id}):
        return True  # committed, resolved or censored: the question moved, even with no update
    was, now = before.question(question_id), after.question(question_id)
    if was is None:
        return now is not None  # a question the night created is a move; one absent on both sides is not
    if now is None:
        return True  # a workspace wipe can drop it: a question that was there and is gone moved
    return was.status != now.status


def _new_result(facts: NightFacts, target_id: str | None) -> bool:
    """A result on the target made after the offer: the target log's row for a study that ran."""
    if target_id is None or facts.at is None:
        return False
    offered_at = facts.at
    return any(result.target_id == target_id and result.created_at >= offered_at
               for result in facts.results)


def _target_studied(facts: NightFacts, target_id: str | None) -> bool:
    """A study paid off when it raised a result on its target, or moved a question about one."""
    if _new_result(facts, target_id):
        return True
    return any(facts.after.question(q.id) is not None and target_id in q.target_ids
               and _question_moved(facts, q.id) for q in facts.after.questions)


def decision_outcome(facts: NightFacts) -> bool | None:
    """Whether the pick paid off. ``None`` for a note or silence (no payoff in the log), for a pick
    whose receipt says it never ran (``declined``, ``budget`` or ``failed``: no chance to pay off),
    and for a night with no workspace log whose pick wasn't a study with a result."""
    choice = facts.choice
    if choice is None or choice.want is None or choice.action not in SCORED_ACTIONS:
        return None
    if facts.pick is not None and facts.pick != RAN:
        return None  # the receipt says the study or inquiry never executed: unresolved, not a miss
    if choice.question_id is not None:
        return _question_moved(facts, choice.question_id) if facts.logged else None
    if _target_studied(facts, choice.target_id):
        return True
    return False if facts.logged else None


def hypothesis_outcome(nights: Sequence[NightFacts], from_night: date, question_id: str,
                       hypothesis_id: str) -> tuple[bool | None, date | None]:
    """Whether the credence was right, and the night it settled: the question's final finding
    resolves this hypothesis (True), retracts it (False), or only suspects it (unresolved)."""
    final = nights[-1].after.finding_of(question_id)
    if final is None or final.hypothesis_id != hypothesis_id:
        return None, None
    if final.kind == "resolution":
        outcome = True
    elif final.kind == "retraction":
        outcome = False
    else:
        return None, None
    for facts in nights:
        if facts.night < from_night:
            continue
        found = facts.after.finding_of(question_id)
        if (found is not None and found.id == final.id and found.kind == final.kind
                and found.hypothesis_id == final.hypothesis_id):
            return outcome, facts.night
    return None, None


def prediction_outcome(nights: Sequence[NightFacts], from_night: date, question_id: str,
                       prediction_id: str, p_true: float) -> tuple[bool | None, date | None]:
    """Whether her true/false call was right, and the night the resolver settled it.

    ``p_true`` is her marginal at the ``Predict`` event, the same one her confidence is derived
    from. A call is right when that marginal's side of 0.5 matches the verdict: a committed 0.26
    that resolves ``false`` is a hit, and the same call resolving ``true`` is a miss. ``undetermined``
    and ``censored`` stay unresolved: the resolver may still be asked, and a censored horizon is
    not a verdict.
    """
    called_true = p_true >= 0.5
    for facts in nights:
        if facts.night < from_night:
            continue
        found = next((p for p in facts.after.predictions
                      if p.id == prediction_id and p.question_id == question_id), None)
        if found is None or found.outcome not in ("true", "false"):
            continue
        return called_true == (found.outcome == "true"), facts.night
    return None, None


# ── All of a week's records, calibrated night by night ───────────────────────


def _hit_rate(prior: Sequence[ConfidenceRecord]) -> float | None:
    resolved = [record for record in prior if record.outcome is not None]
    if len(resolved) < RATE_MIN:
        return None
    return sum(1 for record in resolved if record.outcome) / len(resolved)


def _decision_records(facts: NightFacts, past_rate: float | None) -> list[ConfidenceRecord]:
    choice = facts.choice
    if choice is None or choice.want is None or choice.action not in SCORED_ACTIONS:
        return []  # a note or silence: no payoff the log can see, so no guessed outcome
    question_id = choice.question_id
    if question_id is not None and facts.logged:
        # The decision is scored on what she knew when she made it: its own night's move must not
        # raise the confidence it is about to be measured against.
        evidence = evidence_count(facts.before, question_id)
        held = facts.before.hypotheses_of(question_id)
        entropy: float | None = credence_entropy(facts.before, question_id) if len(held) >= 2 else None
        ready = ready_to_conclude(facts.before, question_id)
    elif question_id is not None:
        evidence, entropy, ready = 0, None, False
    else:
        evidence, entropy, ready = facts.study_evidence, None, False
    raw = decision_confidence(evidence, entropy, ready=ready, past_rate=past_rate)
    outcome = decision_outcome(facts)
    return [ConfidenceRecord(
        night=facts.night, kind="decision", ref=choice.want.id, question_id=question_id,
        target_id=choice.target_id, evidence=evidence, entropy=0.0 if entropy is None else entropy,
        raw=raw, calibrated=raw, calibrator="raw", trials=0, outcome=outcome,
        resolved_on=facts.night if outcome is not None else None,
    )]


def _hypothesis_records(facts: NightFacts, nights: Sequence[NightFacts],
                        past_rate: float | None) -> list[ConfidenceRecord]:
    before = {h.id: h.credence for h in facts.before.hypotheses}
    made: list[ConfidenceRecord] = []
    for hypothesis in facts.after.hypotheses:
        old = before.get(hypothesis.id)
        if old is not None and abs(old - hypothesis.credence) < MOVE_MIN:
            continue  # the night didn't set this credence: it was already there
        question_id = hypothesis.question_id
        alternatives = len(facts.after.hypotheses_of(question_id))
        evidence = evidence_count(facts.after, question_id)
        entropy = credence_entropy(facts.after, question_id)
        raw = hypothesis_confidence(hypothesis.credence, alternatives, evidence, entropy, past_rate)
        outcome, resolved_on = hypothesis_outcome(nights, facts.night, question_id, hypothesis.id)
        made.append(ConfidenceRecord(
            night=facts.night, kind="hypothesis", ref=hypothesis.id, question_id=question_id,
            target_id=_first_target(facts.after, question_id), evidence=evidence, entropy=entropy,
            raw=raw, calibrated=raw, calibrator="raw", trials=0, outcome=outcome,
            resolved_on=resolved_on,
        ))
    return made


def _prediction_records(facts: NightFacts, nights: Sequence[NightFacts],
                        past_rate: float | None) -> list[ConfidenceRecord]:
    known = {p.id for p in facts.before.predictions}
    made: list[ConfidenceRecord] = []
    for prediction in facts.after.predictions:
        if prediction.id in known:
            continue
        commit = _prediction_state(facts, prediction.id)
        if commit is None:
            commit = facts.before  # no event slice: before never holds the night's own outcome
        credences = {h.id: h.credence for h in commit.hypotheses_of(prediction.question_id)}
        p_true = math.fsum(value * credences[h] for h, value in prediction.p_true_given.items() if h in credences)
        evidence = evidence_count(commit, prediction.question_id)
        entropy = credence_entropy(commit, prediction.question_id)
        raw = prediction_confidence(p_true, evidence, entropy, past_rate)
        outcome, resolved_on = prediction_outcome(
            nights, facts.night, prediction.question_id, prediction.id, p_true)
        made.append(ConfidenceRecord(
            night=facts.night, kind="prediction", ref=prediction.id, question_id=prediction.question_id,
            target_id=_first_target(facts.after, prediction.question_id), evidence=evidence,
            entropy=entropy, raw=raw, calibrated=raw, calibrator="raw", trials=0, outcome=outcome,
            resolved_on=resolved_on,
        ))
    return made


def _first_target(ws: Workspace, question_id: str) -> str | None:
    question = ws.question(question_id)
    return None if question is None else question.target_ids[0]


def _ordered(nights: Sequence[NightFacts]) -> tuple[NightFacts, ...]:
    ordered = tuple(sorted(nights, key=lambda facts: facts.night))
    days = [facts.night for facts in ordered]
    if len(days) != len(set(days)):
        raise ValueError("a night appears twice; a week is one view per night")
    return ordered


def records(nights: Sequence[NightFacts]) -> tuple[ConfidenceRecord, ...]:
    """Every second-order confidence in ``nights``, with its outcome and its night's calibrator.

    Pass one settles each record's raw confidence from the state it was set in and joins its
    outcome to what the log shows: a decision's payoff on its own night, a credence's resolution
    on the first night that found it, a prediction's verdict. Pass two walks the nights in order
    and reads each night's records with a calibrator fitted only on records resolved before it,
    so a night is never scored with knowledge of its own outcome.
    """
    ordered = _ordered(nights)
    pending: list[ConfidenceRecord] = []
    pool: list[ConfidenceRecord] = []
    for facts in ordered:
        prior = [record for record in pool
                 if record.resolved_on is not None and record.resolved_on < facts.night]
        rates: Mapping[Kind, float | None] = {
            kind: _hit_rate([record for record in prior if record.kind == kind]) for kind in KINDS
        }
        night_records = _decision_records(facts, rates["decision"])
        if facts.logged:
            night_records += _hypothesis_records(facts, ordered, rates["hypothesis"])
            night_records += _prediction_records(facts, ordered, rates["prediction"])
        pending.extend(night_records)
        pool.extend(night_records)
    out: list[ConfidenceRecord] = []
    for night in sorted({record.night for record in pending}):
        prior = [record for record in pending
                 if record.resolved_on is not None and record.resolved_on < night]
        calibrated: dict[Kind, Calibrator] = {
            kind: fit_calibrator([(record.raw, bool(record.outcome)) for record in prior
                                  if record.kind == kind and record.outcome is not None])
            for kind in KINDS
        }
        for record in pending:
            if record.night != night:
                continue
            calibrator = calibrated[record.kind]
            out.append(replace(record, calibrated=calibrator.calibrate(record.raw),
                               calibrator=calibrator.kind, trials=calibrator.n))
    return tuple(out)
