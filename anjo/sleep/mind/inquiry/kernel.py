"""MIND's kernel: the one transition of her workspace, its fold and its digest (Task 17). Pure.

``apply`` takes a logged step. An operation step without a code is checked (``check``) against
K1-K4, hypothesize-once and the domain denylist, then applied, or returned as ``Rejected(code)``;
every other step is recorded with its effect. ``fold`` replays a log from ``Workspace()`` and
``digest`` hashes the canonical workspace, history included (K6), so a kill loses nothing logged.

- **K1, Bayes over the whole set.** Priors and every ``p_true_given`` are clamped to
  ``PROBABILITY_CLAMP``; likelihoods are never clipped and posteriors are normalized without
  clipping. Two sources, kept apart as ``Update.source``: a resolved prediction (L = p or 1 - p)
  and an assessed item (a bucket from ``ASSESS_BUCKETS``, on an item a search or verified research
  returned for the question). One item updates a question once, by either route. An answer to a
  ``world_query`` is never an item, so it never moves credence.
- **K2:** a prediction's clamped ``p_true_given`` spans at least ``MIN_SPREAD``.
- **K3:** a contradiction cites two items from different days; any other question cites one.
- **K4:** a resolution reports the leading hypothesis at ``RESOLVE_MIN`` or more, with an update
  from outside the question's origin: a resolved prediction, or a search hit returned after the
  question was hypothesized.
- **K5** is the controller's; **K6** is every event's before and after digest.
- **Hypothesize once:** a second set only splits the catch-all while it holds more than
  ``SPLIT_MIN``.
- **A prediction's horizon** is 1 to ``MAX_HORIZON_DAYS`` days; anything longer is ``malformed``.
- **The domain rule:** ``DENYLIST`` holds person-diagnosis terms (traits, knowledge, health,
  feelings, relationships), checked on what she would investigate: a question's subject, a
  hypothesis, a prediction's claim and a tool's query. The LLM screen behind it is the loop's.
  Each version of the list is kept in ``DENYLISTS``; a ``RunStart`` records the version its run
  checked under, and ``check`` reads it from the workspace, so a fold admits every logged
  operation under the terms it was admitted with.
- **A question's life:** a parked question goes live again on a ``Wake`` (new evidence on one of
  its targets). A concluded one stays ``WATCHED``: it still takes searches, assessments and
  resolved predictions, and a ``RETRACT``, so a resolution can become a retraction.

Under ``NO_KERNEL`` (M-kernel) K1, K2 and K4 are off: priors and predictions aren't clamped, an
assessment's buckets, retrieval and reuse aren't checked, ``ASSESS`` may write credences directly,
and a resolved prediction moves nothing.

**Ids** derive from the op counter (``Workspace.ops``, every applied operation and edit, never a
``RunStart`` or an observation), numbering the operation that creates them from 1: question
``q{n}``, prediction ``p{n}``, finding ``f{n}``, and a set's alternatives ``{question}.h{n}a`` to
``d``. So a resumed run assigns the ids the uninterrupted one would, and ``records_digest`` (the
records without the run history) is the same for both.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Final

from anjo.sleep.mind.inquiry.codec import canonical, records_document, workspace_document
from anjo.sleep.mind.inquiry.types import (
    ASSESS_BUCKETS,
    CATCH_ALL,
    FULL_RULES,
    LIVE_STATUSES,
    MAX_ALTERNATIVES,
    MIN_ALTERNATIVES,
    MIN_SPREAD,
    PROBABILITY_CLAMP,
    RESOLVE_MIN,
    RETRACT_BELOW,
    SPLIT_MIN,
    TOOLS,
    AbandonOp,
    Answered,
    Applied,
    AssessOp,
    ConcludeOp,
    Edit,
    EvidenceRef,
    Finding,
    Hypothesis,
    HypothesizeOp,
    InvestigateOp,
    KernelRules,
    NoticeOp,
    Op,
    ParkOp,
    Prediction,
    PredictOp,
    Question,
    QuestionStatus,
    RejectCode,
    Rejected,
    Resolved,
    RetractOp,
    Retrieved,
    RunStart,
    Step,
    Update,
    Wake,
    Workspace,
    WorkspaceEvent,
    catch_all_id,
)

_EPS: Final = 1e-9
_LETTERS: Final = "abcd"  # a set's alternatives, at most MAX_ALTERNATIVES
HIT_TOOLS: Final = frozenset({"evidence_search", "research"})  # what returns items for a question
MAX_HORIZON_DAYS: Final = 30
# A concluded question still takes evidence, so its resolution can be retracted.
WATCHED: Final[frozenset[str]] = LIVE_STATUSES | {"concluded"}
_AFTER_RESOLUTION: Final = (InvestigateOp, AssessOp, RetractOp)  # what a concluded question still takes

# ── The domain rule: data and a pure check ────────────────────────────────────
# Each term matches at a word start, case-insensitively; a term ending in "*" matches any ending.
# They name the person's traits, knowledge, health, feelings and relationships, so a question,
# hypothesis, claim or query that diagnoses the person never stands. It limits what she
# investigates, not what she says: findings and retractions aren't checked. A change to the terms
# is a new version in DENYLISTS, never an edit of a recorded one.
_FIRST_VERSION: Final = "denylist-v1"  # what a RunStart without a recorded version checked under
_DENYLIST_V1: Final[Mapping[str, tuple[str, ...]]] = {
    "traits": (
        "personality", "her character", "his character", "their character", "narcissis*", "lazy",
        "laziness", "careless*", "naive*", "incompeten*", "immatur*", "insecure", "perfectionis*",
        "procrastinat*", "stubborn*", "selfish*", "irresponsib*", "disorganized",
    ),
    "knowledge": (
        "doesn't know", "does not know", "they don't know", "they do not know", "doesn't understand",
        "does not understand", "they don't understand", "doesn't realize", "doesn't realise",
        "hasn't realized", "hasn't realised", "is unaware", "are unaware", "ignoran*", "clueless",
    ),
    "health": (
        "depress*", "anxiety", "anxious*", "adhd", "autis*", "bipolar", "ocd", "ptsd", "trauma*",
        "mental health", "mental illness", "illness", "disorder*", "diagnos*", "therap*",
        "medication*", "burnout", "burned out", "burnt out", "insomnia", "addict*", "suicid*",
        "self-harm", "panic attack*",
    ),
    "feelings": (
        "lonely", "loneliness", "jealous*", "ashamed", "afraid of", "scared of", "frightened of",
        "unhappy", "resent*", "self-esteem", "self-worth", "insecurity", "she feels", "he feels",
        "they feel", "she is feeling", "he is feeling", "they are feeling", "overwhelmed",
        "stressed out",
    ),
    "relationships": (
        "boyfriend*", "girlfriend*", "her partner", "his partner", "their partner", "her relationship",
        "his relationship", "their relationship", "marriage", "divorce*", "dating", "breakup",
        "break up", "broke up", "affair", "sex life",
    ),
}
DENYLISTS: Final[Mapping[str, Mapping[str, tuple[str, ...]]]] = {_FIRST_VERSION: _DENYLIST_V1}
DENYLIST_VERSION: Final = _FIRST_VERSION  # what a new run records and checks under
DENYLIST: Final = DENYLISTS[DENYLIST_VERSION]


def _term(term: str) -> str:
    stem = term[:-1] if term.endswith("*") else term
    tail = r"\w*" if term.endswith("*") else r"\b"
    return r"\b" + r"\s+".join(re.escape(word) for word in stem.split()) + tail


def _pattern(version: str) -> re.Pattern[str]:
    """The version's terms as one pattern (``re`` caches the compile); ``ValueError`` for a
    version this kernel doesn't hold."""
    terms = DENYLISTS.get(version)
    if terms is None:
        raise ValueError(f"the log names denylist version {version!r}, which this kernel doesn't hold")
    return re.compile("|".join(_term(term) for group in terms.values() for term in group), re.IGNORECASE)


def denied(text: str, version: str | None = None) -> bool:
    """Whether ``text`` carries a person-diagnosis term from the denylist ``version`` (the
    current one by default)."""
    return _pattern(version or DENYLIST_VERSION).search(text.replace("’", "'")) is not None


def denylist_version(ws: Workspace) -> str:
    """The version the workspace's latest run recorded; the current one before any run."""
    start = next((step.entry for step in reversed(ws.history) if isinstance(step.entry, RunStart)), None)
    if start is None:
        return DENYLIST_VERSION
    return start.denylist or _FIRST_VERSION


# ── What the workspace holds ──────────────────────────────────────────────────


def known_items(ws: Workspace) -> dict[str, EvidenceRef]:
    """Every item that reached her since the workspace began or was wiped: the night's new
    evidence, search and research hits, and what the resolver was shown."""
    known: dict[str, EvidenceRef] = {}
    for step in ws.history:
        entry = step.entry
        items = entry.items if isinstance(entry, Retrieved) else entry.seen if isinstance(entry, Resolved) else ()
        for item in items:
            known.setdefault(item.id, item)
    return known


def hits(ws: Workspace, question_id: str, *, after: int = -1) -> tuple[EvidenceRef, ...]:
    """Items a search or verified research returned for the question at a step after ``after``,
    each once, in the order they came."""
    seen: dict[str, EvidenceRef] = {}
    for step in ws.history:
        entry = step.entry
        if (
            step.seq > after and isinstance(entry, Retrieved) and entry.question_id == question_id
            and entry.tool in HIT_TOOLS
        ):
            for item in entry.items:
                seen.setdefault(item.id, item)
    return tuple(seen.values())


def research_sources(ws: Workspace, question_id: str) -> frozenset[str]:
    """Verified research sources returned for the question."""
    return frozenset(
        item.id for step in ws.history
        if isinstance(step.entry, Retrieved) and step.entry.question_id == question_id
        and step.entry.tool == "research"
        for item in step.entry.items
    )


def used_items(ws: Workspace, question_id: str) -> frozenset[str]:
    """Items that already updated the question, by either route (K1)."""
    return frozenset(i for u in ws.updates if u.question_id == question_id for i in u.evidence_ids)


def leader(hypotheses: Sequence[Hypothesis]) -> Hypothesis:
    """The hypothesis with the top credence; a tie goes to the earlier one in the set."""
    return max(hypotheses, key=lambda h: h.credence)


def hypothesized_at(ws: Workspace, question_id: str) -> int | None:
    """The seq of the step that first set the question's hypotheses, if it is in the history."""
    return next(
        (
            step.seq for step in ws.history
            if step.applied and isinstance(step.entry, HypothesizeOp) and step.entry.question_id == question_id
        ),
        None,
    )


def outside_origin(ws: Workspace, question: Question) -> bool:
    """K4's evidence: an update from outside the question's origin, from a resolved prediction
    or from a search hit returned after the question was hypothesized."""
    origin = frozenset(question.origin_evidence)
    start = hypothesized_at(ws, question.id)
    later = frozenset(item.id for item in hits(ws, question.id, after=start)) if start is not None else frozenset()
    for update in ws.updates:
        if update.question_id != question.id:
            continue
        outside = frozenset(update.evidence_ids) - origin
        if update.source == "resolved_prediction" and outside:
            return True
        if update.source == "assessed_evidence" and outside & later:
            return True
    return False


def wakes(ws: Workspace, items: Sequence[EvidenceRef]) -> tuple[Wake, ...]:
    """A ``Wake`` for each parked question that one of ``items`` names a target of."""
    out: list[Wake] = []
    for question in ws.questions:
        if question.status != "parked":
            continue
        held = frozenset(question.target_ids)
        on_target = tuple(item for item in items if item.target_id in held)
        if on_target:
            out.append(Wake(question.id, on_target))
    return tuple(out)


def records_digest(ws: Workspace) -> str:
    """The SHA-256 hex digest of the workspace's records (questions, hypotheses, predictions,
    updates, findings) without its run history: what the person's digest covers (Task 3's
    ``replay_digest``), equal for a killed-and-resumed run and an uninterrupted one."""
    return hashlib.sha256(canonical(records_document(ws)).encode("ascii")).hexdigest()


def resolution_ready(ws: Workspace, question_id: str) -> bool:
    """K4 holds for the question's leading hypothesis."""
    question = ws.question(question_id)
    hypotheses = ws.hypotheses_of(question_id)
    if question is None or question.status not in LIVE_STATUSES or not hypotheses:
        return False
    return leader(hypotheses).credence >= RESOLVE_MIN and outside_origin(ws, question)


def retraction_ready(ws: Workspace, question_id: str) -> bool:
    """The question's reported hypothesis has fallen below ``RETRACT_BELOW``, a resolution's
    included."""
    question = ws.question(question_id)
    finding = ws.finding_of(question_id)
    if question is None or question.status not in WATCHED or finding is None:
        return False
    if finding.kind == "retraction" or finding.hypothesis_id is None:
        return False
    reported = next(h for h in ws.hypotheses_of(question_id) if h.id == finding.hypothesis_id)
    return reported.credence < RETRACT_BELOW


def window(prediction: Prediction, as_of: datetime) -> tuple[datetime, datetime]:
    """The times a prediction's resolving evidence falls in: after it was committed, up to its
    horizon or ``as_of``, whichever comes first."""
    return prediction.committed_at, min(prediction.horizon, as_of)


def resolvable(ws: Workspace, prediction: Prediction, as_of: datetime) -> tuple[EvidenceRef, ...]:
    """Every item she holds inside the prediction's window: what the resolver may be shown."""
    start, end = window(prediction, as_of)
    return tuple(item for item in known_items(ws).values() if start < item.created_at <= end)


def unseen(ws: Workspace, prediction: Prediction, as_of: datetime) -> tuple[EvidenceRef, ...]:
    """Items in the window that the resolver hasn't been shown for this prediction."""
    shown = frozenset(prediction.resolved_by)
    return tuple(item for item in resolvable(ws, prediction, as_of) if item.id not in shown)


# ── K1's arithmetic ───────────────────────────────────────────────────────────


def clamp(p: float) -> float:
    low, high = PROBABILITY_CLAMP
    return min(max(p, low), high)


def _normalized(values: Sequence[float]) -> list[float]:
    total = math.fsum(values)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def bayes(hypotheses: Sequence[Hypothesis], likelihoods: Mapping[str, float]) -> tuple[Hypothesis, ...]:
    """p'(H) proportional to p(H) L(E|H) over the whole set, normalized without clipping."""
    posterior = _normalized([h.credence * likelihoods[h.id] for h in hypotheses])
    return tuple(replace(h, credence=p) for h, p in zip(hypotheses, posterior, strict=True))


# ── Checks ────────────────────────────────────────────────────────────────────


def _live(ws: Workspace, op: Op) -> RejectCode | None:
    """``not_live`` unless the op's question is live, or concluded for what a concluded question
    still takes (``_AFTER_RESOLUTION``)."""
    question = None if isinstance(op, NoticeOp) else ws.question(op.question_id)
    if question is None:
        return "unknown_ref"
    allowed = WATCHED if isinstance(op, _AFTER_RESOLUTION) else LIVE_STATUSES
    return None if question.status in allowed else "not_live"


def _unknown(ws: Workspace, ids: Iterable[str]) -> bool:
    known = known_items(ws)
    return any(item not in known for item in ids)


def _k3(kind: str, items: Sequence[EvidenceRef]) -> bool:
    if kind == "contradiction":
        return len({item.id for item in items}) >= 2 and len({item.day for item in items}) >= 2
    return len(items) >= 1


def _check_notice(
    ws: Workspace, op: NoticeOp, target_ids: Collection[str] | None, version: str,
) -> RejectCode | None:
    if not op.target_ids:
        return "no_targets"
    if target_ids is not None and any(target not in target_ids for target in op.target_ids):
        return "unknown_ref"
    known = known_items(ws)
    if any(item not in known for item in op.origin_evidence):
        return "unknown_ref"
    if not _k3(op.kind, [known[item] for item in op.origin_evidence]):
        return "k3_citations"
    return "domain_denylist" if denied(op.subject, version) else None


def _check_hypothesize(ws: Workspace, op: HypothesizeOp, version: str) -> RejectCode | None:
    held = ws.hypotheses_of(op.question_id)
    if not held:
        if not MIN_ALTERNATIVES <= len(op.hypotheses) <= MAX_ALTERNATIVES:
            return "hypothesis_count"
    else:
        catch_all = next(h for h in held if h.is_catch_all)
        if catch_all.credence <= SPLIT_MIN:
            return "hypothesize_once"
        if not op.hypotheses or len(held) - 1 + len(op.hypotheses) > MAX_ALTERNATIVES:
            return "hypothesis_count"
    if any(denied(alt.statement, version) for alt in op.hypotheses):
        return "domain_denylist"
    return None


def _set_ids(ws: Workspace, question_id: str) -> frozenset[str]:
    return frozenset(h.id for h in ws.hypotheses_of(question_id))


def _check_predict(ws: Workspace, op: PredictOp, rules: KernelRules, version: str) -> RejectCode | None:
    if op.horizon_days > MAX_HORIZON_DAYS:
        return "malformed"
    ids = _set_ids(ws, op.question_id)
    if not ids or set(op.p_true_given) != ids:
        return "k1_hypotheses"
    values = [clamp(p) if rules.k1 else p for p in op.p_true_given.values()]
    if rules.k2 and max(values) - min(values) < MIN_SPREAD - _EPS:
        return "k2_not_discriminating"
    return "domain_denylist" if denied(op.claim, version) else None


def _check_assess(ws: Workspace, op: AssessOp, rules: KernelRules) -> RejectCode | None:
    ids = _set_ids(ws, op.question_id)
    if not ids:
        return "k1_hypotheses"
    if _unknown(ws, [op.evidence_id]):
        return "unknown_ref"
    if not rules.k1:
        values = op.likelihoods if op.likelihoods is not None else op.credences
        return None if values is not None and set(values) == ids else "k1_hypotheses"
    if op.credences is not None or op.likelihoods is None:
        return "k1_direct_credence"
    if set(op.likelihoods) != ids:
        return "k1_hypotheses"
    if any(value not in ASSESS_BUCKETS for value in op.likelihoods.values()):
        return "k1_bucket"
    if op.evidence_id not in {item.id for item in hits(ws, op.question_id)}:
        return "k1_not_returned"
    return "k1_evidence_used" if op.evidence_id in used_items(ws, op.question_id) else None


def _check_conclude(ws: Workspace, op: ConcludeOp, rules: KernelRules) -> RejectCode | None:
    hypotheses = ws.hypotheses_of(op.question_id)
    if op.hypothesis_id not in {h.id for h in hypotheses} or _unknown(ws, op.cites):
        return "unknown_ref"
    if op.kind == "resolution" and rules.k4:
        if leader(hypotheses).id != op.hypothesis_id or not resolution_ready(ws, op.question_id):
            return "k4_not_ready"
    return None


def check(
    ws: Workspace,
    op: Op,
    *,
    rules: KernelRules = FULL_RULES,
    target_ids: Collection[str] | None = None,
    tools: Collection[str] = TOOLS,
) -> RejectCode | None:
    """Why ``op`` can't apply to ``ws``, or ``None``. ``target_ids`` are the targets the run holds
    (``None``: not checked, as in a fold) and ``tools`` the tools it has."""
    version = denylist_version(ws)
    if isinstance(op, NoticeOp):
        return _check_notice(ws, op, target_ids, version)
    code = _live(ws, op)
    if code is not None:
        return code
    if isinstance(op, HypothesizeOp):
        return _check_hypothesize(ws, op, version)
    if isinstance(op, PredictOp):
        return _check_predict(ws, op, rules, version)
    if isinstance(op, InvestigateOp):
        if op.tool not in tools:
            return "unavailable"
        return "domain_denylist" if denied(op.query, version) else None
    if isinstance(op, AssessOp):
        return _check_assess(ws, op, rules)
    if isinstance(op, ConcludeOp):
        return _check_conclude(ws, op, rules)
    if isinstance(op, RetractOp):
        return "unknown_ref" if _unknown(ws, op.cites) else None
    return None  # PARK and ABANDON need only a live question, RETRACT a live or concluded one


# ── Effects ───────────────────────────────────────────────────────────────────


def _with_status(ws: Workspace, question_id: str, status: QuestionStatus) -> Workspace:
    return replace(ws, questions=tuple(
        replace(q, status=status) if q.id == question_id else q for q in ws.questions
    ))


def _with_set(ws: Workspace, question_id: str, hypotheses: Sequence[Hypothesis]) -> Workspace:
    others = tuple(h for h in ws.hypotheses if h.question_id != question_id)
    return replace(ws, hypotheses=(*others, *hypotheses))


def _with_finding(ws: Workspace, finding: Finding) -> Workspace:
    others = tuple(f for f in ws.findings if f.question_id != finding.question_id)
    return replace(ws, findings=(*others, finding))


def _alternatives(question_id: str, seq: int, statements: Sequence[str], credences: Sequence[float]) -> list[Hypothesis]:
    return [
        Hypothesis(id=f"{question_id}.h{seq}{_LETTERS[n]}", question_id=question_id, statement=s, credence=c)
        for n, (s, c) in enumerate(zip(statements, credences, strict=True))
    ]


def _inherited(ws: Workspace, question_id: str, new_ids: Sequence[str]) -> tuple[Prediction, ...]:
    """Each new hypothesis takes the catch-all's committed value in every prediction."""
    catch = catch_all_id(question_id)
    return tuple(
        replace(p, p_true_given={**p.p_true_given, **{h: p.p_true_given[catch] for h in new_ids}})
        if p.question_id == question_id else p
        for p in ws.predictions
    )


def _with_split(ws: Workspace, question_id: str, hypotheses: Sequence[Hypothesis], new_ids: Sequence[str]) -> Workspace:
    """The question's set with new alternatives, and every prediction naming them, in one step."""
    others = tuple(h for h in ws.hypotheses if h.question_id != question_id)
    return replace(ws, hypotheses=(*others, *hypotheses), predictions=_inherited(ws, question_id, new_ids))


def _hypothesize(ws: Workspace, op: HypothesizeOp, n: int, rules: KernelRules) -> Workspace:
    statements = [alt.statement for alt in op.hypotheses]
    priors = [alt.credence for alt in op.hypotheses] + [op.catch_all]
    priors = [clamp(p) for p in priors] if rules.k1 else priors
    held = ws.hypotheses_of(op.question_id)
    if not held:
        *credences, rest = _normalized(priors)
        catch = Hypothesis(catch_all_id(op.question_id), op.question_id, CATCH_ALL, rest)
        fresh = _alternatives(op.question_id, n, statements, credences)
        return _with_status(_with_set(ws, op.question_id, [*fresh, catch]), op.question_id, "active")
    old_catch = next(h for h in held if h.is_catch_all)
    *credences, rest = [share * old_catch.credence for share in _normalized(priors)]
    fresh = _alternatives(op.question_id, n, statements, credences)
    kept = [h for h in held if not h.is_catch_all]
    return _with_split(ws, op.question_id, [*kept, *fresh, replace(old_catch, credence=rest)], [h.id for h in fresh])


def _predict(ws: Workspace, op: PredictOp, step: Step, n: int, rules: KernelRules) -> Workspace:
    p_true_given = {h: clamp(p) if rules.k1 else p for h, p in op.p_true_given.items()}
    prediction = Prediction(
        id=f"p{n}", question_id=op.question_id, claim=op.claim, p_true_given=p_true_given,
        committed_at=step.at, horizon=step.at + timedelta(days=op.horizon_days), outcome=None,
        resolved_by=(),
    )
    return replace(ws, predictions=(*ws.predictions, prediction))


def _assess(ws: Workspace, op: AssessOp) -> Workspace:
    held = ws.hypotheses_of(op.question_id)
    if op.likelihoods is None:  # M-kernel writes the vector directly
        credences = op.credences or {}
        written = _normalized([credences[h.id] for h in held])
        return _with_set(ws, op.question_id, [replace(h, credence=c) for h, c in zip(held, written, strict=True)])
    update = Update(
        question_id=op.question_id, ref=op.evidence_id, source="assessed_evidence",
        evidence_ids=(op.evidence_id,), likelihoods=op.likelihoods,
    )
    ws = _with_set(ws, op.question_id, bayes(held, op.likelihoods))
    return replace(ws, updates=(*ws.updates, update))


def _conclude(ws: Workspace, op: ConcludeOp, n: int) -> Workspace:
    reported = next(h for h in ws.hypotheses_of(op.question_id) if h.id == op.hypothesis_id)
    existing = ws.finding_of(op.question_id)
    finding = Finding(
        id=existing.id if existing is not None else f"f{n}", question_id=op.question_id,
        statement=op.statement, p=reported.credence, cites=op.cites, kind=op.kind, hypothesis_id=op.hypothesis_id,
    )
    ws = _with_finding(ws, finding)
    return _with_status(ws, op.question_id, "concluded") if op.kind == "resolution" else ws


def _retract(ws: Workspace, op: RetractOp) -> Workspace:
    existing = ws.finding_of(op.question_id)
    if existing is not None:
        credences = {h.id: h.credence for h in ws.hypotheses_of(op.question_id)}
        p = credences.get(existing.hypothesis_id, existing.p) if existing.hypothesis_id else existing.p
        ws = _with_finding(ws, replace(existing, kind="retraction", statement=op.statement, cites=op.cites, p=p))
    return _with_status(ws, op.question_id, "retracted")


def _effect(ws: Workspace, op: Op, step: Step, rules: KernelRules) -> Workspace:
    """An applied operation's effect; ``n`` numbers it, from 1, for the ids it creates."""
    n = ws.ops + 1
    if isinstance(op, NoticeOp):
        question = Question(
            id=f"q{n}", target_ids=op.target_ids, kind=op.kind, origin_evidence=op.origin_evidence,
            status="open", opened_at=step.at, subject=op.subject,
        )
        return replace(ws, questions=(*ws.questions, question))
    if isinstance(op, HypothesizeOp):
        return _hypothesize(ws, op, n, rules)
    if isinstance(op, PredictOp):
        return _predict(ws, op, step, n, rules)
    if isinstance(op, AssessOp):
        return _assess(ws, op)
    if isinstance(op, ConcludeOp):
        return _conclude(ws, op, n)
    if isinstance(op, RetractOp):
        return _retract(ws, op)
    if isinstance(op, ParkOp):
        return _with_status(ws, op.question_id, "parked")
    if isinstance(op, AbandonOp):
        return _with_status(ws, op.question_id, "abandoned")
    return ws  # INVESTIGATE: its tool's result is the next step


def _resolved(ws: Workspace, entry: Resolved, rules: KernelRules) -> Workspace:
    prediction = next((p for p in ws.predictions if p.id == entry.prediction_id), None)
    if prediction is None or not prediction.pending:
        raise ValueError(f"prediction {entry.prediction_id} isn't pending")
    seen = tuple(item.id for item in entry.seen)
    settled = replace(prediction, outcome=entry.outcome) if entry.outcome == "censored" else replace(
        prediction, outcome=entry.outcome, resolved_by=seen,
    )
    ws = replace(ws, predictions=tuple(settled if p.id == prediction.id else p for p in ws.predictions))
    question_id = prediction.question_id
    moves = (
        entry.outcome in ("true", "false") and rules.k1 and bool(seen)
        and _watched(ws, question_id) and not frozenset(seen) & used_items(ws, question_id)
    )
    if not moves:
        return ws  # recorded; already-used evidence, or no verdict, applies nothing
    likelihoods = {h: p if entry.outcome == "true" else 1.0 - p for h, p in prediction.p_true_given.items()}
    update = Update(question_id, prediction.id, "resolved_prediction", seen, likelihoods)
    ws = _with_set(ws, question_id, bayes(ws.hypotheses_of(question_id), likelihoods))
    return replace(ws, updates=(*ws.updates, update))


def _watched(ws: Workspace, question_id: str) -> bool:
    question = ws.question(question_id)
    return question is not None and question.status in WATCHED


def _wake(ws: Workspace, entry: Wake) -> Workspace:
    question = ws.question(entry.question_id)
    if question is None or question.status != "parked":
        raise ValueError(f"a wake names question {entry.question_id}, which isn't parked")
    if any(item.target_id not in question.target_ids for item in entry.items):
        raise ValueError(f"a wake of {entry.question_id} names an item on none of its targets")
    return _with_status(ws, question.id, "active" if ws.hypotheses_of(question.id) else "open")


def _observe(ws: Workspace, entry: Retrieved | Answered | Resolved, rules: KernelRules) -> Workspace:
    if isinstance(entry, Resolved):
        return _resolved(ws, entry, rules)
    if entry.question_id is not None and ws.question(entry.question_id) is None:
        raise ValueError(f"an observation names question {entry.question_id}, which isn't held")
    return ws


# ── Edits (the intervention test) ─────────────────────────────────────────────


def _edit(ws: Workspace, edit: Edit, n: int) -> Workspace:
    held = ws.hypotheses_of(edit.question_id)
    if not held:
        raise ValueError(f"question {edit.question_id} has no hypotheses to edit")
    if edit.kind == "swap_lead":
        lead = leader(held)
        other = next((h for h in held if h.id == edit.hypothesis_id), None)
        if other is None or other.id == lead.id:
            raise ValueError("swap_lead names a hypothesis in the set other than the leader")
        swapped = {lead.id: other.credence, other.id: lead.credence}
        return _with_set(ws, edit.question_id, [replace(h, credence=swapped.get(h.id, h.credence)) for h in held])
    if edit.kind == "delete_prediction":
        target = next((p for p in ws.predictions if p.id == edit.prediction_id), None)
        if target is None or target.question_id != edit.question_id or not target.pending:
            raise ValueError("delete_prediction names a pending prediction on the question")
        return replace(ws, predictions=tuple(p for p in ws.predictions if p.id != target.id))
    if len(held) - 1 >= MAX_ALTERNATIVES:
        raise ValueError("inject_hypothesis needs a set with room for one more alternative")
    statement = edit.statement or ""
    halved = [replace(h, credence=h.credence / 2) for h in held]
    fresh = _alternatives(edit.question_id, n, [statement], [0.5])
    kept, catch = [h for h in halved if not h.is_catch_all], [h for h in halved if h.is_catch_all]
    return _with_split(ws, edit.question_id, [*kept, *fresh, *catch], [h.id for h in fresh])


# ── The transition, the fold and the digest ───────────────────────────────────


def digest(ws: Workspace) -> str:
    """The SHA-256 hex digest of the canonical workspace, history included (K6)."""
    return hashlib.sha256(canonical(workspace_document(ws)).encode("ascii")).hexdigest()


def _next(ws: Workspace, step: Step, rules: KernelRules) -> Workspace:
    entry = step.entry
    if isinstance(entry, RunStart):
        return Workspace(history=(step,), ops=ws.ops) if entry.wipe else replace(ws, history=(*ws.history, step))
    if isinstance(entry, Edit):
        ws = replace(_edit(ws, entry, ws.ops + 1), ops=ws.ops + 1)
    elif isinstance(entry, Wake):
        ws = _wake(ws, entry)
    elif isinstance(entry, (Retrieved, Answered, Resolved)):
        ws = _observe(ws, entry, rules)
    elif entry is not None and step.code is None:
        ws = replace(_effect(ws, entry, step, rules), ops=ws.ops + 1)
    return replace(ws, history=(*ws.history, step))


_MARKS_AND_OBSERVATIONS: Final = (Retrieved, Answered, Resolved, RunStart, Edit, Wake)


def apply(ws: Workspace, step: Step, *, rules: KernelRules = FULL_RULES) -> Applied | Rejected:
    """The workspace after ``step``, and the event recording it; ``Rejected(code)`` when an
    operation step without a code can't apply. ``ValueError`` when ``step.seq`` isn't ``ws.seq``,
    or an edit or observation can't apply."""
    if step.seq != ws.seq:
        raise ValueError(f"step {step.seq} where {ws.seq} comes next")
    if step.applied and step.entry is not None and not isinstance(step.entry, _MARKS_AND_OBSERVATIONS):
        code = check(ws, step.entry, rules=rules)
        if code is not None:
            return Rejected(code)
    after = _next(ws, step, rules)
    return Applied(ws=after, event=WorkspaceEvent(step=step, before=digest(ws), after=digest(after)))


def fold(events: Iterable[WorkspaceEvent]) -> Workspace:
    """The workspace a log rebuilds: each step applied under the latest ``RunStart``'s rules and
    denylist version. ``ValueError`` when an event's digests don't match, a step logged as applied
    is rejected, or a run names a denylist version this kernel doesn't hold."""
    ws = Workspace()
    rules = FULL_RULES
    for event in events:
        if event.before != digest(ws):
            raise ValueError(f"event {event.step.seq}'s before digest isn't the workspace's")
        if isinstance(event.step.entry, RunStart):
            rules = event.step.entry.rules
        result = apply(ws, event.step, rules=rules)
        if isinstance(result, Rejected):
            raise ValueError(f"step {event.step.seq} is logged as applied but is rejected ({result.code})")
        if result.event.after != event.after:
            raise ValueError(f"event {event.step.seq}'s after digest isn't the one its step produces")
        ws = result.ws
    return ws
