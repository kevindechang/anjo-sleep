"""Does her confidence know when she is right? The type-2 sensitivity report over a results tree.

Fleming & Lau (2014, *How to measure metacognition*) separate first-order sensitivity (is she
right?) from second-order sensitivity (does her confidence say so?) and measure the second with a
type-2 AUROC, meta-d'/d' and calibration, never with self-report. ``mind.metacog`` records a
second-order confidence for every decision the chooser makes and every credence it sets, and joins
each to its outcome when the night's log shows one. This module scores those records:

- **AUROC2:** the probability that a resolved record that paid off carries a higher confidence than
  one that did not, ties at half. 1.0 is a confidence that always knows; 0.5 is a coin.
- **Brier:** the mean squared distance from the confidence to the outcome, for the raw heuristic
  and the nightly calibrated value, so calibration's gain is a number rather than a claim.
- **Calibration bins:** ten equal-width bins of the calibrated confidence with the share that paid
  off in each, so a reader sees where the confidence is high and the outcomes are not.
- **meta-d'/d':** only when there are enough trials with both outcomes and both first-order
  responses. ``d'`` comes from her first-order calls (a confidence at or above chance is a
  "payoff" call) with the log-linear correction; ``meta-d'`` is the equal-variance Gaussian
  sensitivity implied by her type-2 AUROC, ``sqrt(2) * probit(AUROC2)``. A ratio near 1 means her
  confidence carries exactly the evidence her call does, above 1 that it adds some, and below 1
  that it loses some. Below ``TRIALS_MIN`` resolved records, or without ``CLASS_MIN`` in each
  outcome class or on each side of chance, it is reported as insufficient with the reason.

Reads ``<root>/<user>/``: her plaintext workspace log (``workspace.jsonl``, the synthetic lives'
tree, never the app's encrypted stores) and each night's ``nights/<date>/mind_points.jsonl`` for
the chooser's pick and the results. Offline: no port, model, clock or environment read. The root
defaults to ``$ANJO_DATA_DIR/sleep/results``.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import NormalDist
from typing import Any, Final

from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.mind.inquiry.log import JsonlWorkspaceLog
from anjo.sleep.mind.inquiry.types import Workspace, WorkspaceEvent
from anjo.sleep.mind.metacog import (
    BINS,
    KINDS,
    MIN_TRIALS,
    PICK_OUTCOMES,
    ConfidenceRecord,
    NightFacts,
    night_slices,
    records,
)
from anjo.sleep.mind.points import DecisionPoint, point_from_record
from anjo.sleep.night.nightlog import NightLog
from anjo.sleep.reports.results import ResultsError, results_root
from anjo.sleep.types import Result

WORKSPACE: Final = "workspace.jsonl"  # her plaintext log in a synthetic results tree
NIGHTS: Final = "nights"
PRE_STUDY: Final = "pre_study"
CHANCE: Final = 0.5  # her first-order call: a confidence at or above it says the thing paid off
TRIALS_MIN: Final = 20  # resolved records before meta-d'/d' is reported at all
CLASS_MIN: Final = 5  # records in each outcome class, and on each side of chance
PROBIT_FLOOR: Final = 1e-3  # keeps a perfect or empty ranking from an infinite sensitivity
_NORMAL: Final = NormalDist()


def _probit(p: float) -> float:
    return _NORMAL.inv_cdf(p)


# ── The measures ─────────────────────────────────────────────────────────────


def auroc2(pairs: Sequence[tuple[float, bool]]) -> float | None:
    """The type-2 AUROC over (confidence, outcome) pairs, ties counted at half.

    ``None`` when an outcome class is empty: a ranking needs both a record that paid off and one
    that did not, and an absent class is not a perfect score.
    """
    good = [confidence for confidence, outcome in pairs if outcome]
    bad = [confidence for confidence, outcome in pairs if not outcome]
    if not good or not bad:
        return None
    wins = math.fsum(1.0 if g > b else 0.5 if g == b else 0.0 for g in good for b in bad)
    return wins / (len(good) * len(bad))


def brier(pairs: Sequence[tuple[float, bool]]) -> float | None:
    """The mean squared distance from confidence to outcome. ``None`` with nothing scored."""
    if not pairs:
        return None
    return math.fsum((confidence - (1.0 if outcome else 0.0)) ** 2 for confidence, outcome in pairs) / len(pairs)


@dataclass(frozen=True)
class Bin:
    """One calibration bin: its bounds, its size, the mean confidence in it and the share that paid
    off. ``rate`` is ``None`` for an empty bin, never 0."""

    low: float
    high: float
    n: int
    confidence: float | None
    rate: float | None


def bins(pairs: Sequence[tuple[float, bool]], count: int = BINS) -> tuple[Bin, ...]:
    """``count`` equal-width bins over [0, 1]; the last bin includes 1.0."""
    out: list[Bin] = []
    for number in range(count):
        low = number / count
        high = (number + 1) / count
        inside = [(c, o) for c, o in pairs
                  if (low <= c < high) or (number == count - 1 and c == 1.0)]
        out.append(Bin(low, high, len(inside),
                       None if not inside else math.fsum(c for c, _ in inside) / len(inside),
                       None if not inside else sum(1 for _, o in inside if o) / len(inside)))
    return tuple(out)


def d_prime(pairs: Sequence[tuple[float, bool]]) -> float | None:
    """Her first-order sensitivity, from a ``payoff`` call at or above ``CHANCE``.

    The hit rate is P(call | paid off) and the false-alarm rate P(call | did not); each is
    corrected by half a count (Hautus, 1995) so a saturated cell doesn't read as infinite. ``None``
    below ``CLASS_MIN`` on either side of chance: without both kinds of call there is no criterion
    to measure and ``meta-d'/d'`` has no denominator.
    """
    called = [outcome for confidence, outcome in pairs if confidence >= CHANCE]
    not_called = [outcome for confidence, outcome in pairs if confidence < CHANCE]
    if len(called) < CLASS_MIN or len(not_called) < CLASS_MIN:
        return None
    positives = sum(1 for outcome in called + not_called if outcome)
    negatives = len(called) + len(not_called) - positives
    hits = sum(1 for outcome in called if outcome) + 0.5
    false_alarms = sum(1 for outcome in called if not outcome) + 0.5
    return _probit(hits / (positives + 1.0)) - _probit(false_alarms / (negatives + 1.0))


def meta_d_prime(pairs: Sequence[tuple[float, bool]]) -> float | None:
    """The equal-variance Gaussian sensitivity her type-2 ranking implies.

    ``AUROC2 = probit(meta-d' / sqrt(2))`` in that model, so its inverse is the estimate. It is
    ``None`` when ``auroc2`` is, and clamped away from 0 and 1 so a tiny sample can't read as an
    infinite sensitivity.
    """
    area = auroc2(pairs)
    if area is None:
        return None
    return math.sqrt(2.0) * _probit(min(1.0 - PROBIT_FLOOR, max(PROBIT_FLOOR, area)))


@dataclass(frozen=True)
class Sensitivity:
    """meta-d'/d' and why it is or isn't reported. ``why`` is ``None`` when every number stands."""

    n: int
    d_prime: float | None
    meta_d: float | None
    ratio: float | None
    why: str | None


def sensitivity(pairs: Sequence[tuple[float, bool]]) -> Sensitivity:
    """The type-2 sensitivity ratio, with a reason whenever a part of it is missing."""
    n = len(pairs)
    resolved = [outcome for _, outcome in pairs]
    if n < TRIALS_MIN:
        return Sensitivity(n, None, None, None, f"{n} resolved < {TRIALS_MIN} trials")
    if min(sum(resolved), n - sum(resolved)) < CLASS_MIN:
        return Sensitivity(n, None, None, None,
                           f"fewer than {CLASS_MIN} resolved records in an outcome class")
    first = d_prime(pairs)
    if first is None:
        return Sensitivity(n, None, meta_d_prime(pairs), None,
                           f"fewer than {CLASS_MIN} confidence calls on a side of {CHANCE}")
    second = meta_d_prime(pairs)
    if second is None or first == 0:
        return Sensitivity(n, first, second, None, "her first-order calls do not separate outcomes (d' = 0)")
    return Sensitivity(n, first, second, second / first, None)


# ── One group of records ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scores:
    """A set of pairs scored two ways: the raw heuristic and the nightly calibrator."""

    n: int
    auroc2: float | None
    brier: float | None


@dataclass(frozen=True)
class Group:
    """One kind's records, or all of them: raw against calibrated, their bins and sensitivity."""

    label: str
    n: int
    resolved: int
    raw: Scores
    calibrated: Scores
    bins: tuple[Bin, ...]
    sensitivity: Sensitivity

    @property
    def gain(self) -> float | None:
        """Brier's fall from the raw heuristic to the calibrated value: positive is an improvement."""
        if self.raw.brier is None or self.calibrated.brier is None:
            return None
        return self.raw.brier - self.calibrated.brier


def group(label: str, recs: Sequence[ConfidenceRecord]) -> Group:
    """Score ``recs`` as one group. Only resolved records have an outcome to score against."""
    settled = [record for record in recs if record.resolved]
    raw = [(record.raw, bool(record.outcome)) for record in settled]
    calibrated = [(record.calibrated, bool(record.outcome)) for record in settled]
    return Group(
        label=label, n=len(recs), resolved=len(settled),
        raw=Scores(len(raw), auroc2(raw), brier(raw)),
        calibrated=Scores(len(calibrated), auroc2(calibrated), brier(calibrated)),
        bins=bins(calibrated), sensitivity=sensitivity(calibrated),
    )


@dataclass(frozen=True)
class MetacogReport:
    """Everything the report shows: per kind, and pooled, over one person's nights."""

    user: str
    first_night: date | None
    last_night: date | None
    kinds: tuple[Group, ...]
    overall: Group
    unresolved: int


def report(recs: Sequence[ConfidenceRecord], *, user: str = "") -> MetacogReport:
    """Score a week or a life of records: each kind apart (their bases differ) and pooled."""
    nights = [record.night for record in recs]
    return MetacogReport(
        user=user,
        first_night=min(nights) if nights else None,
        last_night=max(nights) if nights else None,
        kinds=tuple(group(kind, [record for record in recs if record.kind == kind]) for kind in KINDS),
        overall=group("overall", recs),
        unresolved=sum(1 for record in recs if not record.resolved),
    )


# ── The words ────────────────────────────────────────────────────────────────

CLAIM: Final = (
    "Her confidence is scored against what her own log shows happened, never against a report of how "
    "sure she says she is. An unresolved record is counted, never guessed, and a night the log cannot "
    "settle is left out of the scores rather than scored as a miss."
)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _number(value: float | None, places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _sense(s: Sensitivity) -> str:
    if s.ratio is not None:
        return f"meta-d' {s.meta_d:.2f} / d' {s.d_prime:.2f} = {s.ratio:.2f}"
    return f"insufficient: {s.why}"


def _bin_table(g: Group) -> list[str]:
    lines = ["| Bin | n | Mean confidence | Share that paid off |", "|---|---|---|---|"]
    for b in g.bins:
        lines.append(f"| {b.low:.1f}-{b.high:.1f} | {b.n} | {_number(b.confidence)} | {_pct(b.rate)} |")
    return lines


def render(r: MetacogReport) -> str:
    """The report as Markdown: the pooled score first, each kind under it, then the bins."""
    span = (f"{r.first_night} through {r.last_night}" if r.first_night is not None
            else "no night with a decision or a credence")
    o = r.overall
    lines = [
        f"# Metacognition over her nights ({r.user or 'person'})", "", CLAIM, "", f"Nights: {span}. "
        f"{o.n} second-order {'confidence' if o.n == 1 else 'confidences'}, {o.resolved} resolved, "
        f"{r.unresolved} unresolved.", "",
        "## Type-2 sensitivity", "",
        f"- All kinds: AUROC2 {_number(o.calibrated.auroc2)}; {_sense(o.sensitivity)}.",
        f"- Brier: raw {_number(o.raw.brier)}, calibrated {_number(o.calibrated.brier)} "
        f"(gain {_number(o.gain)}; positive is better).",
        "- Calibration is Platt scaling fitted only on records resolved before the night it scores, "
        f"and is raw identity below {MIN_TRIALS} resolved outcomes of its kind.",
        "",
        "| Kind | Records | Resolved | AUROC2 raw | AUROC2 calibrated | Brier raw | Brier calibrated |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in (o, *r.kinds):
        lines.append(f"| {g.label} | {g.n} | {g.resolved} | {_number(g.raw.auroc2)} | "
                     f"{_number(g.calibrated.auroc2)} | {_number(g.raw.brier)} | {_number(g.calibrated.brier)} |")
    lines += ["", "### meta-d'/d' by kind", ""]
    lines += [f"- {g.label}: {_sense(g.sensitivity)}." for g in r.kinds]
    lines += ["", "## Calibration bins (calibrated confidence)", ""]
    lines += _bin_table(o) if o.resolved else ["No resolved record: no bin has an outcome."]
    lines += ["", "AUROC2 is unchanged by a monotone calibrator; the Brier gain is what calibration buys. "
              "meta-d'/d' uses the equal-variance Gaussian closed form and is reported only above "
              f"{TRIALS_MIN} trials with {CLASS_MIN} records in each class.", ""]
    return "\n".join(lines)


# ── A results tree, read offline ─────────────────────────────────────────────


def _tree(root: Path | None, user: str) -> Path:
    return results_root(root) / user


def _events(base: Path, user: str) -> tuple[WorkspaceEvent, ...]:
    """Her plaintext workspace log, or nothing when the tree holds none.

    The app's own log is encrypted and lives outside a results tree; reading it here would need
    the account's key, so a log this module can't parse is refused rather than skipped.
    """
    path = base / WORKSPACE
    if not path.is_file():
        return ()
    try:
        return JsonlWorkspaceLog(path, user, codec=PLAIN, guard=NO_GUARD).events()
    except (ValueError, json.JSONDecodeError) as error:
        raise ResultsError(f"{path} is not a plaintext workspace log: {error}") from None


def _views(events: Sequence[WorkspaceEvent]) -> dict[date, tuple[Workspace, Workspace, tuple[WorkspaceEvent, ...]]]:
    if not events:
        return {}
    return {night: (before, after, night_events) for night, before, after, night_events in night_slices(events)}


def _folders(base: Path) -> tuple[Path, ...]:
    """One folder per virtual night: the latest attempt that has decision points."""
    by_day: dict[str, tuple[int, Path]] = {}
    for folder in sorted((base / NIGHTS).glob("*")):
        if not (folder / "summary.json").is_file() or not NightLog(folder).points():
            continue
        day, _, attempt = folder.name.partition(".")
        number = int(attempt) if attempt.isdigit() else 1
        if day not in by_day or number > by_day[day][0]:
            by_day[day] = (number, folder)
    return tuple(by_day[day][1] for day in sorted(by_day))


def _points(folder: Path) -> tuple[DecisionPoint, ...]:
    return tuple(point_from_record(record) for record in NightLog(folder).points())


def _results(points: Sequence[DecisionPoint]) -> tuple[Result, ...]:
    """Every result the night's points carry, once each (the post-result point repeats earlier ones)."""
    found: dict[str, Result] = {}
    for point in points:
        for result in point.raw.results:
            found.setdefault(result.id, result)
    return tuple(found.values())


def _study_evidence(point: DecisionPoint) -> int:
    """The chosen study target's unacted revisions: what a study of a target with no question saw."""
    assert point.choice is not None and point.choice.want is not None
    target_id = point.choice.target_id
    return next((target.unacted for target in point.raw.targets if target.id == target_id), 0)


def _receipt_pick(steps: Sequence[Mapping[str, Any]], pre: DecisionPoint) -> str | None:
    """The choose step's receipt outcome, checked against the point's pick.

    The receipt (``receipt.record``) is written from the same ``Choice`` the pre-study point keeps,
    so a receipt naming a different winner is a tree whose two records disagree: refused rather than
    scored against either. ``None`` is a night before receipts, or a note whose outcome is written
    later; the log's own change still decides those.
    """
    step = next((step for step in reversed(steps)
                 if step.get("step") == "choose" and isinstance(step.get("receipt"), Mapping)), None)
    if step is None:
        return None
    receipt = step["receipt"]
    expected = None if pre.choice is None or pre.choice.want is None else pre.choice.want.id
    if receipt.get("winner") != expected:
        raise ResultsError(f"the choose receipt names {receipt.get('winner')!r}, the point picked {expected!r}")
    outcome = receipt.get("outcome")
    if outcome is None:
        return None
    if outcome not in PICK_OUTCOMES:
        raise ResultsError(f"the choose receipt's outcome {outcome!r} is not one of {sorted(PICK_OUTCOMES)}")
    return str(outcome)


def night_facts(root: Path | None, user: str) -> tuple[NightFacts, ...]:
    """One ``NightFacts`` per night the tree has: the log's views, the pre-study point's pick and the
    choose receipt that says whether it ran."""
    base = _tree(root, user)
    if not base.is_dir():
        raise ResultsError(f"no results for {user!r} under {results_root(root)}")
    views = _views(_events(base, user))
    by_night: dict[date, NightFacts] = {}
    for folder in _folders(base):
        points = _points(folder)
        pre = next((p for p in points if p.phase == PRE_STUDY), None)
        if pre is None:
            continue
        before, after, night_events = views.get(pre.night, (Workspace(), Workspace(), ()))
        by_night[pre.night] = NightFacts(
            night=pre.night, before=before, after=after, results=_results(points), choice=pre.choice,
            study_evidence=0 if pre.choice is None or pre.choice.want is None else _study_evidence(pre),
            logged=pre.night in views, at=pre.raw.as_of, pick=_receipt_pick(NightLog(folder).steps(), pre),
            events=night_events,
        )
    for night, (before, after, night_events) in views.items():
        by_night.setdefault(night, NightFacts(night=night, before=before, after=after, events=night_events))
    return tuple(by_night[night] for night in sorted(by_night))


def load(root: Path | None, user: str) -> tuple[ConfidenceRecord, ...]:
    """Every second-order confidence the tree holds for ``user``, calibrated night by night."""
    return records(night_facts(root, user))


def main(argv: Sequence[str] | None = None, *, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    parser.add_argument("--root", type=Path, help="the results tree (defaults to $ANJO_DATA_DIR/sleep/results)")
    parser.add_argument("--out", type=Path, help="write the Markdown here instead of stdout")
    args = parser.parse_args(argv)
    recs = load(args.root, args.user)
    text = render(report(recs, user=args.user))
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
        out(f"wrote {args.out}")
    else:
        out(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
