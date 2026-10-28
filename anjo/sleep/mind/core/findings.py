"""MIND core's calls as APD finding records, and a deterministic pre-score. Torch-free.

**Finding records.** The core has no words; what it can say is which real events, on which night,
it expects to go wrong. Each call it makes (a night's speak decision, or a dream's ignition) is a
``Raised``: the day, its probability and the event ids its slot cites. ``reports`` turns the calls
that reach the threshold into ``FindingReport`` rows, the shape the APD judge scores, so the core
(N) and the notebook baseline (B) share one blind judge. A call whose cites overlap an earlier
finding's latest cites is that thread again: it updates the finding in place (same id, a later
``seq``), as an arm's ``report_finding`` does. Every report is a suspicion. Its ``problem`` comes
from ``topic`` (the CLI quotes the cited evidence, which the judge sees anyway). A night's speak
decision is kept at the speak threshold, a dream's ignition at the dream's own (``ignite``).

**Another arm in the core's form** (``in_core_form``): for one blind judge pass, another arm's
findings (the notebook baseline B's) are rewritten the way the core writes its own, the cited
evidence quoted as the problem and a fixed conclusion, so the judge can't tell the arms apart by
their prose. Day, id, probability, cites and kind stay as reported.

**The pre-score** costs nothing and needs no judge: a planted item is caught when a report cites
one of its origin or disambiguating ids on a day from its knowable day to its cutoff (the day
before it surfaced). A decoy or reversal cited in that window is flagged (a false alarm); a
finding that cites no item's events at all is unmatched. It reads the items as the data export
writes them (``item_of``), so the same function scores train lives and, given the sealed items,
test lives.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Final, Literal

from anjo.sleep.mind.apd.types import PROBLEM_FAMILIES, FindingReport

Source = Literal["speak", "risk", "missing", "collision"]
Topic = Callable[[tuple[str, ...]], str]
CONCLUSIONS: Final[Mapping[Source, str]] = {
    "speak": "This is likely to go wrong before it comes up again.",
    "risk": "This is likely to go wrong within the week.",
    "missing": "Something this needs is still missing.",
    "collision": "Two of these plans are likely to collide.",
}


@dataclass(frozen=True)
class Raised:
    """One call of the core: a night's speak decision or a dream's ignition."""

    day: date
    p: float
    cites: tuple[str, ...]
    source: Source


def default_topic(cites: tuple[str, ...]) -> str:
    return "The thread of events " + ", ".join(cites)


def reports(
    raised: Sequence[Raised], *, threshold: float, ignite: float | None = None, topic: Topic = default_topic,
) -> tuple[FindingReport, ...]:
    """The speak calls at or above ``threshold`` and the ignitions at or above ``ignite`` (none
    when it is ``None``), in order, as finding reports."""
    out: list[FindingReport] = []
    latest: dict[str, frozenset[str]] = {}
    for call in raised:
        line = threshold if call.source == "speak" else ignite
        if line is None or call.p < line or not call.cites:
            continue
        cited = frozenset(call.cites)
        finding_id = next((fid for fid, held in latest.items() if held & cited), f"core-{len(latest) + 1}")
        latest[finding_id] = cited
        out.append(FindingReport(
            finding_id=finding_id, day=call.day, seq=len(out), problem=topic(call.cites),
            conclusion=CONCLUSIONS[call.source], p=round(min(max(call.p, 0.0), 1.0), 6), cites=call.cites,
            kind="suspicion",
        ))
    return tuple(out)


def in_core_form(report: FindingReport, topic: Topic) -> FindingReport:
    """Another arm's finding as the core writes its own (see the module docstring)."""
    return replace(report, problem=topic(report.cites), conclusion=CONCLUSIONS["speak"])


# ── The pre-score ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ItemRef:
    """What the pre-score reads of a planted item."""

    id: str
    family: str
    ids: frozenset[str]  # its origin and disambiguating events
    knowable_day: date
    cutoff_day: date


def item_of(record: Mapping[str, Any]) -> ItemRef:
    """A planted item as the data export writes it (``labels/<life>.json``'s ``items``)."""
    return ItemRef(
        id=str(record["item_id"]), family=str(record["family"]),
        ids=frozenset(str(i) for i in (*record["origin_ids"], *record["disambiguating_ids"])),
        knowable_day=date.fromisoformat(str(record["knowable_day"])),
        cutoff_day=date.fromisoformat(str(record["cutoff_day"])),
    )


@dataclass(frozen=True)
class PreScore:
    problems: int
    caught: int
    decoys: int  # decoys and reversals
    flagged: int
    findings: int
    unmatched: int

    @property
    def recall(self) -> float | None:
        return self.caught / self.problems if self.problems else None


def _hits(item: ItemRef, report: FindingReport) -> bool:
    return item.knowable_day <= report.day <= item.cutoff_day and bool(item.ids & set(report.cites))


def prescore(found: Sequence[FindingReport], items: Sequence[ItemRef]) -> PreScore:
    """One life's (or many lives') reports against their planted items."""
    problems = [item for item in items if item.family in PROBLEM_FAMILIES]
    decoys = [item for item in items if item.family not in PROBLEM_FAMILIES]
    every = frozenset(i for item in items for i in item.ids)
    by_finding: dict[str, list[FindingReport]] = {}
    for report in found:
        by_finding.setdefault(report.finding_id, []).append(report)
    unmatched = sum(1 for rows in by_finding.values() if not any(every & set(r.cites) for r in rows))
    return PreScore(
        problems=len(problems), caught=sum(1 for item in problems if any(_hits(item, r) for r in found)),
        decoys=len(decoys), flagged=sum(1 for item in decoys if any(_hits(item, r) for r in found)),
        findings=len(by_finding), unmatched=unmatched,
    )
