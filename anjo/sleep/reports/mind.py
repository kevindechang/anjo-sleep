"""The MIND experiment's report (Task 16): what her state studied against four other deciders.

Only D0 ran, so every number is **prospective target relevance on D0's recorded history**: not a
claim about better outcomes for the person, or about how another policy would have played out.

- **Anticipation (primary):** the share of nights on which a decider's study target is relevant,
  by the frozen labels, to a held-out ``new_work`` item arriving in the 7 virtual days after that
  night. Silence is a night with no hit. A night hits when its target is relevant to any such item,
  so a later night that hits the same item counts too. Per life, and pooled over the held-out lives.
  The interval is a stationary bootstrap over nights within each life (Politis & Romano, 1994;
  mean block 7 nights): descriptive uncertainty, never a test.
- **Distinct items anticipated (secondary):** each held-out item counted once per decider, at the
  earliest night that hit it.
- **Silence rate**, and **D0 - D1 on the nights both studied**.
- **Contact discipline:** D0's and D1's note rate and unanchored-note rate per night (a note citing
  no due commitment and no changed target), split by whether her step-4 choice studied.
- **Demand forecasting:** the Brier score of D0's ``p_needed`` on resolved expectations against a
  constant-rate forecast at the development base rate, with n after censoring.

Records are checked whole before scoring. A decider observed on every night of a life is scored; one
never observed there (D1 when no chat port asked it) is reported as not observed, never as silence, and
so is it in a pool any of whose lives lacks it; one observed on some nights only is refused. D0, the
recorded choice, is on every night.

Development lives are reported, labeled, and never pooled with the held-out lives. Every time here
is on the life's virtual clock: a night on day N decides at the start of day N + 1, and an item or a
return on day D happens at the start of day D.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Final, Literal

from anjo.sleep.mind import expect
from anjo.sleep.mind.deciders import CONTACTED, DECIDERS, NightPicks
from anjo.sleep.probes import relevance
from anjo.sleep.probes.relevance import FrozenLabels
from anjo.sleep.types import Expectation, ExpectationOutcome

Split = Literal["development", "held_out"]
HORIZON_DAYS: Final = 7
MEAN_BLOCK: Final = 7.0
RESAMPLES: Final = 10_000
SEED: Final = 20261028
CONFIDENCE: Final = 0.95
UNOBSERVED: Final = "not observed"  # a decider never asked: no number, and never silence
# Copy: the spec asks only that lesions be "reported the same way".
LESION_LINES: Final = (("D2", "Without her confidence (D2)"), ("D3", "Without revision gating (D3)"),
                       ("D4", "By earliest due (D4)"))


@dataclass(frozen=True)
class Arrival:
    """A held-out ``new_work`` item and the virtual day it arrives."""

    item_id: str
    day: date


@dataclass(frozen=True)
class LifeRecord:
    """One life's replayed nights and what no decider saw: its held-out arrivals and their frozen
    relevance labels. ``expectations`` are D0's, on the virtual clock; ``returns`` are the days the
    person's own evidence came back to each target; ``last_day`` is the timeline's last day."""

    life_id: str
    split: Split
    nights: tuple[NightPicks, ...]
    arrivals: tuple[Arrival, ...]
    labels: FrozenLabels
    last_day: date
    expectations: tuple[Expectation, ...] = ()
    returns: Mapping[str, tuple[date, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Share:
    hits: int
    nights: int

    @property
    def rate(self) -> float | None:
        return None if self.nights == 0 else self.hits / self.nights


@dataclass(frozen=True)
class Measured:
    share: Share
    interval: tuple[float, float] | None


@dataclass(frozen=True)
class Paired:
    """D0's and D1's anticipation on the nights both studied."""

    d0: Share
    d1: Share

    @property
    def difference(self) -> float | None:
        if self.d0.rate is None or self.d1.rate is None:
            return None
        return self.d0.rate - self.d1.rate


@dataclass(frozen=True)
class Rates:
    nights: int
    notes: int
    unanchored: int


@dataclass(frozen=True)
class Contact:
    studied: Rates
    not_studied: Rates


@dataclass(frozen=True)
class Demand:
    n: int
    censored: int
    brier: float | None
    constant: float | None
    base_rate: float | None


@dataclass(frozen=True)
class Overhead:
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float | None


@dataclass(frozen=True)
class Outcomes:
    """One life's outcomes, or the held-out lives' pooled. The mappings hold the observed deciders
    only; ``unobserved`` names the rest, and ``paired`` is ``None`` without both D0 and D1."""

    anticipation: Mapping[str, Measured]
    distinct: Mapping[str, int]
    silence: Mapping[str, Share]
    paired: Paired | None
    contact: Mapping[str, Contact]
    demand: Demand
    unobserved: tuple[str, ...] = ()


@dataclass(frozen=True)
class LifeRow(Outcomes):
    life_id: str = ""
    split: Split = "held_out"


@dataclass(frozen=True)
class Pooled(Outcomes):
    lives: tuple[str, ...] = ()


@dataclass(frozen=True)
class MindReport:
    lives: tuple[LifeRow, ...]
    pooled: Pooled
    overhead: Overhead

    @property
    def demand(self) -> Demand:
        return self.pooled.demand


# ── Per night and per life ────────────────────────────────────────────────────


def _virtual(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


def _window(life: LifeRecord, night: date) -> list[Arrival]:
    return [a for a in life.arrivals if night < a.day <= night + timedelta(days=HORIZON_DAYS)]


def observed(life: LifeRecord) -> frozenset[str]:
    """The deciders observed on every night of ``life``, each contacted one with its note on every
    night. One observed on some nights only, or D0 missing anywhere, is refused: a missing
    observation is never silence."""
    nights = len(life.nights)
    seen = set()
    for decider in DECIDERS:
        count = sum(decider in night.study for night in life.nights)
        if count == nights:
            seen.add(decider)
        elif count or decider == "D0":
            raise ValueError(f"{life.life_id}: {decider} was observed on {count} of {nights} nights; "
                             "a missing observation is not silence")
    for decider in sorted(seen & set(CONTACTED)):
        if any(decider not in night.notes for night in life.nights):
            raise ValueError(f"{life.life_id}: {decider} lacks its note on a night it was observed")
    return frozenset(seen)


def hit_items(life: LifeRecord, night: NightPicks, decider: str) -> tuple[str, ...]:
    """The held-out items in the night's horizon that its study target is relevant to."""
    target = night.study[decider]
    if target is None:
        return ()
    return tuple(a.item_id for a in _window(life, night.night) if life.labels.relevant(target, a.item_id))


def _hits(life: LifeRecord, decider: str) -> list[float]:
    return [1.0 if hit_items(life, night, decider) else 0.0 for night in life.nights]


def anticipation(life: LifeRecord, decider: str) -> Share:
    return Share(int(sum(_hits(life, decider))), len(life.nights))


def distinct_items(life: LifeRecord, decider: str) -> dict[str, date]:
    """Each item hit, at the earliest night that hit it."""
    first: dict[str, date] = {}
    for night in sorted(life.nights, key=lambda n: n.night):
        for item in hit_items(life, night, decider):
            first.setdefault(item, night.night)
    return first


def silence(life: LifeRecord, decider: str) -> Share:
    return Share(sum(night.study[decider] is None for night in life.nights), len(life.nights))


def both_studied(life: LifeRecord) -> Paired:
    both = [n for n in life.nights if n.study["D0"] is not None and n.study["D1"] is not None]
    d0 = sum(bool(hit_items(life, n, "D0")) for n in both)
    d1 = sum(bool(hit_items(life, n, "D1")) for n in both)
    return Paired(Share(d0, len(both)), Share(d1, len(both)))


def contact(life: LifeRecord, decider: str) -> Contact:
    def rates(studied: bool) -> Rates:
        nights = [n for n in life.nights if n.studied == studied]
        notes = [n.notes[decider] for n in nights]
        return Rates(len(nights), sum(note is not None for note in notes),
                     sum(note is not None and not note.anchored for note in notes))

    return Contact(studied=rates(True), not_studied=rates(False))


def _resolved(life: LifeRecord) -> list[tuple[Expectation, ExpectationOutcome | None]]:
    end = _virtual(life.last_day + timedelta(days=1))
    out = []
    for made in life.expectations:
        arrivals = [_virtual(a.day) for a in life.arrivals if life.labels.relevant(made.target_id, a.item_id)]
        returns = [_virtual(day) for day in life.returns.get(made.target_id, ())]
        out.append((made, expect.resolve(made, returns=returns, arrivals=arrivals, life_end=end, as_of=end)))
    return out


def demand(lives: Sequence[LifeRecord], base_rate: float | None) -> Demand:
    resolved = [(m, o) for life in lives for m, o in _resolved(life)]
    scored = [(m.p_needed, 1.0 if o == "needed" else 0.0) for m, o in resolved if o in ("needed", "not_needed")]
    censored = sum(o == "censored" for _, o in resolved)
    if not scored:
        return Demand(0, censored, None, None, base_rate)
    brier = math.fsum((p - y) ** 2 for p, y in scored) / len(scored)
    constant = None if base_rate is None else math.fsum((base_rate - y) ** 2 for _, y in scored) / len(scored)
    return Demand(len(scored), censored, brier, constant, base_rate)


# ── The interval ──────────────────────────────────────────────────────────────


def stationary_indices(n: int, mean_block: float, rng: random.Random) -> list[int]:
    """One stationary-bootstrap resample of positions 0..n-1: blocks of geometric length (mean
    ``mean_block``), each from a uniform start, wrapping around."""
    if n <= 0:
        return []
    restart = 1.0 / mean_block
    indices = [rng.randrange(n)]
    while len(indices) < n:
        indices.append(rng.randrange(n) if rng.random() < restart else (indices[-1] + 1) % n)
    return indices


def interval(sequences: Sequence[Sequence[float]], *, resamples: int, rng: random.Random) -> tuple[float, float] | None:
    """The percentile interval of the pooled mean, each life's nights resampled on their own."""
    total = sum(len(s) for s in sequences)
    if total == 0:
        return None
    means = []
    for _ in range(resamples):
        drawn = math.fsum(s[i] for s in sequences for i in stationary_indices(len(s), MEAN_BLOCK, rng))
        means.append(drawn / total)
    means.sort()
    tail = (1.0 - CONFIDENCE) / 2.0
    low = means[max(0, math.floor(tail * resamples))]
    high = means[min(resamples - 1, math.ceil((1.0 - tail) * resamples) - 1)]
    return low, high


# ── The report ────────────────────────────────────────────────────────────────


def _check_labels(life: LifeRecord, registered: Mapping[str, str]) -> None:
    if life.life_id not in registered:
        raise relevance.LabelError(f"{life.life_id}'s labels were not registered before scoring")
    relevance.verify(life.labels, registered[life.life_id])
    picked = {t for n in life.nights for t in n.study.values() if t} | {m.target_id for m in life.expectations}
    for target in sorted(picked):
        for arrival in life.arrivals:
            life.labels.relevant(target, arrival.item_id)  # raises for a pair never labeled


def _outcomes(lives: Sequence[LifeRecord], base_rate: float | None, *, resamples: int, seed: int) -> Outcomes:
    seen = [observed(life) for life in lives]
    available = [d for d in DECIDERS if all(d in s for s in seen)]
    measured = {}
    for decider in available:
        rng = random.Random(f"{seed}:{decider}:{','.join(life.life_id for life in lives)}")
        share = _sum_shares([anticipation(life, decider) for life in lives])
        measured[decider] = Measured(share, interval([_hits(life, decider) for life in lives],
                                                     resamples=resamples, rng=rng))
    pairs = [both_studied(life) for life in lives] if "D1" in available else None
    contacts = {d: [contact(life, d) for life in lives] for d in CONTACTED if d in available}
    return Outcomes(
        anticipation=measured,
        distinct={d: sum(len(distinct_items(life, d)) for life in lives) for d in available},
        silence={d: _sum_shares([silence(life, d) for life in lives]) for d in available},
        paired=None if pairs is None else Paired(_sum_shares([p.d0 for p in pairs]), _sum_shares([p.d1 for p in pairs])),
        contact={d: Contact(_sum_rates([c.studied for c in cs]), _sum_rates([c.not_studied for c in cs]))
                 for d, cs in contacts.items()},
        demand=demand(lives, base_rate),
        unobserved=tuple(d for d in DECIDERS if d not in available),
    )


def _sum_shares(shares: Sequence[Share]) -> Share:
    return Share(sum(s.hits for s in shares), sum(s.nights for s in shares))


def _sum_rates(rates: Sequence[Rates]) -> Rates:
    return Rates(sum(r.nights for r in rates), sum(r.notes for r in rates), sum(r.unanchored for r in rates))


def _overhead(lives: Sequence[LifeRecord], prices: tuple[float, float] | None) -> Overhead:
    answers = [a for life in lives for night in life.nights for a in night.d1]
    tokens_in, tokens_out = sum(a.input_tokens for a in answers), sum(a.output_tokens for a in answers)
    usd = None if prices is None else (tokens_in * prices[0] + tokens_out * prices[1]) / 1_000_000
    return Overhead(len(answers), tokens_in, tokens_out, usd)


def mind_report(
    lives: Sequence[LifeRecord], *, registered: Mapping[str, str], coefficients: expect.Coefficients | None,
    resamples: int = RESAMPLES, seed: int = SEED, prices: tuple[float, float] | None = None,
) -> MindReport:
    """Score every life on labels registered before scoring. ``prices`` are Spark Open's USD per
    million input and output tokens, for D1's overhead."""
    for life in lives:
        _check_labels(life, registered)
        observed(life)
    base_rate = None if coefficients is None else coefficients.base_rate
    rows = []
    for life in lives:
        o = _outcomes([life], base_rate, resamples=resamples, seed=seed)
        rows.append(LifeRow(o.anticipation, o.distinct, o.silence, o.paired, o.contact, o.demand, o.unobserved,
                            life_id=life.life_id, split=life.split))
    held = [life for life in lives if life.split == "held_out"]
    o = _outcomes(held, base_rate, resamples=resamples, seed=seed)
    pooled = Pooled(o.anticipation, o.distinct, o.silence, o.paired, o.contact, o.demand, o.unobserved,
                    lives=tuple(life.life_id for life in held))
    return MindReport(lives=tuple(rows), pooled=pooled, overhead=_overhead(held, prices))


# ── The words ─────────────────────────────────────────────────────────────────

CLAIM: Final = (
    "Only her chooser (D0) ever ran, so this is prospective target relevance on D0's recorded history: "
    "not a claim about better outcomes for the person, or about how another policy would have played out. "
    "D2 and D3 show conditional mechanism effects. No population claim, and no claim about what a real "
    "person needed."
)
INTERVAL_NOTE: Final = (
    "Each interval is a 95% percentile interval from a stationary bootstrap over nights within each life "
    "(Politis & Romano, 1994; mean block 7 nights), shown as descriptive uncertainty, not as a test."
)


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.0f}%"


def _range(bounds: tuple[float, float] | None) -> str:
    return "n/a" if bounds is None else f"{bounds[0] * 100:.0f}-{bounds[1] * 100:.0f}%"


def sentence(o: Outcomes) -> str:
    """The spec's wording for one life, or the pooled held-out lives."""
    d0, d1 = o.anticipation["D0"], o.anticipation.get("D1")
    if d1 is None:
        plain, intervals = UNOBSERVED, f"interval {_range(d0.interval)}"
    else:
        plain, intervals = _pct(d1.share.rate), f"intervals {_range(d0.interval)} and {_range(d1.interval)}"
    return (
        f"On {_pct(d0.share.rate)} of nights her state studied something relevant to work the synthetic "
        f"person brought later; a plain model given the same state, {plain} ({intervals})."
    )


def _lesions(o: Outcomes) -> list[str]:
    lines = []
    for d, words in LESION_LINES:
        m = o.anticipation.get(d)
        lines.append(f"  - {words}: {UNOBSERVED}." if m is None else
                     f"  - {words}: {_pct(m.share.rate)} of nights ({_range(m.interval)}).")
    return lines


def _cell(values: Mapping[str, object], decider: str, shown: Callable[[Any], str]) -> str:
    return UNOBSERVED if decider not in values else shown(values[decider])


def _tables(rows: Sequence[tuple[str, Outcomes]]) -> list[str]:
    deciders = " | ".join(DECIDERS)
    lines = ["### Distinct held-out items anticipated (secondary; each once, at its earliest night)", "",
             f"| Life | {deciders} |", "|---" * (len(DECIDERS) + 1) + "|"]
    lines += [f"| {name} | " + " | ".join(_cell(o.distinct, d, str) for d in DECIDERS) + " |" for name, o in rows]
    lines += ["", "### Silence rate (a silent night is no hit)", "", f"| Life | {deciders} |",
              "|---" * (len(DECIDERS) + 1) + "|"]
    lines += [f"| {name} | " + " | ".join(_cell(o.silence, d, lambda s: _pct(s.rate)) for d in DECIDERS) + " |"
              for name, o in rows]
    lines += ["", "### D0 - D1 on the nights both studied", "", "| Life | Nights | D0 | D1 | D0 - D1 |",
              "|---|---|---|---|---|"]
    for name, o in rows:
        if o.paired is None:
            lines.append(f"| {name} | n/a | n/a | {UNOBSERVED} | n/a |")
            continue
        diff = o.paired.difference
        shown = "n/a" if diff is None else f"{diff * 100:+.0f} points"
        lines.append(f"| {name} | {o.paired.d0.nights} | {_pct(o.paired.d0.rate)} | {_pct(o.paired.d1.rate)} | {shown} |")
    lines += ["", "### Contact discipline (per night; unanchored: no due commitment and no changed target)", "",
              "| Life | Decider | Step 4 studied: nights, notes, unanchored | Not studied: nights, notes, unanchored |",
              "|---|---|---|---|"]
    lines += [f"| {name} | {d} | {_cell(o.contact, d, lambda c: _rates(c.studied))} | "
              f"{_cell(o.contact, d, lambda c: _rates(c.not_studied))} |" for name, o in rows for d in CONTACTED]
    return lines


def _rates(r: Rates) -> str:
    def per(count: int) -> str:
        return _pct(None if r.nights == 0 else count / r.nights)

    return f"{r.nights}, {per(r.notes)}, {per(r.unanchored)}"


def _demand(d: Demand, where: str) -> str:
    if d.brier is None:
        return f"Demand forecasting ({where}): no resolved expectation; {d.censored} censored."
    constant = "n/a" if d.constant is None else f"{d.constant:.3f}"
    return (f"Demand forecasting ({where}): Brier {d.brier:.3f} for D0's p_needed against {constant} for a constant "
            f"forecast at the development base rate ({_pct(d.base_rate)}); n = {d.n} after censoring, "
            f"{d.censored} censored.")


def render(report: MindReport) -> str:
    """The report as Markdown: held-out lives and their pool first, development lives after."""
    held = [row for row in report.lives if row.split == "held_out"]
    dev = [row for row in report.lives if row.split == "development"]
    count = len(report.pooled.lives)
    lines = ["# MIND: what her state studied (Task 16)", "", CLAIM, "", INTERVAL_NOTE, "", "## Held-out lives", ""]
    for row in held:
        lines += [f"- {row.life_id}: {sentence(row)}", *_lesions(row)]
    lines += [f"- Pooled over {count} held-out {'life' if count == 1 else 'lives'}: {sentence(report.pooled)}",
              *_lesions(report.pooled), ""]
    lines += [*_tables([(row.life_id, row) for row in held] + [("Pooled", report.pooled)]), ""]
    lines += [_demand(report.demand, "held-out lives, pooled"), ""]
    o = report.overhead
    usd = "n/a" if o.usd is None else f"${o.usd:.4f}"
    lines += [f"D1's overhead (held-out lives, reported apart): {o.calls} Spark Open calls, {o.input_tokens} input "
              f"and {o.output_tokens} output tokens, {usd}.", ""]
    lines += ["## Development lives", "",
              "Labeled development: they informed decisions before the freeze, and are never pooled with the "
              "held-out lives.", ""]
    for row in dev:
        lines += [f"- {row.life_id} (development): {sentence(row)}", *_lesions(row)]
    if dev:
        lines += ["", *_tables([(row.life_id, row) for row in dev]), ""]
        lines += [_demand(row.demand, f"{row.life_id}, development") for row in dev]
    return "\n".join(lines) + "\n"
