"""The learning curve (hackathon report): a life's nights, arm 3 against arm 1.

    python -m anjo.sleep.reports.curve --root <results> --user <id>

Each night's morning probe run scores arm 3 (the night's checkpoint) against arm 1 (the base) per
dimension, tie-inclusive, with a block-bootstrap 95% interval (``probes.morning`` writes it under
``<results>/<user>/scores/night-NN.json``; ``reports.results`` reads it). The report scores one point
per (night, probe set) from those files, not per night folder: a rerun (``<date>.2``) is another
attempt at one night and never a second point. It prints each point in date order beside the night's
status (``promoted``, ``nothing_new``, ...), then the trend over the same nights: an
ordinary-least-squares slope per calendar day, its bootstrap interval, and the mean of the first
three nights against the last three.

**Why a report of its own.** A single night's interval says whether that night moved; it cannot say
whether the life is learning. The slope does, and its interval says how much of the slope these
nights support. The interval is conditional on these nights, not a population estimate, exactly as
the registered score intervals are.

**Bootstrapping a slope.** Nights are resampled with replacement, as pairs, and the slope refitted
on each resample; the interval is the same nearest-rank percentile convention ``score._interval``
uses. The registered resample count, seed and confidence level are reused (``prereg/thresholds.json``
via ``score.BootstrapConfig``) so the trend speaks the same statistical language as the scores.
Nights are counted on the calendar (the ordinal day), so a missed night widens the x gap rather than
pretending the nights were consecutive.

**Never across probe sets or labels.** A daily morning asks the ``daily-v1`` subset and a full morning
asks every probe; the two scores are not on one scale. Development and final nights answer different
questions too, and ``reports.results`` and ``reports.chart`` keep both apart. The table names each
point's set and label, and a trend is fitted only within one set and one label.

With fewer than three scored nights, or no spread in their days, the trend is ``insufficient``: no
slope is fitted, and the per-night rows are still printed. Nothing here reads a clock, the network or
any file but the results tree.
"""
from __future__ import annotations

import argparse
import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

from anjo.sleep.probes.score import BootstrapConfig
from anjo.sleep.reports import results

ARM: Final = "3"  # the night's checkpoint, against arm 1: did tonight's weights help
BASELINE: Final = "1"
MIN_NIGHTS: Final = 3  # fewer than this cannot say which way a curve goes
EDGE: Final = 3  # the first and last nights the trend compares
MISSING: Final = "-"
SETS: Final = "probe set"
INSUFFICIENT: Final = "insufficient"


@dataclass(frozen=True)
class Point:
    """One score of one night on one probe set: its date, label, calendar day, status and arm 3."""

    night: str
    label: str  # development or final: separately asked, so never one line
    day: int  # date.toordinal: a missed night widens the slope's x gap
    status: str
    probes: str  # the probe set the score came from (``full`` or a subset's name)
    stats: Mapping[str, results.Stat]  # dimension (and ``overall``) -> the night's stat


@dataclass(frozen=True)
class Trend:
    """One series' slope over one probe set and label, with its bootstrap interval and its two ends."""

    label: str
    series: str
    probes: str
    nights: int
    slope: float | None  # score points a calendar day; None: fewer than MIN_NIGHTS points
    ci: tuple[float, float] | None
    first: float | None  # mean of the first EDGE nights
    last: float | None  # mean of the last EDGE nights


@dataclass(frozen=True)
class Curve:
    """A life's scored nights and the trends its records support."""

    user: str
    points: tuple[Point, ...]
    dimensions: tuple[str, ...]
    trends: tuple[Trend, ...]


def _day(night: str, index: int) -> int:
    """The night's calendar day; an unreadable date counts as its place in the file order."""
    try:
        return date.fromisoformat(night).toordinal()
    except ValueError:
        return index


def _summaries(user: str, root: Path | None) -> dict[str, results.Night]:
    """Each night's last summary (a rerun folder sorts after its night), for the status it records."""
    return {night.night: night for night in results.load(user, root)}


def points(user: str, root: Path | None = None) -> tuple[Point, ...]:
    """One point per scored (night, probe set); rerunning a night never adds a point for it.

    The score file names its night and set and carries its label; the summary (one per attempt, so
    a rerun folder repeats its night) supplies the status the table prints.
    """
    base = results.results_root(root) / user
    summaries = _summaries(user, root)
    found: list[Point] = []
    for (night, probes), record in results.score_records(base / "scores").items():
        stats = results._stats(record["arms"]).get(ARM, {})
        if results.OVERALL not in stats:
            continue
        summary = summaries.get(night)
        label = str(record.get("label") or (summary.label if summary else ""))
        if label not in results.LABELS:
            raise results.ResultsError(f"{user} {night}: the score has no development or final label")
        found.append(Point(night, label, _day(night, len(found)),
                           summary.status if summary else "?", probes, dict(stats)))
    return tuple(sorted(found, key=lambda point: (point.night, point.probes)))


def dimensions(points: Sequence[Point]) -> tuple[str, ...]:
    """Every dimension the nights scored, the registered order first and ``overall`` last."""
    names = {name for point in points for name in point.stats}
    seen = names - {results.OVERALL}
    ordered = (*(name for name in results.DIMENSIONS if name in seen),
               *sorted(seen - set(results.DIMENSIONS)))
    return (*ordered, results.OVERALL) if results.OVERALL in names else ordered


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Ordinary least squares: score points a day. Callers give at least two x values apart."""
    mx, my = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
    variance = math.fsum((x - mx) ** 2 for x in xs)
    if variance == 0.0:
        raise ValueError("a slope needs two nights apart")
    return math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / variance


def _rank(q: float, n: int) -> int:
    """Nearest-rank index ``ceil(q * n) - 1``, clamped to the sample (``score._interval``'s rule)."""
    return min(n - 1, max(0, math.ceil(round(q * n, 9)) - 1))


def _interval(values: list[float], ci: float) -> tuple[float, float]:
    """The percentile interval by one convention for both ends, as the score intervals use."""
    if not values:
        return (0.0, 0.0)
    ordered = sorted(values)
    tail = (1.0 - ci) / 2.0
    return ordered[_rank(tail, len(ordered))], ordered[_rank(1.0 - tail, len(ordered))]


def _bootstrap(xs: Sequence[float], ys: Sequence[float], config: BootstrapConfig) -> tuple[float, tuple[float, float]]:
    """The fitted slope and its percentile interval from nights resampled as (x, y) pairs."""
    fitted = _slope(xs, ys)
    rng = random.Random(config.seed)
    found: list[float] = []
    for _ in range(config.resamples):
        picked = rng.choices(range(len(xs)), k=len(xs))
        sample_x = [xs[index] for index in picked]
        if len(set(sample_x)) < 2:  # every pick landed on one night: no slope to fit
            continue
        found.append(_slope(sample_x, [ys[index] for index in picked]))
    return fitted, _interval(found, config.ci)


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def trend(found: Sequence[Point], series: str, label: str, probes: str, *, config: BootstrapConfig) -> Trend:
    """``series``' slope over the nights of one label and probe set, and the two ends it compares."""
    ys = [(point.day, point.stats[series].score) for point in found
          if point.label == label and point.probes == probes and series in point.stats]
    xs = [float(day) for day, _ in ys]
    if len(ys) < MIN_NIGHTS or len(set(xs)) < 2:  # no spread, no slope: say insufficient, never crash
        return Trend(label, series, probes, len(ys), None, None, None, None)
    scores = [score for _, score in ys]
    fitted, ci = _bootstrap(xs, scores, config)
    return Trend(label, series, probes, len(ys), fitted, ci, _mean(scores[:EDGE]), _mean(scores[-EDGE:]))


def curve(user: str, root: Path | None = None, *, config: BootstrapConfig | None = None) -> Curve:
    """A life's curve: its scored points, and a trend per label, probe set and series."""
    found = points(user, root)
    bootstrap = BootstrapConfig.from_thresholds() if config is None else config
    dims = dimensions(found)
    sets = sorted({point.probes for point in found})
    labels = [label for label in results.LABELS if any(point.label == label for point in found)]
    trends = tuple(trend(found, series, label, probes, config=bootstrap)
                   for label in labels for probes in sets for series in dims)
    return Curve(user, found, dims, trends)


def _shown(stat: results.Stat) -> str:
    if stat.ci is None:
        return f"{stat.score:.2f}"
    return f"{stat.score:.2f} [{stat.ci[0]:.2f},{stat.ci[1]:.2f}]"


def _row(cells: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()


def _table(found: Curve) -> Iterator[str]:
    headers = ["night", "label", "status", SETS, *found.dimensions]
    rows = [[point.night, point.label, point.status, point.probes,
             *(MISSING if name not in point.stats else _shown(point.stats[name]) for name in found.dimensions)]
            for point in found.points]
    widths = [max(len(row[column]) for row in [headers, *rows]) for column in range(len(headers))]
    yield _row(headers, widths)
    for row in rows:
        yield _row(row, widths)


def _trend_rows(found: Curve) -> Iterator[str]:
    headers = ["label", "series", SETS, "nights", "slope/day", "95% CI", "first 3", "last 3"]
    rows: list[list[str]] = []
    for entry in found.trends:
        if entry.slope is None or entry.ci is None or entry.first is None or entry.last is None:
            continue
        rows.append([entry.label, entry.series, entry.probes, str(entry.nights), f"{entry.slope:+.4f}",
                     f"[{entry.ci[0]:+.4f},{entry.ci[1]:+.4f}]", f"{entry.first:.3f}", f"{entry.last:.3f}"])
    widths = [max(len(row[column]) for row in [headers, *rows]) for column in range(len(headers))]
    yield _row(headers, widths)
    for row in rows:
        yield _row(row, widths)


def render(found: Curve) -> str:
    """The report as plain text: a row a point, then the trends the nights support."""
    out = [f"learning curve: {found.user}",
           f"(arm {ARM} against arm {BASELINE}; the score is tie-inclusive, 0.50 means neither is ahead)", ""]
    if not found.points:
        return "\n".join([*out, f"{INSUFFICIENT}: no scored nights (three are needed for a trend)"])
    out.extend(_table(found))
    out.append("")
    if len(found.points) < MIN_NIGHTS:
        out.append(f"{INSUFFICIENT}: {len(found.points)} scored night(s), three are needed for a trend")
        return "\n".join(out)
    out.extend(_trend_rows(found))
    out.append("")
    if not any(entry.slope is not None for entry in found.trends):
        out.append(f"{INSUFFICIENT}: no label and probe set has three scored nights on different days")
        return "\n".join(out)
    out.append("a trend row per label and probe set: development and final, and a daily subset and a full "
               "morning, are never fitted on one line")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None, *, root: Path | None = None, out: Callable[[str], None] = print,
         config: BootstrapConfig | None = None) -> int:
    parser = argparse.ArgumentParser(description="A life's learning curve, arm 3 against arm 1.")
    parser.add_argument("--root", type=Path, default=None, help="results tree (default: ANJO_DATA_DIR/sleep/results)")
    parser.add_argument("--user", required=True, help="the life's results folder")
    args = parser.parse_args(argv)
    out(render(curve(args.user, args.root or root, config=config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
