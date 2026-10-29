"""Pick the live demo's notes-only baseline (2C or 2F) from pooled development morning scores
(``python -m anjo.sleep.tools.demo_baseline``).

Oct 31 (spec §6a): compare arm 4 against 2C and against 2F on ``new_work``, across the registered
development lives' full morning scores. The stronger notes-only baseline is the one arm 4 scores
lower against on pooled ``new_work`` probes; that arm is recommended for ``ANJO_SLEEP_DEMO_BASELINE``.
This tool prints one JSON object and never sets env vars itself.

Scores come from each score file's outcome rows (``pair`` and ``kind``), or from the pair's
``knowledge`` dimension when a file has no ``new_work`` rows for that pair but does store the
comparison under ``pairs`` (the morning's dimension layout). Only ``label`` ``development`` nights
on the full probe set (``full``, not a daily subset) are pooled.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from anjo.sleep.demo.starters import BASELINES
from anjo.sleep.probes.morning import NO_VERDICT
from anjo.sleep.probes.score import THRESHOLDS
from anjo.sleep.reports.results import FULL_SET, results_root, score_records

PAIR_2C: Final = "4_vs_2C"
PAIR_2F: Final = "4_vs_2F"
DEFAULT: Final = "2C"
DIMENSION: Final = "knowledge"


@dataclass(frozen=True)
class Count:
    wins: int = 0
    ties: int = 0
    losses: int = 0
    unjudged: int = 0

    def __add__(self, other: Count) -> Count:
        return Count(
            self.wins + other.wins,
            self.ties + other.ties,
            self.losses + other.losses,
            self.unjudged + other.unjudged,
        )

    @property
    def judged(self) -> int:
        return self.wins + self.ties + self.losses

    @property
    def score(self) -> float | None:
        if not self.judged:
            return None
        return (self.wins + self.ties / 2) / self.judged


def _count(row: Mapping[str, Any]) -> Count:
    outcome = row.get("outcome")
    if outcome == NO_VERDICT:
        return Count(unjudged=1)
    if outcome == "win":
        return Count(wins=1)
    if outcome == "loss":
        return Count(losses=1)
    if outcome == "tie":
        return Count(ties=1)
    raise ValueError(f"outcome is win, tie, loss or {NO_VERDICT}, got {outcome!r}")


def _from_outcomes(record: Mapping[str, Any], pair: str, kind: str) -> Count:
    found = Count()
    for row in record.get("outcomes", ()):
        if row.get("pair") == pair and row.get("kind") == kind:
            found += _count(row)
    return found


def _from_pair_dimension(record: Mapping[str, Any], pair: str) -> Count | None:
    """The pair's stored knowledge dimension when there are no per-kind rows (same probes as new_work+vocab)."""
    stat = (record.get("pairs") or {}).get(pair, {}).get(DIMENSION)
    if not isinstance(stat, dict) or not stat.get("n"):
        return None
    return Count(int(stat["wins"]), int(stat["ties"]), int(stat["losses"]))


def _new_work(record: Mapping[str, Any], pair: str, kind: str) -> Count:
    from_outcomes = _from_outcomes(record, pair, kind)
    if from_outcomes.judged:
        return from_outcomes
    stored = _from_pair_dimension(record, pair)
    return stored if stored is not None else Count()


def _fairness_kind() -> str:
    return str(json.loads(THRESHOLDS.read_text(encoding="utf-8"))["fairness"]["kind"])


def pooled_lives() -> tuple[str, ...]:
    raw = json.loads(THRESHOLDS.read_text(encoding="utf-8"))["development"]["pooled_lives"]
    return tuple(str(x) for x in raw)


def pooled_counts(root: Path, lives: Sequence[str], *, kind: str) -> tuple[Count, Count]:
    """Pooled ``new_work`` (or registered fairness kind) counts for both fairness comparisons."""
    total_2c, total_2f = Count(), Count()
    for person in lives:
        folder = root / person / "scores"
        for (_night, probe_set), record in score_records(folder).items():
            if probe_set != FULL_SET or record.get("label") != "development":
                continue
            total_2c += _new_work(record, PAIR_2C, kind)
            total_2f += _new_work(record, PAIR_2F, kind)
    return total_2c, total_2f


def recommend(
    root: Path, lives: Sequence[str] | None = None, *, kind: str | None = None
) -> dict[str, str]:
    """Recommend ``2C`` or ``2F`` from pooled development scores on ``kind`` (default: registered fairness kind)."""
    if kind is None:
        kind = _fairness_kind()
    people = tuple(lives) if lives is not None else pooled_lives()
    count_2c, count_2f = pooled_counts(root, people, kind=kind)
    score_2c, score_2f = count_2c.score, count_2f.score
    margin = None if score_2c is None or score_2f is None else round(abs(score_2c - score_2f), 4)
    if score_2c is None or score_2f is None:
        return {
            "baseline": DEFAULT,
            "reason": f"inconclusive: missing pooled {kind} scores for {PAIR_2C} or {PAIR_2F}; default {DEFAULT}",
        }
    if score_2c == score_2f:
        return {
            "baseline": DEFAULT,
            "reason": f"inconclusive: tied at {score_2c:.4f} on pooled {kind} (margin {margin}); default {DEFAULT}",
        }
    if score_2c < score_2f:
        return {
            "baseline": "2C",
            "reason": f"2C stronger on pooled {kind}: arm 4 scored {score_2c:.4f} vs 2C and {score_2f:.4f} vs 2F"
            f"{f' (margin {margin})' if margin else ''}",
        }
    return {
        "baseline": "2F",
        "reason": f"2F stronger on pooled {kind}: arm 4 scored {score_2f:.4f} vs 2F and {score_2c:.4f} vs 2C"
        f"{f' (margin {margin})' if margin else ''}",
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path | None = None,
    out: Callable[[str], None] = print,
) -> int:
    parser = argparse.ArgumentParser(
        description="Recommend the live demo baseline (2C or 2F) from development scores."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="results root (default: $ANJO_DATA_DIR/sleep/results)",
    )
    args = parser.parse_args(argv)
    chosen = recommend(results_root(args.root or root))
    if chosen["baseline"] not in BASELINES:
        raise ValueError(f"recommended baseline is one of {BASELINES}")
    out(json.dumps(chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
