"""The comparison's mornings by dimension: each arm's stored daily-v1 scores, arm 3 (the person's weights,
no context) against arm 1 (base), split by the spec's dimensions (``python -m anjo.sleep.tools.morning_dimensions``).

    python -m anjo.sleep.tools.morning_dimensions --account seed=<id> --account today=<id> ...

It reads the score files the mornings wrote (``probes.morning``, ``<results>/<account>/scores``), never
anything else, and calls nothing. Per account and daily-v1 morning, for the ``3_vs_1`` comparison:

- **knowledge** (of her world: ``new_work``, ``vocab``) and **interaction** (taste and style:
  ``tone_hold``, ``taste_ab``), the registered dimensions (``prereg/thresholds.json``), each with the
  counts recounted from the file's outcome rows (probes, orders, wins, ties, the ties an order flip made,
  losses, probes with no verdict), the tie-inclusive score, and the block-bootstrap interval the morning's
  scorer stored (``score.PairScore``);
- the same counts per probe kind, with no interval (the scorer computes none per kind).

Per account, the mornings pooled: summed counts and their score, with no interval. Every daily-v1 morning
asks the same probes (``morning.subset_probes``), so its mornings are repeated measures of a handful of
probes, never new ones, and an interval that resampled them as independent would be too narrow. An
account with no daily-v1 morning says so. One JSON line each: ids, codes and numbers only.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from anjo.sleep.probes.morning import DAILY, NO_VERDICT
from anjo.sleep.probes.score import REPORTED, dimension_of
from anjo.sleep.reports.results import results_root, score_records

PAIR: Final = "3_vs_1"  # arm 3 against arm 1: the weights alone against base
ORDERS: Final = 2  # each probe is asked in both orders (a judge's, or a forced choice's options)
NO_POOLED_CI: Final = "the same probes each morning: pooled mornings are repeated measures, so no interval"
MISSING: Final = f"no {DAILY.name} morning scores"


@dataclass(frozen=True)
class Count:
    """One comparison's outcomes over some probes, as the morning's outcome rows give them."""

    wins: int = 0
    ties: int = 0
    flips: int = 0  # the ties an order flip made (the judge, or a forced choice, changed with the order)
    losses: int = 0
    unjudged: int = 0  # no readable verdict: left out of the score, as the morning leaves it out

    def __add__(self, other: Count) -> Count:
        return Count(self.wins + other.wins, self.ties + other.ties, self.flips + other.flips,
                     self.losses + other.losses, self.unjudged + other.unjudged)

    def record(self) -> dict[str, Any]:
        probes = self.wins + self.ties + self.losses + self.unjudged
        judged = self.wins + self.ties + self.losses
        return {"probes": probes, "orders": ORDERS * probes, "wins": self.wins, "ties": self.ties,
                "flips": self.flips, "losses": self.losses, "unjudged": self.unjudged,
                "score": round((self.wins + self.ties / 2) / judged, 4) if judged else None}


def _count(row: Mapping[str, Any]) -> Count:
    outcome = row.get("outcome")
    if outcome == NO_VERDICT:
        return Count(unjudged=1)
    if outcome == "win":
        return Count(wins=1)
    if outcome == "loss":
        return Count(losses=1)
    if outcome == "tie":
        return Count(ties=1, flips=int(bool(row.get("flip"))))
    raise ValueError(f"an outcome row's outcome is win, tie, loss or {NO_VERDICT}, got {outcome!r}")


def kind_counts(record: Mapping[str, Any]) -> dict[str, Count]:
    """The comparison's counts per probe kind, from the score file's outcome rows."""
    found: dict[str, Count] = {}
    for row in record.get("outcomes", ()):
        if row.get("pair") == PAIR:
            found[str(row["kind"])] = found.get(str(row["kind"]), Count()) + _count(row)
    return dict(sorted(found.items()))


def dimension_counts(kinds: Mapping[str, Count]) -> dict[str, Count]:
    """Each dimension's counts: the sum of its kinds' (``pushback``, reported only, has none)."""
    mapping = dimension_of()
    found: dict[str, Count] = {}
    for kind, counted in kinds.items():
        name = mapping[kind]
        if name != REPORTED:
            found[name] = found.get(name, Count()) + counted
    return dict(sorted(found.items()))


def mornings(root: Path, account: str) -> list[dict[str, Any]]:
    """The account's daily-v1 score records with arm 3 scored, in night order."""
    return [record for (_night, probe_set), record in score_records(root / account / "scores").items()
            if probe_set == DAILY.name and "3" in record.get("arms", {})]


def night_line(arm: str, record: Mapping[str, Any]) -> dict[str, Any]:
    kinds = kind_counts(record)
    stored = record.get("arms", {}).get("3", {})
    return {"line": "night", "arm": arm, "night": record["night"], "checkpoint": record.get("checkpoint"),
            "dimensions": {name: {**counted.record(), "ci": (stored.get(name) or {}).get("ci")}
                           for name, counted in dimension_counts(kinds).items()},
            "kinds": {kind: counted.record() for kind, counted in kinds.items()}}


def arm_line(arm: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    kinds: dict[str, Count] = {}
    for record in records:
        for kind, counted in kind_counts(record).items():
            kinds[kind] = kinds.get(kind, Count()) + counted
    return {"line": "arm", "arm": arm, "mornings": len(records), "nights": [r["night"] for r in records],
            "dimensions": {name: {**counted.record(), "ci": None} for name, counted in dimension_counts(kinds).items()},
            "kinds": {kind: counted.record() for kind, counted in sorted(kinds.items())}, "why_no_ci": NO_POOLED_CI}


def probe_line() -> dict[str, Any]:
    """What one daily-v1 morning asks per dimension, and how far one probe moves its score."""
    per_kind = {kind: n for kind, n in DAILY.per_kind.items() if n}
    mapping = dimension_of()
    sizes: dict[str, int] = {}
    for kind, n in per_kind.items():
        if mapping[kind] != REPORTED:
            sizes[mapping[kind]] = sizes.get(mapping[kind], 0) + n
    return {"line": "probes", "set": DAILY.name, "per_kind": per_kind, "dimensions": sizes,
            "one_probe_moves": {name: round(1 / n, 4) for name, n in sizes.items()}, "same_probes_each_morning": True}


def _accounts(parser: argparse.ArgumentParser, pairs: Sequence[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for pair in pairs:
        arm, _, account = pair.partition("=")
        if not arm or not account or arm in found:
            parser.error("--account is <arm>=<account id>, once per arm")
        found[arm] = account
    return found


def main(argv: Sequence[str] | None = None, *, root: Path | None = None, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description="Each arm's daily-v1 mornings, arm 3 against base, by dimension.")
    parser.add_argument("--account", action="append", default=[], help="<arm>=<account id>, once per arm")
    args = parser.parse_args(argv)
    accounts = _accounts(parser, args.account)
    if not accounts:
        parser.error("name at least one --account")
    base = results_root(root)
    out(json.dumps(probe_line()))
    for arm, account in accounts.items():
        found = mornings(base, account)
        if not found:
            out(json.dumps({"line": "missing", "arm": arm, "account": account, "why": MISSING}))
            continue
        for record in found:
            out(json.dumps(night_line(arm, record)))
        out(json.dumps(arm_line(arm, found)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
