"""Development go/no-go read (spec §5.1b, Oct 31): pool the four registered development lives' full
morning scores, re-bootstrap each headline and fairness comparison per dimension, and label win,
loss, inconclusive or (fairness only) match from ``prereg/thresholds.json``.

    python -m anjo.sleep.reports.development --root <dir> --account founder=<id> ...

It reads only score files on the full probe set (``results.FULL_SET``), never a daily subset. One JSON
object per line: ids and numbers only.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal, cast

from anjo.sleep.probes.morning import NO_VERDICT
from anjo.sleep.probes.score import (
    REPORTED,
    BootstrapConfig,
    Outcome,
    dimension_of,
    score_outcomes,
)
from anjo.sleep.reports.results import FULL_SET, results_root
from anjo.sleep.types import Probe, ProbeKind

THRESHOLDS: Final = Path(__file__).parents[1] / "prereg" / "thresholds.json"
HEADLINE: Final = ("3_vs_1", "3_vs_3x")
FAIRNESS: Final = ("4_vs_2C", "4_vs_2F")
DIMENSIONS: Final = ("knowledge", "interaction")
Label = Literal["win", "loss", "inconclusive", "match"]


class DevelopmentError(ValueError):
    """A development read was refused or the results tree is missing something it needs."""


@dataclass(frozen=True)
class WinRule:
    ci_above: float
    score_at_least: float
    decisive_at_least: float


@lru_cache(maxsize=1)
def _thresholds() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(THRESHOLDS.read_text(encoding="utf-8")))


def win_rule() -> WinRule:
    raw = _thresholds()["win"]
    return WinRule(
        float(raw["ci_lower_bound_above"]),
        float(raw["overall_score_at_least"]),
        float(raw["decisive_share_at_least"]),
    )


def fairness_margin() -> float:
    return float(_thresholds()["fairness"]["non_inferiority_margin"])


def pooled_personas() -> tuple[str, ...]:
    return tuple(_thresholds()["development"]["pooled_lives"])


def _accounts(parser: argparse.ArgumentParser, pairs: Sequence[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for pair in pairs:
        persona, _, account = pair.partition("=")
        if not persona or not account or persona in found:
            parser.error("--account is <persona>=<account id>, once per persona")
        found[persona] = account
    return found


def _validate_accounts(accounts: Mapping[str, str]) -> None:
    want = pooled_personas()
    if set(accounts) != set(want):
        raise DevelopmentError(
            f"--account names exactly {', '.join(want)}; got {', '.join(sorted(accounts)) or 'none'}"
        )
    if len(accounts) != len(want):
        raise DevelopmentError(f"the development read needs {len(want)} accounts")


def latest_full_score(root: Path, account: str) -> dict[str, Any]:
    """The newest full morning score file for ``account``; daily subsets are ignored."""
    folder = root / account / "scores"
    best: tuple[tuple[str, int], dict[str, Any]] | None = None
    for path in folder.glob("night-*.json") if folder.is_dir() else ():
        record = json.loads(path.read_text(encoding="utf-8"))
        if str(record.get("subset") or FULL_SET) != FULL_SET:
            continue
        if "night" not in record:
            raise DevelopmentError(f"{path.name} lacks a night")
        number = path.stem.rsplit("-", 1)[1]
        if not number.isdigit():
            raise DevelopmentError(f"{path.name} is not night-NN")
        key = (str(record["night"]), int(number))
        if best is None or key > best[0]:
            best = (key, record)
    if best is None:
        raise DevelopmentError(
            f"{account}: no full morning score file (daily subsets are not read)"
        )
    return best[1]


def _probe(row: Mapping[str, Any], person: str) -> Probe:
    probe_id = str(row["probe"])
    kind = str(row["kind"])
    if kind == "new_work":
        source = str(row.get("source") or f"{probe_id}-src")
    elif kind == "taste_ab":
        source = str(row.get("source") or probe_id)
    else:
        source = probe_id
    return Probe(probe_id, person, cast(ProbeKind, kind), "", None, None, (source,))


def _outcome(row: Mapping[str, Any]) -> Outcome | None:
    raw = row.get("outcome")
    if raw == NO_VERDICT:
        return None
    if raw in ("win", "tie", "loss"):
        return cast(Outcome, raw)
    raise DevelopmentError(f"outcome row outcome is win, tie, loss or {NO_VERDICT}, got {raw!r}")


def pooled_rows(
    root: Path,
    accounts: Mapping[str, str],
    pair: str,
    *,
    kinds: frozenset[str] | None = None,
) -> list[tuple[Probe, Outcome]]:
    """Every judged probe outcome for ``pair`` across the pooled accounts."""
    found: list[tuple[Probe, Outcome]] = []
    for persona, account in accounts.items():
        record = latest_full_score(root, account)
        for row in record.get("outcomes", ()):
            if row.get("pair") != pair:
                continue
            kind = str(row.get("kind", ""))
            if kinds is not None and kind not in kinds:
                continue
            outcome = _outcome(row)
            if outcome is None:
                continue
            found.append((_probe(row, account), outcome))
    return found


def _kinds_for_dimension(dimension: str) -> frozenset[str]:
    raw = _thresholds()["dimensions"][dimension]["kinds"]
    return frozenset(str(k) for k in raw)


def score_pooled(
    rows: Sequence[tuple[Probe, Outcome]],
    dimension: str,
    *,
    bootstrap: BootstrapConfig | None = None,
) -> tuple[int, int, int, float, tuple[float, float], float | None]:
    if not rows:
        return 0, 0, 0, 0.5, (0.5, 0.5), None
    probes, outcomes = zip(*rows, strict=True)
    dims = dimension_of()
    if any(dims[p.kind] not in (dimension, REPORTED) for p in probes):
        raise DevelopmentError(f"pooled rows for {dimension} include another dimension's kinds")
    done = score_outcomes(
        probes, outcomes, bootstrap=bootstrap or BootstrapConfig.from_thresholds()
    )
    tally = done.per_dimension[dimension]
    assert tally.ci is not None
    return tally.wins, tally.ties, tally.losses, tally.score, tally.ci, tally.decisive_share


def _loss_rule(rule: WinRule) -> WinRule:
    return WinRule(1.0 - rule.ci_above, 1.0 - rule.score_at_least, 1.0 - rule.decisive_at_least)


def label_headline(
    *,
    ci: tuple[float, float],
    score: float,
    decisive: float | None,
    rule: WinRule | None = None,
) -> Label:
    registered = rule or win_rule()
    loss = _loss_rule(registered)
    if (
        ci[0] > registered.ci_above
        and score >= registered.score_at_least
        and decisive is not None
        and decisive >= registered.decisive_at_least
    ):
        return "win"
    if (
        ci[1] < loss.ci_above
        and score <= loss.score_at_least
        and decisive is not None
        and decisive <= loss.decisive_at_least
    ):
        return "loss"
    return "inconclusive"


def label_fairness(
    *,
    ci: tuple[float, float],
    score: float,
    decisive: float | None,
    rule: WinRule | None = None,
    margin: float | None = None,
) -> Label:
    headline = label_headline(ci=ci, score=score, decisive=decisive, rule=rule)
    if headline != "inconclusive":
        return headline
    if ci[0] > (margin if margin is not None else fairness_margin()):
        return "match"
    return "inconclusive"


def _result_line(
    pair: str,
    dimension: str,
    wins: int,
    ties: int,
    losses: int,
    score: float,
    ci: tuple[float, float],
    decisive: float | None,
    label: Label,
) -> dict[str, Any]:
    n = wins + ties + losses
    line: dict[str, Any] = {
        "line": "comparison",
        "pair": pair,
        "dimension": dimension,
        "label": label,
        "n": n,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "score": round(score, 4),
        "ci": [round(ci[0], 4), round(ci[1], 4)],
    }
    if decisive is not None:
        line["decisive_share"] = round(decisive, 4)
    return line


def comparisons(
    root: Path,
    accounts: Mapping[str, str],
    *,
    bootstrap: BootstrapConfig | None = None,
) -> list[dict[str, Any]]:
    """Every registered development comparison and dimension, pooled over ``accounts``."""
    _validate_accounts(accounts)
    rule = win_rule()
    margin = fairness_margin()
    lines: list[dict[str, Any]] = []
    for pair in HEADLINE:
        for dimension in DIMENSIONS:
            rows = pooled_rows(root, accounts, pair, kinds=_kinds_for_dimension(dimension))
            wins, ties, losses, score, ci, decisive = score_pooled(
                rows, dimension, bootstrap=bootstrap
            )
            label = label_headline(ci=ci, score=score, decisive=decisive, rule=rule)
            lines.append(
                _result_line(pair, dimension, wins, ties, losses, score, ci, decisive, label)
            )
    for pair in FAIRNESS:
        rows = pooled_rows(root, accounts, pair, kinds=frozenset({"new_work"}))
        wins, ties, losses, score, ci, decisive = score_pooled(
            rows, "knowledge", bootstrap=bootstrap
        )
        label = label_fairness(ci=ci, score=score, decisive=decisive, rule=rule, margin=margin)
        lines.append(
            _result_line(pair, "knowledge", wins, ties, losses, score, ci, decisive, label)
        )
    return lines


def read(
    root: Path,
    accounts: Mapping[str, str],
    *,
    bootstrap: BootstrapConfig | None = None,
) -> list[dict[str, Any]]:
    """Header line plus every comparison line."""
    _validate_accounts(accounts)
    header = {
        "line": "development",
        "personas": dict(sorted(accounts.items())),
        "probe_set": FULL_SET,
    }
    return [header, *comparisons(root, accounts, bootstrap=bootstrap)]


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path | None = None,
    out: Callable[[str], None] = print,
    bootstrap: BootstrapConfig | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Development go/no-go from pooled full morning scores."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="results tree (default: ANJO_DATA_DIR/sleep/results)",
    )
    parser.add_argument(
        "--account", action="append", default=[], help="<persona>=<account id>, four personas"
    )
    args = parser.parse_args(argv)
    accounts = _accounts(parser, args.account)
    base = results_root(args.root or root)
    for line in read(base, accounts, bootstrap=bootstrap):
        out(json.dumps(line, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
