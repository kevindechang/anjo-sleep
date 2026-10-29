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
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal, cast

from anjo.sleep.app.night import SEALED_ENV
from anjo.sleep.probes.generate import load_sealed
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


def _donor_personas() -> frozenset[str]:
    return frozenset(p for p in pooled_personas() if p.endswith("_donor"))


def _unpromoted_donor(root: Path, persona: str, account: str) -> bool:
    if persona not in _donor_personas():
        return False
    basis = str(latest_full_score(root, account).get("checkpoint_basis") or "promoted")
    return basis == "not_promoted"


def _pool_accounts(pair: str, root: Path, accounts: Mapping[str, str]) -> dict[str, str]:
    """Accounts whose outcomes count toward the pooled score and pool-ready gate."""
    if pair != "3_vs_3x":
        return dict(accounts)
    return {
        persona: account
        for persona, account in accounts.items()
        if not _unpromoted_donor(root, persona, account)
    }


def min_probes_per_dimension() -> int:
    return int(_thresholds()["development"]["min_probes_per_dimension"])


def sealed_dir(path: Path | None = None) -> Path:
    """Directory of ``<persona>.jsonl`` sealed probes (same layout as the morning run)."""
    if path is not None:
        return path
    raw = os.environ.get(SEALED_ENV, "").strip()
    if not raw:
        raise DevelopmentError(f"{SEALED_ENV} is not set; probe ids need the registered sealed set")
    return Path(raw)


@lru_cache(maxsize=32)
def _registered_probes(directory: str, persona: str) -> dict[str, Probe]:
    path = Path(directory) / f"{persona}.jsonl"
    if not path.is_file():
        raise DevelopmentError(f"sealed probes missing for persona {persona}: {path.name}")
    by_id = {probe.id: probe for probe in load_sealed(path)}
    if not by_id:
        raise DevelopmentError(f"sealed probes for persona {persona} are empty")
    return by_id


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


def _probe(
    row: Mapping[str, Any],
    person: str,
    persona: str,
    *,
    probes_root: Path,
) -> Probe:
    probe_id = str(row["probe"])
    kind = str(row["kind"])
    registered = _registered_probes(str(probes_root.resolve()), persona)
    if probe_id not in registered:
        raise DevelopmentError(f"probe {probe_id!r} is not in the sealed set for {persona}")
    sealed = registered[probe_id]
    if kind != sealed.kind:
        raise DevelopmentError(
            f"score row kind {kind!r} disagrees with sealed {sealed.kind!r} for {probe_id}"
        )
    return Probe(probe_id, person, cast(ProbeKind, kind), "", None, None, sealed.source_ids)


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
    probes_root: Path | None = None,
) -> list[tuple[Probe, Outcome]]:
    """Every judged probe outcome for ``pair`` across the pooled accounts."""
    pending: list[tuple[Mapping[str, Any], str, str, Outcome]] = []
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
            pending.append((row, account, persona, outcome))
    if not pending:
        return []
    sealed = sealed_dir(probes_root)
    return [
        (_probe(row, account, persona, probes_root=sealed), outcome)
        for row, account, persona, outcome in pending
    ]


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
        return 0, 0, 0, 0.5, (0.0, 0.0), None
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
    return WinRule(
        round(1.0 - rule.ci_above, 12),
        round(1.0 - rule.score_at_least, 12),
        round(1.0 - rule.decisive_at_least, 12),
    )


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


def _pool_ready(
    rows: Sequence[tuple[Probe, Outcome]],
    accounts: Mapping[str, str],
) -> bool:
    """Every pooled life contributed at least one judged probe, with enough per dimension."""
    n = len(rows)
    if n < min_probes_per_dimension():
        return False
    lives = {probe.person_id for probe, _ in rows}
    return len(lives) == len(accounts)


def _label_comparison(
    pair: str,
    *,
    rows: Sequence[tuple[Probe, Outcome]],
    accounts: Mapping[str, str],
    wins: int,
    ties: int,
    losses: int,
    score: float,
    ci: tuple[float, float],
    decisive: float | None,
    rule: WinRule,
    margin: float,
) -> Label:
    if not _pool_ready(rows, accounts):
        return "inconclusive"
    if pair in FAIRNESS:
        return label_fairness(ci=ci, score=score, decisive=decisive, rule=rule, margin=margin)
    return label_headline(ci=ci, score=score, decisive=decisive, rule=rule)


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


def _life_result_line(
    persona: str,
    pair: str,
    dimension: str,
    wins: int,
    ties: int,
    losses: int,
    score: float,
    decisive: float | None,
) -> dict[str, Any]:
    n = wins + ties + losses
    line: dict[str, Any] = {
        "line": "life",
        "persona": persona,
        "pair": pair,
        "dimension": dimension,
        "n": n,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }
    if n:
        line["score"] = round(score, 4)
        if decisive is not None:
            line["decisive_share"] = round(decisive, 4)
    else:
        line["score"] = None
    return line


def life_scores(
    root: Path,
    accounts: Mapping[str, str],
    *,
    probes_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Each development life's tie-inclusive score per registered comparison and dimension (§5.1b)."""
    lines: list[dict[str, Any]] = []
    for pair in HEADLINE:
        for dimension in DIMENSIONS:
            for persona, account in sorted(accounts.items()):
                rows = pooled_rows(
                    root,
                    {persona: account},
                    pair,
                    kinds=_kinds_for_dimension(dimension),
                    probes_root=probes_root,
                )
                wins, ties, losses, score, _ci, decisive = score_pooled(rows, dimension)
                lines.append(
                    _life_result_line(persona, pair, dimension, wins, ties, losses, score, decisive)
                )
    for pair in FAIRNESS:
        for persona, account in sorted(accounts.items()):
            rows = pooled_rows(
                root,
                {persona: account},
                pair,
                kinds=frozenset({"new_work"}),
                probes_root=probes_root,
            )
            wins, ties, losses, score, _ci, decisive = score_pooled(rows, "knowledge")
            lines.append(
                _life_result_line(persona, pair, "knowledge", wins, ties, losses, score, decisive)
            )
    return lines


def comparisons(
    root: Path,
    accounts: Mapping[str, str],
    *,
    bootstrap: BootstrapConfig | None = None,
    probes_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Every registered development comparison and dimension, pooled over ``accounts``."""
    _validate_accounts(accounts)
    rule = win_rule()
    margin = fairness_margin()
    lines: list[dict[str, Any]] = []
    for pair in HEADLINE:
        pool = _pool_accounts(pair, root, accounts)
        for dimension in DIMENSIONS:
            rows = pooled_rows(
                root,
                pool,
                pair,
                kinds=_kinds_for_dimension(dimension),
                probes_root=probes_root,
            )
            wins, ties, losses, score, ci, decisive = score_pooled(
                rows, dimension, bootstrap=bootstrap
            )
            label = _label_comparison(
                pair,
                rows=rows,
                accounts=pool,
                wins=wins,
                ties=ties,
                losses=losses,
                score=score,
                ci=ci,
                decisive=decisive,
                rule=rule,
                margin=margin,
            )
            lines.append(
                _result_line(pair, dimension, wins, ties, losses, score, ci, decisive, label)
            )
    for pair in FAIRNESS:
        rows = pooled_rows(
            root,
            accounts,
            pair,
            kinds=frozenset({"new_work"}),
            probes_root=probes_root,
        )
        wins, ties, losses, score, ci, decisive = score_pooled(
            rows, "knowledge", bootstrap=bootstrap
        )
        label = _label_comparison(
            pair,
            rows=rows,
            accounts=accounts,
            wins=wins,
            ties=ties,
            losses=losses,
            score=score,
            ci=ci,
            decisive=decisive,
            rule=rule,
            margin=margin,
        )
        lines.append(
            _result_line(pair, "knowledge", wins, ties, losses, score, ci, decisive, label)
        )
    return lines


def read(
    root: Path,
    accounts: Mapping[str, str],
    *,
    bootstrap: BootstrapConfig | None = None,
    probes_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Header line plus every comparison line."""
    _validate_accounts(accounts)
    header = {
        "line": "development",
        "personas": dict(sorted(accounts.items())),
        "probe_set": FULL_SET,
    }
    return [
        header,
        *life_scores(root, accounts, probes_root=probes_root),
        *comparisons(root, accounts, bootstrap=bootstrap, probes_root=probes_root),
    ]


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path | None = None,
    out: Callable[[str], None] = print,
    bootstrap: BootstrapConfig | None = None,
    probes_root: Path | None = None,
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
    parser.add_argument(
        "--sealed-dir",
        type=Path,
        default=None,
        help=f"sealed probe files (default: ${SEALED_ENV})",
    )
    args = parser.parse_args(argv)
    accounts = _accounts(parser, args.account)
    base = results_root(args.root or root)
    sealed = sealed_dir(args.sealed_dir or probes_root)
    for line in read(base, accounts, bootstrap=bootstrap, probes_root=sealed):
        out(json.dumps(line, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
