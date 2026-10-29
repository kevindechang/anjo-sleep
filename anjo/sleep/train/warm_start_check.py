"""Warm start vs clean-slate on development mornings (Task 5, spec §5.1b).

Given two sets of morning score records for the same development nights — warm compound
training vs clean-slate retrain (``metadata.training_policy`` when present) — pool each
policy's probes over the registered development lives and compare ``3_vs_1`` and ``3_vs_3x``
per dimension with the development win rule (same CI, score and decisive share as final;
no lives-ahead rule). Offline only: reads score JSON, never judges.

    python -m anjo.sleep.train.warm_start_check --warm warm.jsonl --clean clean.jsonl

One JSON object per line on stdout.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, cast

from anjo.sleep.probes.score import BootstrapConfig, Outcome, score_outcomes
from anjo.sleep.types import Probe

Policy = Literal["warm_start", "clean_slate"]
Pick = Literal["warm_start", "clean_slate", "inconclusive"]
Outcome3 = Literal["win", "loss", "inconclusive"]
Dimension = Literal["knowledge", "interaction"]

THRESHOLDS: Final = Path(__file__).parents[1] / "prereg" / "thresholds.json"
COMP_3V1: Final = "3_vs_1"
COMP_3V3X: Final = "3_vs_3x"
COMPARISONS: Final = (COMP_3V1, COMP_3V3X)
DIMENSIONS: Final = ("knowledge", "interaction")
KNOWLEDGE_KINDS: Final = frozenset({"new_work", "vocab"})
INTERACTION_KINDS: Final = frozenset({"tone_hold", "taste_ab"})
DONOR_UNAVAILABLE: Final = "donor_unavailable"
WARM_LABEL: Final = "warm_compound"
CLEAN_LABEL: Final = "clean_slate"


class WarmStartCheckError(ValueError):
    """The comparison refuses to run."""


def _kinds(dimension: Dimension) -> frozenset[str]:
    return KNOWLEDGE_KINDS if dimension == "knowledge" else INTERACTION_KINDS


def _probe(life_id: str, row: Mapping[str, Any]) -> Probe:
    pid = str(row["probe"])
    kind = str(row["kind"])
    source = pid if kind != "new_work" else pid.rsplit("-", 1)[0] if "-" in pid else pid
    return Probe(
        id=pid,
        person_id=life_id,
        kind=kind,
        prompt="",
        options=None,
        expected=None,
        source_ids=(source,),
    )


def _life_id(record: Mapping[str, Any]) -> str:
    for key in ("life", "life_id", "person", "account"):
        if record.get(key):
            return str(record[key])
    raise WarmStartCheckError("each score record needs life, life_id, person or account")


def _night(record: Mapping[str, Any]) -> str:
    night = record.get("night")
    if night is None:
        raise WarmStartCheckError("each score record needs a night")
    return str(night)


def _donor_excluded(records: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    excluded: set[str] = set()
    for record in records:
        unavailable = record.get("unavailable")
        if isinstance(unavailable, dict) and unavailable.get("3x") == DONOR_UNAVAILABLE:
            excluded.add(_life_id(record))
    return frozenset(excluded)


def pool_comparison(
    records: Sequence[Mapping[str, Any]],
    comparison: str,
    dimension: Dimension,
    *,
    exclude_lives: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    """Pooled development stats for one comparison and dimension from morning outcome rows."""
    probes: list[Probe] = []
    outcomes: list[Outcome] = []
    per_life: dict[str, dict[str, Any]] = {}
    allowed = _kinds(dimension)
    for record in records:
        life_id = _life_id(record)
        if life_id in exclude_lives:
            continue
        rows = record.get("outcomes")
        if not isinstance(rows, list):
            continue
        life_outcomes: list[Outcome] = []
        for row in rows:
            if not isinstance(row, dict) or row.get("pair") != comparison:
                continue
            kind = row.get("kind")
            if kind not in allowed:
                continue
            outcome = row.get("outcome")
            if outcome not in ("win", "tie", "loss"):
                continue
            o = cast(Outcome, outcome)
            life_outcomes.append(o)
            probes.append(_probe(life_id, row))
            outcomes.append(o)
        if life_outcomes:
            n = len(life_outcomes)
            wins = life_outcomes.count("win")
            ties = life_outcomes.count("tie")
            losses = life_outcomes.count("loss")
            per_life[life_id] = {
                "n": n,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "score": (wins + 0.5 * ties) / n,
                "ahead": (wins + 0.5 * ties) / n > 0.5,
            }
    if not probes:
        return None
    scored = score_outcomes(probes, outcomes, bootstrap=BootstrapConfig.from_thresholds())
    dim = scored.per_dimension.get(dimension)
    if dim is None:
        return None
    return {
        "n": dim.wins + dim.ties + dim.losses,
        "wins": dim.wins,
        "ties": dim.ties,
        "losses": dim.losses,
        "score": dim.score,
        "decisive_share": dim.decisive_share,
        "ci": list(dim.ci) if dim.ci else None,
        "per_life": per_life,
    }


def headline_outcome(
    stats: Mapping[str, Any] | None, win: Mapping[str, Any], *, final: bool
) -> Outcome3:
    """Registered win rule; ``final`` applies the lives-ahead requirement, development does not."""
    if stats is None or stats.get("n", 0) == 0:
        return "inconclusive"
    ci = stats.get("ci")
    lower = float(ci[0]) if isinstance(ci, list) and len(ci) == 2 else None
    score = float(stats["score"])
    share = stats.get("decisive_share")
    decisive = float(share) if share is not None else None
    lives_ahead = sum(1 for life in stats.get("per_life", {}).values() if life.get("ahead"))
    need = int(win["lives_ahead_at_least"])
    total = int(win["lives_total"])
    checks_win = (
        lower is not None
        and lower > float(win["ci_lower_bound_above"])
        and score >= float(win["overall_score_at_least"])
        and decisive is not None
        and decisive >= float(win["decisive_share_at_least"])
    )
    if final:
        checks_win = checks_win and lives_ahead >= need
    checks_loss = (
        lower is not None
        and lower < (1.0 - float(win["ci_lower_bound_above"]))
        and score <= (1.0 - float(win["overall_score_at_least"]))
        and decisive is not None
        and decisive <= (1.0 - float(win["decisive_share_at_least"]))
    )
    if final:
        checks_loss = checks_loss and lives_ahead <= (total - need)
    if checks_win:
        return "win"
    if checks_loss:
        return "loss"
    return "inconclusive"


def development_outcome(stats: Mapping[str, Any] | None, thresholds: Mapping[str, Any]) -> Outcome3:
    return headline_outcome(stats, thresholds["win"], final=False)


def final_outcome(stats: Mapping[str, Any] | None, thresholds: Mapping[str, Any]) -> Outcome3:
    return headline_outcome(stats, thresholds["win"], final=True)


def pick_policy(
    warm: Mapping[str, Any] | None,
    clean: Mapping[str, Any] | None,
    thresholds: Mapping[str, Any],
) -> Pick:
    """Which training policy wins this comparison for one dimension, development rule."""
    wo = development_outcome(warm, thresholds)
    co = development_outcome(clean, thresholds)
    if wo == "win" and co != "win":
        return "warm_start"
    if co == "win" and wo != "win":
        return "clean_slate"
    if wo == "loss" and co != "loss":
        return "clean_slate"
    if co == "loss" and wo != "loss":
        return "warm_start"
    return "inconclusive"


def _match_nights(warm: Sequence[Mapping[str, Any]], clean: Sequence[Mapping[str, Any]]) -> None:
    warm_nights = {(_life_id(r), _night(r)) for r in warm}
    clean_nights = {(_life_id(r), _night(r)) for r in clean}
    if warm_nights != clean_nights:
        raise WarmStartCheckError("warm and clean scores must cover the same development nights")


def _check_policy_labels(records: Sequence[Mapping[str, Any]], want: str, name: str) -> None:
    for record in records:
        meta = record.get("metadata")
        if isinstance(meta, dict) and meta.get("training_policy") not in (None, want):
            raise WarmStartCheckError(
                f"{name} record has metadata.training_policy {meta['training_policy']!r}, want {want!r}"
            )


def compare_policies(
    warm_scores: Sequence[Mapping[str, Any]],
    clean_scores: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare warm compound vs clean-slate retrain; return recommendation and per-dimension table."""
    if not warm_scores or not clean_scores:
        raise WarmStartCheckError("need at least one warm and one clean score record")
    _check_policy_labels(warm_scores, WARM_LABEL, "warm")
    _check_policy_labels(clean_scores, CLEAN_LABEL, "clean")
    _match_nights(warm_scores, clean_scores)
    excluded_warm = _donor_excluded(warm_scores)
    excluded_clean = _donor_excluded(clean_scores)
    excluded_3x = excluded_warm | excluded_clean
    table: dict[str, Any] = {}
    dim_winners: list[Pick] = []
    for dimension in DIMENSIONS:
        row: dict[str, Any] = {"dimension": dimension, "comparisons": {}}
        picks: list[Pick] = []
        for comparison in COMPARISONS:
            exclude = excluded_3x if comparison == COMP_3V3X else frozenset()
            warm_stats = pool_comparison(warm_scores, comparison, dimension, exclude_lives=exclude)
            clean_stats = pool_comparison(
                clean_scores, comparison, dimension, exclude_lives=exclude
            )
            choice = pick_policy(warm_stats, clean_stats, thresholds)
            picks.append(choice)
            row["comparisons"][comparison] = {
                "warm": warm_stats,
                "clean": clean_stats,
                "warm_outcome": development_outcome(warm_stats, thresholds),
                "clean_outcome": development_outcome(clean_stats, thresholds),
                "winner": choice,
            }
        if picks[0] == picks[1] != "inconclusive":
            dim_winner: Pick = picks[0]
        else:
            dim_winner = "inconclusive"
        row["winner"] = dim_winner
        dim_winners.append(dim_winner)
        table[dimension] = row
    if dim_winners[0] == dim_winners[1] == "warm_start":
        recommendation: Pick = "warm_start"
    elif dim_winners[0] == dim_winners[1] == "clean_slate":
        recommendation = "clean_slate"
    else:
        recommendation = "inconclusive"
    return {"recommendation": recommendation, "dimensions": table}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise WarmStartCheckError(f"{path}: each line is a JSON object")
        found.append(row)
    return found


def main(argv: Sequence[str] | None = None, *, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(
        description="Warm start vs clean-slate on development morning scores."
    )
    parser.add_argument(
        "--warm", type=Path, required=True, help="JSONL of warm-compound score records"
    )
    parser.add_argument(
        "--clean", type=Path, required=True, help="JSONL of clean-slate score records"
    )
    parser.add_argument(
        "--thresholds", type=Path, default=THRESHOLDS, help="thresholds.json (default: prereg)"
    )
    args = parser.parse_args(argv)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    result = compare_policies(_read_jsonl(args.warm), _read_jsonl(args.clean), thresholds)
    out(json.dumps({"line": "recommendation", "policy": result["recommendation"]}))
    for dimension in DIMENSIONS:
        row = result["dimensions"][dimension]
        out(json.dumps({"line": "dimension", **row}, default=_json_default))
    return 0


def _json_default(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


if __name__ == "__main__":
    raise SystemExit(main())
