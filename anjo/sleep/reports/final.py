"""Final test report (Task 12b): held-out lives from morning score files and ``thresholds.json``.

Reads each life's last ``final`` morning score under ``<results_root>/<life>/scores/night-NN.json``,
never a earlier night (serving checkpoint after the last night, not the best-scoring night). Writes
``<results_root>/<life>/scores/final.json`` once; a second run refuses. The git commit must match
``freeze_tag`` (``sleep-freeze``). No live judging: aggregates registered rules only.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal, cast

from anjo.sleep.probes.morning import MORNING_PROBES, SUBSETS, subset_probes
from anjo.sleep.probes.score import BootstrapConfig, Outcome, score_outcomes
from anjo.sleep.reports.results import FULL_SET
from anjo.sleep.types import Probe

Outcome3 = Literal["win", "loss", "inconclusive"]
FairnessOutcome = Literal["win", "loss", "matches", "inconclusive"]
Dimension = Literal["knowledge", "interaction"]
REPO: Final = Path(__file__).parents[3]
THRESHOLDS: Final = Path(__file__).parents[1] / "prereg" / "thresholds.json"
SCORES: Final = "scores"
FINAL_NAME: Final = "final.json"
NIGHT_PREFIX: Final = "night-"
COMP_3V1: Final = "3_vs_1"
COMP_3V3X: Final = "3_vs_3x"
COMP_4V2C: Final = "4_vs_2C"
COMP_4V2F: Final = "4_vs_2F"
DIMENSIONS: Final = ("knowledge", "interaction")
KNOWLEDGE_KINDS: Final = frozenset({"new_work", "vocab"})
INTERACTION_KINDS: Final = frozenset({"tone_hold", "taste_ab"})
NEW_WORK: Final = "new_work"
DONOR_UNAVAILABLE: Final = "donor_unavailable"
GIT_TIMEOUT_S: Final = 30.0


class FinalError(ValueError):
    """The final report refuses to run."""


@dataclass(frozen=True)
class FinalReport:
    """The registered final analysis (also returned as a plain dict from ``run``)."""

    freeze_tag: str
    freeze_sha: str
    git_sha: str
    donors: Mapping[str, str]
    pair_lives: tuple[str, str]
    single_life: str
    dimensions: Mapping[str, Mapping[str, Any]]
    fairness: Mapping[str, Mapping[str, Any]]
    qualifiers: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "freeze_tag": self.freeze_tag,
            "freeze_sha": self.freeze_sha,
            "git_sha": self.git_sha,
            "donors": dict(self.donors),
            "pair_lives": list(self.pair_lives),
            "single_life": self.single_life,
            "dimensions": {k: dict(v) for k, v in self.dimensions.items()},
            "fairness": {k: dict(v) for k, v in self.fairness.items()},
            "qualifiers": [dict(q) for q in self.qualifiers],
        }


@lru_cache(maxsize=1)
def _thresholds() -> dict[str, Any]:
    found = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    if not isinstance(found, dict):
        raise FinalError("thresholds.json is not a record")
    return found


def freeze_sha(tag: str, repo: Path = REPO) -> str:
    """Resolve ``tag`` to a commit sha (local ``git rev-parse`` only)."""
    done = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{tag}^{{commit}}"],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_S,
        check=False,
    )
    sha = done.stdout.strip()
    if done.returncode != 0 or len(sha) != 40:
        raise FinalError(f"freeze tag {tag!r} does not resolve to a commit")
    return sha


def git_sha(repo: Path = REPO) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_S,
        check=False,
    )
    sha = done.stdout.strip()
    if done.returncode != 0 or len(sha) != 40:
        raise FinalError("this checkout's git sha can't be read")
    return sha


def score_path(results_root: Path, life_id: str) -> Path:
    """Per-life final analysis output (written once)."""
    return results_root / life_id / SCORES / FINAL_NAME


def _json(path: Path) -> dict[str, Any]:
    found = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(found, dict):
        raise FinalError(f"{path.name} is not a record")
    return found


def _night_number(path: Path) -> int:
    stem = path.stem
    if not stem.startswith(NIGHT_PREFIX):
        raise FinalError(f"{path.name} is not a night score file")
    suffix = stem[len(NIGHT_PREFIX) :]
    if not suffix.isdigit():
        raise FinalError(f"{path.name} lacks a night number")
    return int(suffix)


def _night_summary_probe_set(results_root: Path, life_id: str, night_iso: str) -> str | None:
    """The accelerated night's ``morning_probes`` when the score file names the night only."""
    folder = results_root / life_id / "nights"
    if not folder.is_dir():
        return None
    for summary in folder.glob("*/summary.json"):
        record = _json(summary)
        if str(record.get("night", summary.parent.name)) != night_iso:
            continue
        named = record.get(MORNING_PROBES)
        if named is not None:
            return str(named)
    return None


def _probe_set_name(record: Mapping[str, Any], results_root: Path, life_id: str) -> str:
    subset = record.get("subset")
    if subset is not None:
        return str(subset)
    if record.get("label") == "final":
        return str(FULL_SET)
    night = record.get("night")
    if isinstance(night, str):
        from_summary = _night_summary_probe_set(results_root, life_id, night)
        if from_summary is not None:
            return str(from_summary)
    return str(FULL_SET)


def registered_probes(
    life_id: str,
    record: Mapping[str, Any],
    sealed: Callable[[str], Sequence[Probe]],
    *,
    results_root: Path,
) -> dict[str, Probe]:
    """Probe ids in the set the morning score registered, with held-out ``source_ids`` for blocks."""
    name = _probe_set_name(record, results_root, life_id)
    probes = list(sealed(life_id))
    if name != FULL_SET:
        subset = SUBSETS.get(name)
        if subset is None:
            raise FinalError(f"{life_id}: probe set {name!r} is not registered")
        probes = subset_probes(probes, subset)
    return {p.id: p for p in probes}


def serving_morning_path(results_root: Path, life_id: str) -> Path:
    """The morning score for the checkpoint serving after the life's last night."""
    folder = results_root / life_id / SCORES
    if not folder.is_dir():
        raise FinalError(f"{life_id} has no scores folder")
    best: tuple[int, Path] | None = None
    for path in folder.glob(f"{NIGHT_PREFIX}*.json"):
        record = _json(path)
        if record.get("label") != "final":
            raise FinalError(f"{life_id} {path.name}: not a final morning score")
        basis = record.get("checkpoint_basis")
        if basis == "best":
            raise FinalError(f"{life_id} {path.name}: the best night is never scored")
        number = _night_number(path)
        if best is None or number > best[0]:
            best = (number, path)
    if best is None:
        raise FinalError(f"{life_id} has no final morning score file")
    return best[1]


def _probe(life_id: str, row: Mapping[str, Any], registry: Mapping[str, Probe]) -> Probe:
    pid = str(row["probe"])
    if pid not in registry:
        raise FinalError(f"{life_id}: probe {pid!r} is not in the registered set for this score")
    found = registry[pid]
    if found.person_id != life_id:
        return replace(found, person_id=life_id)
    return found


def _pool_comparison(
    life_records: Mapping[str, Mapping[str, Any]],
    registries: Mapping[str, Mapping[str, Probe]],
    comparison: str,
    dimension: Dimension,
    *,
    kinds: frozenset[str] | None = None,
    exclude_lives: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    probes: list[Probe] = []
    outcomes: list[Outcome] = []
    per_life: dict[str, dict[str, Any]] = {}
    allowed = (
        kinds
        if kinds is not None
        else (KNOWLEDGE_KINDS if dimension == "knowledge" else INTERACTION_KINDS)
    )
    for life_id, record in life_records.items():
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
            typed = cast(Outcome, outcome)
            life_outcomes.append(typed)
            probes.append(_probe(life_id, row, registries[life_id]))
            outcomes.append(typed)
        if life_outcomes:
            n = len(life_outcomes)
            wins = life_outcomes.count("win")
            ties = life_outcomes.count("tie")
            losses = life_outcomes.count("loss")
            life_score = (wins + 0.5 * ties) / n
            per_life[life_id] = {
                "n": n,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "score": life_score,
                "ahead": life_score > 0.5,
                "behind": life_score < 0.5,
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


def _stat_from_record(
    record: Mapping[str, Any],
    comparison: str,
    dimension: Dimension,
) -> dict[str, Any] | None:
    if comparison == COMP_3V1:
        arms = record.get("arms")
        if not isinstance(arms, dict):
            return None
        arm = arms.get("3")
        if not isinstance(arm, dict):
            return None
        raw = arm.get(dimension)
    else:
        pairs = record.get("pairs")
        if not isinstance(pairs, dict):
            return None
        raw = (pairs.get(comparison) or {}).get(dimension)
    if not isinstance(raw, dict) or raw.get("n", 0) == 0:
        return None
    ci = raw.get("ci")
    return {
        "n": int(raw["n"]),
        "wins": int(raw["wins"]),
        "ties": int(raw["ties"]),
        "losses": int(raw["losses"]),
        "score": float(raw["score"]),
        "decisive_share": raw.get("decisive_share"),
        "ci": list(ci) if isinstance(ci, list) else None,
    }


def _life_ahead(record: Mapping[str, Any], comparison: str, dimension: Dimension) -> bool | None:
    pooled = _stat_from_record(record, comparison, dimension)
    if pooled is None:
        rows = record.get("outcomes")
        if not isinstance(rows, list):
            return None
        kinds = KNOWLEDGE_KINDS if dimension == "knowledge" else INTERACTION_KINDS
        life_outcomes = [
            row["outcome"]
            for row in rows
            if isinstance(row, dict)
            and row.get("pair") == comparison
            and row.get("kind") in kinds
            and row.get("outcome") in ("win", "tie", "loss")
        ]
        if not life_outcomes:
            return None
        n = len(life_outcomes)
        wins = sum(1 for o in life_outcomes if o == "win")
        ties = sum(1 for o in life_outcomes if o == "tie")
        return (wins + 0.5 * ties) / n > 0.5
    return float(pooled["score"]) > 0.5


def _life_behind(cell: Mapping[str, Any]) -> bool:
    if "behind" in cell:
        return bool(cell["behind"])
    if "score" in cell:
        return float(cell["score"]) < 0.5
    return False


def _min_probes_for_dimension(dimension: Dimension) -> int:
    counts = _thresholds()["counts_per_life"]
    kinds = _thresholds()["dimensions"][dimension]["kinds"]
    return sum(int(counts[kind]) for kind in kinds)


def _registered_per_life_counts_met(
    stats: Mapping[str, Any],
    *,
    min_per_life: int,
    lives_total: int,
) -> bool:
    per_life = stats.get("per_life", {})
    if len(per_life) < lives_total:
        return False
    return all(int(cell.get("n", 0)) >= min_per_life for cell in per_life.values())


def _mirror_threshold(above: float) -> float:
    """Complement of a registered lower bound (``1 - x``), rounded for inclusive loss at the margin."""
    return round(1.0 - above, 10)


def _headline_outcome(
    stats: Mapping[str, Any] | None,
    *,
    reverse: bool = False,
    min_per_life: int | None = None,
) -> Outcome3:
    if stats is None or stats.get("n", 0) == 0:
        return "inconclusive"
    raw = _thresholds()["win"]
    lives_total = int(raw["lives_total"])
    need_min = min_per_life if min_per_life is not None else 1
    if not _registered_per_life_counts_met(stats, min_per_life=need_min, lives_total=lives_total):
        return "inconclusive"
    ci = stats.get("ci")
    lower = float(ci[0]) if isinstance(ci, list) and len(ci) == 2 else None
    upper = float(ci[1]) if isinstance(ci, list) and len(ci) == 2 else None
    score = float(stats["score"])
    share = stats.get("decisive_share")
    decisive = float(share) if share is not None else None
    per_life = stats.get("per_life", {})
    lives_ahead = sum(1 for life in per_life.values() if life.get("ahead"))
    lives_behind = sum(1 for life in per_life.values() if _life_behind(life))
    need_lives = int(raw["lives_ahead_at_least"])
    ci_floor = float(raw["ci_lower_bound_above"])
    score_floor = float(raw["overall_score_at_least"])
    share_floor = float(raw["decisive_share_at_least"])
    checks_win = (
        lower is not None
        and lower > ci_floor
        and score >= score_floor
        and decisive is not None
        and decisive >= share_floor
        and lives_ahead >= need_lives
    )
    checks_loss = (
        upper is not None
        and upper < _mirror_threshold(ci_floor)
        and round(score, 10) <= _mirror_threshold(score_floor)
        and decisive is not None
        and round(decisive, 10) <= _mirror_threshold(share_floor)
        and lives_behind >= need_lives
    )
    if reverse:
        checks_win, checks_loss = checks_loss, checks_win
    if checks_win:
        return "win"
    if checks_loss:
        return "loss"
    return "inconclusive"


def _fairness_outcome(stats: Mapping[str, Any] | None) -> FairnessOutcome:
    if stats is None or stats.get("n", 0) == 0:
        return "inconclusive"
    new_work_min = int(_thresholds()["counts_per_life"][NEW_WORK])
    if _headline_outcome(stats, min_per_life=new_work_min) == "win":
        return "win"
    if _headline_outcome(stats, reverse=True, min_per_life=new_work_min) == "win":
        return "loss"
    lives_total = int(_thresholds()["win"]["lives_total"])
    if not _registered_per_life_counts_met(
        stats, min_per_life=new_work_min, lives_total=lives_total
    ):
        return "inconclusive"
    ci = stats.get("ci")
    margin = float(_thresholds()["fairness"]["non_inferiority_margin"])
    if isinstance(ci, list) and len(ci) == 2 and float(ci[0]) > margin:
        return "matches"
    return "inconclusive"


def _donor_excluded_lives(life_records: Mapping[str, Mapping[str, Any]]) -> frozenset[str]:
    """Lives left out of the pooled 3 vs 3x score when the donor never promoted."""
    excluded: set[str] = set()
    for life_id, record in life_records.items():
        unavailable = record.get("unavailable")
        if isinstance(unavailable, dict) and unavailable.get("3x") == DONOR_UNAVAILABLE:
            excluded.add(life_id)
    return frozenset(excluded)


def _pair_ahead_on_3x(
    life_records: Mapping[str, Mapping[str, Any]],
    pair_lives: Sequence[str],
    dimension: Dimension,
) -> tuple[bool, bool]:
    """Both matched-pair lives ahead on 3 vs 3x (unpromoted donor counts as not ahead)."""
    ahead: list[bool] = []
    for life_id in pair_lives:
        record = life_records[life_id]
        unavailable = record.get("unavailable")
        if isinstance(unavailable, dict) and unavailable.get("3x") == DONOR_UNAVAILABLE:
            ahead.append(False)
            continue
        stat = _life_ahead(record, COMP_3V3X, dimension)
        ahead.append(bool(stat))
    return ahead[0], ahead[1]


def _claim_row(vs1: Outcome3, vs3x: Outcome3, pair_ok: bool) -> str:
    rows = _thresholds()["claim_table"]["rows"]
    if vs1 == "win" and vs3x == "win" and pair_ok:
        return str(rows[0]["id"])
    if vs1 == "win" and vs3x in ("inconclusive", "loss"):
        return str(rows[1]["id"])
    return str(rows[2]["id"])


def _claim_text(row_id: str, dimension: Dimension) -> str:
    rows = _thresholds()["claim_table"]["rows"]
    row = next(r for r in rows if r["id"] == row_id)
    text = str(row["claim"])
    if dimension == "knowledge":
        return text.replace("[know her world / talk her way]", "know her world").replace(
            "[in knowledge / in interaction]",
            "in knowledge",
        )
    return text.replace("[know her world / talk her way]", "talk her way").replace(
        "[in knowledge / in interaction]",
        "in interaction",
    )


def _qualifiers(
    dimension: Dimension,
    vs1: Outcome3,
    style_neutral: Outcome3 | None,
) -> list[dict[str, Any]]:
    quals = _thresholds()["claim_table"]["qualifiers"]
    found: list[dict[str, Any]] = []
    if vs1 == "win" and style_neutral is not None and style_neutral != "win":
        row = next(q for q in quals if q["id"] == "style_neutral")
        found.append({"id": row["id"], "claim": row["claim"], "outcome": style_neutral})
    for q in quals:
        if q["id"] in ("cost", "memory"):
            found.append({"id": q["id"], "claim": q["claim"]})
    return found


def _fairness_new_work(
    life_records: Mapping[str, Mapping[str, Any]],
    registries: Mapping[str, Mapping[str, Probe]],
    comparison: str,
) -> dict[str, Any] | None:
    return _pool_comparison(
        life_records,
        registries,
        comparison,
        "knowledge",
        kinds=frozenset({NEW_WORK}),
        exclude_lives=frozenset(),
    )


def run(
    freeze_tag: str,
    results_root: Path,
    lives: Sequence[str],
    *,
    donors: Mapping[str, str],
    pair_lives: Sequence[str],
    single_life: str,
    repo: Path = REPO,
    git: str | None = None,
    rerun_reason: str | None = None,
    style_neutral: Mapping[str, Outcome3 | None] | None = None,
    second_judge: Mapping[str, Any] | None = None,
    sealed: Callable[[str], Sequence[Probe]] | None = None,
) -> dict[str, Any]:
    """Score held-out lives once; refuse wrong code or an existing per-life ``final.json``."""
    expected = freeze_sha(freeze_tag, repo)
    found = git if git is not None else git_sha(repo)
    if found != expected:
        raise FinalError("this checkout is not the freeze tag the final test registered")

    if len(lives) != 3 or len(pair_lives) != 2 or single_life not in lives:
        raise FinalError("final test expects 3 lives, a pair of 2 and one single life")
    for life_id in lives:
        if score_path(results_root, life_id).exists():
            raise FinalError(f"{life_id} already has {FINAL_NAME}; the first score stands")

    life_records: dict[str, dict[str, Any]] = {}
    for life_id in lives:
        path = serving_morning_path(results_root, life_id)
        record = _json(path)
        record = {**record, "life_id": life_id, "_morning_path": str(path)}
        life_records[life_id] = record

    if sealed is None:
        raise FinalError("sealed probes are required to map score rows to bootstrap blocks")
    registries = {
        life_id: registered_probes(
            life_id, life_records[life_id], sealed, results_root=results_root
        )
        for life_id in lives
    }

    excluded = _donor_excluded_lives(life_records)
    dim_report: dict[str, Any] = {}
    all_quals: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        vs1_stats = _pool_comparison(life_records, registries, COMP_3V1, dimension)
        vs3x_stats = _pool_comparison(
            life_records,
            registries,
            COMP_3V3X,
            dimension,
            exclude_lives=excluded,
        )
        min_n = _min_probes_for_dimension(dimension)
        vs1 = _headline_outcome(vs1_stats, min_per_life=min_n)
        vs3x = _headline_outcome(vs3x_stats, min_per_life=min_n)
        if excluded:
            vs3x = "inconclusive"
        pair_ok = all(_pair_ahead_on_3x(life_records, pair_lives, dimension))
        row_id = _claim_row(vs1, vs3x, pair_ok)
        neutral = None if style_neutral is None else style_neutral.get(dimension)
        quals = _qualifiers(dimension, vs1, neutral)
        all_quals.extend(quals)
        dim_report[dimension] = {
            COMP_3V1: {"outcome": vs1, "stats": vs1_stats},
            COMP_3V3X: {"outcome": vs3x, "stats": vs3x_stats, "excluded_lives": sorted(excluded)},
            "claim_row": row_id,
            "claim": _claim_text(row_id, dimension),
            "pair_lives_ahead_on_3x": pair_ok,
            "qualifiers": quals,
        }
        if second_judge is not None and dimension in second_judge:
            dim_report[dimension]["second_judge"] = second_judge[dimension]

    fairness_report: dict[str, Any] = {}
    for comp in (COMP_4V2C, COMP_4V2F):
        stats = _fairness_new_work(life_records, registries, comp)
        outcome = _fairness_outcome(stats)
        fairness_report[comp] = {"outcome": outcome, "new_work": stats}

    report = FinalReport(
        freeze_tag=freeze_tag,
        freeze_sha=expected,
        git_sha=found,
        donors=dict(donors),
        pair_lives=(pair_lives[0], pair_lives[1]),
        single_life=single_life,
        dimensions=dim_report,
        fairness=fairness_report,
        qualifiers=tuple(all_quals),
    )
    body = report.to_dict()
    for life_id in lives:
        folder = results_root / life_id / SCORES
        folder.mkdir(parents=True, exist_ok=True)
        target = score_path(results_root, life_id)
        pending = target.with_name(f".{target.name}.tmp")
        pending.write_text(
            json.dumps({**body, "life_id": life_id}, sort_keys=True, indent=1), encoding="utf-8"
        )
        pending.replace(target)
    return body


__all__ = [
    "FINAL_NAME",
    "FinalError",
    "FinalReport",
    "freeze_sha",
    "git_sha",
    "registered_probes",
    "run",
    "score_path",
    "serving_morning_path",
]
