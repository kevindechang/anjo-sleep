"""The live comparison of the sleep objective's terms: plan first, then, with ``--yes``, one
accelerated life per arm on fresh synthetic accounts, then ``--report`` over what they wrote.

    python -m anjo.sleep.tools.sleep_objective_live --account seed=<id> --account fold=<id> \\
        --account today=<id> --persona founder --days 5 --night-usd <a measured night's cost_usd> [--yes]
    python -m anjo.sleep.tools.sleep_objective_live --account seed=<id> ... --report

**Arms**, each a whole accelerated run (``sim.accelerated``, the same persona, seed and virtual dates):

- ``seed``: ``ANJO_SLEEP_OBJECTIVE=sft_dpo_seed`` (SFT, DPO, knowledge seeding, imitation);
- ``fold``: the same with ``ANJO_SLEEP_ADAPTER_FOLD=1``, the nightly fold into one rank-32 adapter;
- ``dreams``: the same with ``ANJO_SLEEP_DREAMS=1``; only once the app wires the core's dream
  seeds into ``NightDeps.dreams``, and only when asked for (``--account dreams=<id>``);
- ``today``: ``sft_dpo``, padded with SFT to the seed arm's mean training minutes
  (``ANJO_SLEEP_MATCH_MINUTES``), so the comparison holds the GPU budget equal. It runs last.

**Never a real night.** Every account must be new: one that already has a results folder is refused,
so no arm ever trains on Teo's real-time nights or any other running life. Each run names a persona;
the wiring refuses a persona on an account that isn't registered synthetic. The runs inherit this
shell's environment, the night's own (``ANJO_SLEEP_HOST=1``, ``ANJO_DATA_DIR``, the Modal and price
settings), with the four switches replaced by the arm's.

**Holds.** Each night takes its own holds as every night does (``night/holds.py``), so no run can pass
a pool's cap. Before the first run, ``--yes`` checks ``modal_usd``'s remaining balance against the
whole plan priced at ``--night-usd`` (a measured night's ``cost_usd``, today's objective) plus each
arm's added A100 minutes (``ADDED_A100_MIN``, estimates anchored to the 2.83 training minutes measured
on 2026-10-28), every dollar counted against ``modal_usd`` as the worst case. An arm that fails stops
the rest.

**The report** reads each account's night logs and morning scores (numbers only): per night the
training receipt (minutes and rows by term, the dream filter's keep rate by seed kind), the general
gate's lost points, promoted or not and the night's cost; per arm the mean weights-only score (arm 3
against base) and arm 4's. Forgetting is read off arm 3 across nights and the general gate, fold
against no fold.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from anjo.sleep.budget import default_ledger
from anjo.sleep.reports.results import results_root
from anjo.sleep.train import objective

POOL: Final = "modal_usd"
A100_USD_PER_HOUR: Final = 2.50  # Modal's A100-80GB list price, checked 2026-10-27
ORDER: Final = ("seed", "fold", "dreams", "today")
SEED: Final = {objective.OBJECTIVE_ENV: objective.SEED_OBJECTIVE}
ENVS: Final[Mapping[str, Mapping[str, str]]] = {
    "seed": SEED, "fold": {**SEED, objective.FOLD_ENV: "1"}, "dreams": {**SEED, objective.DREAMS_ENV: "1"},
    "today": {objective.OBJECTIVE_ENV: "sft_dpo"},
}
# Estimates per night (2026-10-29), not measurements: seeding's teacher passes 0.9, fresh samples 1.3,
# imitation 0.8, the fold 0.1, dreams' ranking and block 1.3 plus 1.5 minutes of serving for their text.
ADDED_A100_MIN: Final = {"seed": 3.0, "fold": 3.1, "dreams": 5.8, "today": 3.0}  # today pads to the seed arm
SWITCHES: Final = (objective.OBJECTIVE_ENV, objective.FOLD_ENV, objective.DREAMS_ENV, objective.MATCH_ENV)
WIRING: Final = ("--wiring", "anjo.sleep.app.night:build", "--simulate", "anjo.sleep.app.sim_transport:simulate",
                 "--morning", "anjo.sleep.app.morning:build")


def arms(accounts: Mapping[str, str]) -> list[str]:
    """The arms asked for, in run order: the matched control last."""
    unknown = sorted(set(accounts) - set(ORDER))
    if unknown or "seed" not in accounts:
        raise ValueError(f"arms are {', '.join(ORDER)}, and the seed arm is always run")
    if len(set(accounts.values())) != len(accounts):
        raise ValueError("each arm lives on its own account")
    return [arm for arm in ORDER if arm in accounts]


def plan_usd(chosen: Sequence[str], days: int, night_usd: float) -> dict[str, float]:
    """Each arm's worst case: its nights at today's cost plus its added A100 minutes."""
    return {arm: round(days * (night_usd + ADDED_A100_MIN[arm] / 60.0 * A100_USD_PER_HOUR), 2) for arm in chosen}


def command(account: str, persona: str, days: int, start: date) -> list[str]:
    return [sys.executable, "-m", "anjo.sleep.sim.accelerated", *WIRING, "--user", account, "--persona", persona,
            "--days", str(days), "--start", start.isoformat()]


def arm_env(arm: str, base: Mapping[str, str], match_minutes: float | None) -> dict[str, str]:
    """The run's environment: the switches cleared, then the arm's own."""
    env = {key: value for key, value in base.items() if key not in SWITCHES} | dict(ENVS[arm])
    if arm == "today":
        if match_minutes is None:
            raise ValueError("the matched control needs the seed arm's training minutes")
        env[objective.MATCH_ENV] = f"{match_minutes:.2f}"
    return env


def nights(root: Path, account: str) -> list[dict[str, Any]]:
    """Each finished night of ``account``: its train and gates records, its cost and promotion."""
    folder = root / account / "nights"
    found = []
    for night in sorted(folder.iterdir()) if folder.is_dir() else ():
        log, summary = night / "log.jsonl", night / "summary.json"
        if not summary.is_file():
            continue
        steps = {r["step"]: r for r in (json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x)}
        done = json.loads(summary.read_text(encoding="utf-8"))
        general = steps.get("gates", {}).get("gates", {}).get("general", {})
        found.append({"night": night.name, "train": steps.get("train", {}), "dataset": steps.get("dataset", {}),
                      "general_lost": general.get("value"), "cost_usd": done.get("cost_usd"),
                      "promoted": done.get("promoted") is not None})
    return found


def scores(root: Path, account: str) -> list[dict[str, Any]]:
    folder = root / account / "scores"
    files = sorted(folder.glob("night-*.json")) if folder.is_dir() else []
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def _arm_score(record: Mapping[str, Any], arm: str) -> float | None:
    stat = record.get("arms", {}).get(arm, {}).get("overall", {})
    value = stat.get("score")
    return float(value) if isinstance(value, int | float) else None


def _mean(values: Sequence[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return round(sum(kept) / len(kept), 4) if kept else None


def matched_minutes(root: Path, account: str) -> float:
    """The seed arm's mean training minutes a night, for the control's padding."""
    minutes = [n["train"].get("train_minutes") for n in nights(root, account)]
    kept = [float(m) for m in minutes if isinstance(m, int | float)]
    if not kept:
        raise ValueError("the seed arm trained no night: nothing to match")
    return sum(kept) / len(kept)


def report(root: Path, accounts: Mapping[str, str], out: Callable[[str], None]) -> dict[str, Any]:
    """Per arm: the night series and the means. Numbers only."""
    table: dict[str, Any] = {}
    for arm in arms(accounts):
        done, scored = nights(root, accounts[arm]), scores(root, accounts[arm])
        terms = [n["train"].get("terms") or {} for n in done]
        dreams: list[dict[str, Any]] = [t["dreams"] for t in terms if t.get("dreams")]
        table[arm] = {
            "nights": len(done), "promoted": sum(n["promoted"] for n in done),
            "arm3_by_night": [_arm_score(s, "3") for s in scored], "arm4_by_night": [_arm_score(s, "4") for s in scored],
            "arm3_mean": _mean([_arm_score(s, "3") for s in scored]), "arm4_mean": _mean([_arm_score(s, "4") for s in scored]),
            "general_lost_by_night": [n["general_lost"] for n in done],
            "train_minutes_mean": _mean([n["train"].get("train_minutes") for n in done]),
            "minutes_by_term": [t.get("minutes") for t in terms], "rows_by_term": [t.get("rows") for t in terms],
            "dreams_by_seed": [d.get("by_seed") for d in dreams], "dream_blocks_kept": sum(bool(d.get("block_kept")) for d in dreams),
            "cost_usd": round(sum(float(n["cost_usd"] or 0.0) for n in done), 2),
        }
        out(f"{arm}: {json.dumps(table[arm], sort_keys=True)}")
    return table


def _accounts(pairs: Sequence[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for pair in pairs:
        arm, _, account = pair.partition("=")
        if not account or arm in found:
            raise ValueError("--account is <arm>=<account id>, once per arm")
        found[arm] = account
    return found


def _plan(args: argparse.Namespace, chosen: Sequence[str], start: date, out: Callable[[str], None]) -> float:
    priced = plan_usd(chosen, args.days, args.night_usd or 0.0)
    out(f"sleep objective live comparison: plan only, nothing run, nothing held ({args.persona}, {args.days} days "
        f"from {start.isoformat()}, arms {', '.join(chosen)})")
    for arm in chosen:
        env = " ".join(f"{k}={v}" for k, v in ENVS[arm].items())
        match = f" {objective.MATCH_ENV}=<seed arm's mean train minutes>" if arm == "today" else ""
        out(f"  {arm}: {env}{match} {' '.join(command('<' + arm + ' account>', args.persona, args.days, start)[1:])}")
        out(f"    worst case ${priced[arm]:.2f}: {args.days} nights at ${args.night_usd or 0.0:.2f} plus "
            f"{ADDED_A100_MIN[arm]:.1f} A100 min each (estimate)")
    if "dreams" in chosen:
        out("  dreams: the app must wire the core's dream seeds into NightDeps.dreams first; unwired, its "
            "dataset step logs 'no dream source' and the arm equals the seed arm")
    total = round(sum(priced.values()), 2)
    out(f"total worst case ${total:.2f} on {POOL}; each night still takes its own holds")
    out("run again with --yes to run (needs --night-usd), or --report to read the results")
    return total


def main(argv: Sequence[str] | None = None, *, run: Callable[..., Any] = subprocess.run,
         out: Callable[[str], None] = print, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> int:
    parser = argparse.ArgumentParser(prog="sleep_objective_live", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--account", action="append", default=[], help="<arm>=<synthetic account id>")
    parser.add_argument("--persona", default="founder")
    parser.add_argument("--days", type=int, default=5, choices=range(1, 15), metavar="N")
    parser.add_argument("--night-usd", type=float, default=None, help="a measured night's cost_usd")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    try:
        accounts = _accounts(args.account)
        chosen = arms(accounts)
    except ValueError as error:
        parser.error(str(error))
    if args.report:
        report(results_root(), accounts, out)
        return 0
    start = (now() - timedelta(days=args.days + 1)).date()  # one past start for every arm: the same virtual nights
    total = _plan(args, chosen, start, out)
    if not args.yes:
        return 0
    root = results_root()
    if args.night_usd is None or args.night_usd <= 0:
        parser.error("--yes needs --night-usd, a measured night's cost_usd")
    used = [accounts[arm] for arm in chosen if (root / accounts[arm]).exists()]
    if used:
        parser.error(f"{len(used)} account(s) already have results: every arm runs on a new synthetic account")
    left = default_ledger().remaining(POOL)
    if left < total:
        parser.error(f"{POOL} has ${left:.2f} left, under the plan's ${total:.2f}")
    match: float | None = None
    for arm in chosen:
        if arm == "today":
            match = matched_minutes(root, accounts["seed"])
        done = run(command(accounts[arm], args.persona, args.days, start), env=arm_env(arm, os.environ, match),
                   check=False)
        if done.returncode != 0:
            out(f"{arm}: the run exited with {done.returncode}; the arms after it don't run")
            return 1
    report(root, accounts, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
