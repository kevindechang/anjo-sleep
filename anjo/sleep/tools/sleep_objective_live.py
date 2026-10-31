"""The live comparison of the sleep objective's terms: plan first, then, with ``--yes``, one
accelerated life per arm on fresh synthetic accounts, then ``--report`` over what they wrote.

    python -m anjo.sleep.tools.sleep_objective_live --account seed=<id> --account today=<id> \\
        [--account fold=<id>] --token-file seed=<path> --token-file today=<path> [--token-file fold=<path>] \\
        --persona founder --days 5 --night-usd <a measured night's cost_usd> [--yes]
    python -m anjo.sleep.tools.sleep_objective_live --account seed=<id> ... --report

**Arms**, each a whole accelerated run (``sim.accelerated``, the same persona, seed and virtual dates):

- ``seed``: ``ANJO_SLEEP_OBJECTIVE=sft_dpo_seed`` (SFT, DPO, knowledge seeding, imitation);
- ``today``: ``sft_dpo``, the seed arm's control, matched night by night: its SFT trains, by the clock,
  for the seed arm's training minutes that night less its DPO block, with one warmup and one decay
  over them, and today's DPO pass follows (``colab_train.StretchClock``);
- ``fold``: the seed arm with ``ANJO_SLEEP_ADAPTER_FOLD=1``. No control: the fold adds under 0.1 A100
  minutes a night, within 5% of the seed arm's, and the report checks that it stayed so;
- ``dreams``: the seed arm with ``ANJO_SLEEP_DREAMS=1``, only with its control ``seed_matched``: the
  seed arm's objective, its main block stretched to the dreams arm's minutes, serving GPU minutes for
  the dreams' text included. Unwired dreams (the dataset step's ``dreams: not wired``) skip the control.
- ``protect``: the fold arm with ``ANJO_SLEEP_PROTECT=anchors``: tonight's slot trains off the forgetting
  anchors' activation basis (``train/sleep_protect.py``). Only with ``fold`` (the same arm unprojected) and
  its twin ``protect_random``: the same job with a random orthonormal basis of the same rank and width in
  place of the anchors' (the decisive ablation). A twin needs no match file: the same rows, step plan and
  sampling make its minutes the same by construction, and the report checks they stayed within 5%.
- ``seed_cap``: the seed arm with ``ANJO_SLEEP_CAPACITY=1`` (rank 32, three passes over the night's rows at
  one row a step, ``train/objective.py``): what more training does, against the seed arm (``--reuse seed``
  reads a seed arm that already lived). No control: the budget is the claim, and the report gives the minutes.

A control runs after the arm it is matched to, with a match file of that arm's minutes per night
(``--match-file``, per run: no host setting can pad a real night). Every arm runs with
``ANJO_SLEEP_PRESSURE=0``, so sleep pressure never changes how much a night trains.

**Never a real night.** Every account must be new: one that already has a results folder is refused,
so no arm ever trains on a running life. The one exception is ``--reuse <arm>``: a reference arm (``seed``
or ``protect``, what a control or twin is matched to) that already lived all ``--days`` nights from
``--start`` (by default its own first night), each promoted or rejected, is read, not run again: its
account may have results, it needs no token, and its worst case isn't held. Anything less (a night
missing, failed or outside those dates) is refused. Each run holds under a key of its own, so a run
that stopped and released its holds can be continued. Each run names a persona; the wiring refuses a persona on an
account that isn't registered synthetic. Each arm chats as its own account: ``--token-file <arm>=<path>``
names a file holding a signed-in token for that arm's account, read when ``--yes`` starts and never
printed or logged (the plan says only whether each is there and non-empty). ``--yes`` needs one for
every arm. The day itself refuses a token that isn't its account's (``app.sim_transport``). The runs
inherit this shell's environment, with the switches replaced by the arm's and the shell's own
``ANJO_SLEEP_SIM_TOKEN`` by the arm's token; the thought channel (``ANJO_SLEEP_CHANNEL``) is the
shell's, so on or off it is the same in every arm, and a claim and its control differ in their
objective alone.

**Holds.** The plan's worst case is priced at ``--night-usd`` (a measured night's ``cost_usd``, today's
objective) plus each arm's added A100 minutes (``ADDED_A100_MIN``, estimates; a control at its match
file's minutes once its reference has run), all against ``modal_usd``. ``--yes`` holds it whole before
the first run; before each arm the hold shrinks to the arms still to come, the running arm's nights
taking their own holds (``night/holds.py``) from what was freed, and the last is released at the end.
No room for a hold stops the rest; so does an arm that fails.

**The report** reads each account's night logs and morning scores (numbers only): per night the
training receipt (minutes and rows by term, the dream filter's keep rate by seed kind), the general
gate's lost points, promoted or not and the night's cost; per arm and probe set (``full``, ``daily-v1``:
``ANJO_SLEEP_MORNING_SUBSET``) the weights-only score (arm 3 against base) and arm 4's by night date, and
their means, never pooled across sets; per control and night its budget and the minutes it used, flagged past 5%.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from anjo.sleep.budget import Ledger, Reservation, default_ledger
from anjo.sleep.night.core_snapshot import CHANNEL_ENV
from anjo.sleep.night.holds import usd
from anjo.sleep.probes.morning import MORNING_SUBSET_ENV, morning_subset
from anjo.sleep.reports.results import FULL_SET, results_root, score_records
from anjo.sleep.train import objective
from anjo.sleep.train.terms import DREAMS_NOT_WIRED

POOL: Final = "modal_usd"
A100_USD_PER_HOUR: Final = 2.50  # Modal's A100-80GB list price, checked 2026-10-27
ORDER: Final = ("seed", "today", "fold", "dreams", "seed_matched", "protect", "protect_random", "seed_cap")
CONTROL_OF: Final = {"today": "seed", "seed_matched": "dreams"}  # a control, and the arm its minutes match
TWIN_OF: Final = {"protect_random": "protect"}  # the same job with one thing swapped: equal minutes by construction
NEEDS: Final = {"seed": ("today",), "dreams": ("seed_matched",), "protect": ("protect_random", "fold")}  # a claim's comparisons
PRESSURE_ENV: Final = "ANJO_SLEEP_PRESSURE"
SEED: Final = {objective.OBJECTIVE_ENV: objective.SEED_OBJECTIVE, objective.CAPACITY_ENV: "0"}
ENVS: Final[Mapping[str, Mapping[str, str]]] = {
    "seed": SEED, "fold": {**SEED, objective.FOLD_ENV: "1"}, "dreams": {**SEED, objective.DREAMS_ENV: "1"},
    "today": {objective.OBJECTIVE_ENV: "sft_dpo", objective.CAPACITY_ENV: "0"}, "seed_matched": SEED,
    "protect": {**SEED, objective.FOLD_ENV: "1", objective.PROTECT_ENV: "anchors"},
    "protect_random": {**SEED, objective.FOLD_ENV: "1", objective.PROTECT_ENV: "random"},
    "seed_cap": {objective.OBJECTIVE_ENV: objective.SEED_OBJECTIVE, objective.CAPACITY_ENV: "1"},
}
# Estimates per night (2026-10-29), not measurements: seeding's teacher passes 0.9, fresh samples 1.3,
# imitation 0.8, the fold 0.1, dreams' ranking and block 1.3 plus 1.5 minutes of serving for their text.
# The protected slot adds 1.5 to the fold's: the anchors' passes 0.3, the bases 0.5, the projection in the
# loop and the protected cut 0.7; each night's receipt measures the first two (terms.protect.minutes).
# Capacity runs the seed arm's main block (about 5.3 of its 5.8 training minutes) three times over: +10.7.
ADDED_A100_MIN: Final = {"seed": 3.0, "fold": 3.1, "dreams": 5.8, "protect": 4.6, "protect_random": 4.6, "seed_cap": 13.7}
TODAY_TRAIN_MIN: Final = 2.83  # today's training minutes, measured 2026-10-28: a control's budget holds them too
MISMATCH: Final = 0.05
SWITCHES: Final = (objective.OBJECTIVE_ENV, objective.FOLD_ENV, objective.DREAMS_ENV, objective.PROTECT_ENV,
                   objective.CAPACITY_ENV, PRESSURE_ENV)
SIM_TOKEN_ENV: Final = "ANJO_SLEEP_SIM_TOKEN"  # app.sim_transport's: the account a day's lines post as
HOLD_KEY: Final = "sleep-objective-live"
REUSABLE: Final = ("seed", "protect")  # reference arms: what a control or a twin is matched to
FINISHED: Final = frozenset({"promoted", "rejected"})  # a reused arm's nights each ended so
WIRING: Final = ("--wiring", "anjo.sleep.app.night:build", "--simulate", "anjo.sleep.app.sim_transport:simulate",
                 "--morning", "anjo.sleep.app.morning:build")


def arms(accounts: Mapping[str, str], reused: Collection[str] = ()) -> list[str]:
    """The arms asked for, in run order: each control and twin after the arm it is matched to. A reused
    arm makes no claim in this run, so its comparisons need not be named."""
    unknown = sorted(set(accounts) - set(ORDER))
    if unknown or "seed" not in accounts:
        raise ValueError(f"arms are {', '.join(ORDER)}, and the seed arm is always run")
    if len(set(accounts.values())) != len(accounts):
        raise ValueError("each arm lives on its own account")
    missing = [f"{arm} needs {other}" for arm, others in NEEDS.items() if arm in accounts and arm not in reused
               for other in others if other not in accounts]
    missing += [f"{control} needs {arm}" for control, arm in {**CONTROL_OF, **TWIN_OF}.items()
                if control in accounts and arm not in accounts]
    if missing:
        raise ValueError(f"every claim runs with its matched control: {'; '.join(missing)}")
    return [arm for arm in ORDER if arm in accounts]


def arm_usd(arm: str, days: int, night_usd: float, budgets: Mapping[str, float] | None = None) -> float:
    """An arm's worst case: its nights at today's cost plus its added A100 minutes; a control's whole
    budget (its match file's once known, else the estimate for the arm it matches)."""
    if arm in CONTROL_OF:
        added = sum(budgets.values()) if budgets else days * (TODAY_TRAIN_MIN + ADDED_A100_MIN[CONTROL_OF[arm]])
    else:
        added = days * ADDED_A100_MIN[arm]
    return round(days * night_usd + added / 60.0 * A100_USD_PER_HOUR, 2)


def command(account: str, persona: str, days: int, start: date, match_file: Path | None = None) -> list[str]:
    return [sys.executable, "-m", "anjo.sleep.sim.accelerated", *WIRING, "--user", account, "--persona", persona,
            "--days", str(days), "--start", start.isoformat(),
            *(["--match-file", str(match_file)] if match_file is not None else [])]


def arm_env(arm: str, base: Mapping[str, str], token: str) -> dict[str, str]:
    """The run's environment: the switches cleared, then the arm's own, sleep pressure off, and its
    account's token in place of the shell's."""
    kept = {key: value for key, value in base.items() if key not in SWITCHES}
    return kept | dict(ENVS[arm]) | {PRESSURE_ENV: "0", SIM_TOKEN_ENV: token}


def nights(root: Path, account: str) -> list[dict[str, Any]]:
    """Each finished night of ``account``: its training receipt (the train step keeps what the job
    measured under ``receipt``; the dataset step is flat), dataset and gates records, cost and status."""
    folder = root / account / "nights"
    found = []
    for night in sorted(folder.iterdir()) if folder.is_dir() else ():
        log, summary = night / "log.jsonl", night / "summary.json"
        if not summary.is_file():
            continue
        if not log.is_file():  # finished, yet nothing to read: refused, never guessed
            raise ValueError(f"night {night.name} has a summary but no log.jsonl: its training can't be read")
        steps = {r.get("step"): r for r in (json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x)}
        done = json.loads(summary.read_text(encoding="utf-8"))
        general = steps.get("gates", {}).get("gates", {}).get("general", {})
        found.append({"night": night.name, "train": steps.get("train", {}).get("receipt") or {},
                      "dataset": steps.get("dataset", {}), "general_lost": general.get("value"),
                      "cost_usd": done.get("cost_usd"), "status": done.get("status"),
                      "promoted": done.get("status") == "promoted"})
    return found


def reused_nights(root: Path, account: str, start: date, days: int) -> int:
    """How many nights a reused arm lived: all ``days`` of them from ``start``, each promoted or
    rejected, and no other; anything else is refused."""
    folder = root / account / "nights"
    wanted = [(start + timedelta(days=n)).isoformat() for n in range(days)]
    have = sorted(path.name for path in folder.iterdir()) if folder.is_dir() else []
    status = {night["night"]: night["status"] for night in nights(root, account)}
    if extra := [name for name in have if name not in wanted]:
        raise ValueError(f"the reused arm has nights outside the {days} from {start.isoformat()} ({', '.join(extra)}): "
                         "not this life")
    if missing := [name for name in wanted if name not in status]:
        raise ValueError(f"the reused arm lived {days - len(missing)} of {days} nights from {start.isoformat()} "
                         f"(none finished on {', '.join(missing)}): reuse needs all of them")
    if bad := [f"{name} {status[name]}" for name in wanted if status[name] not in FINISHED]:
        raise ValueError(f"the reused arm has a night that wasn't promoted or rejected ({', '.join(bad)})")
    return len(wanted)


def _first_night(root: Path, account: str) -> date:
    folder = root / account / "nights"
    found = sorted(path.name for path in folder.iterdir()) if folder.is_dir() else []
    if not found:
        raise ValueError("the reused arm has no nights")
    return date.fromisoformat(found[0])


def budgets(root: Path, account: str) -> dict[str, float]:
    """Each night's budget for the control matched to ``account``'s arm: its training minutes by term
    without the DPO block (the control runs its own after), plus her serving GPU's minutes on dreams."""
    found = {}
    for night in nights(root, account):
        minutes = (night["train"].get("terms") or {}).get("minutes")
        if not isinstance(minutes, Mapping):
            raise ValueError(f"night {night['night']} of the matched arm has no training minutes by term")
        serving = float((night["dataset"].get("terms") or {}).get("dream_seconds") or 0.0) / 60.0
        found[night["night"]] = round(sum(float(v) for k, v in minutes.items() if k != "dpo") + serving, 4)
    if not found:
        raise ValueError("the matched arm trained no night: nothing to match")
    return found


def dreams_wired(root: Path, account: str) -> bool:
    return not any((n["dataset"].get("terms") or {}).get("dreams") == DREAMS_NOT_WIRED for n in nights(root, account))


def scores(root: Path, account: str) -> list[dict[str, Any]]:
    """Each scored morning, one per (night, probe set) in that order (``results.score_records``): a backfill
    (``tools.morning_backfill``) may score an earlier night last, or a night again on another set."""
    return list(score_records(root / account / "scores").values())


def by_set(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per probe set: arm 3's and arm 4's score by night date, and their means. A mean never spans two sets."""
    sets = {name: [s for s in scored if str(s.get("subset") or FULL_SET) == name]
            for name in sorted({str(s.get("subset") or FULL_SET) for s in scored})}
    return {name: {"by_night": {str(s["night"]): {"3": _arm_score(s, "3"), "4": _arm_score(s, "4")} for s in mine},
                   "arm3_mean": _mean([_arm_score(s, "3") for s in mine]), "arm4_mean": _mean([_arm_score(s, "4") for s in mine])}
            for name, mine in sets.items()}


def _arm_score(record: Mapping[str, Any], arm: str) -> float | None:
    stat = record.get("arms", {}).get(arm, {}).get("overall", {})
    value = stat.get("score")
    return float(value) if isinstance(value, int | float) else None


def _mean(values: Sequence[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return round(sum(kept) / len(kept), 4) if kept else None


def _off(a: float, b: float) -> bool:
    return abs(a - b) > MISMATCH * max(b, 1e-9)


def matches(done: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A control's nights: its budget, the minutes it used, and whether they part by more than 5%."""
    out = []
    for night in done:
        match = night["train"].get("match") or {}
        budget, used = match.get("budget_minutes"), match.get("minutes")
        off = not isinstance(budget, int | float) or not isinstance(used, int | float) or _off(float(used), float(budget))
        out.append({"night": night["night"], "budget": budget, "used": used, "mismatch": off})
    return out


def report(root: Path, accounts: Mapping[str, str], out: Callable[[str], None],
           reused: Collection[str] = ()) -> dict[str, Any]:
    """Per arm: the night series and the means; per control, its match night by night. Numbers only."""
    table: dict[str, Any] = {}
    for arm in arms(accounts, reused):
        done, scored = nights(root, accounts[arm]), scores(root, accounts[arm])
        terms = [n["train"].get("terms") or {} for n in done]
        dreams: list[dict[str, Any]] = [t["dreams"] for t in terms if t.get("dreams")]
        table[arm] = {
            "nights": len(done), "promoted": sum(n["promoted"] for n in done),
            "scores": by_set(scored),  # per probe set, never pooled: a night may be scored on two
            "general_lost_by_night": [n["general_lost"] for n in done],
            "train_minutes_mean": _mean([n["train"].get("train_minutes") for n in done]),
            "minutes_by_term": [t.get("minutes") for t in terms], "rows_by_term": [t.get("rows") for t in terms],
            "dreams_by_seed": [d.get("by_seed") for d in dreams], "dream_blocks_kept": sum(bool(d.get("block_kept")) for d in dreams),
            "cost_usd": round(sum(float(n["cost_usd"] or 0.0) for n in done), 2),
            **({"protect_by_night": [t.get("protect") for t in terms]} if "protect" in arm else {}),
            **({"match": matches(done)} if arm in CONTROL_OF else {}),
        }
        out(f"{arm}: {json.dumps(table[arm], sort_keys=True)}")
        if any(m["mismatch"] for m in table[arm].get("match", ())):
            out(f"  {arm}: a night's minutes part from its budget by more than {MISMATCH:.0%}: that night isn't matched")
    if "fold" in table:
        fold, seed = budgets(root, accounts["fold"]), budgets(root, accounts["seed"])
        if _off(sum(fold.values()), sum(seed.values())):
            out(f"  fold: its training minutes part from the seed arm's by more than {MISMATCH:.0%}: it needs a control")
    for twin, arm in TWIN_OF.items():
        if twin in table and _off(sum(budgets(root, accounts[twin]).values()), sum(budgets(root, accounts[arm]).values())):
            out(f"  {twin}: its training minutes part from {arm}'s by more than {MISMATCH:.0%}: the twin isn't matched")
    return table


def _accounts(pairs: Sequence[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for pair in pairs:
        arm, _, account = pair.partition("=")
        if not account or arm in found:
            raise ValueError("--account is <arm>=<account id>, once per arm")
        found[arm] = account
    return found


def _token_files(pairs: Sequence[str], chosen: Sequence[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for pair in pairs:
        arm, _, path = pair.partition("=")
        if not path or arm in found or arm not in chosen:
            raise ValueError("--token-file is <arm>=<path>, once per arm that runs")
        found[arm] = Path(path)
    return found


def _token(path: Path | None) -> str | None:
    """The token in ``path``, or None when it isn't given, can't be read or is empty. Never printed."""
    if path is None:
        return None
    try:
        token = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return token or None


def _token_line(arm: str, path: Path | None) -> str:
    if path is None:
        return f"    {arm}: no token file"
    return f"    {arm}: token file {'ok' if _token(path) else 'missing or empty'}"


def _plan(args: argparse.Namespace, chosen: Sequence[str], start: date, files: Mapping[str, Path],
          out: Callable[[str], None], reused: Mapping[str, int] | None = None) -> dict[str, float]:
    reused = reused or {}
    priced = {arm: arm_usd(arm, args.days, args.night_usd or 0.0) for arm in chosen if arm not in reused}
    out(f"sleep objective live comparison: plan only, nothing run, nothing held ({args.persona}, {args.days} days "
        f"from {start.isoformat()}, {len(chosen)} arms: {', '.join(chosen)})")
    for arm in chosen:
        if arm in reused:  # read, not run: no command, no token, nothing held
            out(f"  {arm}: reused, {reused[arm]} nights found")
            continue
        env = " ".join(f"{k}={v}" for k, v in {**ENVS[arm], PRESSURE_ENV: "0"}.items())
        match = f" --match-file <{CONTROL_OF[arm]} arm's minutes per night>" if arm in CONTROL_OF else ""
        out(f"  {arm}: {env} {' '.join(command('<' + arm + ' account>', args.persona, args.days, start)[1:])}{match}")
        out(f"    worst case ${priced[arm]:.2f} (estimate)")
        out(_token_line(arm, files.get(arm)))
    out(f"  thought channel: {os.environ.get(CHANNEL_ENV) or 'off'} ({CHANNEL_ENV}, this shell's), the same in every arm")
    out(f"  morning: {os.environ.get(MORNING_SUBSET_ENV, '').strip() or 'the full set'} ({MORNING_SUBSET_ENV}, this "
        "shell's, passed through), the same in every arm")
    if "dreams" in chosen:
        out("  dreams: the app must wire the core's dream seeds into NightDeps.dreams first; unwired, its dataset "
            f"step says dreams: {DREAMS_NOT_WIRED}, and its control doesn't run")
    out(f"total worst case ${sum(priced.values()):.2f} on {POOL}, held whole before the first run")
    out("run again with --yes to run (needs --night-usd), or --report to read the results")
    return priced


def _hold(ledger: Ledger, key: str, amount: float) -> Reservation | None:
    """Hold ``amount`` (the arms still to come) under ``key``; None when nothing is left to come."""
    if amount <= 0:
        return None
    found = ledger.reserve(key, (POOL,), usd(amount))
    if found is None:
        raise ValueError(f"{POOL} can't hold the ${amount:.2f} the arms still to come may cost")
    return found


def _run_arms(args: argparse.Namespace, accounts: Mapping[str, str], tokens: Mapping[str, str], chosen: Sequence[str],
              start: date, priced: dict[str, float], run: Callable[..., Any], out: Callable[[str], None],
              folder: Path) -> int:
    root, ledger = results_root(), default_ledger()
    key = f"{HOLD_KEY}:{accounts['seed']}:{secrets.token_hex(4)}"  # each run its own: a stopped run's keys have ended
    held: Reservation | None = None
    chosen = [arm for arm in chosen if arm in priced]  # a reused arm is read, not run
    try:
        held = _hold(ledger, f"{key}:all", sum(priced.values()))
        for i, arm in enumerate(chosen):
            match_file = None
            if arm in CONTROL_OF:
                if CONTROL_OF[arm] == "dreams" and not dreams_wired(root, accounts["dreams"]):
                    out(f"dreams: {DREAMS_NOT_WIRED}, so that arm is the seed arm again; {arm} doesn't run")
                    continue
                found = budgets(root, accounts[CONTROL_OF[arm]])
                priced[arm] = arm_usd(arm, args.days, args.night_usd, found)
                match_file = folder / f"{arm}.json"
                match_file.write_text(json.dumps(found), encoding="utf-8")
            if held is not None:  # the hold shrinks to the arms still to come; this arm's nights hold for it
                held.release()
                held = None
            held = _hold(ledger, f"{key}:{i}", sum(priced[a] for a in chosen[i + 1:]))
            if ledger.remaining(POOL) < priced[arm]:
                out(f"{arm}: {POOL} has under its ${priced[arm]:.2f}; it and the arms after it don't run")
                return 1
            done = run(command(accounts[arm], args.persona, args.days, start, match_file),
                       env=arm_env(arm, os.environ, tokens[arm]), check=False)
            if done.returncode != 0:
                out(f"{arm}: the run exited with {done.returncode}; the arms after it don't run")
                return 1
    except ValueError as error:
        out(f"stopped: {error}")
        return 1
    finally:
        if held is not None:
            held.release()
    try:
        report(root, accounts, out, args.reuse)
    except ValueError as error:  # every arm ran; only the reading stopped
        out(f"report stopped: {error}")
        return 1
    return 0


def main(argv: Sequence[str] | None = None, *, run: Callable[..., Any] = subprocess.run,
         out: Callable[[str], None] = print, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> int:
    parser = argparse.ArgumentParser(prog="sleep_objective_live", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--account", action="append", default=[], help="<arm>=<synthetic account id>")
    parser.add_argument("--token-file", action="append", default=[],
                        help="<arm>=<file holding a signed-in token for that arm's account>")
    parser.add_argument("--persona", default="founder")
    parser.add_argument("--days", type=int, default=5, choices=range(1, 15), metavar="N")
    parser.add_argument("--night-usd", type=float, default=None, help="a measured night's cost_usd")
    parser.add_argument("--start", type=date.fromisoformat, default=None,
                        help="the first virtual night, YYYY-MM-DD (default: a reused arm's first, else today less --days + 1)")
    parser.add_argument("--reuse", action="append", default=[], choices=REUSABLE,
                        help="a reference arm whose account already lived every night: read, not run again")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    try:
        accounts = _accounts(args.account)
        chosen = arms(accounts, args.reuse)
        morning_subset(os.environ)  # an unknown ANJO_SLEEP_MORNING_SUBSET stops the plan, as it stops each run
        files = _token_files(args.token_file, chosen)
    except ValueError as error:
        parser.error(str(error))
    if args.report:
        try:
            report(results_root(), accounts, out, args.reuse)
        except ValueError as error:
            parser.error(f"--report: {error}")
        return 0
    if absent := sorted(set(args.reuse) - set(chosen)):
        parser.error(f"--reuse names an arm of this run with its --account: {', '.join(absent)}")
    start = args.start or (now() - timedelta(days=args.days + 1)).date()  # the same virtual nights for every arm
    reused: dict[str, int] = {}
    try:
        if args.reuse:
            root = results_root()
            start = args.start or _first_night(root, accounts[next(a for a in REUSABLE if a in args.reuse)])
            reused = {arm: reused_nights(root, accounts[arm], start, args.days) for arm in args.reuse}
    except ValueError as error:
        parser.error(f"--reuse: {error}")
    priced = _plan(args, chosen, start, files, out, reused)
    if not args.yes:
        return 0
    if args.night_usd is None or args.night_usd <= 0:
        parser.error("--yes needs --night-usd, a measured night's cost_usd")
    tokens = {arm: token for arm in chosen if (token := _token(files.get(arm)))}
    lacking = [arm for arm in chosen if arm not in tokens and arm not in reused]
    if lacking:
        parser.error(f"--yes needs a non-empty --token-file for every arm, so each chats as its own account: "
                     f"{', '.join(lacking)}")
    root = results_root()
    used = [accounts[arm] for arm in chosen if arm not in reused and (root / accounts[arm]).exists()]
    if used:
        parser.error(f"{len(used)} account(s) already have results: every arm runs on a new synthetic account")
    left = default_ledger().remaining(POOL)
    if left < sum(priced.values()):
        parser.error(f"{POOL} has ${left:.2f} left, under the plan's ${sum(priced.values()):.2f}")
    with tempfile.TemporaryDirectory(prefix="sleep-objective-match-") as folder:
        return _run_arms(args, accounts, tokens, chosen, start, priced, run, out, Path(folder))


if __name__ == "__main__":
    raise SystemExit(main())
