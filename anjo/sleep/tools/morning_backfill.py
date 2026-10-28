"""Score a promoted night whose morning failed, afterwards: the night's own morning, run later.

    python -m anjo.sleep.tools.morning_backfill --user <account> --persona film_founder --night 2026-10-24
    python -m anjo.sleep.tools.morning_backfill --user <account> --persona film_founder --night 2026-10-24 --go

Without ``--go`` it only plans: the night's promoted adapter, what serves now, the morning it would run,
its holds, its expected cost (the judge and the serving GPU) and what it would write. Nothing is called,
held or written. With ``--go`` it runs that morning (``probes.morning.run_morning``) on the night's own
promoted checkpoint, pinned for the whole run (``pinned``): what serves never changes, so a later
promotion no longer blocks scoring an earlier night.

**The same morning the night would have run.** The persona's ``Morning`` from the same builder
(``--morning``, ``app.morning:build`` by default: the same sealed probes, judge and reply seeds, dated on
the night), the label the night was stamped with, and the night's probe set: the one an accelerated
night's summary records (``MORNING_PROBES``, ``ANJO_SLEEP_MORNING_SUBSET``'s); one that records none (a
night run before the record) the full set, and the plan says so; a daily night the committed subset
(``DAILY``) in development, every probe when final. ``--subset full|daily-v1`` asks that set instead. Arm
3x, when a donor is registered, reads the donor's adapter serving now, not the one it served that night:
if the donor promoted since, 3x and the ``3_vs_3x`` pair may differ from what the night's morning gave.

**Holds.** The judge and the arms are held as the night's morning holds them, under a key of its own
(``morning:<night>:<account>:backfill-<set>``): the failed morning's settled spend still counts in the
person's month (``night/past.latest_attempts`` keeps a key's last attempt), beside the backfill's.

**What it writes.** The score file where the night's morning would have put it
(``<results>/<account>/scores/night-NN.json``, marked ``pinned``), then a ``morning`` step marked
``backfilled: true`` appended to that night's ``log.jsonl``, and the night's story told again, as the
accelerated runner does after a morning. A backfill that fails is logged too: a failed ``morning``
step marked ``backfilled``, with its reason.

**Refused** before anything is called or held: a night that wasn't promoted, one whose morning
already succeeded on the probe set asked (an ``ok`` morning step on it, or a score file for the night on
it: a full backfill may follow a daily-v1 morning, never a second daily-v1 one), a checkpoint that isn't in
the person's promoted history, an account not registered synthetic unless ``--real`` (the default
builder refuses a real account itself), and whenever the night lock is held: the run holds it
throughout, so no night or simulated day starts under it.

The engine never imports the app: the sleep host's ports are named here and imported when the
command runs (``app_ports``), as the accelerated runner names its builders.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

from anjo.sleep.budget import Ledger, default_ledger
from anjo.sleep.checkpoints import CheckpointRegistry, ref_from_record
from anjo.sleep.night.cost import Units
from anjo.sleep.night.daily import morning_receipt
from anjo.sleep.night.nightlog import LOG, SUMMARY, NightLog, safe_error
from anjo.sleep.night.story import MORNING_FAILED
from anjo.sleep.night.sweep import NightBusy, held_elsewhere, night_lock
from anjo.sleep.probes.morning import (
    ARM_CALL_S,
    ARM_ORDER,
    DAILY,
    JUDGE_RETRY_SHARE,
    JUDGE_USD_PER_CALL,
    MORNING_PROBES,
    SCORES,
    SUBSETS,
    Morning,
    MorningError,
    MorningResult,
    ProbeSubset,
    TooManyNoVerdicts,
    _next_file,  # the morning's own rules: the next score file, refusing a night already scored
    _worst_judge_calls,  # and the judge calls its hold counts
    run_morning,
    subset_probes,
)
from anjo.sleep.reports.results import FULL_SET, results_root
from anjo.sleep.serving.config import SCALEDOWN_S
from anjo.sleep.types import CheckpointRef

MORNING: Final = "anjo.sleep.app.morning:build"  # the live comparison's and the daily's morning builder
SYNTHETIC: Final = "anjo.sleep.app.cleared_sessions:is_synthetic"
LOGS: Final = "anjo.sleep.app.ownership:night_logs"
REGISTRY: Final = "anjo.sleep.app.storage:checkpoint_registry"
PROMOTED: Final = "promoted"


class Refused(RuntimeError):
    """A backfill that may not run; its message names the rule."""


@dataclass(frozen=True)
class Ports:
    """What a backfill reaches outside through (``app_ports``: the sleep host's)."""

    morning: Callable[[str, str, date], Morning]  # the night's own morning builder
    is_synthetic: Callable[[str], bool]
    logs: Callable[[str], Path]  # the person's night logs (``<date>``, ``<date>.2`` on a rerun)
    registry: CheckpointRegistry
    ledger: Ledger  # the night lock lives beside it


@dataclass(frozen=True)
class Plan:
    user: str
    night: date
    folder: Path  # the night's log folder
    checkpoint: CheckpointRef  # the one the night promoted
    serving: CheckpointRef | None  # the one serving now, left as it is
    label: str
    subset: ProbeSubset | None  # None: the full set
    basis: str  # how the set is known: the night's record, its kind, or --subset
    morning: Morning
    probes: int
    replies: int
    judge_calls: int  # every comparison both ways, as the judge's hold counts them
    judge_hold_usd: float
    arms_hold_usd: float
    gpu_s: float  # expected: the wake, the arms at the night's pace, and Modal's idle window after
    target: Path  # the score file it will write


def _night(logs: Path, night: date) -> tuple[Path, dict[str, Any]]:
    """The night's one promoted log folder and its summary."""
    day = night.isoformat()
    finished = [(folder, json.loads((folder / SUMMARY).read_text(encoding="utf-8")))
                for folder in (sorted(logs.iterdir()) if logs.is_dir() else [])
                if (folder.name == day or folder.name.startswith(f"{day}.")) and (folder / SUMMARY).is_file()]
    promoted = [(folder, summary) for folder, summary in finished if summary.get("status") == PROMOTED]
    if not finished:
        raise Refused(f"no night finished on {day}")
    if not promoted:
        raise Refused(f"the night of {day} wasn't promoted ({', '.join(str(s.get('status')) for _, s in finished)})")
    if len(promoted) > 1:
        raise Refused(f"{len(promoted)} nights of {day} were promoted: which morning is owed is unclear")
    return promoted[0]


def _scored(folder: Path, asked: str | None, own: str | None) -> bool:
    """An ``ok`` morning step on the set ``asked`` (a name; None: the full set). A step that names no
    ``subset`` asked the night's own set, ``own``: the receipt names a subset only when one was asked."""
    steps = NightLog(folder).steps() if (folder / LOG).is_file() else []
    return any(step.get("step") == "morning" and step.get("status") == "ok" and step.get("subset", own) == asked
               for step in steps)


def _checkpoint(summary: dict[str, Any], user: str, registry: CheckpointRegistry) -> CheckpointRef:
    record = summary.get("checkpoint")
    if not isinstance(record, dict):
        raise Refused("the night's summary names no checkpoint")
    checkpoint = ref_from_record(record)
    if checkpoint.person_id != user or checkpoint.id not in {c.id for c in registry.history(user)}:
        raise Refused(f"{checkpoint.id} isn't in the person's promoted history")
    return checkpoint


def _own(summary: dict[str, Any], label: str) -> tuple[ProbeSubset | None, str]:
    """The probe set the night's morning asked, and how that is known: an accelerated night's summary
    records it (``MORNING_PROBES``), and one that records none asked the full set, as every accelerated
    morning did before the record; a daily night asks the daily subset in development (``night/daily.py``)."""
    recorded = summary.get(MORNING_PROBES)
    if recorded is not None:
        if str(recorded) != FULL_SET and str(recorded) not in SUBSETS:
            raise Refused(f"the night's summary names a probe set this backfill doesn't know: {recorded!r}")
        return SUBSETS.get(str(recorded)), "as the night's summary records"
    if summary.get("accelerated"):
        return None, ("the night's summary records none, so the full set (run before the record): "
                      "--subset daily-v1 if its morning asked that")
    return (DAILY, "a daily development night's") if label == "development" else (None, "a daily final night's")


def _set(subset: ProbeSubset | None) -> str:
    return "the full set" if subset is None else f"the {subset.name} subset"


def plan(user: str, persona: str, night: date, ports: Ports, *, real: bool = False,
         subset: str | None = None) -> Plan:
    """The backfill of ``user``'s ``night``, or ``Refused``: no model call, no hold, nothing written.
    ``subset`` names the probe set to ask (``full`` or a subset's name); None: the night's own."""
    if held_elsewhere(ports.ledger):
        raise Refused("the night lock is held: a night or a simulated day is running")
    if not real and not ports.is_synthetic(user):
        raise Refused("the account isn't registered synthetic (--real to score a real person's night)")
    folder, summary = _night(ports.logs(user), night)
    label = str(summary.get("label") or "development")
    own, basis = _own(summary, label)
    if subset is not None and subset != FULL_SET and subset not in SUBSETS:
        raise Refused(f"the probe set is {FULL_SET} or {', '.join(SUBSETS)}, not {subset!r}")
    asked = own if subset is None else SUBSETS.get(subset)
    basis = basis if subset is None else f"named with --subset (the night's own: {_set(own)})"
    name = None if asked is None else asked.name
    if _scored(folder, name, None if own is None else own.name):
        raise Refused(f"the morning of {night.isoformat()} already succeeded on {_set(asked)}")
    checkpoint = _checkpoint(summary, user, ports.registry)
    try:
        morning = ports.morning(user, persona, night)
        target = _next_file(results_root(morning.root) / user / SCORES, night, name, any_set=False)
    except (MorningError, ValueError) as error:  # already scored, or the builder's own refusal
        raise Refused(str(error)) from None
    probes = list(morning.sealed(user))
    probes = probes if asked is None else subset_probes(probes, asked)
    arms = ARM_ORDER if asked is None else tuple(a for a in ARM_ORDER if a in asked.arms)
    judge_calls = _worst_judge_calls(probes, arms)
    replies = len(probes) * len(arms)
    units, daily_replies = Units(), sum(DAILY.per_kind.values()) * len(DAILY.arms)
    serve = morning.prices.serve_gpu_per_hour
    return Plan(user, night, folder, checkpoint, ports.registry.last_promoted(user), label, asked, basis, morning,
                len(probes), replies, judge_calls,
                (judge_calls + math.ceil(judge_calls * JUDGE_RETRY_SHARE)) * JUDGE_USD_PER_CALL,
                serve * replies * ARM_CALL_S / 3600.0,
                units.wake_s + units.morning_arms_s * replies / daily_replies + SCALEDOWN_S, target)


def describe(chosen: Plan) -> list[str]:
    serve = chosen.morning.prices.serve_gpu_per_hour
    judge, gpu = chosen.judge_calls * JUDGE_USD_PER_CALL, chosen.gpu_s * serve / 3600.0
    asked = _set(chosen.subset)
    serving = "the base model" if chosen.serving is None else chosen.serving.id
    return [
        f"morning backfill: plan only, nothing run, nothing held (night {chosen.night.isoformat()}, "
        f"folder {chosen.folder.name})",
        f"  scores {chosen.checkpoint.id}, the adapter the night promoted, pinned; {serving} keeps serving",
        f"  the night's morning: label {chosen.label}, {asked}: {chosen.probes} probes, {chosen.replies} replies, "
        f"up to {chosen.judge_calls} judge calls",
        f"  the probe set: {asked}, {chosen.basis}",
        f"  holds: the judge ${chosen.judge_hold_usd:.2f} on meta_usd, the arms ${chosen.arms_hold_usd:.2f} "
        "on modal_usd and baseten_sleep_usd",
        f"  expected: ${judge + gpu:.2f} (the judge ${judge:.2f} at ${JUDGE_USD_PER_CALL:g} a call, the cost "
        f"model's estimate, settled at its tokens; the serving GPU ${gpu:.2f}, {chosen.gpu_s / 60.0:.0f} min: "
        f"the wake, the arms at the night's pace and Modal's idle window, at ${serve:g}/h)",
        f"  writes: {chosen.target}, then a morning step (backfilled: true) in {chosen.folder / LOG}, and the "
        "night's story again",
        "run again with --go to run it",
    ]


def _failed(log: NightLog, night: date, error: Exception, seconds: float) -> None:
    unread: dict[str, Any] = {"no_verdict": error.no_verdict} if isinstance(error, TooManyNoVerdicts) else {}
    log.step("morning", "failed", uses="new", system=None, backfilled=True, seconds=round(seconds, 3),
             why=f"{MORNING_FAILED}{safe_error(error)}", **unread)


def backfill(chosen: Plan, ports: Ports, *, timer: Callable[[], float] = time.monotonic) -> MorningResult:
    """Run the planned morning under the night lock and log it in the night's log, ``backfilled``."""
    log = NightLog(chosen.folder)
    try:
        with night_lock(ports.ledger):
            began = timer()
            try:
                done = run_morning(chosen.user, chosen.night, chosen.checkpoint, morning=chosen.morning,
                                   label=chosen.label, subset=chosen.subset, pinned=True)
            except MorningError:
                raise  # refused before any call: no morning ran
            except Exception as error:  # logged as the night's morning failure, then raised
                _failed(log, chosen.night, error, timer() - began)
                raise
            log.step("morning", "ok", uses="new", system=None, backfilled=True, checkpoint=chosen.checkpoint.id,
                     **morning_receipt(done, timer() - began))  # its subset among them, when one
            log.tell(chosen.night, PROMOTED)
    except NightBusy as busy:
        raise Refused(str(busy)) from None
    return done


def _named(path: str) -> Any:
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def app_ports(morning: str = MORNING) -> Ports:
    """The sleep host's ports, imported when the command runs."""
    return Ports(morning=_named(morning), is_synthetic=_named(SYNTHETIC), logs=_named(LOGS),
                 registry=_named(REGISTRY)(), ledger=default_ledger())


def main(argv: Sequence[str] | None = None, *, ports: Ports | None = None,
         out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description="Score a promoted night whose morning failed.")
    parser.add_argument("--user", required=True, help="the account")
    parser.add_argument("--persona", required=True, help="the persona the account plays (its sealed probes)")
    parser.add_argument("--night", required=True, type=date.fromisoformat, help="the night, YYYY-MM-DD")
    parser.add_argument("--go", action="store_true", help="run it (without: the plan only)")
    parser.add_argument("--real", action="store_true", help="allow an account that isn't registered synthetic")
    parser.add_argument("--subset", choices=sorted([*SUBSETS, FULL_SET]), default=None,
                        help="ask this probe set instead of the night's own")
    parser.add_argument("--morning", default=MORNING, help="the morning builder, <module>:<function>")
    args = parser.parse_args(argv)
    found = ports or app_ports(args.morning)
    try:
        chosen = plan(args.user, args.persona, args.night, found, real=args.real, subset=args.subset)
        for line in describe(chosen):
            out(line)
        if not args.go:
            return 0
        done = backfill(chosen, found)
    except Refused as refused:
        out(f"refused: {refused}")
        return 1
    out(f"scored: {done.path} ({done.replies} replies, {done.judge_calls} judge calls, judge ${done.judge_usd:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
