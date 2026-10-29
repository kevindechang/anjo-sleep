"""A second-family judge on the comparison's mornings (spec 5.1): Gemini 3.8 Flash on Vertex beside Muse
Spark, on the same pairs (``python -m anjo.sleep.tools.morning_second_judge``).

    python -m anjo.sleep.tools.morning_second_judge --persona film_founder --account seed=<id> ...   # the plan
    python -m anjo.sleep.tools.morning_second_judge --persona film_founder --account seed=<id> ... --out DIR --go

**The pairs are asked again.** A morning keeps no reply (its score file holds ids, codes and numbers), so
for each daily-v1 morning of each account (``morning_dimensions.mornings``) arms 1 and 3 answer that
morning's judged probes again: ``new_work``, ``vocab`` and ``tone_hold`` (``taste_ab`` is a log-probability
choice no judge reads). The persona's morning is built as ``tools.morning_backfill`` builds it (the same
sealed probes, reply seeds and night date), with arm 3 pinned to the checkpoint that morning scored.
Replies are sampled (``arms.REPLY_TEMPERATURE``, seeded) and the server's batches move them, so they may
differ from the morning's: Spark's verdicts on them are set beside the stored ones (``stored_agreement``),
which says how far asking again moved the result.

**Two judges, blind, both orders.** Each pair goes to Muse Spark (the morning's own judge) and to Gemini
(``probes.second_judge``: the same ``judge_v1`` prompt and brief, through ``mind.apd.backend:_gemini``'s
Vertex call), arm 3's reply first and then second; neither judge is told which arm wrote which. A verdict
that flips with the order is a tie. An answer that can't be read is asked once more (the morning's rule);
one still unread, or a call that failed, is no verdict, counted under ``none``.

**The report** (JSON lines: ids, codes and numbers): per life (account) and dimension, and pooled over the
lives, each judge's wins, ties and losses for arm 3 against base and its score, and how often the two
judges gave the same outcome; every disagreement (night, probe, kind, each judge's outcome); Spark's
agreement with the stored morning. Knowledge is ``new_work`` and ``vocab``; interaction is ``tone_hold``
alone here. The pairs themselves (the prompt, both replies, each judge's outcome) go to
``--out/pairs.jsonl``, synthetic people only, never to a person's sleep folder.

**Spend**, held before the first call and settled at what it used: the serving GPU on the morning's pools
(``ARM_CALL_S`` a reply, the wake and the idle window), Muse Spark on ``meta_usd`` (``JUDGE_USD_PER_CALL`` a
call and the retry share, settled at its tokens), Gemini on ``vertex_usd`` (its worst call: a full Spark
prompt's tokens and its reply cap, settled at its tokens). **Refused** before anything is called, held or
written: an account not registered synthetic, a morning whose checkpoint isn't in the account's promoted
history or whose probes aren't the ones daily-v1 asks now, a held night lock, and an ``--out`` inside the
sleep folders; the run holds it. A Glimmer that doesn't wake is refused the same way, its holds settled at
nothing.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Final

from anjo.sleep.budget import Bound, Pool
from anjo.sleep.model_api import ModelApiError
from anjo.sleep.night.cost import Units
from anjo.sleep.night.holds import SPARK_MAX_IN_TOKENS, held, usd
from anjo.sleep.night.sweep import NightBusy, held_elsewhere, night_lock
from anjo.sleep.parallel import Calls
from anjo.sleep.probes.arms import Arm
from anjo.sleep.probes.morning import (
    ARM_CALL_S,
    ARM_POOLS,
    DAILY,
    JUDGE_POOLS,
    JUDGE_RETRY_SHARE,
    JUDGE_USD_PER_CALL,
    Morning,
    _answer,  # the morning's own: each arm's questions prepared, then asked side by side
    _Counting,  # its judge wrapper: calls counted, an unreadable answer asked once more
    _judge_spend,  # and what the judge spent: its tokens, and the calls that reported none at the bound
    _judge_usage,
    _Pinned,  # and the checkpoint a morning scored, fixed for the run
    subset_probes,
)
from anjo.sleep.probes.score import JudgeError, MuseJudge, NoVerdict, dimension_of, judge_orders
from anjo.sleep.probes.second_judge import MODEL_ID, SecondJudge
from anjo.sleep.reports.results import results_root
from anjo.sleep.serving.backend import phase
from anjo.sleep.serving.client import WAKE_TOTAL_TIMEOUT_S, GlimmerNotReady
from anjo.sleep.serving.config import SCALEDOWN_S
from anjo.sleep.tools import morning_backfill
from anjo.sleep.tools.morning_dimensions import MISSING, PAIR, mornings
from anjo.sleep.tools.practice_ab import JUDGE_CALL_USD, JUDGE_IN_PER_M, JUDGE_MAX_OUT, JUDGE_OUT_PER_M
from anjo.sleep.types import CheckpointRef, Probe

JUDGED: Final = ("new_work", "vocab", "tone_hold")  # daily-v1's kinds a judge reads
ASKED: Final[tuple[Arm, ...]] = ("1", "3")
JUDGES: Final = 2  # Muse Spark and Gemini: each allows its own retries on each morning (``_judge_job``)
GEMINI: Final = "anjo.sleep.mind.apd.backend:_gemini"  # Vertex, thinking low, its reply cap (practice_ab's judge call)
VERTEX_POOL: Final[Pool] = "vertex_usd"
GEMINI_WORST_USD: Final[float] = (SPARK_MAX_IN_TOKENS * JUDGE_IN_PER_M + JUDGE_MAX_OUT * JUDGE_OUT_PER_M) / 1e6
# Muse Spark's expected call, an estimate: its median reply on Teo's night 1 (2026-10-28, 257 calls), a 3,000-token prompt.
SPARK_IN_TOKENS: Final = 3_000
SPARK_OUT_TOKENS: Final = 2_558
DAILY_REPLIES: Final = sum(DAILY.per_kind.values()) * len(DAILY.arms)  # what ``Units.morning_arms_s`` measured
PAIRS_FILE: Final = "pairs.jsonl"


class Refused(RuntimeError):
    """A run that may not start; its message names the rule."""


@dataclass(frozen=True)
class Ports:
    backfill: morning_backfill.Ports  # the persona's morning builder, the synthetic check, the registry, the ledger
    gemini: Callable[[str], tuple[str, int, int]]  # a prompt -> Gemini's reply, its tokens in and out


@dataclass(frozen=True)
class Job:
    arm: str
    account: str
    night: date
    checkpoint: CheckpointRef  # the one the morning scored
    morning: Morning
    probes: tuple[Probe, ...]  # its judged probes
    stored: Mapping[str, str]  # probe id -> the morning's stored outcome for arm 3 against base


@dataclass(frozen=True)
class Spend:
    replies: int
    calls: int  # each judge's: every pair in both orders
    rate: float  # the serving GPU's $ an hour
    gpu_worst_s: float
    gpu_expected_s: float
    spark_hold: float
    spark_expected: float
    gemini_hold: float
    gemini_expected: float


def _job(arm: str, account: str, persona: str, record: Mapping[str, Any], ports: morning_backfill.Ports) -> Job:
    night = date.fromisoformat(str(record["night"]))
    checkpoint = {c.id: c for c in ports.registry.history(account)}.get(str(record.get("checkpoint")))
    if checkpoint is None:
        raise Refused(f"{arm} {night}: the checkpoint its morning scored isn't in the account's promoted history")
    morning = ports.morning(account, persona, night)
    if not isinstance(morning.judge, MuseJudge):
        raise Refused("the morning's judge isn't Muse Spark's: there is no brief to share with Gemini")
    probes = tuple(p for p in subset_probes(list(morning.sealed(account)), DAILY) if p.kind in JUDGED)
    stored = {str(row["probe"]): str(row["outcome"]) for row in record.get("outcomes", ())
              if row.get("pair") == PAIR and row.get("kind") in JUDGED}
    if set(stored) != {p.id for p in probes}:
        raise Refused(f"{arm} {night}: the morning's probes aren't the ones {DAILY.name} asks now")
    return Job(arm, account, night, checkpoint, morning, probes, stored)


def plan(accounts: Mapping[str, str], persona: str, ports: Ports,
         root: Path | None = None) -> tuple[list[Job], list[dict[str, str]]]:
    """Every daily-v1 morning to judge again, and the accounts with none; or ``Refused``. Nothing is called."""
    found = ports.backfill
    if held_elsewhere(found.ledger):
        raise Refused("the night lock is held: a night, its gates or a morning is serving on this host")
    jobs: list[Job] = []
    missing: list[dict[str, str]] = []
    for arm, account in accounts.items():
        if not found.is_synthetic(account):
            raise Refused(f"{account} isn't registered synthetic: Vertex sees synthetic people only")
        records = mornings(results_root(root), account)
        if not records:
            missing.append({"line": "missing", "arm": arm, "account": account, "why": MISSING})
        jobs += [_job(arm, account, persona, record, found) for record in records]
    return jobs, missing


def spend(jobs: Sequence[Job]) -> Spend:
    pairs = sum(len(job.probes) for job in jobs)
    replies, calls = len(ASKED) * pairs, 2 * pairs
    # Each judge may ask its unreadable answers once more, per morning: the worst case ``_judge_job`` allows.
    retries = JUDGES * sum(math.ceil(2 * len(job.probes) * JUDGE_RETRY_SHARE) for job in jobs)
    held_calls = calls + retries
    prices, units = jobs[0].morning.prices, Units()
    spark_call = (SPARK_IN_TOKENS * prices.spark_in_per_m + SPARK_OUT_TOKENS * prices.spark_out_per_m) / 1e6
    return Spend(replies, calls, prices.serve_gpu_per_hour, replies * ARM_CALL_S + WAKE_TOTAL_TIMEOUT_S + SCALEDOWN_S,
                 units.morning_arms_s * replies / DAILY_REPLIES + units.wake_s + SCALEDOWN_S,
                 held_calls * JUDGE_USD_PER_CALL, calls * spark_call, held_calls * GEMINI_WORST_USD, calls * JUDGE_CALL_USD)


def describe(jobs: Sequence[Job], cost: Spend) -> list[str]:
    by_arm: dict[str, list[str]] = {}
    for job in jobs:
        by_arm.setdefault(job.arm, []).append(job.night.isoformat())
    gpu, gpu_held = cost.rate * cost.gpu_expected_s / 3600.0, cost.rate * cost.gpu_worst_s / 3600.0
    return [
        "second judge: plan only, nothing called, held or written",
        f"  mornings  {len(jobs)} mornings ({'; '.join(f'{a}: {', '.join(n)}' for a, n in by_arm.items())}), "
        f"{DAILY.name}'s judged probes ({', '.join(JUDGED)}; taste_ab is a log-probability choice no judge reads)",
        f"  replies   {cost.replies} replies, arms 1 and 3 asked again (a morning keeps none), arm 3 pinned to the "
        "checkpoint each morning scored",
        f"  judges    {cost.calls} Muse Spark calls and {cost.calls} Gemini calls ({MODEL_ID}): each pair blind, "
        "in both orders",
        f"  holds     the GPU ${gpu_held:.2f} on {' and '.join(ARM_POOLS)}; Muse Spark ${cost.spark_hold:.2f} on "
        f"{JUDGE_POOLS[0]}; Gemini ${cost.gemini_hold:.2f} on {VERTEX_POOL}",
        f"  expected  ~${gpu + cost.spark_expected + cost.gemini_expected:.2f}, estimates: the GPU ${gpu:.2f} "
        f"(~{cost.gpu_expected_s / 60:.0f} min: the arms at the daily pace, the wake, the idle window, at "
        f"${cost.rate:g}/h), Muse Spark ${cost.spark_expected:.2f}, Gemini ${cost.gemini_expected:.2f}",
        f"  writes    the pairs to --out/{PAIRS_FILE}; prints agreement per life and dimension, and every disagreement",
        "run again with --out DIR --go to run it",
    ]


class _Metered:
    """A ``VertexClient`` over a prompt -> (reply, tokens in, tokens out) call, its spend summed. A call that
    failed is counted at the worst a call can cost."""

    def __init__(self, ask: Callable[[str], tuple[str, int, int]]) -> None:
        self._ask, self._lock = ask, threading.Lock()
        self._in = self._out = self._failed = 0

    def generate(self, model: str, prompt: str, *, temperature: float) -> str:
        try:
            text, tokens_in, tokens_out = self._ask(prompt)
        except ModelApiError:
            with self._lock:
                self._failed += 1
            raise
        with self._lock:
            self._in, self._out = self._in + tokens_in, self._out + tokens_out
        return text

    def usd(self) -> float:
        with self._lock:
            spent: float = (self._in * JUDGE_IN_PER_M + self._out * JUDGE_OUT_PER_M) / 1e6 + self._failed * GEMINI_WORST_USD
            return spent


def _judged(judge: Any, probe: Probe, first: str, second: str) -> tuple[str, bool] | None:
    try:
        found: tuple[str, bool] = judge_orders(judge, probe, first, second)
        return found
    except (NoVerdict, JudgeError, ModelApiError):
        return None  # no verdict from this judge on this probe: counted, never scored


def _judge_job(job: Job, ports: Ports, meter: _Metered, spark_spent: list[float]) -> list[dict[str, Any]]:
    """One morning's pairs asked again and judged by both judges: the pairs, its Muse Spark spend added to
    ``spark_spent`` in a ``finally`` so a morning that fails partway still settles the calls it made."""
    morning, muse = job.morning, job.morning.judge
    assert isinstance(muse, MuseJudge)  # the plan refused any other
    if morning.wake is not None and not morning.wake().ok:
        raise GlimmerNotReady()
    deps = replace(morning.arms, adapters=_Pinned(morning.arms.adapters, job.account, job.checkpoint),
                   until=job.night, context_cache={})
    calls = Calls(morning.concurrency)
    replies, _ = _answer(job.account, job.probes, deps, calls, ASKED)
    retries = math.ceil(2 * len(job.probes) * JUDGE_RETRY_SHARE)
    spark = _Counting(muse, retries)
    gemini = _Counting(SecondJudge(meter, muse.brief_for, ports.backfill.is_synthetic), retries)
    before = _judge_usage(morning)
    tasks = [(index, judge) for index in range(len(job.probes)) for judge in (spark, gemini)]
    try:
        found = calls.map(lambda task: _judged(task[1], job.probes[task[0]], replies["3"][task[0]].text,
                                               replies["1"][task[0]].text), tasks)
    finally:  # ``Calls.map`` waits for in-flight calls before it raises: the reading is complete
        spark_spent[0] += _judge_spend(spark, morning, before)[0]
    dims, pairs = dimension_of(), []
    for index, probe in enumerate(job.probes):
        by_spark, by_gemini = found[2 * index], found[2 * index + 1]
        pairs.append({"arm": job.arm, "account": job.account, "night": job.night.isoformat(), "probe": probe.id,
                      "kind": probe.kind, "dimension": dims[probe.kind], "stored": job.stored[probe.id],
                      "spark": by_spark and by_spark[0], "spark_flip": bool(by_spark and by_spark[1]),
                      "gemini": by_gemini and by_gemini[0], "gemini_flip": bool(by_gemini and by_gemini[1]),
                      "prompt": probe.prompt, "replies": {arm: replies[arm][index].text for arm in ASKED}})
    return pairs


def run(jobs: Sequence[Job], ports: Ports, folder: Path, cost: Spend, *,
        timer: Callable[[], float] = time.monotonic) -> list[dict[str, Any]]:
    """Every planned morning judged again under the night lock, each paid step held first; the pairs."""
    ledger, meter, spark_spent = ports.backfill.ledger, _Metered(ports.gemini), [0.0]
    key = f"second_judge:{jobs[0].account}:{int(time.time())}"
    pairs: list[dict[str, Any]] = []
    try:
        with night_lock(ledger), phase(), ExitStack() as holds:
            started = timer()
            holds.enter_context(held(ledger, f"{key}:arms", ARM_POOLS, Bound(cost.rate, cost.gpu_worst_s),
                                     lambda: cost.rate * (max(0.0, timer() - started) + SCALEDOWN_S) / 3600.0))
            holds.enter_context(held(ledger, f"{key}:spark", JUDGE_POOLS, usd(cost.spark_hold), lambda: spark_spent[0]))
            holds.enter_context(held(ledger, f"{key}:gemini", (VERTEX_POOL,), usd(cost.gemini_hold), meter.usd))
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / PAIRS_FILE).open("w", encoding="utf-8") as written:
                for job in jobs:
                    with phase(job.morning.sleep):  # Glimmer sleeps once, when the outer phase ends
                        found = _judge_job(job, ports, meter, spark_spent)
                    written.writelines(json.dumps(pair, ensure_ascii=False) + "\n" for pair in found)
                    pairs += found
    except NightBusy as busy:
        raise Refused(str(busy)) from None
    return pairs


def _tally(rows: Sequence[Mapping[str, Any]], judge: str) -> dict[str, Any]:
    outcomes = [row[judge] for row in rows]
    wins, ties, losses = outcomes.count("win"), outcomes.count("tie"), outcomes.count("loss")
    judged = wins + ties + losses
    return {"wins": wins, "ties": ties, "losses": losses, "none": outcomes.count(None),
            "flips": sum(bool(row[f"{judge}_flip"]) for row in rows),
            "score": round((wins + ties / 2) / judged, 4) if judged else None}


def _agreement(arm: str, dimension: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    both = [row for row in rows if row["spark"] is not None and row["gemini"] is not None]
    same = sum(row["spark"] == row["gemini"] for row in both)
    return {"line": "agreement", "arm": arm, "dimension": dimension, "kinds": sorted({row["kind"] for row in rows}),
            "probes": len(rows), "spark": _tally(rows, "spark"), "gemini": _tally(rows, "gemini"),
            "both_judged": len(both), "same": same, "agreement": round(same / len(both), 4) if both else None}


def report(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per life and dimension, then pooled: each judge's tally for arm 3 and their agreement; every
    disagreement; Spark's agreement with the stored mornings. Ids, codes and numbers only."""
    lines: list[dict[str, Any]] = []
    for arm in [*dict.fromkeys(row["arm"] for row in pairs), "all"]:
        mine = [row for row in pairs if arm in ("all", row["arm"])]
        lines += [_agreement(arm, dim, [row for row in mine if row["dimension"] == dim])
                  for dim in sorted({row["dimension"] for row in mine})]
        compared = [row for row in mine if row["spark"] is not None and row["stored"] in ("win", "tie", "loss")]
        lines.append({"line": "stored_agreement", "arm": arm, "both": len(compared),
                      "same": sum(row["spark"] == row["stored"] for row in compared)})
    lines += [{"line": "disagreement", **{key: row[key] for key in ("arm", "night", "probe", "kind", "spark", "gemini")}}
              for row in pairs if row["spark"] is not None and row["gemini"] is not None and row["spark"] != row["gemini"]]
    return lines


def _named(path: str) -> Any:
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def app_ports() -> Ports:
    """The sleep host's ports and the Vertex call, imported when the command runs."""
    return Ports(morning_backfill.app_ports(), _named(GEMINI))


def _accounts(parser: argparse.ArgumentParser, pairs: Sequence[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for pair in pairs:
        arm, _, account = pair.partition("=")
        if not arm or not account or arm in found:
            parser.error("--account is <arm>=<account id>, once per arm")
        found[arm] = account
    return found


def main(argv: Sequence[str] | None = None, *, ports: Ports | None = None, root: Path | None = None,
         out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description="Gemini beside Muse Spark on the comparison's daily mornings.")
    parser.add_argument("--account", action="append", default=[], help="<arm>=<account id>, once per arm")
    parser.add_argument("--persona", required=True, help="the persona the accounts play (its sealed probes)")
    parser.add_argument("--out", type=Path, help="the folder the pairs are written to (needed with --go)")
    parser.add_argument("--go", action="store_true", help="run it (without: the plan only)")
    args = parser.parse_args(argv)
    accounts = _accounts(parser, args.account)
    if not accounts:
        parser.error("name at least one --account")
    if args.go and args.out is None:
        parser.error("--go writes the pairs to --out DIR, never to a person's sleep folder")
    found = ports or app_ports()
    try:
        folders = results_root(root)
        if args.out is not None and args.out.resolve().is_relative_to(folders.resolve()):
            raise Refused(f"--out {args.out} is inside the sleep folders ({folders}): the pairs go in a "
                          "folder of their own, never a person's sleep folder")
        jobs, missing = plan(accounts, args.persona, found, root)
        for line in missing:
            out(json.dumps(line))
        if not jobs:
            out(f"nothing to judge: no {DAILY.name} morning in these accounts")
            return 0
        cost = spend(jobs)
        for text in describe(jobs, cost):
            out(text)
        if not args.go:
            return 0
        pairs = run(jobs, found, args.out, cost)
    except Refused as refused:
        out(f"refused: {refused}")
        return 1
    except GlimmerNotReady:  # the holds' context managers settled them on the way out
        out("refused: the serving app did not wake: nothing was judged")
        return 1
    for line in report(pairs):
        out(json.dumps(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
