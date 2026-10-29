"""Score each arm's nightly adapters on a persona's frozen knowledge set, after the runs: post-hoc, so it
never delays or changes a run (``python -m anjo.sleep.tools.knowledge_score``).

    python -m anjo.sleep.tools.knowledge_score --account seed=<id> --account today=<id> ...   # the plan
    python -m anjo.sleep.tools.knowledge_score --account seed=<id> --account today=<id> ... --go
    python -m anjo.sleep.tools.knowledge_score --ceiling recipe=<adapter> --go   # an oracle ceiling (below)

**What it reads:** each account's night folders under the results root (``reports.results``): the adapter
a night trained (``summary.json``'s ``trained_adapter``, scored whether or not it was promoted, and
flagged), and the simulated day's wall-clock window (the ``close_day`` step). A night that trained
nothing is listed, not scored.

**What it scores:** the frozen set (``sets/knowledge.py``) on the base model (no adapter) and on every
trained adapter. Each choice is one log-probability call through Glimmer's client
(``VllmClient.continuation_logprobs``: ``/v1/completions`` with prompt log-probabilities, prefill only),
summed over its tokens; the first choice wins a tie (the general gate's rule). The prompt is the set's
one line, alone: no system turn, brief, memory or thought channel, so what an adapter picks is in its
weights. Every prompt carries the gates' fixed date, so every side is asked the same thing.

**Seen or unseen:** an item no simulated day mentioned can't have been learned from the days. A line of a
day mentions an item when it holds the item's key (a number as a word or in digits; a phrase's content
words) and at least one other content word of its statement. Per arm, an item is seen by night N when a
line of days 1..N mentions it; a pair compares the items both arms' days mentioned. The days are read
through the app's own reader, ``anjo.sleep.app.history_source:synthetic_day_text`` (``DAYS``, loaded by
name so the engine never imports the app; no other reader can be named), which refuses a real person;
they are read before anything is held or woken.

**The report** (one JSON line each: item ids and numbers, never a statement's or a day's words): ``set``;
``base`` (accuracy, the floor every arm starts from, beside chance; mean margin: the true choice's
log-probability less the best counterfeit's);
``arm_night`` per arm and night (1, 2, ... in date order): trained and promoted, accuracy on all items
and on seen ones (the main number), the mean margin gained over base (all, seen), the items newly right
and newly wrong against base, and the unseen ones; ``pair`` per night date for each claim in ``PAIRS``
both of whose arms trained that night: the paired differences in accuracy and margin, on the items both
arms saw (the main number) and on all items, each with its item and statement counts and a 95% paired
cluster bootstrap interval (each resample draws statements, since a statement's items move together, and
serves both sides; ``RESAMPLES`` of them from a fixed seed).

**A ceiling** (``--ceiling <label>=<adapter>``, ``tools.knowledge_ceiling``'s adapter, with or without
arms): scored as a night is, on every item (it was trained on every statement), and printed as a
``ceiling`` line marked CEILING with its paired difference from the base. Contamination by design: it
says what the pipeline can store when handed the answers, never a result.

**Reproducible** (2026-10-29): the app serves Glimmer at FP8 with activations quantized on the fly, one
scale for each layer's whole batch (``config.SERVING``, ``--quantization fp8``), and vLLM batches every
request in flight together, other adapters' and other clients' included. So a line's log-probability
depends on what shared its batch: two runs 16 at a time moved one adapter's accuracy by 14.5 points. The
scorer sends one request at a time (``SCORING_WORKERS``), so each line is computed alone, the same way
every time; its prefill is never a prefix-cache hit (vLLM computes a prompt-log-probability request
whole). That holds only with nothing else on the app (its one container batches every client together),
so a run holds the night lock (``night.sweep.night_lock``) from before its hold to its last request: it is
refused while a night, its gates, a simulated day or a morning runs on this host, and none starts while
it scores. Traffic from another host (the live app's chat) isn't covered; ``--repeat`` is the receipt: the
base and the first adapter scored again at the end, with the largest change in any choice's
log-probability, the accuracy change and the items that flipped. Nights keep their own concurrency, so
the other log-probability scorers, the general gate (``gates/general.py``) and the log-probability
probes of the mornings (``probes/logprob.py``), still send 16 at a time and carry the same batch noise.

**Spend:** it wakes the app (meant for a warm one) and scores for at most ``SCORE_MAX_S``; no adapter
starts after it. That, the wake and Modal's idle window after the last request are held on
``modal_usd`` before the first request, and settled at what ran plus the idle window. The URL and token
come from ``ANJO_SLEEP_MODAL_URL`` (or ``--url``) and ``ANJO_SLEEP_MODAL_TOKEN``, read only with ``--go``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Protocol

from anjo.sleep.budget import Bound, default_ledger
from anjo.sleep.gates import EVAL_DAY, in_order
from anjo.sleep.night.holds import hold
from anjo.sleep.night.sweep import NightBusy, night_lock
from anjo.sleep.reports.results import results_root
from anjo.sleep.serving import backend, config, gpus
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.sets import knowledge
from anjo.sleep.sets.knowledge import KnowledgeItem, KnowledgeSet
from anjo.sleep.sets.loader import SetError
from anjo.sleep.types import CheckpointRef

POOL: Final = "modal_usd"
SCORE_MAX_S: Final = 60 * 60.0
SCORING_WORKERS: Final = 1  # one request in flight, so nothing of the scorer's shares its batch (Reproducible)
LOCKED: Final = "the night lock is held: a night, its gates or a morning is serving on this host; nothing was held or sent"
REPRODUCIBLE_NATS: Final = 1e-3  # a repeat this close, choice by choice, with no pick changed, is the same number
WAKE_S: Final = 600.0  # client.WAKE_TOTAL_TIMEOUT_S: a cold start's worst
CALL_S: Final = 0.5  # an estimate, not measured: a choice's three round trips (two tokenizations, one prefill), one at a time
LOAD_S: Final = 10.0  # an estimate: loading one adapter from the volume
RESAMPLES: Final = 2000
CI_SEED: Final = 20261029
DAYS: Final = "anjo.sleep.app.history_source:synthetic_day_text"
# Each claim and what it is compared with: the live comparison's arms (tools.sleep_objective_live).
CEILING: Final = "contamination by design: trained on the set's own statements, never a result"
PAIRS: Final = (("seed", "today"), ("fold", "seed"), ("protect", "fold"), ("protect", "protect_random"),
                ("dreams", "seed_matched"), ("seed_cap", "seed"))
_ARM: Final = re.compile(r"[a-z][a-z_]{0,31}")
_ACCOUNT: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_STOP: Final = frozenset(
    "about after also always been before being both but does each every from have into just like more most "
    "much never only over same such than that their them then there these they this those under very what "
    "when where which while will with would your yours".split())
_NUMBERS: Final = {word: n for n, word in enumerate(knowledge.SMALL, 2)} | {
    word: 10 * n for n, word in enumerate(knowledge.TENS, 2)}
_ONES: Final = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_COMPOUND: Final = re.compile(rf"\b({'|'.join(knowledge.TENS)})-({'|'.join(_ONES)})\b")

DayText = Callable[[str, tuple[datetime, datetime]], Sequence[str]]
Scores = Mapping[str, tuple[float, ...]]  # item id: each choice's summed log-probability


class LogprobClient(Protocol):
    """``VllmClient`` as the scorer calls it."""

    @property
    def rendering(self) -> str: ...

    def on(self, day: date) -> LogprobClient: ...

    def continuation_logprobs(self, adapter: CheckpointRef | None, messages: list[dict[str, str]],
                              continuation: str) -> Sequence[float]: ...


@dataclass(frozen=True)
class Night:
    number: int
    date: str
    adapter: str | None  # the adapter it trained, None when it trained none
    promoted: bool
    window: tuple[datetime, datetime] | None


@dataclass(frozen=True)
class Paired:
    n: int
    clusters: int
    diff: float | None
    low: float | None
    high: float | None

    def record(self, prefix: str) -> dict[str, Any]:
        rounded = [None if v is None else round(v, 4) for v in (self.diff, self.low, self.high)]
        return {f"{prefix}_diff": rounded[0], f"{prefix}_ci95": rounded[1:]}


# ── reading the runs ─────────────────────────────────────────────────────────────────────────────────


def _window(log: Path) -> tuple[datetime, datetime] | None:
    if not log.is_file():
        return None
    for line in log.read_text(encoding="utf-8").splitlines():
        record = json.loads(line) if line.strip() else {}
        if record.get("step") == "close_day" and isinstance(window := record.get("window"), list) and len(window) == 2:
            return datetime.fromisoformat(window[0]), datetime.fromisoformat(window[1])
    return None


def nights(root: Path, account: str) -> list[Night]:
    """Each finished night of ``account``, in date order."""
    folder = root / account / "nights"
    found: list[Night] = []
    for path in sorted(folder.iterdir()) if folder.is_dir() else ():
        summary = path / "summary.json"
        if not summary.is_file():
            continue
        done = json.loads(summary.read_text(encoding="utf-8"))
        adapter = done.get("trained_adapter")
        if adapter is not None and not (isinstance(adapter, str) and config.ADAPTER_NAME.fullmatch(adapter)):
            raise ValueError(f"night {path.name}: its trained adapter isn't named p-<hash>-n<night>")
        found.append(Night(len(found) + 1, path.name, adapter, done.get("status") == "promoted",
                           _window(path / "log.jsonl")))
    return found


# ── what the days mentioned ──────────────────────────────────────────────────────────────────────────


def _tokens(text: str) -> set[str]:
    lowered = _COMPOUND.sub(lambda m: str(_NUMBERS[m.group(1)] + _ONES.index(m.group(2)) + 1), text.lower())
    found = set()
    for word in re.findall(r"\d[\d,]*(?:\.\d+)?|[a-z]+", lowered):
        if word[0].isdigit():
            found.add(word.replace(",", ""))
            continue
        word = str(_NUMBERS.get(word, word))
        found.add(word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word)
    return found


def _content(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOP and (len(t) >= 4 or t[0].isdigit())}


def seen(items: Sequence[KnowledgeItem], lines: Sequence[str]) -> frozenset[str]:
    """The ids of the items some line mentions: its key, and one other content word of its statement."""
    read = [_tokens(line) for line in lines]
    found = set()
    for item in items:
        key = {t for t in _tokens(item.key) if t not in _STOP}
        anchors = _content(item.statement) - key
        if key and any(key <= words and (not anchors or anchors & words) for words in read):
            found.add(item.id)
    return frozenset(found)


# ── scoring ──────────────────────────────────────────────────────────────────────────────────────────


def score(client: LogprobClient, adapter: CheckpointRef | None, items: Sequence[KnowledgeItem],
          workers: int = SCORING_WORKERS) -> dict[str, tuple[float, ...]]:
    """Every choice's summed log-probability, the prompt alone as the person's one turn."""
    calls = [(item.prompt, choice) for item in items for choice in item.choices]
    sums = in_order(lambda call: math.fsum(client.continuation_logprobs(
        adapter, [{"role": "user", "content": call[0]}], call[1])), calls, workers)
    out, at = {}, 0
    for item in items:
        out[item.id] = tuple(sums[at:at + len(item.choices)])
        at += len(item.choices)
    return out


def _right(item: KnowledgeItem, scores: Sequence[float]) -> bool:
    return max(range(len(scores)), key=lambda n: (scores[n], -n)) == item.answer


def _margin(item: KnowledgeItem, scores: Sequence[float]) -> float:
    return scores[item.answer] - max(s for n, s in enumerate(scores) if n != item.answer)


def chance(items: Sequence[KnowledgeItem]) -> float | None:
    """The accuracy of a pick at random: the mean of one over each item's choices."""
    return _mean([1.0 / len(item.choices) for item in items])


def _mean(values: Sequence[float]) -> float | None:
    return round(math.fsum(values) / len(values), 4) if values else None


def paired(a: Mapping[str, float], b: Mapping[str, float], *, clusters: Mapping[str, str] | None = None,
           resamples: int = RESAMPLES, seed: int = CI_SEED) -> Paired:
    """``a`` less ``b`` over the item ids both hold, with a percentile cluster bootstrap: each resample
    draws clusters (``clusters`` maps an item id to its statement; an item missing from it is its own),
    and takes both sides' values for every item of each. Items of one statement move together, so the
    statement, not the item, is what is drawn."""
    shared = sorted(set(a) & set(b))
    groups: dict[str, list[float]] = {}
    for key in shared:
        groups.setdefault((clusters or {}).get(key, key), []).append(a[key] - b[key])
    if not groups:
        return Paired(0, 0, None, None, None)
    sums = [math.fsum(values) for values in groups.values()]
    sizes = [len(values) for values in groups.values()]
    k, rng = len(sums), random.Random(seed)
    means = []
    for _ in range(resamples):
        drawn = [rng.randrange(k) for _ in range(k)]
        means.append(math.fsum(sums[i] for i in drawn) / sum(sizes[i] for i in drawn))
    means.sort()
    return Paired(len(shared), k, math.fsum(sums) / len(shared), means[int(0.025 * (resamples - 1))],
                  means[math.ceil(0.975 * (resamples - 1))])


def arm_line(items: Sequence[KnowledgeItem], base: Scores, arm: Scores, seen_ids: frozenset[str]) -> dict[str, Any]:
    right = {i.id: _right(i, arm[i.id]) for i in items}
    base_right = {i.id: _right(i, base[i.id]) for i in items}
    gain = {i.id: _margin(i, arm[i.id]) - _margin(i, base[i.id]) for i in items}
    looked = [i.id for i in items if i.id in seen_ids]
    return {"items": len(items), "accuracy": _mean([float(v) for v in right.values()]),
            "seen_items": len(looked), "seen_accuracy": _mean([float(right[k]) for k in looked]),
            "margin_vs_base": _mean(list(gain.values())), "seen_margin_vs_base": _mean([gain[k] for k in looked]),
            "newly_right": sorted(k for k in right if right[k] and not base_right[k]),
            "newly_wrong": sorted(k for k in right if base_right[k] and not right[k]),
            "unseen": sorted(i.id for i in items if i.id not in seen_ids)}


def repeat_line(items: Sequence[KnowledgeItem], first: Scores, again: Scores) -> dict[str, Any]:
    """One scored side against its repeat: the largest change in a choice's summed log-probability, the
    accuracy change, the items whose pick changed, and whether it reproduced."""
    worst = max((abs(a - b) for i in items for a, b in zip(first[i.id], again[i.id], strict=True)), default=0.0)
    flipped = sorted(i.id for i in items if _right(i, first[i.id]) != _right(i, again[i.id]))
    accuracy = [_mean([float(_right(i, side[i.id])) for i in items]) or 0.0 for side in (first, again)]
    return {"items": len(items), "max_logprob_diff": round(worst, 6), "accuracy_diff": round(accuracy[1] - accuracy[0], 4),
            "flipped": flipped, "reproducible": worst <= REPRODUCIBLE_NATS and not flipped}


def pair_line(items: Sequence[KnowledgeItem], a: Scores, b: Scores, seen_both: frozenset[str]) -> dict[str, Any]:
    right = [{i.id: float(_right(i, side[i.id])) for i in items} for side in (a, b)]
    margin = [{i.id: _margin(i, side[i.id]) for i in items} for side in (a, b)]
    sources = {i.id: i.source for i in items}

    def only(values: Mapping[str, float]) -> dict[str, float]:
        return {k: v for k, v in values.items() if k in seen_both}

    seen_right = paired(only(right[0]), only(right[1]), clusters=sources)
    every_right = paired(right[0], right[1], clusters=sources)
    return {"items": every_right.n, "clusters": every_right.clusters, "seen_items": seen_right.n,
            "seen_clusters": seen_right.clusters, **seen_right.record("seen_accuracy"),
            **paired(only(margin[0]), only(margin[1]), clusters=sources).record("seen_margin"),
            **every_right.record("accuracy"), **paired(margin[0], margin[1], clusters=sources).record("margin")}


# ── the run ──────────────────────────────────────────────────────────────────────────────────────────


def _seen_by_night(items: Sequence[KnowledgeItem], account: str, life: Sequence[Night],
                   day_text: DayText) -> dict[int, frozenset[str]]:
    lines: list[str] = []
    found = {}
    for night in life:
        if night.window is not None:
            lines += day_text(account, night.window)
        found[night.number] = seen(items, lines)
    return found


def _report(the_set: KnowledgeSet, lives: Mapping[str, Sequence[Night]], scored: Mapping[str | None, Scores],
            seen_by: Mapping[str, Mapping[int, frozenset[str]]], out: Callable[[str], None],
            ceilings: Mapping[str, str]) -> None:
    items, base = the_set.items, scored[None]
    out(json.dumps({"line": "base", "items": len(items), "accuracy": _mean([float(_right(i, base[i.id])) for i in items]),
                    "chance": chance(items), "mean_margin": _mean([_margin(i, base[i.id]) for i in items])}))
    every = frozenset(i.id for i in items)
    for label, name in ceilings.items():
        if name in scored:
            out(json.dumps({"line": "ceiling", "CEILING": CEILING, "label": label, "adapter": name,
                            **arm_line(items, base, scored[name], every),
                            "vs_base": pair_line(items, scored[name], base, every)}))
    for arm, life in lives.items():
        for night in life:
            head = {"line": "arm_night", "arm": arm, "night": night.number, "date": night.date,
                    "trained": night.adapter is not None, "promoted": night.promoted}
            done = night.adapter in scored
            out(json.dumps({**head, **(arm_line(items, base, scored[night.adapter], seen_by[arm][night.number])
                                       if done else {"scored": False})}))
    for a, b in PAIRS:
        if a not in lives or b not in lives:
            continue
        theirs = {n.date: n for n in lives[b] if n.adapter in scored}  # the arms share virtual dates
        for night in lives[a]:
            other = theirs.get(night.date)
            if night.adapter not in scored or other is None or other.adapter is None:
                continue
            both = seen_by[a][night.number] & seen_by[b][other.number]
            out(json.dumps({"line": "pair", "a": a, "b": b, "date": night.date, "night": night.number,
                            "b_night": other.number, **pair_line(items, scored[night.adapter], scored[other.adapter], both)}))


def _accounts(parser: argparse.ArgumentParser, pairs: Sequence[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for pair in pairs:
        arm, _, account = pair.partition("=")
        if not _ARM.fullmatch(arm) or not _ACCOUNT.fullmatch(account) or arm in found:
            parser.error("--account is <arm>=<account id> (letters, digits, - and _), once per arm")
        found[arm] = account
    if len(set(found.values())) != len(found):
        parser.error("each arm lives on its own account")
    return found


def _ceilings(parser: argparse.ArgumentParser, pairs: Sequence[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for pair in pairs:
        label, _, adapter = pair.partition("=")
        if not _ARM.fullmatch(label) or not config.ADAPTER_NAME.fullmatch(adapter) or label in found:
            parser.error("--ceiling is <label>=<adapter> (p-<hash>-n<night>, as knowledge_ceiling prints it), once per label")
        found[label] = adapter
    return found


def _app_day_text() -> DayText:
    module, _, name = DAYS.partition(":")
    reader: DayText = getattr(importlib.import_module(module), name)  # by name: the engine never imports the app
    return reader


def _setting(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


def plan(the_set: KnowledgeSet, lives: Mapping[str, Sequence[Night]], gpu: gpus.Gpu, rate: float,
         ceilings: Mapping[str, str] | None = None, repeat: bool = False) -> str:
    ceilings = ceilings or {}
    adapters = 1 + sum(n.adapter is not None for life in lives.values() for n in life) + len(ceilings)
    repeats = (1 + (adapters > 1)) if repeat else 0
    adapters += repeats
    choices = sum(len(item.choices) for item in the_set.items)
    seconds = adapters * choices * CALL_S + adapters * LOAD_S
    expected = rate * (seconds + config.SCALEDOWN_S) / 3600.0
    held = rate * (SCORE_MAX_S + WAKE_S + config.SCALEDOWN_S) / 3600.0
    arms = "; ".join(f"{arm}: {len(life)} nights, {sum(n.adapter is not None for n in life)} trained, "
                     f"{sum(n.promoted for n in life)} promoted" for arm, life in lives.items())
    return "\n".join([
        "plan (nothing is sent; add --go to run it):",
        f"  set       {the_set.persona} v{the_set.version}, {len(the_set.items)} items from "
        f"{len({item.source for item in the_set.items})} statements, {choices} choices, sha {the_set.sha[:12]}",
        f"  base      scored first: its accuracy is the floor every arm starts from (chance {100 * (chance(the_set.items) or 0):.0f}%); "
        "the report's base line prints both",
        f"  arms      {arms or 'none'}",
        *([f"  ceilings  {', '.join(f'{label}={name}' for label, name in ceilings.items())}: CEILING, {CEILING}"]
          if ceilings else []),
        f"  calls     {adapters} adapters (the base, every trained night and each ceiling"
        f"{', and the repeats' if repeats else ''}) x {choices} choices = {adapters * choices} log-probability calls, "
        "prefill only, one at a time, under the night lock (refused while a night, its gates or a morning runs)",
        *([f"  repeat    the base{' and the first adapter' if repeats > 1 else ''} scored again at the end: the "
           "receipt (largest log-probability change, accuracy change, items flipped)"] if repeats else []),
        f"  expected  ~{seconds / 60:.1f} GPU min scoring, an estimate ({CALL_S} s a call, {LOAD_S:.0f} s an adapter "
        f"load), plus the {config.SCALEDOWN_S / 60:.0f}-min idle window: ~${expected:.2f} at ${rate:.2f}/h on "
        f"{gpu.name} with the app warm; a cold start adds its wake",
        f"  hold      ${held:.2f} on {POOL} ({SCORE_MAX_S / 60:.0f} min cap, the wake and the idle window), "
        "settled at what ran plus the idle window",
        f"  days      read first with --go through {DAYS}, for the seen flags (a synthetic person's only)",
        "  prints    per arm and night, and per pair: accuracy, margins, intervals and item ids; never a "
        "statement's or a day's words",
        f"  needs     {backend.MODAL_URL_ENV} (or --url) and {backend.MODAL_TOKEN_ENV} (never printed)",
    ])


def main(
    argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None, client: Any = None,
    wake: Callable[[], WakeResult] | None = None, ledger: Any = ..., clock: Callable[[], float] = time.monotonic,
    root: Path | None = None, day_text: DayText | None = None, out: Callable[[str], None] = print,
) -> int:
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description="Score each arm's nightly adapters on a frozen knowledge set.")
    parser.add_argument("--account", action="append", default=[], help="<arm>=<account id>, once per arm")
    parser.add_argument("--ceiling", action="append", default=[],
                        help="<label>=<adapter>: an oracle ceiling (tools.knowledge_ceiling), never a result")
    parser.add_argument("--persona", default="film_founder", help="the persona the arms lived (its frozen set)")
    parser.add_argument("--gpu", choices=tuple(gpus.GPUS), default=config.serve_precision(env).gpu)
    parser.add_argument("--usd-per-hour", type=float, help="the GPU's price an hour, if not the listed one")
    parser.add_argument("--url", help=f"the app's URL (default {backend.MODAL_URL_ENV})")
    parser.add_argument("--repeat", action="store_true",
                        help="score the base and the first adapter again at the end: the reproducibility receipt")
    parser.add_argument("--go", action="store_true", help="send the requests; without it, print the plan only")
    args = parser.parse_args(argv)
    accounts, ceilings = _accounts(parser, args.account), _ceilings(parser, args.ceiling)
    if not accounts and not ceilings:
        parser.error("name at least one --account or --ceiling")
    try:
        the_set = knowledge.load(args.persona)
    except SetError as exc:
        parser.error(str(exc))
    base_root = results_root(root)
    lives = {arm: nights(base_root, account) for arm, account in accounts.items()}
    if empty := [arm for arm, life in lives.items() if not life]:
        parser.error(f"no finished night under the results root for {', '.join(empty)}")
    gpu = gpus.GPUS[args.gpu]
    rate = args.usd_per_hour or gpu.usd_per_hour
    out(plan(the_set, lives, gpu, rate, ceilings, args.repeat))
    if not args.go:
        return 0
    reader = day_text or _app_day_text()
    seen_by = {arm: _seen_by_night(the_set.items, accounts[arm], life, reader) for arm, life in lives.items()}
    if client is None:
        url, token = args.url or _setting(env, backend.MODAL_URL_ENV), _setting(env, backend.MODAL_TOKEN_ENV)
        serving = backend.ModalServing(url, token)
        client, wake = serving.client(), wake or serving.wake
    assert wake is not None
    book = default_ledger(env) if ledger is ... else ledger
    with ExitStack() as locked:  # the night lock for the run: no night, gate or morning on this host shares the batch
        try:
            locked.enter_context(nullcontext() if book is None else night_lock(book))
        except NightBusy:
            out(json.dumps({"line": "refused", "why": LOCKED}))
            return 2
        reservation = None if book is None else hold(book, f"knowledge_score:{the_set.sha[:12]}:{int(time.time())}",
                                                     (POOL,), Bound(rate, SCORE_MAX_S + WAKE_S + config.SCALEDOWN_S))
        started = clock()
        scored: dict[str | None, Scores] = {}
        repeated: dict[str | None, Scores] = {}
        try:
            woke = wake()
            out(json.dumps({"line": "wake", "ok": woke.ok, "seconds": round(woke.seconds, 1)}))
            if not woke.ok:
                return 1
            dated, deadline = client.on(EVAL_DAY), clock() + SCORE_MAX_S
            wanted = list(dict.fromkeys([None, *(n.adapter for life in lives.values() for n in life if n.adapter is not None),
                                         *ceilings.values()]))
            again = [None, *wanted[1:2]] if args.repeat else []  # the base and the first adapter, at the end
            for n, name in enumerate([*wanted, *again]):
                if clock() >= deadline:
                    out(json.dumps({"line": "stopped", "why": "the scorer's cap", "unscored": len(wanted) + len(again) - n}))
                    break
                ref = None if name is None else CheckpointRef(name, "knowledge", EVAL_DAY, f"{config.ADAPTERS_DIR}/{name}",
                                                              None, rendering=client.rendering)
                (scored if n < len(wanted) else repeated)[name] = score(dated, ref, the_set.items)
        finally:
            if reservation is not None:  # what ran, and the idle window Modal bills after the last request
                reservation.settle(rate * (max(0.0, clock() - started) + config.SCALEDOWN_S) / 3600.0)
    out(json.dumps({"line": "set", "persona": the_set.persona, "version": the_set.version, "sha": the_set.sha,
                    "items": len(the_set.items)}))
    if None in scored:
        _report(the_set, lives, scored, seen_by, out, ceilings)
    for name, scores in repeated.items():
        out(json.dumps({"line": "repeat", "scored": "base" if name is None else name,
                        **repeat_line(the_set.items, scored[name], scores)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
