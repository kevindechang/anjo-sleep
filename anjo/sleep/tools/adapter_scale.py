"""Arm 4 with its adapter scaled down, against arm 2C on the same knowledge probes (adapter-scale).

    python -m anjo.sleep.tools.adapter_scale --user <account> --persona <name> --night 2026-10-24 \
        --scales 0.25,0.5,0.75
    python -m anjo.sleep.tools.adapter_scale --user <account> --persona <name> --night 2026-10-24 \
        --scales 0.25,0.5,0.75 --go

The development read found arm 4 below arm 2C on knowledge while arm 3 beat arm 1. The cheapest
test of "at full strength the adapter overrides what C says" is the same pair with the adapter's
LoRA delta multiplied by ``s``. Without ``--go`` this only plans: the adapter copies, the replies
it will generate, the judge calls, the GPU minutes and the holds, and the file it will write.
Nothing is called, held, uploaded or written.

**Both replies are generated in the run.** No morning stores reply text (its score file holds ids,
codes and numbers), so the default reads nothing back: for each knowledge probe it generates arm
2C (base weights, context C, no adapter) and arm 4 at every scale. The 2C reply takes the morning's
own per-probe, per-arm seed (``arms.reply_seed``), so it is the reply the morning itself would
have drawn; arm 4 uses the same seed it always does, and the only change from it is the scale. The
scale set always includes ``s=1`` as a control: it is the adapter as trained, and should reproduce
arm 4. ``--replies`` names a stored 2C file (JSON lines: an optional header with ``persona`` and
``checkpoint``, then one ``{"probe", "arm": "2C", "text"}`` a probe): then those texts are reused
and only arm 4 is generated. Without ``--persona`` the run reads the host's account-to-persona map
(``ANJO_SLEEP_DAILY_USERS``). Rows outside knowledge (``new_work``, ``vocab``) are left out, since a
``taste_ab`` answer is a log-probability choice, not stored text.

A run generates ``probes x (1 + scales)`` replies (the 2C once a probe, arm 4 at each scale), or
``probes x scales`` when the stored 2C replies are reused, and judges ``2 x probes x scales`` pairs
(both orders, DeepSeek when ``ANJO_SLEEP_JUDGE=deepseek``), against the full morning's six arms.

**What it writes.** One file beside that night's scores, ``scores/adapter-scale-<night>.json``,
always labelled ``development`` and never the final: per scale the adapter's name, the wins, ties
and losses, the tie-inclusive score, the decisive share and the block-bootstrap 95% interval. It
holds ids, codes and numbers, never a reply or a probe's words.

**Adapters.** Each copy is written under a new ``p-<hash>-n<night>`` in the local adapter cache and
uploaded once (``serving.scale`` says how the B matrices are scaled and why that is exact). The name
carries the source adapter's content hash and the scale, so a copy already on the serving side is the
same bytes and is reused rather than uploaded again. A copy is loaded for this run only: it is never
promoted, never added to the person's checkpoint history, and never changes what serves. The night's
checkpoint is pinned for the run, so a later promotion changes nothing; the original folder is never
modified. A copy already written locally is reused, so a rerun pays only for what it generates.

**Holds and spend** are the morning's: Glimmer on ``modal_usd`` and ``baseten_sleep_usd`` at
``ARM_CALL_S`` a reply, and the judge on its provider pool plus ``judge_usd`` (DeepSeek holds each
call's full context separately), settled at what it metered. The holds are reserved before any
upload, so a failed upload still settles them and a rerun can reserve again. This tool's own holds
are keyed ``morning:adapter-scale:...``: the night sweep already settles that prefix, and a run
settles its own leftovers at their full bound before it reserves. Refused before any call, hold,
upload or write: an account not registered synthetic unless ``--real``, a missing or unreadable
reply file named by ``--replies``, a missing persona when there is no store to name one, a
checkpoint outside the person's promoted history, a night already read, a held night lock, and a
Glimmer that does not wake. The run holds the night lock throughout.

The engine never imports the app: the sleep host's ports are named here and loaded when the command
runs (``app_ports``), as ``tools.morning_backfill`` names its builders.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Final

from anjo.sleep import fireworks_api
from anjo.sleep.budget import Bound, Ledger, default_ledger
from anjo.sleep.checkpoints import CheckpointRegistry
from anjo.sleep.night.cost import Units
from anjo.sleep.night.holds import held, usd
from anjo.sleep.night.sweep import NightBusy, held_elsewhere, night_lock
from anjo.sleep.parallel import Calls
from anjo.sleep.probes.arms import ArmReply, Question, ask, prepare
from anjo.sleep.probes.morning import (
    ARM_CALL_S,
    JUDGE_RETRY_SHARE,
    NO_VERDICT_MAX_SHARE,
    SCORES,
    Morning,
    MorningError,
    TooManyNoVerdicts,
    _Counting,
    _HeldDeepseekChat,
    _judge_spend,
    _judge_usage,
    _Pinned,
)
from anjo.sleep.probes.score import (
    BootstrapConfig,
    MuseJudge,
    NoVerdict,
    PairScore,
    probe_outcome,
    score_outcomes,
)
from anjo.sleep.reports.results import results_root
from anjo.sleep.serving import config, scale
from anjo.sleep.serving.backend import phase
from anjo.sleep.serving.client import GlimmerNotReady
from anjo.sleep.serving.scale import ScaleError
from anjo.sleep.types import CheckpointRef, Probe

MORNING: Final = "anjo.sleep.app.morning:build"
SYNTHETIC: Final = "anjo.sleep.app.cleared_sessions:is_synthetic"
REGISTRY: Final = "anjo.sleep.app.storage:checkpoint_registry"
DATA_ROOT: Final = "anjo.sleep.app.storage:data_root"
TRAINING_DIR: Final = "anjo.sleep.app.ownership:TRAINING_DIR"
PERSONAS: Final = "anjo.sleep.night.daily:daily_users"  # persona -> account, ANJO_SLEEP_DAILY_USERS
BASELINE: Final = "2C"
PERSONAL: Final = "4"
LABEL: Final = "development"  # this read is never the final one
KNOWLEDGE: Final = frozenset({"new_work", "vocab"})
ARTIFACT: Final = "adapter-scale-"
DEFAULT_SCALES: Final = "0.25,0.5,0.75"
CONTROL: Final = 1.0  # the adapter as trained: its copy is a byte-for-byte control for arm 4
MAX_SCALES: Final = 8
# The night sweep settles every ``morning:`` hold before a night starts (``night.sweep.NIGHT_KINDS``),
# so a hard-killed read's holds are closed there; this tool also settles its own before it reserves.
KEY: Final = "morning:adapter-scale:"
TAKEN: Final = ("already on the volume", "already in the bucket")  # an adapter already on the serving side


class Refused(RuntimeError):
    """A read that may not run; its message names the rule."""


class ReplyError(ValueError):
    """The stored 2C replies can't be read."""


@dataclass(frozen=True)
class StoredReplies:
    """The night's stored 2C replies: which persona and checkpoint they belong to, and one text a
    probe id. ``outside`` counts stored rows this read leaves out (another kind, or another arm)."""

    persona: str | None
    checkpoint: str | None
    texts: Mapping[str, str]
    outside: int


@dataclass(frozen=True)
class Ports:
    """What a read reaches outside through (``app_ports``: the sleep host's)."""

    morning: Callable[[str, str, date], Morning]
    is_synthetic: Callable[[str], bool]
    registry: CheckpointRegistry
    ledger: Ledger
    adapters: Path  # the local adapter cache: ``<adapters>/<name>`` holds one adapter
    publish: Callable[[Path, str], None]  # copies one checked adapter to the serving side
    persona: Callable[[str], str | None] | None = None  # the account's persona, when the host knows it


@dataclass(frozen=True)
class Plan:
    user: str
    night: date
    persona: str
    checkpoint: CheckpointRef
    serving: CheckpointRef | None
    replies_path: Path | None  # the stored 2C file named by --replies; None: generate them
    reply_count: int  # the stored 2C replies reused (0 when they are generated)
    left_out: int  # stored rows this read leaves out: another arm, or a kind outside knowledge
    probes: tuple[Probe, ...]
    scales: tuple[float, ...]
    names: Mapping[float, str]  # scale -> the copy's adapter name
    morning: Morning
    stored: StoredReplies | None  # the reused 2C texts, None when this run generates them
    replies: int  # generated: 2C once a probe, then arm 4 at every scale
    judge_calls: int  # both orders, before retries
    judge_hold_usd: float
    arms_hold_usd: float
    gpu_s: float  # expected: the wake, the replies, and Modal's idle window
    target: Path


def check_scales(raw: str) -> tuple[float, ...]:
    """``--scales``'s comma-separated numbers: distinct, in [0, 1], at most ``MAX_SCALES`` including
    the ``1`` control, which is always asked so full strength is in the same read."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("--scales names at least one scale from 0 to 1")
    found: list[float] = []
    for part in parts:
        try:
            value = scale.check_scale(float(part))
        except (TypeError, ValueError) as error:
            raise ValueError(f"--scales is a comma-separated list from 0 to 1, not {part!r}") from error
        if value in found:
            raise ValueError(f"--scales names {value:g} twice")
        found.append(value)
    if CONTROL not in found:
        found.append(CONTROL)  # the control: arm 4 as trained, so the read says a scaled copy is exact
    if len(found) > MAX_SCALES:
        raise ValueError(f"--scales names at most {MAX_SCALES} scales, the {CONTROL:g} control included")
    return tuple(found)


def load_replies(path: Path) -> StoredReplies:
    """The JSON lines at ``path``: an optional header, then the 2C text a probe.

    A row is ``{"probe": <id>, "text": <reply>}``, with ``"arm"`` ``"2C"`` (or absent), or a stored
    pair row carrying ``"replies"`` (as the second judge writes them). A header names the ``persona``
    and ``checkpoint`` it was written against. Rows of another arm are counted, never used, and the
    probes are narrowed to knowledge later: this read is arm 4 against 2C on knowledge alone.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ReplyError(f"the stored 2C replies can't be read ({type(error).__name__})") from None
    persona: str | None = None
    checkpoint: str | None = None
    texts: dict[str, str] = {}
    outside = 0
    for number, line in enumerate((line for line in lines if line.strip()), start=1):
        try:
            row = json.loads(line)
        except ValueError:
            raise ReplyError(f"the stored 2C replies: line {number} is not JSON") from None
        if not isinstance(row, dict):
            raise ReplyError(f"the stored 2C replies: line {number} is not an object")
        if persona is None and isinstance(row.get("persona"), str):
            persona = str(row["persona"])
        if checkpoint is None and isinstance(row.get("checkpoint"), str):
            checkpoint = str(row["checkpoint"])
        if "probe" not in row:
            continue
        paired = row.get("replies")
        if "text" not in row and isinstance(paired, dict) and isinstance(paired.get(BASELINE), str):
            text = paired[BASELINE]  # a stored pair row, as the second judge writes them
        else:
            if row.get("arm", BASELINE) != BASELINE:
                outside += 1
                continue
            text = row.get("text")
        probe = row.get("probe")
        if not isinstance(probe, str) or not probe:
            raise ReplyError(f"the stored 2C replies: line {number} names no probe")
        if not isinstance(text, str) or not text.strip():
            raise ReplyError(f"the stored 2C replies: line {number} ({probe}) has no text")
        if probe in texts:
            raise ReplyError(f"the stored 2C replies name {probe} twice")
        texts[probe] = text
    if not texts:
        raise ReplyError("the stored 2C replies hold no 2C reply")
    return StoredReplies(persona, checkpoint, texts, outside)


def _target(folder: Path, night: date) -> Path:
    return folder / f"{ARTIFACT}{night.isoformat()}.json"


def _checkpoint(user: str, stored: StoredReplies, registry: CheckpointRegistry) -> CheckpointRef:
    """The adapter the stored 2C replies were compared with, pinned; else the one serving now."""
    if stored.checkpoint is not None:
        found = next((ckpt for ckpt in registry.history(user) if ckpt.id == stored.checkpoint), None)
        if found is None:
            raise Refused(f"{stored.checkpoint} isn't in the person's promoted history")
        return found
    serving = registry.last_promoted(user)
    if serving is None:
        raise Refused("the person has no promoted adapter to scale")
    return serving


def _probes(user: str, morning: Morning, stored: StoredReplies | None) -> tuple[Probe, ...]:
    """The knowledge probes to ask, in sealed order. With a store, the ones it has a 2C reply for."""
    sealed = list(morning.sealed(user))
    if stored is None:
        chosen = tuple(probe for probe in sealed if probe.kind in KNOWLEDGE)
        if not chosen:
            raise Refused(f"{user} has no sealed knowledge probe")
        return chosen
    by_id = {probe.id: probe for probe in sealed}
    unknown = sorted(set(stored.texts) - set(by_id))
    if unknown:
        raise Refused(f"the stored 2C replies name probes that aren't sealed for {user}: {', '.join(unknown[:3])}")
    chosen = tuple(probe for probe in sealed if probe.id in stored.texts and probe.kind in KNOWLEDGE)
    if not chosen:
        raise Refused("the stored 2C replies leave no knowledge probe to score")
    return chosen


def _persona_and_checkpoint(
    user: str, persona: str | None, stored: StoredReplies | None, registry: CheckpointRegistry,
    find: Callable[[str], str | None] | None = None,
) -> tuple[str, CheckpointRef]:
    """The persona and the adapter to scale: the store's own, else ``--persona``, else the host's
    account-to-persona map (``ANJO_SLEEP_DAILY_USERS``); and the night's promoted adapter."""
    if stored is not None:
        who = persona or stored.persona
        if not who:
            raise Refused("the stored replies name no persona; pass --persona")
        return who, _checkpoint(user, stored, registry)
    who = persona or (find(user) if find is not None else None)
    if not who:
        raise Refused("no stored replies name the persona; pass --persona or set ANJO_SLEEP_DAILY_USERS")
    serving = registry.last_promoted(user)
    if serving is None:
        raise Refused("the person has no promoted adapter to scale")
    return who, serving


def plan(
    user: str, night: date, ports: Ports, *, scales: Sequence[float], persona: str | None = None,
    replies: Path | None = None, real: bool = False,
) -> Plan:
    """The read of ``user``'s ``night``, or ``Refused``: no model call, no hold, nothing written."""
    scales = tuple(dict.fromkeys([*scales, CONTROL]))  # the control is always in the same read
    if held_elsewhere(ports.ledger):
        raise Refused("the night lock is held: a night or a simulated day is running")
    if not real and not ports.is_synthetic(user):
        raise Refused("the account isn't registered synthetic (--real to read a real person's night)")
    path: Path | None = None
    stored: StoredReplies | None = None
    if replies is not None:
        path = replies
        try:
            stored = load_replies(path)
        except ReplyError as error:
            raise Refused(str(error)) from None
    who, checkpoint = _persona_and_checkpoint(user, persona, stored, ports.registry, ports.persona)
    try:
        morning = ports.morning(user, who, night)
    except ValueError as error:  # the builder's own refusal
        raise Refused(str(error)) from None
    probes = _probes(user, morning, stored)
    folder = ports.adapters / checkpoint.id
    if not folder.is_dir():
        raise Refused(f"the adapter {checkpoint.id} isn't in the local cache ({ports.adapters})")
    try:
        digest = scale.content_digest(folder)
    except ScaleError as error:
        raise Refused(str(error)) from None
    names: dict[float, str] = {}
    for value in scales:
        try:
            names[value] = scale.scaled_name(checkpoint.id, value, digest=digest)
        except ScaleError as error:
            raise Refused(str(error)) from None
    try:
        scores = results_root(morning.root) / user / SCORES
    except ValueError as error:  # no results root named in the environment
        raise Refused(str(error)) from None
    target = _target(scores, night)
    if target.exists():
        raise Refused(f"{night.isoformat()} already has an adapter-scale read ({target.name})")
    calls = 2 * len(probes) * len(scales)
    replies_needed = len(probes) * len(scales) + (0 if stored is not None else len(probes))
    serve = morning.prices.serve_gpu_per_hour
    units = Units()
    return Plan(
        user=user, night=night, persona=who, checkpoint=checkpoint,
        serving=ports.registry.last_promoted(user), replies_path=path,
        reply_count=0 if stored is None else len(stored.texts),
        left_out=0 if stored is None else stored.outside + len(stored.texts) - len(probes),
        probes=probes, scales=tuple(scales), names=names, morning=morning, stored=stored,
        replies=replies_needed, judge_calls=calls,
        judge_hold_usd=(calls + math.ceil(calls * JUDGE_RETRY_SHARE)) * morning.judge_call_usd,
        arms_hold_usd=serve * replies_needed * ARM_CALL_S / 3600.0,
        gpu_s=units.wake_s + ARM_CALL_S * replies_needed + config.SCALEDOWN_S, target=target,
    )


def describe(chosen: Plan) -> list[str]:
    serve = chosen.morning.prices.serve_gpu_per_hour
    judge = chosen.judge_calls * chosen.morning.judge_call_usd
    gpu = chosen.gpu_s * serve / 3600.0
    scales = ", ".join(f"{value:g}" for value in chosen.scales)
    serving = "the base model" if chosen.serving is None else chosen.serving.id
    if chosen.stored is None:
        replies = (
            f"  2C replies: generated in this run, arm 2C with the morning's own reply seed "
            f"({len(chosen.probes)} replies), beside arm 4 at every scale "
            f"({len(chosen.probes) * len(chosen.scales)} replies); {chosen.replies} in all")
    else:
        replies = (
            f"  2C replies: {chosen.reply_count} stored at {chosen.replies_path}, reused; only arm 4 is "
            f"generated ({len(chosen.probes) * len(chosen.scales)} replies over {len(chosen.probes)} knowledge "
            f"probes; {chosen.left_out} stored rows left out)")
    return [
        f"adapter scale: plan only, nothing run, nothing held (night {chosen.night.isoformat()}, "
        f"{chosen.user}, {len(chosen.probes)} knowledge probes)",
        f"  scales {chosen.checkpoint.id} by {scales}: the B matrices, one copy each "
        f"({', '.join(chosen.names[value] for value in chosen.scales)}); the original is untouched; "
        f"{CONTROL:g} is the control, the adapter as trained",
        f"  the night's checkpoint is pinned; {serving} keeps serving",
        replies,
        f"  judge: up to {chosen.judge_calls} calls, both orders, {chosen.morning.judge_model}",
        f"  holds: the judge ${chosen.judge_hold_usd:.2f} on {'/'.join(chosen.morning.judge_pools)}, "
        f"the arms ${chosen.arms_hold_usd:.2f} on modal_usd and baseten_sleep_usd, reserved before any upload",
        f"  expected: ${judge + gpu:.2f} (the judge ${judge:.2f} at ${chosen.morning.judge_call_usd:g} "
        f"a call, settled at its tokens; the serving GPU ${gpu:.2f}, {chosen.gpu_s / 60.0:.0f} min: "
        f"the wake, {chosen.replies} replies at {ARM_CALL_S:g} s and Modal's idle window, at ${serve:g}/h)",
        f"  writes: {chosen.target} (label {LABEL}, never the final)",
        "run again with --go to run it",
    ]


def _ref(chosen: Plan, name: str) -> CheckpointRef:
    return CheckpointRef(name, chosen.user, chosen.night, f"{config.ADAPTERS_DIR}/{name}",
                         chosen.checkpoint.parent_id, rendering=chosen.checkpoint.rendering)


def _questions(
    chosen: Plan, deps: Any, *, stored: bool,
) -> tuple[list[Question], list[tuple[float, Probe, Question]]]:
    """The prepared questions: 2C once a probe (unless its reply is reused), and arm 4 at every scale.

    C is built once a probe and shared by 2C and every scaled arm, so the only difference between the
    pair is the adapter, exactly as in the morning. ``ask`` adds the morning's own per-probe, per-arm
    reply seed (``arms.reply_seed``), so 2C is the reply the morning would have drawn.
    """
    prepared = [prepare(PERSONAL, chosen.user, probe, deps=deps) for probe in chosen.probes]
    baseline = [] if stored else [prepare(BASELINE, chosen.user, probe, deps=deps) for probe in chosen.probes]
    asked: list[tuple[float, Probe, Question]] = []
    for value in chosen.scales:
        adapter = _ref(chosen, chosen.names[value])
        asked += [(value, probe, replace(question, adapter=adapter))
                  for probe, question in zip(chosen.probes, prepared, strict=True)]
    return baseline, asked


def _settle_stale(ledger: Ledger, night: date, user: str) -> int:
    """Settle this tool's own holds a killed run left open, at their full bound, before reserving.

    The key begins with ``morning:``, which the night sweep settles too; a run does this itself so a
    read that is rerun without a night start doesn't leave its pools reserved and meet ``BudgetStop``.
    The night lock is held, so none of these is a live run's. Returns how many were settled.
    """
    prefix = f"{KEY}{night.isoformat()}:{user}"
    stale = [hold for hold in ledger.open_holds() if hold.job.startswith(prefix)]
    for hold in stale:
        hold.settle(hold.bound.amount)
    return len(stale)


def _upload(ports: Ports, folder: Path, name: str) -> None:
    """Put one checked copy on the serving side, or reuse it when the name is already there.

    The name carries the source adapter's content hash and the scale (``serving.scale``), so an
    existing name is the same bytes at the same ``s``. Any other refusal is raised for a clean exit.
    """
    try:
        ports.publish(folder, name)
    except config.ServeConfigError as error:
        if not any(text in str(error) for text in TAKEN):
            raise
        return  # the same copy is already there: a rerun after a failed upload pays no second upload


def _stat(done: PairScore) -> dict[str, object]:
    return {"n": done.wins + done.ties + done.losses, "wins": done.wins, "ties": done.ties,
            "losses": done.losses, "score": round(done.score, 4),
            "decisive_share": None if done.decisive_share is None else round(done.decisive_share, 4),
            "ci": [round(done.ci[0], 4), round(done.ci[1], 4)]}


@dataclass(frozen=True)
class Result:
    path: Path
    replies: int
    judge_calls: int
    judge_usd: float


def read(
    chosen: Plan, ports: Ports, *, timer: Callable[[], float] = time.monotonic,
) -> Result:
    """Write the scaled copies, generate 2C (when it isn't reused) and arm 4 at each scale, judge the
    pairs, and write one development record beside the night's scores."""
    try:
        with night_lock(ports.ledger), phase(chosen.morning.sleep):
            return _read(chosen, ports, timer)
    except NightBusy as busy:
        raise Refused(str(busy)) from None


def _read(chosen: Plan, ports: Ports, timer: Callable[[], float]) -> Result:
    if chosen.target.exists():  # another run may have planned before this one took the lock
        raise Refused(f"{chosen.night.isoformat()} already has an adapter-scale read ({chosen.target.name})")
    stored: StoredReplies | None = None
    if chosen.replies_path is not None:
        try:
            stored = load_replies(chosen.replies_path)
        except ReplyError as error:
            raise Refused(str(error)) from None
    if chosen.morning.wake is not None and not chosen.morning.wake().ok:
        raise GlimmerNotReady()  # nothing was called or uploaded
    folders: dict[float, Path] = {}
    for value in chosen.scales:
        folders[value] = scale.write_scaled(ports.adapters / chosen.checkpoint.id, ports.adapters, value)
    deps = replace(chosen.morning.arms, adapters=_Pinned(chosen.morning.arms.adapters, chosen.user,
                                                         chosen.checkpoint),
                   until=chosen.night, context_cache={})
    judged = len(chosen.probes) * len(chosen.scales)
    retries = math.ceil(2 * judged * JUDGE_RETRY_SHARE)
    key = f"{KEY}{chosen.night.isoformat()}:{chosen.user}"
    book = chosen.morning.ledger or ports.ledger
    if book is not None:
        _settle_stale(book, chosen.night, chosen.user)  # a killed run's holds, before this one reserves
    inner = chosen.morning.judge
    per_call = chosen.morning.judge_model == fireworks_api.DEEPSEEK_V4P1_FLASH and book is not None
    if per_call:
        if not isinstance(inner, MuseJudge):
            raise Refused("DeepSeek requires the configured MuseJudge")
        inner = replace(inner, chat=_HeldDeepseekChat(inner.chat, book, f"{key}:judge"))
        with held(book, f"{key}:judge:preflight", chosen.morning.judge_pools,
                  usd(chosen.morning.judge_call_usd), lambda: 0.0):
            pass
    judge = _Counting(inner, retries)
    before = _judge_usage(chosen.morning)  # the judge's counters before this read's calls

    def spend() -> tuple[float, int]:
        return _judge_spend(judge, chosen.morning, before)

    started = timer()
    outcomes: dict[float, list[Any]] = {value: [] for value in chosen.scales}
    with held(
        None if per_call else book, f"{key}:judge", chosen.morning.judge_pools,
        usd((2 * judged + retries) * chosen.morning.judge_call_usd), lambda: spend()[0],
    ), held(
        book, f"{key}:arms", ("modal_usd", "baseten_sleep_usd"),
        Bound(chosen.morning.prices.serve_gpu_per_hour, chosen.replies * ARM_CALL_S),
        lambda: chosen.morning.prices.serve_gpu_per_hour * max(0.0, timer() - started) / 3600.0,
    ):
        for value, folder in folders.items():  # every hold is reserved before the first upload
            _upload(ports, folder, chosen.names[value])
        started = timer()  # the GPU's billed time starts with the first arm call, not the uploads
        baseline, asked = _questions(chosen, deps, stored=stored is not None)
        questions = [question for _value, _probe, question in asked] + baseline
        answers = Calls(chosen.morning.concurrency).map(lambda question: ask(question, deps=deps), questions)
        arm_four, generated = answers[:len(asked)], answers[len(asked):]
        by_scale: dict[float, list[ArmReply]] = {value: [] for value in chosen.scales}
        for (value, _probe, _question), reply in zip(asked, arm_four, strict=True):
            by_scale[value].append(reply)  # map returns (scale, probe) order
        two_c = ({probe.id: ArmReply(stored.texts[probe.id], 0) for probe in chosen.probes}
                 if stored is not None else dict(zip((probe.id for probe in chosen.probes), generated,
                                                    strict=True)))
        for value in chosen.scales:
            for probe, reply in zip(chosen.probes, by_scale[value], strict=True):
                try:
                    outcomes[value].append(probe_outcome(probe, reply, two_c[probe.id], judge))
                except NoVerdict:
                    outcomes[value].append(None)
    record: dict[str, object] = {
        "night": chosen.night.isoformat(), "label": LABEL, "kind": "adapter_scale",
        "user": chosen.user, "persona": chosen.persona, "baseline": BASELINE, "arm": PERSONAL,
        "dimension": "knowledge", "checkpoint": chosen.checkpoint.id,
        "probe_set": "generated-2C" if stored is None else "stored-2C",
        "probes": len(chosen.probes), "replies_stored": chosen.reply_count,
        "judge_model": chosen.morning.judge_model, "judge_calls": judge.calls,
        "scales": {},
    }
    bootstrap = BootstrapConfig.from_thresholds()
    record["ci_level"] = bootstrap.ci
    record["replies_generated"] = chosen.replies  # 2C once a probe when generated, plus arm 4 at every scale
    record["replies_arm4"] = judged
    record["scaled"] = "lora_B"  # the delta is (B @ A) * alpha / r, so scaling B scales the delta
    scores: dict[str, object] = {}
    for value in chosen.scales:
        found = outcomes[value]
        missing = sum(1 for outcome in found if outcome is None)
        if missing > NO_VERDICT_MAX_SHARE * len(chosen.probes):
            raise TooManyNoVerdicts(
                f"the judge gave no readable verdict on {missing} of {len(chosen.probes)} probes at "
                f"scale {value:g}, more than {NO_VERDICT_MAX_SHARE:.0%}: nothing was scored",
                no_verdict=missing)
        kept = [(probe, outcome) for probe, outcome in zip(chosen.probes, found, strict=True)
                if outcome is not None]
        done = score_outcomes([probe for probe, _ in kept], [outcome.outcome for _, outcome in kept],
                              bootstrap=bootstrap)
        stats = _stat(done)
        stats["adapter"] = chosen.names[value]
        stats["scale"] = value
        stats["kinds"] = {kind: _stat(score_outcomes(
            [probe for probe, _ in kept if probe.kind == kind],
            [outcome.outcome for probe, outcome in kept if probe.kind == kind], bootstrap=bootstrap))
            for kind in sorted({probe.kind for probe, _ in kept})}
        scores[f"{value:g}"] = stats
    record["scales"] = scores
    record["seconds"] = round(timer() - started, 3)
    folder = chosen.target.parent
    folder.mkdir(parents=True, exist_ok=True)
    pending = chosen.target.with_name(f".{chosen.target.name}.tmp")
    pending.write_text(json.dumps(record, sort_keys=True, indent=1), encoding="utf-8")
    os.replace(pending, chosen.target)
    j_usd, _estimated = spend()
    return Result(chosen.target, chosen.replies, judge.calls, j_usd)


def _named(path: str) -> Any:
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def _publisher() -> Callable[[Path, str], None]:
    def publish(folder: Path, name: str) -> None:
        from anjo.sleep.serving import backend  # imported at --go, as the command names it
        backend.from_env().upload_adapter(folder, name)
    return publish


def app_ports(morning: str = MORNING) -> Ports:
    """The sleep host's ports, imported when the command runs."""
    data = _named(DATA_ROOT)()

    def find(user: str) -> str | None:
        pairs = _named(PERSONAS)()  # persona -> account, from ANJO_SLEEP_DAILY_USERS
        return next((persona for persona, account in pairs.items() if account == user), None)

    return Ports(morning=_named(morning), is_synthetic=_named(SYNTHETIC), registry=_named(REGISTRY)(),
                 ledger=default_ledger(), adapters=data / "sleep" / _named(TRAINING_DIR) / "adapters",
                 publish=_publisher(), persona=find)


def main(argv: Sequence[str] | None = None, *, ports: Ports | None = None,
         out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description="Arm 4 with its adapter scaled down, against arm 2C.")
    parser.add_argument("--user", required=True, help="the account")
    parser.add_argument("--night", required=True, type=date.fromisoformat, help="the night, YYYY-MM-DD")
    parser.add_argument("--scales", default=DEFAULT_SCALES,
                        help=f"comma-separated LoRA scales from 0 to 1 (default {DEFAULT_SCALES}; "
                             f"{CONTROL:g}, the control, is always added)")
    parser.add_argument("--persona", default=None,
                        help="the persona (default: the reply store's, else the host's account map)")
    parser.add_argument("--replies", type=Path, default=None,
                        help="reuse a stored 2C reply file (default: generate arm 2C in this run)")
    parser.add_argument("--go", action="store_true", help="run it (without: the plan only)")
    parser.add_argument("--real", action="store_true", help="allow an account that isn't registered synthetic")
    parser.add_argument("--morning", default=MORNING, help="the morning builder, <module>:<function>")
    args = parser.parse_args(argv)
    try:
        scales = check_scales(args.scales)
    except ValueError as error:
        parser.error(str(error))
    found = ports or app_ports(args.morning)
    try:
        chosen = plan(args.user, args.night, found, scales=scales, persona=args.persona,
                      replies=args.replies, real=args.real)
        for line in describe(chosen):
            out(line)
        if not args.go:
            return 0
        done = read(chosen, found)
    except Refused as refused:
        out(f"refused: {refused}")
        return 1
    except GlimmerNotReady:  # the holds' context managers settled them on the way out
        out("refused: the serving app did not wake: nothing was generated")
        return 1
    except config.ServeConfigError as error:
        out(f"refused: {error}")
        return 1
    except (MorningError, ScaleError) as error:
        out(f"refused: {error}")
        return 1
    except TooManyNoVerdicts as error:
        out(f"failed: {error}")
        return 1
    out(f"read: {done.path} ({done.replies} replies, {done.judge_calls} judge calls, judge ${done.judge_usd:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
