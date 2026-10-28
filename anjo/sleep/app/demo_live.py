"""The live page's wiring on the sleep host (Task 13a): Anjo's crisis checks, the arms, Glimmer's
status and the demo's files, for ``anjo.sleep.demo.service``.

- **Before any model:** every visitor message passes the server floor chat runs before it replies
  (``anjo.core.floor.check_message``, with chat's own harm-routing switch). A severe or harm message
  gets the fixed protocol, an ideation message its fixed resource line, and no model is called for
  any of them: a blind comparison is no place to answer someone at risk with two generated replies.
  Each referral is counted, with no words, in the crisis ledger under the ``sleep_demo`` surface.
- **Before a reply is shown:** ``reply_crisis_reason``. A side that trips is withheld and the severe
  protocol stands in its place, as chat does, counted under ``reply_check``.
- **The night slider** (the full step): ``GlimmerArms.answer_at`` answers notes off at one stop,
  arm 3 on that stop's checkpoint (``slider.Pinned``) or arm 1 at night 0, through the same
  ``arms.reply``; the stops, the panel and the demo's answer files are read in ``demo_mind``.
- **The arms:** ``arms.reply`` for arm 4 and the baseline, over the ports the morning run uses
  (``morning.build``), for the account that plays the film student (``ANJO_SLEEP_DEMO_USER_ID``).
  That account must be registered synthetic, or the page answers nothing live: no real person's
  context ever reaches a stranger. Context C is built once and both sides read it. ``reply`` only
  reads, so nothing is written to history, evidence or memory, and the question is never logged.
- **Glimmer is never woken by default.** ``glimmer_status`` asks without waking: on SageMaker a ping
  finds no copy running and says so. On Modal any request starts a container, so the page never
  pings it: Glimmer is up inside a keep-warm window (``serving.wake``'s window file, whose pings
  still answer) or the scale-down window after the page's own last answer or wake. Waking on demand
  needs ``ANJO_SLEEP_DEMO_WAKE=1`` (off by default) and is capped per day (``Limits.wakes_per_day``).
- **Warm time is bought, then capped.** Each live answer and wake first buys the GPU time it keeps
  Glimmer up (``Upkeep``: the call, then the backend's idle window), except what a keep-warm window
  already covers: debited at the serving GPU's list price to ``modal_usd`` on the host's budget
  ledger, which holds serving on either backend, and capped per day (``Limits.warm_minutes_per_day``).
  Without a list price none is bought. ``ANJO_SLEEP_DEMO_LIVE_HOURS`` (``HH-HH``, UTC) limits buying
  to those hours. Past either, the page runs on its starter pairs and says so; a keep-warm window
  Kevin runs still answers, its cost his window's.

The baseline is ``ANJO_SLEEP_DEMO_BASELINE`` (``2C`` or ``2F``, default ``2C``; Kevin picks it on
Oct 31). Cost per turn is a side's measured seconds at the serving GPU's list price
(``ANJO_SLEEP_PRICES``); without it the reveal says it wasn't measured. The demo's files are
``<data dir>/sleep/demo/``: the day's counters, the votes and the starter pairs.
"""
from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Final, cast

from anjo.sleep.app.cleared_sessions import is_synthetic
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.storage import SLEEP_DIR, data_root
from anjo.sleep.budget import LEDGER_NAME, Bound, BudgetError, Ledger, Pool
from anjo.sleep.demo.limits import JsonCounterStore, Limiter, Limits, Refused
from anjo.sleep.demo.pairs import PairRegistry, Vote
from anjo.sleep.demo.service import Answer, ArmsError, DemoConfig, DemoService, LiveArms, Upkeep
from anjo.sleep.demo.slider import Pinned
from anjo.sleep.demo.starters import BASELINES, PRODUCT, StarterSet
from anjo.sleep.demo.starters import load as load_starters
from anjo.sleep.evidence.jsonl import PLAIN, LogError, append_records
from anjo.sleep.probes.arms import Arm, ArmDeps, DonorError, reply
from anjo.sleep.serving import client as vllm
from anjo.sleep.serving.backend import Serving
from anjo.sleep.serving.client import ServingError, VllmClient, WakeResult
from anjo.sleep.serving.config import SCALEDOWN_S, ServeConfigError
from anjo.sleep.serving.sagemaker import settings as sagemaker
from anjo.sleep.serving.wake import read_window, window_path
from anjo.sleep.types import CheckpointRef, Probe

logger = logging.getLogger(__name__)

USER_ENV: Final = "ANJO_SLEEP_DEMO_USER_ID"
BASELINE_ENV: Final = "ANJO_SLEEP_DEMO_BASELINE"
WAKE_ENV: Final = "ANJO_SLEEP_DEMO_WAKE"
LIVE_HOURS_ENV: Final = "ANJO_SLEEP_DEMO_LIVE_HOURS"
WARM_POOLS: Final[tuple[Pool, ...]] = ("modal_usd",)
DEFAULT_BASELINE: Final = "2C"
DEMO_PERSONA: Final = "film_founder"
DEMO_DIR: Final = "demo"
COUNTERS_FILE: Final = "counters.json"
VOTES_FILE: Final = "votes.jsonl"
STARTERS_FILE: Final = "starter_pairs.json"
SURFACE: Final = "sleep_demo"
LANGUAGE: Final = "en"  # the page is English; the crisis texts use the English lines
VISITOR: Final = "sleep-demo-visitor"  # whose harm-routing switch is read: a visitor has no account
WARM_MARGIN_S: Final = 60.0
KEY_INFO: Final = b"anjo-sleep-demo-v1"


def demo_dir() -> Path:
    return data_root() / SLEEP_DIR / DEMO_DIR


# ── Anjo's crisis checks ──────────────────────────────────────────────────────


def _count_referral(tier: str, surface: str) -> None:
    from anjo.core import crisis_ledger
    from anjo.core.best_effort import best_effort
    from anjo.core.crisis_render import resource_language

    with best_effort("count a live demo referral"):
        crisis_ledger.register_surface(SURFACE)
        crisis_ledger.record_referral(crisis_ledger.utc_day(), tier, resource_language(LANGUAGE), surface)


class AnjoFloor:
    """``Floor``: chat's normalization and server floor, called, never changed."""

    def normalize(self, text: str) -> str:
        from anjo.core.floor import normalize_message

        return str(normalize_message(text))

    def fixed_reply(self, text: str) -> str | None:
        from anjo.core.crisis import NO_CRISIS
        from anjo.core.crisis_harm import harm_others_enabled
        from anjo.core.floor import FloorRules, check_message

        decision = check_message(text, LANGUAGE, FloorRules(harm_routing=harm_others_enabled(VISITOR)))
        if decision.tier == NO_CRISIS:
            return None
        fixed = decision.protocol or decision.resource_line
        if not fixed:
            raise RuntimeError(f"the floor's {decision.tier} tier came with no fixed text")
        _count_referral(decision.tier, SURFACE)
        return str(fixed)


class AnjoReplyCheck:
    """``ReplyCheck``: ``reply_crisis_reason``; a tripped reply gives way to the severe protocol."""

    def replacement(self, reply_text: str) -> str | None:
        from anjo.core.crisis_render import severe_reply
        from anjo.core.crisis_reply import reply_crisis_reason

        if reply_crisis_reason(reply_text) is None:
            return None
        _count_referral("severe", "reply_check")
        return str(severe_reply(LANGUAGE, None))


# ── Glimmer ───────────────────────────────────────────────────────────────────


class Warmth:
    """On Modal: Glimmer counts as up within the scale-down window after the page's own last answer
    or wake, since asking would start a container."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last: float | None = None

    def mark(self) -> None:
        self._last = self._clock()

    def warm(self) -> bool:
        return self._last is not None and self._clock() - self._last < SCALEDOWN_S - WARM_MARGIN_S


def window_until() -> float | None:
    """The end of the keep-warm window running now (epoch seconds), or None."""
    window = read_window(window_path(data_root()))
    return window.until if window is not None and window.open_at(time.time()) else None


def glimmer_status(
    serving: Serving, warmth: Warmth | None = None, window: Callable[[], float | None] = window_until,
) -> bool:
    """Whether Glimmer is up, found without waking her."""
    if serving.name == "modal":
        return window() is not None or (warmth is not None and warmth.warm())
    try:
        return serving.ping()
    except ServingError as error:
        logger.warning("the live demo can't ask Glimmer's status: %s", error)
        return False


def _today() -> date:
    return datetime.now(timezone.utc).date()


class GlimmerArms:
    """``LiveArms`` over ``arms.reply``: arm 4 and the baseline for the film student's account."""

    def __init__(
        self, user_id: str, *, deps: Callable[[date], ArmDeps], status: Callable[[], bool],
        today: Callable[[], date] = _today, clock: Callable[[], float] = time.monotonic,
        on_answer: Callable[[], None] = lambda: None,
    ) -> None:
        self._user = user_id
        self._deps = deps
        self._status = status
        self._today = today
        self._clock = clock
        self._on_answer = on_answer

    def awake(self) -> bool:
        if not is_synthetic(self._user):
            logger.error("the live demo's account is not registered synthetic; it answers nothing live")
            return False
        return self._status()

    def answer_pair(self, question: str, baseline: str) -> tuple[Answer, Answer]:
        probe = Probe(f"demo-{secrets.token_hex(8)}", self._user, "new_work", question, None, None, ())
        deps = self._deps(self._today())
        deps.context_cache[(self._user, probe.id)] = deps.contexts.build(self._user, probe)  # one C for both
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="sleep-demo") as pool:
            product = pool.submit(self._one, PRODUCT, probe, deps)
            other = pool.submit(self._one, baseline, probe, deps)
            answers = product.result(), other.result()
        self._on_answer()
        return answers

    def answer_at(self, question: str, checkpoint: CheckpointRef | None) -> Answer:
        """Notes off at one stop: arm 3 pinned to ``checkpoint``, or arm 1 (the base) when None."""
        probe = Probe(f"demo-{secrets.token_hex(8)}", self._user, "new_work", question, None, None, ())
        deps = replace(self._deps(self._today()), adapters=Pinned(self._user, checkpoint))
        answer = self._one("1" if checkpoint is None else "3", probe, deps)
        self._on_answer()
        return answer

    def _one(self, arm: str, probe: Probe, deps: ArmDeps) -> Answer:
        started = self._clock()
        try:
            got = reply(cast(Arm, arm), self._user, probe, deps=deps)  # arm 4, or a checked baseline
        except (ServingError, DonorError) as error:
            raise ArmsError(f"arm {arm}: {type(error).__name__}") from None
        tokens = got.prompt_tokens + vllm.MAX_TOKENS  # the reply's ceiling: its real length isn't reported
        return Answer(got.text, got.prompt_tokens, tokens, self._clock() - started)


def glimmer_deps(user_id: str, serving: Serving) -> Callable[[date], ArmDeps]:
    """The arms' ports for ``user_id``, as the morning run wires them, on one client."""
    from anjo.sleep.app.context_builder import RECALL, anjo_context_builder
    from anjo.sleep.app.storage import brief_store, checkpoint_registry, evidence_store
    from anjo.sleep.context.brief_store import ServedBriefs

    made: list[VllmClient] = []  # one client, made at the first answer, never at startup

    def deps(until: date) -> ArmDeps:
        if not made:
            made.append(serving.client())
        model = made[0]
        return ArmDeps(
            model=model, adapters=checkpoint_registry(), logprobs=model, donors={}, until=until,
            contexts=anjo_context_builder(ServedBriefs(brief_store), until=until, settings=RECALL),
            evidence=evidence_store,
        )

    return deps


class BackgroundWake:
    """``Waker``: one wake at a time, off the request thread."""

    def __init__(
        self, wake: Callable[[], WakeResult], on_awake: Callable[[], None],
        spawn: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self._wake = wake
        self._on_awake = on_awake
        self._spawn = spawn or _in_thread
        self._lock = threading.Lock()
        self._busy = False

    def running(self) -> bool:
        return self._busy

    def start(self) -> None:
        with self._lock:
            if self._busy:
                return
            self._busy = True
        self._spawn(self._run)

    def _run(self) -> None:
        try:
            result = self._wake()
        except (ServingError, ServeConfigError) as error:
            logger.warning("the live demo's wake failed: %s", error)
            return
        finally:
            self._busy = False
        if result.ok:
            self._on_awake()
        else:
            logger.warning("the live demo's wake timed out after %.0f s", result.seconds)


def _in_thread(run: Callable[[], None]) -> None:
    threading.Thread(target=run, name="sleep-demo-wake", daemon=True).start()


# ── the service ───────────────────────────────────────────────────────────────


class Unavailable:
    """``LiveArms`` when the page has no account or no backend: always asleep, never woken."""

    def awake(self) -> bool:
        return False

    def answer_pair(self, question: str, baseline: str) -> tuple[Answer, Answer]:
        raise ArmsError("the live demo has no arms")

    def answer_at(self, question: str, checkpoint: CheckpointRef | None) -> Answer:
        raise ArmsError("the live demo has no arms")


def baseline_from(environ: Mapping[str, str]) -> str:
    chosen = environ.get(BASELINE_ENV, "").strip() or DEFAULT_BASELINE
    if chosen not in BASELINES:
        raise ValueError(f"{BASELINE_ENV} is one of {BASELINES}")
    return chosen


def serve_rate() -> float | None:
    from anjo.sleep.app.night import prices

    try:
        return prices().serve_gpu_per_hour
    except ValueError:
        return None  # no list price set: the reveal says cost wasn't measured


def ledger_pay(rate: float | None) -> Callable[[float], bool]:
    """``Upkeep.pay``: ``seconds`` of the serving GPU at ``rate`` dollars an hour, held on
    ``modal_usd`` and settled at once. False when there is no rate or the ledger refuses."""

    def pay(seconds: float) -> bool:
        if rate is None:
            return False
        job = f"sleep-demo-warm-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{secrets.token_hex(4)}"
        try:
            held = Ledger(data_root() / SLEEP_DIR / LEDGER_NAME).reserve(job, WARM_POOLS, Bound(rate, seconds))
            if held is None:
                logger.warning("the budget ledger refused the live demo's warm time")
                return False
            held.settle(held.bound.amount)
        except (BudgetError, LogError, OSError) as error:
            logger.error("the live demo's warm time wasn't debited: %s", type(error).__name__)
            return False
        return True

    return pay


def upkeep(serving: Serving | None, rate: float | None) -> Upkeep:
    """What a live answer and a wake keep Glimmer up for, on the backend in use."""
    if serving is not None and serving.name == "sagemaker":
        idle, wake = sagemaker.IDLE_MINUTES * 60.0, sagemaker.WAKE_TOTAL_TIMEOUT_S
    else:
        idle, wake = float(SCALEDOWN_S), vllm.WAKE_TOTAL_TIMEOUT_S
    return Upkeep(idle_s=idle, answer_s=vllm.REQUEST_TIMEOUT_S, wake_s=wake, covered_until=window_until,
                  pay=ledger_pay(rate))


def live_hours(environ: Mapping[str, str]) -> tuple[int, int] | None:
    """``ANJO_SLEEP_DEMO_LIVE_HOURS``: ``HH-HH`` in UTC, from and to; unset is every hour."""
    raw = environ.get(LIVE_HOURS_ENV, "").strip()
    if not raw:
        return None
    start, sep, end = raw.partition("-")
    if not sep or not start.strip().isdigit() or not end.strip().isdigit():
        raise ValueError(f"{LIVE_HOURS_ENV} is two UTC hours, HH-HH")
    return int(start), int(end)


def _limits(given: Limits | None, rate: float | None) -> Limits:
    chosen = given or Limits(live_hours_utc=live_hours(os.environ))
    return chosen if rate is not None else replace(chosen, warm_minutes_per_day=0)  # no price: none bought


def _demo_key() -> bytes:
    from anjo.core.crypto import _get_key

    return bytes(_get_key(KEY_INFO))


def _record_vote(vote: Vote) -> None:
    try:
        append_records(demo_dir() / VOTES_FILE, PLAIN, [vote.record()])
    except (OSError, LogError) as error:
        logger.error("a live demo vote wasn't stored: %s", type(error).__name__)
        raise Refused(503, "resting") from None


def _starters() -> StarterSet | None:
    return load_starters(demo_dir() / STARTERS_FILE)


def _from_env(serving: Serving | None) -> tuple[Serving | None, str | None]:
    user = os.environ.get(USER_ENV, "").strip() or None
    if serving is not None:
        return serving, user
    try:
        from anjo.sleep.serving import backend

        return backend.from_env(), user
    except ServeConfigError as error:
        logger.warning("the live demo has no serving backend: %s", error)
        return None, user


def build_service(
    *, serving: Serving | None = None, arms: LiveArms | None = None,
    spawn: Callable[[Callable[[], None]], None] | None = None, limits: Limits | None = None,
) -> DemoService:
    """The page's service from the host's settings. ``serving``, ``arms`` and ``limits`` replace the
    ones the settings would give (tests)."""
    from anjo.sleep.app import demo_mind

    require_sleep_host()
    serving, user = _from_env(serving)
    rate = serve_rate()
    warmth = Warmth()
    if arms is None:
        arms = Unavailable() if serving is None or user is None else GlimmerArms(
            user, deps=glimmer_deps(user, serving), status=lambda: glimmer_status(serving, warmth),
            on_answer=warmth.mark)
    waker = None
    if os.environ.get(WAKE_ENV) == "1" and serving is not None:
        waker = BackgroundWake(serving.wake, warmth.mark, spawn)
    folder = demo_dir()
    return DemoService(
        config=DemoConfig(baseline_from(os.environ), _demo_key(), rate),
        limiter=Limiter(JsonCounterStore(folder / COUNTERS_FILE), _limits(limits, rate)), pairs=PairRegistry(),
        floor=AnjoFloor(), reply_check=AnjoReplyCheck(), arms=arms, starters=_starters,
        votes=_record_vote, waker=waker, upkeep=upkeep(serving, rate),
        stops=lambda: demo_mind.demo_stops(user), night_answers=demo_mind.night_answers,
        swaps=lambda: demo_mind.swaps(user), stop_arms=arms if isinstance(arms, (GlimmerArms, Unavailable)) else None,
        mind=demo_mind.PanelCache(user).payload,
    )
