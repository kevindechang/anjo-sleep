"""The live page's wiring on the sleep host (Task 13a): Anjo's crisis checks, the arms, Glimmer's
status and the demo's files, for ``anjo.sleep.demo.service``.

- **Before any model:** every visitor message passes the server floor chat runs before it replies
  (``anjo.core.floor.check_message``, with chat's own harm-routing switch). A severe or harm message
  gets the fixed protocol, an ideation message its fixed resource line, and no model is called for
  any of them: a blind comparison is no place to answer someone at risk with two generated replies.
  Each referral is counted, with no words, in the crisis ledger under the ``sleep_demo`` surface.
- **Before a reply is shown:** ``reply_crisis_reason``. A side that trips is withheld and the severe
  protocol stands in its place, as chat does, counted under ``reply_check``.
- **The arms:** ``arms.reply`` for arm 4 and the baseline, over the ports the morning run uses
  (``morning.build``), for the account that plays the film student (``ANJO_SLEEP_DEMO_USER_ID``).
  That account must be registered synthetic, or the page answers nothing live: no real person's
  context ever reaches a stranger. Context C is built once and both sides read it. ``reply`` only
  reads, so nothing is written to history, evidence or memory, and the question is never logged.
- **Glimmer is never woken by default.** ``glimmer_status`` asks without waking: on SageMaker a ping
  finds no copy running and says so. On Modal any request starts a container, so the page never
  pings it and treats Glimmer as up only within the scale-down window after its own last answer or
  wake. Waking on demand needs ``ANJO_SLEEP_DEMO_WAKE=1`` (off by default) and is capped per day
  (``Limits.wakes_per_day``): the seam for judging week, whose warm windows Kevin sets.

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
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Final, cast

from anjo.sleep.app.cleared_sessions import is_synthetic
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.storage import SLEEP_DIR, data_root
from anjo.sleep.demo.limits import JsonCounterStore, Limiter, Refused
from anjo.sleep.demo.pairs import PairRegistry, Vote
from anjo.sleep.demo.service import Answer, ArmsError, DemoConfig, DemoService, LiveArms
from anjo.sleep.demo.starters import BASELINES, PRODUCT, StarterSet
from anjo.sleep.demo.starters import load as load_starters
from anjo.sleep.evidence.jsonl import PLAIN, LogError, append_records
from anjo.sleep.probes.arms import Arm, ArmDeps, DonorError, reply
from anjo.sleep.serving import client as vllm
from anjo.sleep.serving.backend import Serving
from anjo.sleep.serving.client import ServingError, VllmClient, WakeResult
from anjo.sleep.serving.config import SCALEDOWN_S, ServeConfigError
from anjo.sleep.types import Probe

logger = logging.getLogger(__name__)

USER_ENV: Final = "ANJO_SLEEP_DEMO_USER_ID"
BASELINE_ENV: Final = "ANJO_SLEEP_DEMO_BASELINE"
WAKE_ENV: Final = "ANJO_SLEEP_DEMO_WAKE"
DEFAULT_BASELINE: Final = "2C"
DEMO_PERSONA: Final = "film_student"
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


def glimmer_status(serving: Serving, warmth: Warmth | None = None) -> bool:
    """Whether Glimmer is up, found without waking her."""
    if serving.name == "modal":
        return warmth is not None and warmth.warm()
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
    spawn: Callable[[Callable[[], None]], None] | None = None,
) -> DemoService:
    """The page's service from the host's settings. ``serving`` and ``arms`` replace the ones the
    settings would give (tests)."""
    require_sleep_host()
    serving, user = _from_env(serving)
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
        config=DemoConfig(baseline_from(os.environ), _demo_key(), serve_rate()),
        limiter=Limiter(JsonCounterStore(folder / COUNTERS_FILE)), pairs=PairRegistry(),
        floor=AnjoFloor(), reply_check=AnjoReplyCheck(), arms=arms, starters=_starters,
        votes=_record_vote, waker=waker,
    )
