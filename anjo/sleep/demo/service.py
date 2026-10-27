"""One visit to the live page (Task 13a): the flow, over ports the app fills.

- ``start_visit``: refused unless the visitor confirmed they are 18 or older (``adult is True``);
  then the visit token every chat call needs.
- ``starter``: a starter scene's pre-generated pair. No model, no GPU.
- ``ask``: the visitor's own question. In this order:
  1. the crisis floor (``Floor.fixed_reply``), before any model sees it: a crisis message gets the
     existing fixed text and nothing else;
  2. the limits hold both calls at their worst case (``Limiter.admit``);
  3. Glimmer must already be up (``LiveArms.awake``). If she isn't, the hold goes back and the page
     says she's asleep. Nothing wakes her unless the app gave a ``Waker`` (an opt-in setting, off
     by default), the day has a wake left and the wake's warm time can be bought;
  4. the warm time the answer keeps her up is bought (``Limiter.buy_warm_time``, paid through
     ``Upkeep.pay``): past the day's warm minutes, or outside the live hours, nothing is asked and
     the page says so, and runs on its starter pairs;
  5. arm 4 and the baseline answer (``LiveArms.answer_pair``), and the hold is settled.
- Every reply, pre-generated or live, passes the reply check (``ReplyCheck``) before it is shown. A
  side that trips is withheld and the check's own text stands in its place; that pair takes no vote.
- ``vote``: through ``pairs.PairRegistry``, which reveals the sides only after the vote is stored
  and caps a visit's votes and open pairs. A starter scene takes one vote per address a day
  (``Limiter.claim_scene_vote``, claimed first and given back if the vote isn't stored).

Stateless: the question lives only in this call. Nothing here stores, logs or returns it, and the
ports only read (``LiveArms`` answers through ``arms.reply``, which writes nothing).
"""
from __future__ import annotations

import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, Protocol

from anjo.sleep.demo import words
from anjo.sleep.demo.limits import Limiter, Refused
from anjo.sleep.demo.pairs import PairRegistry, Reveal, SideFacts, Vote
from anjo.sleep.demo.starters import BASELINES, PRODUCT, Side, StarterError, StarterSet
from anjo.sleep.demo.visits import mint_visit, visit_nonce

logger = logging.getLogger(__name__)

MAX_CHARS: Final = 1_000
CALLS_PER_ASK: Final = 2  # arm 4 and the baseline: every model call counts


class Floor(Protocol):
    def normalize(self, text: str) -> str:
        """The text every check reads (the chat path's normalization)."""

    def fixed_reply(self, text: str) -> str | None:
        """The existing fixed answer when ``text`` is a crisis message, else None."""


class ReplyCheck(Protocol):
    def replacement(self, reply: str) -> str | None:
        """None when ``reply`` may be shown, else the text shown in its place."""


@dataclass(frozen=True)
class Answer:
    """One live side: its reply, prompt length, the tokens the limits settle and its seconds."""

    text: str
    prompt_tokens: int
    tokens: int
    seconds: float


class ArmsError(RuntimeError):
    """A side couldn't be answered. The message names a cause, never a word of the question."""


class LiveArms(Protocol):
    def awake(self) -> bool:
        """Whether Glimmer is up now, found without waking her."""

    def answer_pair(self, question: str, baseline: str) -> tuple[Answer, Answer]:
        """Arm 4's answer, then the baseline's, from one shared context. Raises ``ArmsError``."""


class Waker(Protocol):
    def running(self) -> bool: ...

    def start(self) -> None:
        """Begin waking Glimmer in the background."""


@dataclass(frozen=True)
class Upkeep:
    """What keeping Glimmer up costs the demo, in seconds of GPU time, and who pays. ``idle_s``: how
    long one call keeps her up (the backend's idle window); ``answer_s``: the longest a live pair
    takes; ``wake_s``: the longest a wake takes. ``covered_until`` is the end of a keep-warm window
    someone else holds (epoch seconds), or None; ``pay`` debits seconds to the budget ledger."""

    idle_s: float
    answer_s: float
    wake_s: float
    covered_until: Callable[[], float | None]
    pay: Callable[[float], bool]


@dataclass(frozen=True)
class SideView:
    text: str
    withheld: bool


@dataclass(frozen=True)
class Source:
    """Where a pair came from: ``pre-generated`` (with its checkpoint) or ``live``."""

    kind: Literal["pre-generated", "live"]
    checkpoint: str | None = None
    night: str | None = None
    generated_at: str | None = None


@dataclass(frozen=True)
class PairView:
    """What the page gets for a pair, in side order. Nothing in it names an arm."""

    pair_id: str
    left: SideView
    right: SideView
    votable: bool
    source: Source


@dataclass(frozen=True)
class Fixed:
    """A crisis message's answer: the existing fixed text. No model saw the message."""

    text: str


@dataclass(frozen=True)
class Asleep:
    waking: bool


@dataclass(frozen=True)
class SceneView:
    scene: words.Scene
    ready: bool


@dataclass(frozen=True)
class PageView:
    """The page's opening state: the scenes, where their pairs came from, and whether a question of
    the visitor's own can be answered live (``open``, ``spent`` for today, or ``closed`` now)."""

    scenes: tuple[SceneView, ...]
    source: Source | None
    live: str


@dataclass(frozen=True)
class DemoConfig:
    baseline: str
    secret: bytes
    serve_usd_per_hour: float | None  # the serving GPU's list price; None: cost isn't shown

    def __post_init__(self) -> None:
        if self.baseline not in BASELINES:
            raise ValueError(f"the demo baseline is one of {BASELINES}")
        if not self.secret:
            raise ValueError("the demo key is empty")


def _coin() -> bool:
    return secrets.randbelow(2) == 0


def _new_id() -> str:
    return secrets.token_urlsafe(16)


class DemoService:
    def __init__(
        self, *, config: DemoConfig, limiter: Limiter, pairs: PairRegistry, floor: Floor,
        reply_check: ReplyCheck, arms: LiveArms, starters: Callable[[], StarterSet | None],
        votes: Callable[[Vote], None], upkeep: Upkeep, waker: Waker | None = None,
        coin: Callable[[], bool] = _coin, new_id: Callable[[], str] = _new_id,
    ) -> None:
        self.config = config
        self._upkeep = upkeep
        self._limiter = limiter
        self._pairs = pairs
        self._floor = floor
        self._reply_check = reply_check
        self._arms = arms
        self._starters = starters
        self._votes = votes
        self._waker = waker
        self._coin = coin
        self._new_id = new_id

    def scenes(self, client: str, now: datetime) -> PageView:
        self._limiter.request(client, now)
        starters = self._current_starters()
        views = tuple(SceneView(s, starters is not None and starters.pair(s) is not None) for s in words.SCENES)
        live = self._limiter.live_state(now, self._answer_s, covered_until=self._upkeep.covered_until())
        return PageView(views, None if starters is None else _source(starters), live)

    def start_visit(self, client: str, adult: bool, now: datetime) -> str:
        self._limiter.request(client, now)
        if adult is not True:
            raise Refused(403, "adult_required")
        self._limiter.visit(client, now)
        return mint_visit(self.config.secret, self._new_id(), now)

    def starter(self, client: str, token: str, scene_id: str, now: datetime) -> PairView:
        visit = self._visit(token, now)
        self._limiter.request(client, now)
        scene = words.scene(scene_id)
        if scene is None:
            raise Refused(404, "unknown_scene")
        starters = self._current_starters()
        pair = None if starters is None else starters.pair(scene)
        if starters is None or pair is None:
            raise Refused(404, "not_ready")
        return self._open(pair.product, pair.baseline, _source(starters), now, visit=visit, scene=scene.id)

    def ask(self, client: str, token: str, raw: str, now: datetime) -> PairView | Fixed | Asleep:
        visit = self._visit(token, now)
        self._limiter.request(client, now)
        text = self._floor.normalize(raw).strip()
        if not text:
            raise Refused(400, "empty")
        if len(text) > MAX_CHARS:
            raise Refused(400, "too_long")
        fixed = self._floor.fixed_reply(text)
        if fixed is not None:
            return Fixed(fixed)
        live = self._limiter.live_state(now, self._answer_s, covered_until=self._upkeep.covered_until())
        if live != "open":  # before even a status ping: on SageMaker a ping keeps her up too
            raise Refused(429, f"live_{live}")
        self._pairs.check_room(visit, now)  # before the calls are spent on a pair it couldn't open
        hold = self._limiter.admit(client, visit, CALLS_PER_ASK, now)
        ran, used = False, hold.tokens
        try:
            if not self._arms.awake():
                return Asleep(self._wake(now))
            self._buy(now, self._answer_s)
            ran = True
            product, baseline = self._arms.answer_pair(text, self.config.baseline)
            used = product.tokens + baseline.tokens
        except ArmsError as error:
            logger.warning("a live demo pair failed: %s", error)
            raise Refused(503, "lost_thread") from None
        finally:
            if ran:
                self._limiter.settle(hold, used, now)  # a failed pair keeps its worst case
            else:
                self._limiter.release(hold, now)
        return self._open(self._side(product), self._side(baseline), Source("live"), now, visit=visit)

    def vote(self, client: str, token: str, pair_id: str, choice: str, now: datetime) -> Reveal:
        visit = self._visit(token, now)
        self._limiter.request(client, now)
        scene = self._pairs.scene_of(pair_id, visit=visit)
        if scene is None:
            return self._pairs.vote(pair_id, choice, visit=visit, now=now, record=self._votes)
        self._limiter.claim_scene_vote(client, scene, now)
        try:
            return self._pairs.vote(pair_id, choice, visit=visit, now=now, record=self._votes)
        except Refused:
            self._limiter.release_scene_vote(client, scene, now)
            raise

    # ── helpers ───────────────────────────────────────────────────────────────

    def _visit(self, token: str, now: datetime) -> str:
        nonce = visit_nonce(token, self.config.secret, now)
        if nonce is None:
            raise Refused(401, "visit_expired")
        return nonce

    def _current_starters(self) -> StarterSet | None:
        """The pre-generated pairs, if they were generated against the demo's current baseline."""
        try:
            starters = self._starters()
        except StarterError as error:
            logger.error("the demo's starter pairs can't be read: %s", error)
            return None
        return starters if starters is not None and starters.baseline == self.config.baseline else None

    @property
    def _answer_s(self) -> float:
        """How long a live answer keeps her up: the answer, then the backend's idle window."""
        return self._upkeep.answer_s + self._upkeep.idle_s

    def _buy(self, now: datetime, seconds: float, *, wake: bool = False) -> None:
        upkeep = self._upkeep
        self._limiter.buy_warm_time(now, seconds, covered_until=upkeep.covered_until(), pay=upkeep.pay, wake=wake)

    def _wake(self, now: datetime) -> bool:
        if self._waker is None:
            return False
        if self._waker.running():
            return True
        self._buy(now, self._upkeep.wake_s + self._upkeep.idle_s, wake=True)  # refused: the page says why
        self._waker.start()
        return True

    def _side(self, answer: Answer) -> Side:
        rate = self.config.serve_usd_per_hour
        cost = None if rate is None else round(answer.seconds * rate / 3600.0, 6)
        return Side(answer.text, answer.prompt_tokens, cost)

    def _shown(self, side: Side) -> SideView:
        replacement = self._reply_check.replacement(side.text)
        return SideView(side.text, False) if replacement is None else SideView(replacement, True)

    def _open(
        self, product: Side, baseline: Side, source: Source, now: datetime, *, visit: str, scene: str | None = None,
    ) -> PairView:
        product_view, baseline_view = self._shown(product), self._shown(baseline)
        votable = not (product_view.withheld or baseline_view.withheld)
        product_facts = SideFacts(PRODUCT, product.prompt_tokens, product.cost_usd)
        baseline_facts = SideFacts(self.config.baseline, baseline.prompt_tokens, baseline.cost_usd)
        pair_id = self._new_id()
        product_left = self._coin()
        left, right = (product_facts, baseline_facts) if product_left else (baseline_facts, product_facts)
        self._pairs.open(pair_id, visit=visit, scene=scene, left=left, right=right, votable=votable, now=now)
        if product_left:
            return PairView(pair_id, product_view, baseline_view, votable, source)
        return PairView(pair_id, baseline_view, product_view, votable, source)


def _source(starters: StarterSet) -> Source:
    return Source("pre-generated", starters.checkpoint, starters.night, starters.generated_at)
