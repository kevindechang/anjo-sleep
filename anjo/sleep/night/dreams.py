"""Dreams as data, and which of the sleep objective's terms a night builds rows for.

**Terms.** ``Terms`` says which extra rows the night's dataset step builds for ``sft_dpo_seed``
(``train/objective.py`` reads the switches; the night command sets them). The default is all off:
the night is today's, and nothing here runs.

**One dream step.** MIND's core dreams once a night (its ``dream_states`` call, which the core
owns and which also picks the night's practice curriculum). Each state it returns reaches the night
as a ``DreamSeed``. Nothing here rolls the core.

**Ignition-boundary replay** (2026-10-29). The seeds worth dreaming are states at the edge of a
decision: the smallest plausible change of state that flips her intervention (speak or wait, or
which thread). The core returns both branches of such a boundary, each anchored to the real events
it rests on (``event_ids``), with the branch she did not take marked ``hypothetical``; a few seeds on
random threads (``kind="random"``) ride along as the control, so the training job's receipt can
show whether targeting the boundary matters (its keep rate by seed kind). A seed with no event under
it is never dreamed: an unanchored dream would teach what never happened.

**From MIND's night receipt** (``seeds_from_dream``). The core's night keeps each imagined state by
id: ``dreams`` (the states about the curriculum's picked threads) and ``boundaries`` (where a
rollout first flips her decision: ``speak`` and the thread ``slot``). A boundary becomes a
``boundary`` seed whose branch is the flipped decision, always imagined; a picked state a ``picked``
seed with no branch, labelled imagined too, since every dream state is a continuation, not the day.
The threads are slots, so their text and ancestry come from the real events the state rests on
(``describe``); a state resting on none is dropped.

**Glimmer speaks from them.** ``GlimmerDreams`` asks her own Glimmer (the adapter that serves her)
for ``per_state`` candidates per seed: one message the person might send about the thread, then her
reply with the seed in her system turn (``state_text``), as the channel will carry it. The text says
which branch it is and, for the imagined one, that it is hypothetical, not something that happened;
otherwise a wrong dream would read as a fact. A message or reply cut at its cap is no dream. The
seed is the student's input and never its target: the weights learn to answer from the state, not
to recall it, so live threads stay in the channel (Kevin, 2026-10-29). A dream's ancestry is
Glimmer plus the seed's own, so a seed built from an uncleared model's text never trains
(``train/terms.py``). Which candidates train is the training job's call: the top by gradient norm
plus a few at random, kept only if the anchor check holds (``train/sleep_loss.py``).
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal, Protocol, TypedDict, cast

from anjo.sleep.model_api import ChatResult
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.types import CheckpointRef

PER_STATE: Final = 3
MAX_STATES: Final = 8  # seeds a night: 24 candidates at most, of which the training job keeps 8
DREAM_TEMPERATURE: Final = 1.0  # the paper's randomness comes from routing; ours from the state and sampling
MESSAGE_TOKENS: Final = 160
REPLY_TOKENS: Final = 400
NORMAL_STOP: Final = "stop"
SEED_KINDS: Final = ("boundary", "picked", "random")
HYPOTHETICAL: Final = "Imagined, not something that happened"
TAKEN: Final = "What happened"
MESSAGE_PROMPT: Final = (
    "Something you are holding about the person right now: {thread}\n\n"
    "Write one message they might send you about it tomorrow, in their own voice, one to three "
    "sentences. Write only the message."
)


@dataclass(frozen=True)
class Terms:
    """Which of ``sft_dpo_seed``'s extra rows the night builds: knowledge seeding's, and dreams'."""

    seed: bool = False
    dreams: bool = False


SeedKind = Literal["boundary", "picked", "random"]


@dataclass(frozen=True)
class DreamSeed:
    """One seed from the core's ``dream_states``: the state it comes from (``state_ref``), its thread
    or slot as text (its targets' labels), the real events it rests on, the branch it rehearses
    (``speak``, ``wait``, or the thread she turns to), whether that branch is the imagined one, and
    whether it sits at a decision boundary, on a thread the curriculum picked, or on a random thread
    (the control)."""

    state_ref: str
    thread: str
    event_ids: tuple[str, ...]
    branch: str
    hypothetical: bool
    kind: SeedKind
    ancestry: frozenset[str] | None
    slots: tuple[int, ...] = ()  # the core's thread slots, the one a boundary turns to first; never in her text

    def __post_init__(self) -> None:
        if self.kind not in SEED_KINDS:
            raise ValueError(f"a dream seed's kind is one of {', '.join(SEED_KINDS)}")


@dataclass(frozen=True)
class Dream:
    """One candidate: the state as her system turn (``thread``), the imagined message, her reply."""

    id: str
    state_ref: str
    thread: str
    message: str
    reply: str
    ancestry: frozenset[str] | None
    kind: SeedKind
    event_ids: tuple[str, ...]


class DreamSource(Protocol):
    available: bool

    def dream(self, user_id: str, night: date, adapter: CheckpointRef | None) -> tuple[Dream, ...]: ...


class NoDreams:
    """No dream source (today's night, tests, the public example): no dream rows."""

    available: bool = False

    def dream(self, user_id: str, night: date, adapter: CheckpointRef | None) -> tuple[Dream, ...]:
        return ()


class DreamClient(Protocol):
    def complete(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *, temperature: float,
        seed: int | None = None, max_tokens: int | None = None,
    ) -> ChatResult: ...


def state_text(seed: DreamSeed) -> str:
    """The seed as the channel's text stand-in: the thread, then the branch, marked as imagined when
    it is the one she didn't take (an imagined state with no branch is marked as imagined too)."""
    if not seed.branch:
        return f"{seed.thread}\n{HYPOTHETICAL}" if seed.hypothetical else seed.thread
    return f"{seed.thread}\n{HYPOTHETICAL if seed.hypothetical else TAKEN}: {seed.branch}"


class KeptState(TypedDict):
    """One imagined state as MIND's night receipt keeps it (``dreams``)."""

    state: str
    threads: list[int]
    rests_on: list[str]


class KeptBoundary(KeptState):
    """A decision boundary as the receipt keeps it (``boundaries``): the state there, and the decision it flips to."""

    steps: int
    speak: bool
    slot: int | None


class Describe(Protocol):
    def __call__(self, event_ids: Sequence[str]) -> tuple[str, frozenset[str] | None]:
        """The thread as text from the real events ``event_ids`` (what they say she holds), and their ancestry."""
        ...


SPEAK: Final = "you bring it up now"
WAIT: Final = "you wait, and don't bring it up yet"


def seeds_from_dream(receipt: Mapping[str, object], describe: Describe) -> tuple[DreamSeed, ...]:
    """The night's dream seeds from MIND's night receipt: its boundaries first, then its picked
    states; a state resting on no real event is dropped before ``describe`` reads anything."""
    boundaries = cast("Sequence[KeptBoundary]", receipt.get("boundaries") or ())
    picked = cast("Sequence[KeptState]", receipt.get("dreams") or ())
    made = [_from_state(b, "boundary", SPEAK if b["speak"] else WAIT, (b["slot"],) if b["slot"] is not None else (), describe)
            for b in boundaries]
    made += [_from_state(d, "picked", "", (), describe) for d in picked]
    return tuple(seed for seed in made if seed is not None)


def _from_state(kept: KeptState, kind: SeedKind, branch: str, slot: tuple[int, ...], describe: Describe) -> DreamSeed | None:
    if not kept["rests_on"]:
        return None
    events = tuple(kept["rests_on"])
    thread, ancestry = describe(events)
    return DreamSeed(state_ref=kept["state"], thread=thread, event_ids=events, branch=branch, hypothetical=True,
                     kind=kind, ancestry=ancestry, slots=tuple(dict.fromkeys((*slot, *kept["threads"]))))


def _seed(seed: int, user_id: str, night: date, state_ref: str, i: str, part: str) -> int:
    key = f"{seed}:{user_id}:{night.isoformat()}:{state_ref}:{i}:{part}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


class GlimmerDreams:
    """Her Glimmer's candidates from the core's dream seeds (``seeds(user_id, night)``); a seed with
    no event under it is skipped before any call."""

    available: bool = True

    def __init__(
        self, client: DreamClient, seeds: Callable[[str, date], Sequence[DreamSeed]], *, seed: int,
        per_state: int = PER_STATE, max_states: int = MAX_STATES,
    ) -> None:
        if per_state < 1 or max_states < 1:
            raise ValueError("a dream source asks at least one candidate of at least one state")
        self._client, self._seeds, self._seed = client, seeds, seed
        self._per_state, self._max_states = per_state, max_states

    def dream(self, user_id: str, night: date, adapter: CheckpointRef | None) -> tuple[Dream, ...]:
        made: list[Dream] = []
        anchored = [seed for seed in self._seeds(user_id, night) if seed.event_ids]
        for i_seed, seed in enumerate(anchored[: self._max_states]):
            for i in range(self._per_state):
                dream = self._candidate(user_id, night, adapter, seed, f"{i_seed}-{i}")
                if dream is not None:
                    made.append(dream)
        return tuple(made)

    def _candidate(
        self, user_id: str, night: date, adapter: CheckpointRef | None, state: DreamSeed, i: str,
    ) -> Dream | None:
        thread = state_text(state)
        asked = self._client.complete(
            adapter, [{"role": "user", "content": MESSAGE_PROMPT.format(thread=thread)}],
            temperature=DREAM_TEMPERATURE, seed=_seed(self._seed, user_id, night, state.state_ref, i, "message"),
            max_tokens=MESSAGE_TOKENS,
        )
        message = asked.text.strip()
        if asked.finish_reason != NORMAL_STOP or not message:
            return None
        answered = self._client.complete(
            adapter, [{"role": "system", "content": thread}, {"role": "user", "content": message}],
            temperature=DREAM_TEMPERATURE, seed=_seed(self._seed, user_id, night, state.state_ref, i, "reply"),
            max_tokens=REPLY_TOKENS,
        )
        reply = answered.text.strip()
        if answered.finish_reason != NORMAL_STOP or not reply:
            return None
        ancestry = None if state.ancestry is None else state.ancestry | {GLIMMER}
        return Dream(id=f"dream-{i}", state_ref=state.state_ref, thread=thread, message=message, reply=reply,
                     ancestry=ancestry, kind=state.kind, event_ids=state.event_ids)
