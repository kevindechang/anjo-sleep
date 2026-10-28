"""One simulated day before a synthetic person's night (Task 5 over Task 2's ``simulate_day``).

Glimmer is woken before the first turn (``wake``): a server that can't be woken means the day does
not start, and the night records ``glimmer_not_ready`` as a failure, not a short day.

A cleared session is answered only by Glimmer: when it can't answer, the turn fails and nothing is
stored (Task 4). The transport raises ``TurnFailed`` for such a turn, and the day ends there as a
short day, never an error: the night still runs on what was stored, and its log records how many
turns were answered and failed. The window is the wall-clock span around the day, which the
night's extraction reads.

``replies`` is required: without it every reply the simulator reads counts as unknown ancestry,
and none of the person's simulated lines could train.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final

from anjo.sleep import baseten_api
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.personas.model import Persona
from anjo.sleep.serving.backend import phase
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.sim.persona_sim import ChatTransport, ReplyAncestry, SimModel, Turn, simulate_day

SIM_TIMEOUT_S: Final = 120.0
NOT_READY: Final = "glimmer_not_ready"


UNKNOWN_REASON: Final = "turn_failed"


class TurnFailed(RuntimeError):
    """A simulated turn the dev API couldn't answer; nothing of it was stored.

    ``reason`` is a short code (an HTTP refusal's code, ``stale session``, a transport error class
    name), never the message: it is what a day's marker may carry."""

    def __init__(self, message: str, *, reason: str = UNKNOWN_REASON) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class DayRun:
    window: tuple[datetime, datetime]
    answered: int
    failed: int
    turns: tuple[Turn, ...]  # every line, for a day that ran to its end; empty for a short day
    wake_seconds: float | None = None  # how long waking Glimmer took, when the day woke it
    failure: str | None = None  # a code when the day never started (``glimmer_not_ready``)
    failure_detail: str | None = None  # the code of the turn that ended the day short, when one did

    @property
    def short(self) -> bool:
        return self.failed > 0


class _Counting:
    def __init__(self, inner: ChatTransport) -> None:
        self._inner = inner
        self.answered = 0

    def send(self, session_id: str, text: str, *, cleared: bool) -> str:
        reply = self._inner.send(session_id, text, cleared=cleared)
        self.answered += 1
        return reply


def run_day(
    person: Persona, day_index: int, seed: int, *, model: SimModel, transport: ChatTransport,
    replies: ReplyAncestry, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    wake: Callable[[], WakeResult] | None = None, sleep: Callable[[], None] | None = None,
) -> DayRun:
    """One simulated day. ``wake`` wakes Glimmer first; ``sleep`` puts it back when the day ends,
    however it ends (``serving.backend.phase``)."""
    with phase(sleep):
        counting = _Counting(transport)
        start = clock()
        woke = None if wake is None else wake()
        if woke is not None and not woke.ok:
            return DayRun((start, clock()), 0, 0, (), wake_seconds=woke.seconds, failure=NOT_READY)
        try:
            turns = tuple(simulate_day(person, day_index, seed, model=model, transport=counting, replies=replies))
            failed, detail = 0, None
        except TurnFailed as error:
            turns, failed, detail = (), 1, error.reason
        return DayRun((start, clock()), counting.answered, failed, turns,
                      wake_seconds=None if woke is None else woke.seconds, failure_detail=detail)


class SparkSim:
    """``SimModel`` on Spark Open (Baseten). Its API takes no seed, so a day replays only as far as
    the sampling allows; the seed still picks the day's topics and length. ``tokens`` is what the
    day has read and written so far, for its budget hold."""

    def __init__(self, chat: Callable[..., Any] = baseten_api.chat) -> None:
        self._chat = chat
        self.tokens: tuple[int, int] = (0, 0)

    def complete(self, system: str, messages: list[dict[str, str]], *, temperature: float, seed: int) -> str:
        result = self._chat(SPARK_OPEN, [{"role": "system", "content": system}, *messages], None, SIM_TIMEOUT_S,
                            temperature=temperature)
        self.tokens = (self.tokens[0] + int(result.input_tokens), self.tokens[1] + int(result.output_tokens))
        return str(result.text)
