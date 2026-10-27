"""One simulated day before a synthetic person's night (Task 5 over Task 2's ``simulate_day``).

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
from anjo.sleep.sim.persona_sim import ChatTransport, ReplyAncestry, SimModel, Turn, simulate_day

SIM_TIMEOUT_S: Final = 120.0


class TurnFailed(RuntimeError):
    """A simulated turn the dev API couldn't answer; nothing of it was stored."""


@dataclass(frozen=True)
class DayRun:
    window: tuple[datetime, datetime]
    answered: int
    failed: int
    turns: tuple[Turn, ...]  # every line, for a day that ran to its end; empty for a short day

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
) -> DayRun:
    counting = _Counting(transport)
    start = clock()
    try:
        turns = tuple(simulate_day(person, day_index, seed, model=model, transport=counting, replies=replies))
        failed = 0
    except TurnFailed:
        turns, failed = (), 1
    return DayRun((start, clock()), counting.answered, failed, turns)


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
