"""Day dreams: the messages the person might plausibly send next, written from the consolidated day
(``ANJO_SLEEP_DAY_DREAMS``, on under ``sft_dpo_seed`` unless set to ``0``).

The paper's dreaming starts from the knowledge being consolidated, not from a task: the model writes
its own practice about what it just learned. Here the day is the person's own material (her notes on
them and the day's lessons, corrections, preferences and tone requests), and her Glimmer writes
``DAY_DREAMS`` messages in the person's voice: new work in their field and project, questions that
need their vocabulary, and choices their stated preferences bear on. Her reply to each message is
written with the same day as system context: that reply is the seeding teacher's sample and the
training target.

One call returns the messages, one JSON object a line. A cut answer loses only its last line (an
array cut mid-way would lose every message), and an unreadable line is skipped: a bad answer yields
fewer dreams, never a failed night. A failed answer to one message loses only that dream; the replies
stop after ``DREAM_FAILURES_MAX`` failures in a row, so an unreachable server doesn't hold the night on
call after call. A failed call for the messages yields none. The training job holds every dream to the
decontaminator's rules and the probes' 13-gram rule exactly as it holds MIND's dreams
(``train/terms.py``); this module only writes them down.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, Protocol

from anjo.sleep.context.template import render
from anjo.sleep.model_api import ModelApiError
from anjo.sleep.night.dreams import NORMAL_STOP, DreamClient
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.serving.client import ServingError
from anjo.sleep.types import CheckpointRef

SWITCH_ENV: Final = "ANJO_SLEEP_DAY_DREAMS"  # objective.DAY_DREAMS_ENV aliases it: one name, read here too
DAY_DREAMS: Final = 16  # messages a night asks for
DREAM_MIX: Final = (8, 4, 4)  # new work, field questions, choices their preferences bear on
MESSAGE_TOKENS: Final = 2400  # one answer holding every JSON line, with room for sixteen messages
REPLY_TOKENS: Final = 400
DREAM_FAILURES_MAX: Final = 3  # replies that fail in a row end the night's asking; one failure loses one dream
DREAM_TEMPERATURE: Final = 1.0  # as MIND's dreams: her own sampling, never a judge's
DAY_DREAM_GROUP: Final = "dream"  # SeedRow.group, the job's keep-if-improves trial set
PROMPT_PATH: Final = Path(__file__).parent / "prompts" / "day_dreams_v1.md"


def day_dreams_on(environ: Mapping[str, str]) -> bool:
    """Whether the host builds her day dream source: off only for ``0`` (unset or ``1`` is on). The
    recipe's objective still decides whether the rows are asked (``objective.night_terms``)."""
    return environ.get(SWITCH_ENV, "").strip() != "0"


@dataclass(frozen=True)
class DreamDay:
    """The day the dreams start from: its consolidated text and the ancestry of everything in it."""

    text: str
    ancestry: frozenset[str] | None


@dataclass(frozen=True)
class DayDream:
    """One message the person might send, her reply to it, and what the two derive from."""

    id: str
    message: str
    reply: str
    ancestry: frozenset[str] | None


class DayDreamSource(Protocol):
    available: bool

    def dream(
        self, user_id: str, night: date, adapter: CheckpointRef | None, day: DreamDay,
    ) -> tuple[DayDream, ...]: ...


class NoDayDreams:
    """No day dreams (tests, the public example, the switch off): no rows."""

    available: bool = False

    def dream(
        self, user_id: str, night: date, adapter: CheckpointRef | None, day: DreamDay,
    ) -> tuple[DayDream, ...]:
        return ()


def _seed(seed: int, user_id: str, night: date, i: str, part: str) -> int:
    key = f"{seed}:{user_id}:{night.isoformat()}:{i}:{part}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def messages_from(text: str) -> list[str]:
    """The messages a JSONL answer holds, in order: a whole array too, a bad line skipped, duplicates
    once. A cut answer keeps its complete lines, which is why the call asks for one object a line."""
    stripped = text.strip()
    if stripped.startswith("```"):  # a model's code fence: the lines inside still parse
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()
    if not stripped:
        return []
    try:
        whole = json.loads(stripped)
    except ValueError:
        whole = None
    found: list[str] = []
    items: Sequence[object]
    if isinstance(whole, list):
        items = whole
    elif isinstance(whole, dict):
        nested = whole.get("messages")
        items = nested if isinstance(nested, list) else [whole]
    else:
        items = []
        for line in stripped.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict):
                items.append(item)
    for item in items:
        raw = item.get("message") if isinstance(item, dict) else item
        if not isinstance(raw, str):
            continue
        message = raw.strip()
        if message and message not in found:
            found.append(message)
    return found


class GlimmerDayDreams:
    """Her Glimmer writes the day's messages and answers each with the day in context. The adapter is
    the one that serves her (``registry.last_promoted``), as MIND's dreams use it."""

    available: bool = True

    def __init__(self, client: DreamClient, *, seed: int, count: int = DAY_DREAMS) -> None:
        if count < 1:
            raise ValueError("a day dream source asks for at least one message")
        self._client, self._seed, self._count = client, seed, count

    def dream(
        self, user_id: str, night: date, adapter: CheckpointRef | None, day: DreamDay,
    ) -> tuple[DayDream, ...]:
        if not day.text.strip():
            return ()
        prompt = render(PROMPT_PATH.read_text(encoding="utf-8"), {
            "COUNT": str(self._count), "NEW_WORK": str(DREAM_MIX[0]), "FIELD": str(DREAM_MIX[1]),
            "CHOICES": str(DREAM_MIX[2]),
        })
        try:
            asked = self._client.complete(
                adapter, [{"role": "system", "content": day.text}, {"role": "user", "content": prompt}],
                temperature=DREAM_TEMPERATURE, seed=_seed(self._seed, user_id, night, "messages", "messages"),
                max_tokens=MESSAGE_TOKENS,
            )
        except (ModelApiError, ServingError):
            return ()
        made: list[DayDream] = []
        failures = 0
        for i, message in enumerate(messages_from(asked.text)[: self._count]):
            try:
                answered = self._client.complete(
                    adapter, [{"role": "system", "content": day.text}, {"role": "user", "content": message}],
                    temperature=DREAM_TEMPERATURE, seed=_seed(self._seed, user_id, night, str(i), "reply"),
                    max_tokens=REPLY_TOKENS,
                )
            except (ModelApiError, ServingError):
                failures += 1  # one lost reply is one fewer dream, not the end of them all
                if failures >= DREAM_FAILURES_MAX:
                    break
                continue
            failures = 0
            reply = answered.text.strip()
            if answered.finish_reason != NORMAL_STOP or not reply:
                continue
            ancestry = None if day.ancestry is None else day.ancestry | {GLIMMER}
            made.append(DayDream(id=f"day-dream-{i}", message=message, reply=reply, ancestry=ancestry))
        return tuple(made)
