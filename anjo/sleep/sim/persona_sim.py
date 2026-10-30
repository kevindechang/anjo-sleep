"""The persona simulator: Spark Open plays a synthetic person for one day.

It takes only the visible `Persona` (RF5). The model client and the chat transport are
injected ports, so the engine never imports a provider. Every request is sent in a cleared
session (`cleared=True`) while `hosted_in_training` is false, which is always, here.
The dev-API side that honors the flag is Task 4's.

Each simulated line derives from Spark Open and from every Anjo reply it read before it in its
session (`simulated_ancestry`), so one reply of unknown ancestry leaves every line after it
unknown, and extraction never trains on them. The replies' recorded ancestry comes through the
`ReplyAncestry` port (the app's reply log); without it, a reply is unknown.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from anjo.sleep import spark_provider
from anjo.sleep.personas.model import Persona
from anjo.sleep.provenance import union

PROMPT_PATH = Path(__file__).parent / "prompts" / "sim_v1.md"
END_MARKER = "[END]"
MIN_TURNS = 4
MAX_TURNS = 9
PREFS_PER_DAY = 4
TOPICS_PER_DAY = 3
SIM_TEMPERATURE = 0.8


@dataclass(frozen=True)
class Turn:
    """One line of the day. ``ancestry`` is what its text derives from (``None`` when unknown)."""

    role: Literal["user", "assistant"]
    text: str
    day: int
    index: int
    ancestry: frozenset[str] | None = None


def simulated_ancestry(read: Iterable[frozenset[str] | None]) -> frozenset[str] | None:
    """A simulated line's ancestry: Spark Open's, and that of every reply it read (unknown if
    any reply's is)."""
    return union([frozenset({spark_provider.current().model}), *read])


class SimModel(Protocol):
    """A Spark Open chat client."""

    def complete(
        self, system: str, messages: list[dict[str, str]], *, temperature: float, seed: int
    ) -> str: ...


class ChatTransport(Protocol):
    """A client for the dev API. `cleared` marks a cleared session."""

    def send(self, session_id: str, text: str, *, cleared: bool) -> str: ...


class ReplyAncestry(Protocol):
    """The recorded ancestry of the reply ``send`` just returned in a session (``None`` when
    unknown)."""

    def of_reply(self, session_id: str) -> frozenset[str] | None: ...


def prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _pick(items: list[str], n: int, label: str, seed: int) -> list[str]:
    """Deterministic sample of n items, independent of Python's random module."""
    scored = sorted(
        items, key=lambda t: hashlib.sha256(f"{seed}:{label}:{t}".encode()).hexdigest()
    )
    return scored[:n]


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {t}" for t in items) if items else "- (nothing specific)"


def _topics(person: Persona, day_index: int, seed: int) -> list[str]:
    todays = [p.text for p in person.plans if p.day == day_index]
    prefs = _pick([s.text for s in person.preferences], PREFS_PER_DAY, f"pref{day_index}", seed)
    views = _pick([s.text for s in person.stated_views], 1, f"view{day_index}", seed)
    pool = [f"Something about your work where this matters to you: {t}" for t in prefs]
    pool += [f"A view you hold: {t}" for t in views]
    pool.append(f"Where {person.project.title} stands today")
    return todays + _pick(pool, TOPICS_PER_DAY, f"topic{day_index}", seed)


def build_system(person: Persona, day_index: int, seed: int) -> str:
    todays = [p.text for p in person.plans if p.day == day_index]
    plans = (
        "Something is on your calendar: " + " ".join(todays)
        if todays
        else "Nothing special is scheduled."
    )
    values = {
        "NAME": person.name,
        "LINE": person.line,
        "PROJECT_TITLE": person.project.title,
        "PROJECT_LOGLINE": person.project.logline,
        "CONSTRAINTS": _bullets(list(person.project.constraints)),
        "PREFERENCES": _bullets(
            _pick([s.text for s in person.preferences], 12, f"known{day_index}", seed)
        ),
        "TONE_REQUESTS": _bullets([s.text for s in person.tone_requests]),
        "VIEWS": _bullets([s.text for s in person.stated_views]),
        "DAY": str(day_index + 1),
        "TODAY_PLANS": plans,
        "TOPICS": _bullets(_topics(person, day_index, seed)),
    }
    text = prompt_text()
    for key, val in values.items():
        text = text.replace("{{" + key + "}}", val)
    return text


def simulate_day(
    person: Persona,
    day_index: int,
    seed: int,
    *,
    model: SimModel,
    transport: ChatTransport,
    replies: ReplyAncestry | None = None,
) -> list[Turn]:
    """Play one day. Only the visible part is read; every send is cleared."""
    system = build_system(person, day_index, seed)
    span = MAX_TURNS - MIN_TURNS + 1
    budget = MIN_TURNS + int(hashlib.sha256(f"{seed}:{day_index}:n".encode()).hexdigest(), 16) % span
    session = f"sim-{person.person_id}-d{day_index}"
    messages: list[dict[str, str]] = [{"role": "user", "content": "Begin the conversation."}]
    turns: list[Turn] = []
    read: list[frozenset[str] | None] = []
    for n in range(budget):
        raw = model.complete(
            system,
            messages,
            temperature=SIM_TEMPERATURE,
            seed=seed + day_index * 1000 + n,
        ).strip()
        if not raw or raw == END_MARKER or raw.startswith(END_MARKER):
            break
        said = simulated_ancestry(read)
        reply = transport.send(session, raw, cleared=True)
        heard = replies.of_reply(session) if replies is not None else None
        read.append(heard)
        turns.append(Turn("user", raw, day_index, len(turns), said))
        turns.append(Turn("assistant", reply, day_index, len(turns), heard))
        # From the model's side it is the speaker: its own lines are "assistant".
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": reply})
    return turns
