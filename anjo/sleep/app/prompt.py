"""A flagged person's open-model prompt (Task 4): context C on a served turn, and the cleared
context in a cleared session.

``sleep_turn_hook`` is an open-model turn hook (``open_model_turn.TURN_HOOKS``). It acts only on a
turn the sleep route serves, one whose pins are Glimmer pins; every other turn passes through as
the same object, so nobody else's prompt changes.

**Crisis first.** The chat route's floor has already answered a severe or harm message with its
fixed text, and the harness refuses those tiers. The hook also refuses (``SleepTurnRefused``) any
turn with a crisis tier, a session in distress, or a person who is not flagged. It raises before
any model call, so the chat route's fallback answers with the frontier talker, and the floor's
resource line follows that reply as on any turn.

**Context C.** One ``ContextBuilder`` (``anjo.sleep.app.context_builder``) builds it, the one arms
2C and 4 use, so a served reply and an arm's reply see the same C for the same message. It goes
in as one fenced optional part.

**Served turn.** The prompt the route built, whose blocks carry Anjo's memory, plus C from the
served brief. Its ancestry is unknown, because Anjo's memory has no recorded source.

**Cleared turn.** In a cleared session (Task 2) nothing of Anjo's memory, homework or notes reaches
the model. The instructions are Anjo's fixed persona text and medical boundary; C is the cleared
builder's (the training brief, cleared recall, and the person's own preferences and corrections);
the history is the recent turns of their cleared sessions only. No exemplars, no voice plan and
no tools.

Each turn records what its context was read from, and its ancestry (``ancestry.record_context``).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any, Final

from anjo.agent.records import TurnFacts
from anjo.core.anjo_model.pin import HostPin
from anjo.core.anjo_model.turn_input import AnjoTurnInput, Exchange, Segment, exchanges_from_messages
from anjo.core.best_effort import best_effort
from anjo.core.crisis import NO_CRISIS
from anjo.core.retrieved_context import RetrievedContext
from anjo.sleep.app.ancestry import ContextPart, record_context, reply_ancestries, union
from anjo.sleep.app.cleared_sessions import cleared_sessions, is_synthetic
from anjo.sleep.app.context_builder import RECALL, NoBrief, anjo_context_builder, cleared_context_builder
from anjo.sleep.app.flags import sleep_enabled_for
from anjo.sleep.app.runtime import GlimmerPin
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.builder import BriefSource, ComposedContext, ComposedContextBuilder
from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.types import Probe

CLEARED_HISTORY_DAYS: Final = 14
CLEARED_HISTORY_ROWS: Final = 24
MEMORY_SOURCE: Final = "anjo_memory"
CONTEXT_SOURCE: Final = "sleep_context"
# A synthetic person's words are Spark Open's (Task 2's simulator plays them).
_SIMULATED: Final = frozenset({SPARK_OPEN})

ToolDefs = Sequence[Mapping[str, Any]]


class SleepTurnRefused(RuntimeError):
    """A turn the sleep runtime must not answer. Raised before any model call, so the chat
    route's fallback answers it."""


@dataclasses.dataclass(frozen=True)
class ContextSetup:
    """The brief stores C reads (Task 5): the served brief for served turns, the training brief
    for cleared ones. Without a store, C has no brief."""

    served_briefs: BriefSource = dataclasses.field(default_factory=NoBrief)
    training_briefs: BriefSource = dataclasses.field(default_factory=NoBrief)


_setup: list[ContextSetup] = [ContextSetup()]
_builders: dict[tuple[date, bool], ComposedContextBuilder] = {}


def set_context_setup(setup: ContextSetup | None) -> None:
    """Set the brief stores (``None`` for none). Builders made before are dropped."""
    _setup[:] = [setup if setup is not None else ContextSetup()]
    _builders.clear()


def _builder(*, cleared: bool) -> ComposedContextBuilder:
    """Today's builder: one a day per kind, so recall embeds each stored item once a day."""
    today = datetime.now(timezone.utc).date()
    if (today, cleared) not in _builders:
        for stale in [key for key in list(_builders) if key[0] != today]:
            _builders.pop(stale, None)
        setup = _setup[0]
        _builders[(today, cleared)] = (
            cleared_context_builder(setup.training_briefs, until=today, settings=RECALL)
            if cleared else anjo_context_builder(setup.served_briefs, until=today, settings=RECALL)
        )
    return _builders[(today, cleared)]


def _context(user_id: str, turn: AnjoTurnInput, *, cleared: bool) -> ComposedContext:
    probe = Probe(
        id=turn.turn_id, person_id=user_id, kind="new_work", prompt=turn.user_text,
        options=None, expected=None, source_ids=(),
    )
    return _builder(cleared=cleared).compose(user_id, probe)


def _part(turn: AnjoTurnInput, text: str) -> RetrievedContext:
    """C as one optional part, ranked above every part the turn holds (ranks are unique)."""
    ranks = [
        *(part.rank for part in turn.optional), *(() if turn.carry is None else (turn.carry.rank,)),
        *(exchange.rank for exchange in (*turn.exemplars, *turn.history)),
    ]
    return RetrievedContext(
        source_ref=CONTEXT_SOURCE, revision=0, rank=max(ranks, default=0) + 1,
        token_cost=estimate_tokens(text), text=text,
    )


def _read_from(context: ComposedContext, preferences: frozenset[str] | None) -> list[ContextPart]:
    """Where C's parts came from. Empty parts read "none", which derives from no model."""
    brief = context.brief
    return [
        ContextPart("brief", brief.ancestry if brief is not None else frozenset()),
        ContextPart("preferences", preferences if context.since else frozenset()),
        *(ContextPart(f"evidence:{item.id}", item.ancestry) for item in context.found),
    ]


def _cleared_history(user_id: str, sessions: frozenset[str]) -> tuple[tuple[Exchange, ...], ContextPart]:
    """The recent turns of the person's cleared sessions, and their ancestry: the person's words
    are simulated, and each of Anjo's replies carries its recorded ancestry (unknown if none)."""
    from anjo.core.db import get_db
    from anjo.core.history import get_history_between

    now = datetime.now(timezone.utc)
    rows = get_history_between(
        user_id, now - timedelta(days=CLEARED_HISTORY_DAYS), now + timedelta(seconds=1),
        session_ids=sessions,
    )[-CLEARED_HISTORY_ROWS:]
    replies = [row["id"] for row in rows if row["role"] == "assistant"]
    turn_ids: dict[int, str | None] = {}
    if replies:
        marks = ", ".join("?" for _ in replies)
        found = get_db().execute(
            f"SELECT id, turn_id FROM messages WHERE user_id = ? AND id IN ({marks})", (user_id, *replies),
        ).fetchall()
        turn_ids = {row["id"]: row["turn_id"] for row in found}
    recorded = reply_ancestries(user_id)

    def of(row: Mapping[str, Any]) -> frozenset[str] | None:
        if row["role"] == "user":
            return _SIMULATED
        known = recorded.get(turn_ids.get(row["id"]) or "")
        return None if known is None else known.reply

    exchanges, _unanswered = exchanges_from_messages(
        [(str(row["role"]), str(row["content"])) for row in rows], id_prefix="c",
    )
    return exchanges, ContextPart("cleared_history", union(of(row) for row in rows))


def _persona() -> tuple[Segment, ...]:
    from anjo.core.medical_boundary import MEDICAL_BOUNDARY_RULE
    from anjo.core.voice_contract import ANJO_IDENTITY_OPENER, character_core

    return (
        Segment("persona", f"{ANJO_IDENTITY_OPENER}\n\n{character_core()}", "static"),
        Segment("safety_rule", MEDICAL_BOUNDARY_RULE.strip(), "static"),
    )


def _refuse_unless_clear(facts: TurnFacts, pins: Sequence[GlimmerPin]) -> None:
    if facts.crisis_tier != NO_CRISIS or facts.session_in_distress:
        raise SleepTurnRefused("a crisis or distress turn goes to the frontier route")
    if any(pin.user_id != facts.user_id for pin in pins) or not sleep_enabled_for(facts.user_id):
        raise SleepTurnRefused("the sleep route serves only the person it flagged")


def sleep_turn_hook(
    turn: AnjoTurnInput, tool_defs: ToolDefs, facts: TurnFacts, session_id: str | None,
    pins: Sequence[HostPin],
) -> tuple[AnjoTurnInput, ToolDefs]:
    glimmer = [pin for pin in pins if isinstance(pin, GlimmerPin)]
    if not glimmer:
        return turn, tool_defs
    _refuse_unless_clear(facts, glimmer)
    user_id = facts.user_id
    tools: ToolDefs = tool_defs if glimmer[0].traits.tool_calls == "native" else ()
    sessions = cleared_sessions(user_id)
    cleared = session_id is not None and session_id in sessions
    context = _context(user_id, turn, cleared=cleared)
    if cleared:
        history, history_part = _cleared_history(user_id, sessions)
        bare = dataclasses.replace(
            turn, instructions=_persona(), corrections=(), optional=(), carry=None,
            exemplars=(), history=history, voice_plan=None, tools=(), tool_steps=(),
        )
        built = dataclasses.replace(bare, optional=(_part(bare, context.text),))
        # Only the person's own words reach a cleared C's preferences.
        own = _SIMULATED if is_synthetic(user_id) else frozenset()
        recorded = [*_read_from(context, own), history_part]
        tools = ()
    else:
        built = dataclasses.replace(turn, optional=(*turn.optional, _part(turn, context.text)))
        recorded = [ContextPart(MEMORY_SOURCE, None), *_read_from(context, None)]
    with best_effort("record a sleep turn's context ancestry", user_id=user_id):
        record_context(user_id, turn.turn_id, session_id, cleared=cleared, parts=recorded)
    return built, tools
