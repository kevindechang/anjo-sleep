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

**Served turn (context C).** The prompt the route built, whose blocks carry Anjo's memory, plus,
as fenced optional context: the training brief when a source gives one, recall from the person's
evidence, and their explicit preferences and corrections. Its ancestry is unknown, because Anjo's
memory has no recorded source.

**Cleared turn.** In a cleared session (Task 2) nothing of Anjo's memory, homework or notes reaches
the model. The instructions are Anjo's fixed persona text and medical boundary; the context is
the brief, cleared recall (items that may train, homework left out) and the person's own
preferences and corrections; the history is the recent turns of their cleared sessions only.
No exemplars, no voice plan and no tools.

Each turn records its context's ancestry (``ancestry.record_context``).
"""
from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, Final, Protocol

from anjo.agent.records import TurnFacts
from anjo.core.anjo_model.pin import HostPin
from anjo.core.anjo_model.turn_input import AnjoTurnInput, Exchange, Segment, exchanges_from_messages
from anjo.core.best_effort import best_effort
from anjo.core.crisis import NO_CRISIS
from anjo.core.retrieved_context import RetrievedContext
from anjo.sleep.app.ancestry import ContextPart, record_context, reply_ancestries, union
from anjo.sleep.app.cleared_sessions import cleared_sessions, is_synthetic
from anjo.sleep.app.flags import sleep_enabled_for
from anjo.sleep.app.runtime import GlimmerPin
from anjo.sleep.app.storage import evidence_store
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.provenance import may_train

RECALL_K: Final = 6
RECALL_BUDGET_TOKENS: Final = 1200
CLEARED_HISTORY_DAYS: Final = 14
CLEARED_HISTORY_ROWS: Final = 24
CHARS_PER_TOKEN: Final = 4
MEMORY_SOURCE: Final = "anjo_memory"
# A synthetic person's words are Spark Open's (Task 2's simulator plays them).
_SIMULATED: Final = frozenset({SPARK_OPEN})
_SOURCE_CHARS = re.compile(r"[^A-Za-z0-9._:/%+#~-]")

ToolDefs = Sequence[Mapping[str, Any]]


class SleepTurnRefused(RuntimeError):
    """A turn the sleep runtime must not answer. Raised before any model call, so the chat
    route's fallback answers it."""


class BriefSource(Protocol):
    """The person's served training brief (Task 2), with its ancestry, or None before one."""

    def __call__(self, user_id: str) -> ContextPart | None: ...


_brief: list[BriefSource] = []


def set_brief_source(source: BriefSource | None) -> None:
    _brief[:] = [] if source is None else [source]


def _tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN + 1


def _recall(user_id: str, query: str, *, cleared: bool) -> list[ContextPart]:
    items = evidence_store(user_id).search(query, datetime.now(timezone.utc), RECALL_K)
    if cleared:
        items = [
            item for item in items
            if item.kind != "homework" and may_train(item.ancestry, hosted_in_training=False)
        ]
    parts: list[ContextPart] = []
    budget = RECALL_BUDGET_TOKENS
    for item in items:
        budget -= _tokens(item.text)
        if budget < 0:
            break
        parts.append(ContextPart(f"evidence:{item.id}", item.text, item.ancestry))
    return parts


def _own_words(user_id: str, session_id: str | None) -> list[ContextPart]:
    """Explicit preferences that apply now, and stored corrections, in the person's own words."""
    from anjo.core.personal_preferences import list_explicit_preferences
    from anjo.sleep.app.history_source import own_imports

    own = _SIMULATED if is_synthetic(user_id) else frozenset()
    parts: list[ContextPart] = []
    for preference in list_explicit_preferences(user_id, session_id=session_id):
        if not preference.get("applies_now"):
            continue
        quote = str(preference.get("receipt") or "").strip()
        text = f"{preference['field']}: {preference['value']}" + (f' ("{quote}")' if quote else "")
        parts.append(ContextPart(f"preference:{preference['preference_id']}", text, own if quote else None))
    for imported in own_imports(user_id):
        if imported.kind == "correction":
            ancestry = None if imported.ancestry is None else imported.ancestry | own
            parts.append(ContextPart(imported.source, imported.text, ancestry))
    return parts


def context_parts(user_id: str, query: str, session_id: str | None, *, cleared: bool) -> list[ContextPart]:
    brief = _brief[0](user_id) if _brief else None
    return [
        *([brief] if brief is not None else []),
        *_recall(user_id, query, cleared=cleared),
        *_own_words(user_id, session_id),
    ]


def _retrieved(parts: Sequence[ContextPart]) -> tuple[RetrievedContext, ...]:
    return tuple(
        RetrievedContext(
            source_ref=_SOURCE_CHARS.sub("-", part.source)[:256], revision=0,
            rank=len(parts) - index, token_cost=_tokens(part.text), text=part.text,
        )
        for index, part in enumerate(parts) if part.text.strip()
    )


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
    return exchanges, ContextPart("cleared_history", "", union(of(row) for row in rows))


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
    parts = context_parts(user_id, turn.user_text, session_id, cleared=cleared)
    if cleared:
        history, history_part = _cleared_history(user_id, sessions)
        built = dataclasses.replace(
            turn, instructions=_persona(), corrections=(), optional=_retrieved(parts), carry=None,
            exemplars=(), history=history, voice_plan=None, tools=(), tool_steps=(),
        )
        recorded = [*parts, history_part]
        tools = ()
    else:
        built = dataclasses.replace(turn, optional=(*turn.optional, *_retrieved(parts)))
        recorded = [ContextPart(MEMORY_SOURCE, "", None), *parts]
    with best_effort("record a sleep turn's context ancestry", user_id=user_id):
        record_context(user_id, turn.turn_id, session_id, cleared=cleared, parts=recorded)
    return built, tools
