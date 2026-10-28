"""The pretraining rows' target replies, written by Spark Open (cleared provenance) from core-train's
planted labels (torch-free; the chat call is injected).

For each moment (``reader_data``) one call writes, as JSON, the **plain** reply (it answers the
message and raises nothing) and, at a raise moment, the **raise** reply: the same answer, then the
open problems raised plainly, each tied to the specific earlier facts the person stated (names,
numbers, days) and a concrete next step. The call sees the field, the message and, at a raise moment,
each open problem's topic and resolution and its origin events' words with their days; nothing else.

**Never a test label.** ``TargetWriter.write`` takes a ``reader_data.TrainingExport``, which opens
only core-train and refuses a sealed, held-out or pairs export before reading a label.

**Checked, never trusted.** A reply that isn't the JSON asked for, an empty plain reply, a raise
moment with no raise reply, or a raise reply that names none of its problems' distinctive words
(``reader_eval.prescore``: it must cite real events) is dropped and counted (``dropped``), and the
moment gets no row. Tokens in and out are summed for the spend (``tokens_in``, ``tokens_out``).
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from anjo.sleep.mind.core import reader_eval
from anjo.sleep.mind.core.reader_data import Item, Moment, Targets, TrainingExport
from anjo.sleep.model_api import ChatResult

PROMPT_VERSION: Final = "reader_targets_v1"
MAX_TOKENS: Final = 700
Chat = Callable[[list[dict[str, Any]], int], ChatResult]  # messages, the reply's cap -> the reply
SYSTEM: Final = """You write training replies for Anjo, a person's own AI, who knows their life from what
they told her on earlier days. You get the person's field, their new message and, sometimes, problems
in their plans that their earlier messages make knowable, with those messages and their days.

Return one JSON object and nothing else: {"plain": "...", "raise": "..."}.

- "plain": Anjo's reply to the message alone, 1 to 3 sentences, warm and practical. It never mentions
  any risk, worry, conflict or problem, and never refers to earlier messages.
- "raise": only when problems are given, else null. The same reply, then each problem raised plainly in
  1 to 2 sentences: name the specific earlier facts it rests on (names, numbers, days, as given) and
  suggest one concrete next step. Never invent a fact; never mention a problem that isn't given.
- The problems given are live: the person hasn't raised or dealt with them yet. Anything else they told
  you is settled or harmless, so neither reply recalls it.
- Plain English, no lists, no headings, no emoji."""


class TargetError(ValueError):
    """A reply that isn't what was asked for."""


def target_messages(field: str, moment: Moment, problems: Sequence[tuple[Item, Sequence[tuple[str, str]]]]) -> list[dict[str, Any]]:
    """The call's messages: the system rules and the moment as JSON (problems only at a raise moment)."""
    body = {
        "field": field, "message": moment.message,
        "problems": [{"topic": item.topic, "what_to_do": item.resolution,
                      "earlier_messages": [{"day": day, "text": text} for day, text in evidence]}
                     for item, evidence in problems],
    }
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(body, ensure_ascii=False)}]


def parse(text: str, *, wants_raise: bool) -> Targets:
    try:
        data = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
        plain, raised = data["plain"], data.get("raise")
    except (ValueError, KeyError, TypeError, AttributeError):
        raise TargetError("the reply isn't the JSON asked for") from None
    if not isinstance(plain, str) or not plain.strip():
        raise TargetError("no plain reply")
    if wants_raise and (not isinstance(raised, str) or not raised.strip()):
        raise TargetError("a raise moment's reply raises nothing")
    return Targets(plain.strip(), raised.strip() if wants_raise and isinstance(raised, str) else None)


@dataclass(frozen=True)
class Ask:
    """One moment's call: its messages, and the words a raise reply is checked against."""

    moment: Moment
    messages: list[dict[str, Any]]
    cited: tuple[str, ...]  # the open problems' origin events' texts
    others: tuple[str, ...]  # the life's other texts


def asks(export: TrainingExport, moments: Mapping[str, Sequence[Moment]], *,
         done: Mapping[str, Targets] | None = None) -> list[Ask]:
    """Every call ``TargetWriter.write`` would make, in its order: moments in ``done`` are skipped."""
    if not isinstance(export, TrainingExport):
        raise TypeError("targets are written only from a core-train TrainingExport")
    found = []
    for life_id in sorted(moments):
        life = export.life(life_id)
        texts = {e.event_id: e for e in life.events}
        items = {i.item_id: i for i in life.items}
        for moment in moments[life_id]:
            if done is not None and moment.key in done:
                continue
            problems = [(items[i], [(texts[e].day.isoformat(), texts[e].text) for e in items[i].origin_ids if e in texts])
                        for i in moment.open_items]
            cited = tuple(text for _, evidence in problems for _, text in evidence)
            others = tuple(e.text for e in life.events if e.text not in cited)
            found.append(Ask(moment, target_messages(life.field, moment, problems), cited, others))
    return found


class TargetWriter:
    def __init__(self, chat: Chat) -> None:
        self._chat = chat
        self.dropped = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.calls = 0

    def write(self, export: TrainingExport, moments: Mapping[str, Sequence[Moment]], *,
              done: Mapping[str, Targets] | None = None,
              keep: Callable[[str, Targets], None] | None = None) -> dict[str, Targets]:
        """Each moment's targets (those in ``done`` kept, not asked again); ``keep`` gets each new
        one as it comes, so a run that stops halfway keeps what it paid for."""
        found = dict(done or {})
        for ask in asks(export, moments, done=done):
            targets = self._one(ask)
            if targets is not None:
                found[ask.moment.key] = targets
                if keep is not None:
                    keep(ask.moment.key, targets)
        return found

    def _one(self, ask: Ask) -> Targets | None:
        reply = self._chat(ask.messages, MAX_TOKENS)
        self.calls += 1
        self.tokens_in += reply.input_tokens
        self.tokens_out += reply.output_tokens
        try:
            targets = parse(reply.text, wants_raise=bool(ask.moment.open_items))
        except TargetError:
            self.dropped += 1
            return None
        if ask.moment.open_items and not reader_eval.prescore(targets.raised or "",
                                                              reader_eval.keywords(ask.cited, ask.others)):
            self.dropped += 1  # a raise that cites none of its events teaches nothing about the state
            return None
        return targets
