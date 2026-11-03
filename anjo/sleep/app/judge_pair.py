"""The wake's before and after on a judge's night: the same question asked of base Glimmer and of
tonight's adapter, asked inside the night and saved once it is over (``judge_cohort.save_before_after``).

**Personal first.** The question is made from what this judge said, by a template, never by a model:
her chooser's target from the latest live pass (the pick, else the strongest want it weighed), as
"Thinking about <target>: what should I focus on next?"; else the judge's own most recent
substantive message in the night's window, in their words. Both replies come from the night's own
serving client as the collapse gate hands over the adapter it checked (``GatePairs``, called inside
``CollapseGate.check``): so inside the gates step, while the batch's GPU reservation is open, and
their serving time is in the gates' cost part, the night's cost and the judge's settlement. Base
Glimmer with no adapter, then tonight's adapter by name, each at most ``REPLY_TOKENS``. Saved with
``question_source`` ``personal``.

**Else the gate's everyday pair.** When there is no question, or serving can't answer either side,
the collapse gate's own pair is kept (``GatePairs``: base Glimmer and the adapter on one of the
gate's everyday prompts, the one scored most in the adapter's favour, else the first), with
``question_source`` ``gate_everyday``.

Either way it is saved whether the gate promoted the adapter or not, and says which. A night whose
collapse gate didn't run (skipped, or never reached) asks nothing; a night that saves no pair clears
the last one, so an older night's pair is never read as its own.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Final, Protocol

from anjo.sleep.app.judge_cohort import clear_before_after, save_before_after
from anjo.sleep.gates import EVAL_SEED, TEMPERATURE
from anjo.sleep.gates.collapse import CollapsePair
from anjo.sleep.types import CheckpointRef, NightResult

BASE_MODEL: Final = "base Glimmer"  # what the base side answers with, in both kinds of pair
PERSONAL: Final = "personal"
GATE_EVERYDAY: Final = "gate_everyday"
REPLY_TOKENS: Final = 200  # each side's cap: two short generations on the warm server
MAX_TARGET: Final = 120
MAX_QUESTION: Final = 300
MIN_WORDS: Final = 5  # a message shorter than this ("ok thanks") isn't a question about them
TARGET_ASK: Final = "Thinking about {target}: what should I focus on next?"
ASK: Final = " What should I do next?"
logger = logging.getLogger(__name__)

Reply = Callable[[CheckpointRef | None, str], str]


@dataclass(frozen=True)
class PersonalPorts:
    """``question(user_id)``: tonight's personal question, or None; ``reply(adapter, question)``: one
    short reply (adapter None is base Glimmer), raising when serving can't answer."""

    question: Callable[[str], str | None]
    reply: Reply


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:")


def from_target(label: str | None) -> str | None:
    """A stored target label as a question the judge would recognise."""
    text = " ".join((label or "").split()).rstrip(".!?;:,")
    if not text:
        return None
    first = text.split(" ", 1)[0]
    if first[:1].isupper() and first[1:].islower() and first != "I":  # "Hackathon demo", not "NYC move"
        text = text[:1].lower() + text[1:]
    return TARGET_ASK.format(target=_cut(text, MAX_TARGET))


def from_message(message: str | None) -> str | None:
    """The judge's own message as the question, in their words; None when it isn't substantive."""
    text = " ".join((message or "").split())
    if len(text.split(" ")) < MIN_WORDS:
        return None
    text = _cut(text, MAX_QUESTION)
    if text.endswith("?"):
        return text
    return f"{text.rstrip('.!;:,')}.{ASK}"


def _target_ids(record: Mapping[str, Any] | None) -> list[str]:
    """Her chooser's targets on the latest live pass: the pick first, then the wants by strength."""
    choice = record.get("choice") if isinstance(record, Mapping) else None
    if not isinstance(choice, Mapping):
        return []
    rows = [r for r in choice.get("considered") or () if isinstance(r, Mapping) and r.get("target_id")]
    def strength(row: Mapping[str, Any]) -> float:
        value = row.get("strength")
        return float(value) if isinstance(value, int | float) else 0.0

    ranked = sorted(rows, key=lambda r: -strength(r))
    picked = [choice["target_id"]] if choice.get("target_id") else []
    return [str(t) for t in dict.fromkeys([*picked, *(r["target_id"] for r in ranked)])]


def personal_question(record: Callable[[], Mapping[str, Any] | None], label: Callable[[str], str | None],
                      messages: Callable[[], Sequence[str]]) -> Callable[[], str | None]:
    """Tonight's question: from her chooser's target, else the judge's latest substantive message."""
    def ask() -> str | None:
        for target_id in _target_ids(record()):
            question = from_target(label(target_id))
            if question:
                return question
        for message in reversed(list(messages())):
            question = from_message(message)
            if question:
                return question
        return None

    return ask


class Completer(Protocol):
    def complete(self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *, temperature: float,
                 seed: int | None = None, max_tokens: int | None = None) -> Any: ...


def client_reply(client: Completer) -> Reply:
    """One short reply from the night's serving client; an empty one counts as no answer."""
    def reply(adapter: CheckpointRef | None, question: str) -> str:
        text = str(client.complete(adapter, [{"role": "user", "content": question}], temperature=TEMPERATURE,
                                   seed=EVAL_SEED, max_tokens=REPLY_TOKENS).text).strip()
        if not text:
            raise ValueError("an empty reply")
        return text

    return reply


def default_ports(user_id: str, night: date, window: tuple[datetime, datetime]) -> PersonalPorts:
    """The app's own reads: the live pass's record, her target labels, the judge's turns in the night's
    window, and the serving client the night used, dated on the night."""
    from anjo.core.history import get_history_between
    from anjo.sleep.app.live_mind import load_record
    from anjo.sleep.app.storage import target_store
    from anjo.sleep.serving import backend

    def label(target_id: str) -> str | None:
        found = target_store(user_id).get(target_id)
        return found.label if found is not None else None

    def messages() -> list[str]:
        rows = get_history_between(user_id, *window)
        return [str(r["content"]) for r in rows if r.get("role") == "user"]

    ask = personal_question(lambda: load_record(user_id), label, messages)
    return PersonalPorts(question=lambda _uid: ask(), reply=client_reply(backend.from_env().client().on(night)))


def _personal(user_id: str, adapter: CheckpointRef,
              ports: PersonalPorts | Callable[[], PersonalPorts]) -> tuple[str, str, str] | None:
    from anjo.sleep.serving.client import ServingError

    try:
        ports = ports if isinstance(ports, PersonalPorts) else ports()  # the app's reads, built only now
        question = ports.question(user_id)
        if not question:
            return None
        return question, ports.reply(None, question), ports.reply(adapter, question)
    except (ServingError, OSError, TimeoutError, ValueError, LookupError) as error:
        logger.warning("tonight's personal before-and-after couldn't be asked (%s)", type(error).__name__)
        return None


class GatePairs:
    """What tonight's collapse gate hands over (inside its check, so inside the night's metered
    window): the adapter it checked, its paired replies, and, with ``ports``, the personal question
    asked of both sides right then. It never raises into the gate: the night stands either way."""

    def __init__(self, user_id: str | None = None, ports: PersonalPorts | Callable[[], PersonalPorts] | None = None) -> None:
        self._user_id, self._ports = user_id, ports
        self.adapter: CheckpointRef | None = None
        self.pairs: tuple[CollapsePair, ...] = ()
        self.asked: tuple[str, str, str] | None = None

    def __call__(self, ckpt: CheckpointRef, pairs: list[CollapsePair]) -> None:
        self.adapter, self.pairs = ckpt, tuple(pairs)
        if self._ports is None or self._user_id is None:
            return
        try:
            self.asked = _personal(self._user_id, ckpt, self._ports)
        except Exception as error:  # an unforeseen failure: the gate's verdict stands, the wake falls back
            logger.warning("tonight's personal before-and-after broke (%s)", type(error).__name__)
            self.asked = None

    def best(self) -> CollapsePair | None:
        """The pair most in the adapter's favour (base dragged, adapter not), else the first."""
        def score(pair: CollapsePair) -> int:
            return int(pair.base_dragged) - int(pair.dragged)

        return max(self.pairs, key=score) if self.pairs else None  # max keeps the first of equals


def save_pair(user_id: str, night: str, result: NightResult, pairs: GatePairs) -> None:
    """Tonight's before and after for the wake, from what the gate handed over (no call is made here);
    when there is none, the last night's pair is cleared."""
    gate = result.gates.get("collapse")
    adapter = pairs.adapter or result.checkpoint
    everyday = pairs.best()
    try:
        if gate is None or gate.skipped or adapter is None:
            clear_before_after(user_id)
        elif pairs.asked is not None:
            _save(user_id, night, result, adapter, pairs.asked, PERSONAL)
        elif everyday is not None:
            _save(user_id, night, result, adapter, (everyday.prompt, everyday.base_reply, everyday.reply), GATE_EVERYDAY)
        else:
            clear_before_after(user_id)
    except (OSError, ValueError) as error:  # the night stands; the wake shows only a pair from its own night
        logger.warning("tonight's before-and-after wasn't saved (%s)", type(error).__name__)


def _save(user_id: str, night: str, result: NightResult, adapter: CheckpointRef, asked: tuple[str, str, str],
          source: str) -> None:
    question, before, after = asked
    save_before_after(user_id, question=question, before=before, after=after, night=night, before_model=BASE_MODEL,
                      after_model=adapter.id, promoted=result.status == "promoted", question_source=source)
