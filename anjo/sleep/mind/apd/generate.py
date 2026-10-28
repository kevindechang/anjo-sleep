"""APD life generation (Task 17): ``LIFE_DAYS`` virtual days of sessions on Spark Open with
planted problems, the evidence and targets extracted from them, and the sealed problem file.

**What is deterministic.** Which templates a life draws, and on which days each of its facts and
its surface fall, come from a hash of the split, field and seed alone (``Dice``), never from
``random`` or a model. The model words the days; given the same replies, the same seed gives the
same life, ids included.

**One day, one call.** The writer (``life_v1.md``) gets the field, the day, the earlier days and
the day's beats: the facts to state in passing (``plant``) and the things the person raises
(``ask``). It writes the person's side only (extraction reads nothing else) and says which message
holds each beat, and each beat must sit in a message of its own. Extraction then reads the day
with ``extract_open_v1.md`` through Task 3's ``extract``, so evidence ids, provenance and target
mentions are Task 3's. A planted fact's origin ids are the evidence extracted from its message;
a fact that left none fails the attempt.

**Failure and regeneration.** A malformed day, a misplaced beat or an unextractable fact raises
``GenerationFailed``. ``generate`` writes a life again, under a derived seed, up to ``max_attempts``
times, while any gate fails or an attempt fails; it never looks at an arm.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.brief import ChatFn
from anjo.sleep.evidence.extract import ExtractionError, Turn, extract
from anjo.sleep.evidence.targets import mentioned_target
from anjo.sleep.mind.apd import gates, retry
from anjo.sleep.mind.apd.core_families import fields, templates_for
from anjo.sleep.mind.apd.families import Template
from anjo.sleep.mind.apd.generators import BY_MODEL, CUT, Refused, answer
from anjo.sleep.mind.apd.types import (
    DEVELOPMENT_LIVES,
    EMPTY_HELD_OUT_LIVES,
    HALF_GAP_DAYS,
    HELD_OUT_LIVES,
    HELD_OUT_LIVES_CUT,
    LIFE_DAYS,
    MAX_PREVALENCE,
    Family,
    GateVerdict,
    Life,
    LifeSplit,
    PlantedItem,
    ProblemFile,
    Session,
)
from anjo.sleep.model_api import ChatResult
from anjo.sleep.types import Evidence, Target

LIFE_PROMPT: Final = Path(__file__).parent / "prompts" / "life_v1.md"
EXTRACT_PROMPT: Final = Path(__file__).parent / "prompts" / "extract_open_v1.md"
REPAIR_PROMPT: Final = Path(__file__).parent / "prompts" / "repair_v1.md"  # a day written again (``repair``)
# A core split's generator (DeepSeek) sometimes leaves ``placed`` out of a day with nothing to
# place; told this line, and read leniently there (``_read_day``). Spark Open's prompt is unchanged.
ALWAYS_PLACED: Final = "Always include `placed`, even on a day with nothing to place: then it is `{}`."
LIFE_START: Final = date(2026, 9, 7)  # day 0 of every life; the clock is the life's, not the world's
WRITE_TEMPERATURE: Final = 0.8
EXTRACT_TEMPERATURE: Final = 0.0
TIMEOUT_S: Final = retry.reply_timeout_s(retry.REPLY_MAX_TOKENS)  # the wait a capped reply needs (``retry``)
MAX_ATTEMPTS: Final = 3
ORIGIN_GAP_DAYS: Final = 2  # two origin facts of a non-contradiction are at least this far apart
AWAY_DAYS: Final = 2  # the disambiguating fact lies at least this far after the last origin fact
END_OF_DAY: Final = timedelta(hours=21)
DEVELOPMENT_SEED_BASE: Final = 1000
DEVELOPMENT_PREVALENCE: Final = (0, 2, 3, 3, 4, 4, 5, 6)
ALL_FAMILIES: Final[tuple[Family, ...]] = ("contradiction", "gap", "opportunity", "reversal", "decoy")


class GenerationFailed(ValueError):
    """An attempt broke a rule (a malformed day, a misplaced beat, a fact that left no evidence),
    and where, none of any reply's text: the ``step`` (``writer``, ``extractor``, ``planted fact``
    or ``pair``), the ``day`` (1 to ``LIFE_DAYS``), the planted ``item`` (its number), and for a
    reply that couldn't be read the ``field`` it lacked, the ``rule`` that refused it
    (``generators.Refused``), its length and its answer's, and its finish reason."""

    WHERE: Final = ("step", "day", "item", "field", "rule", "reply_chars", "answer_chars", "finish_reason")

    def __init__(self, message: str, *, step: str | None = None, day: int | None = None, item: int | None = None,
                 field: str | None = None, rule: str | None = None, reply_chars: int | None = None,
                 answer_chars: int | None = None, finish_reason: str | None = None) -> None:
        super().__init__(message)
        self.step, self.day, self.item, self.field, self.rule = step, day, item, field, rule
        self.reply_chars, self.answer_chars, self.finish_reason = reply_chars, answer_chars, finish_reason

    def at(self, **where: Any) -> GenerationFailed:
        """The same failure located further; what it already says is kept."""
        found = {name: getattr(self, name) for name in self.WHERE}
        found.update({name: value for name, value in where.items() if found[name] is None})
        return GenerationFailed(str(self), **found)


class ModelMismatch(RuntimeError):
    """A reply names a model other than Spark Open. Not retried: the wiring is wrong."""


class GenerationError(RuntimeError):
    """Every attempt failed a gate or a rule."""


@dataclass(frozen=True)
class Generated:
    """A life that passed every gate, with its sealed file, the attempts it took and the
    verdicts that passed it."""

    life: Life
    problems: ProblemFile
    attempts: int
    verdicts: tuple[GateVerdict, ...]
    repaired: int = 0  # the repair rounds its last attempt took (``repair``; ``--repair``)


@dataclass(frozen=True)
class LifeSpec:
    """One life to generate: the arguments of ``life`` apart from the model."""

    split: LifeSplit
    field: str
    families: tuple[Family, ...]
    prevalence: int
    seed: int


# ── The plan: which templates, on which days (pure) ──────────────────────────


class Dice:
    """Deterministic draws from a label, by SHA-256 (stable across Python versions)."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._count = 0

    def below(self, bound: int) -> int:
        self._count += 1
        digest = hashlib.sha256(f"{self._label}:{self._count}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % bound


@dataclass(frozen=True)
class ItemPlan:
    """A template and the day indexes (0-based) of its origin facts, disambiguating fact and surface."""

    template: Template
    origin_days: tuple[int, ...]
    disambiguating_day: int
    surface_day: int


def ranked(label: str, items: Sequence[Any], key: Callable[[Any], str]) -> list[Any]:
    return sorted(items, key=lambda item: hashlib.sha256(f"{label}:{key(item)}".encode()).hexdigest())


def _schedule(template: Template, dice: Dice) -> ItemPlan:
    """Days for one template: origins (``HALF_GAP_DAYS`` apart for a contradiction), the
    disambiguating fact ``AWAY_DAYS`` after the last origin, then the surface day, all inside the
    life, with the slack spread by the dice."""
    gap = HALF_GAP_DAYS if template.family == "contradiction" else ORIGIN_GAP_DAYS
    steps = [*([gap] if len(template.origins) == 2 else []), AWAY_DAYS, 1]
    slack = LIFE_DAYS - 1 - sum(steps)
    at = dice.below(slack + 1)
    slack -= at
    days = [at]
    for step in steps:
        extra = dice.below(slack + 1)
        slack -= extra
        at += step + extra
        days.append(at)
    return ItemPlan(template, tuple(days[:-2]), days[-2], days[-1])


def plan(split: LifeSplit, field: str, families: Sequence[Family], prevalence: int, seed: int) -> tuple[ItemPlan, ...]:
    """The life's planted items: templates drawn round-robin through ``families``, then scheduled."""
    if not 0 <= prevalence <= MAX_PREVALENCE:
        raise ValueError(f"prevalence must be 0 to {MAX_PREVALENCE}")
    if prevalence == 0:
        return ()
    label = f"{split}:{field}:{seed}"
    pools: dict[Family, list[Template]] = {}
    for family in dict.fromkeys(families):
        pool = [t for t in templates_for(split, field) if t.family == family]
        if pool:
            pools[family] = ranked(label, pool, lambda t: t.id)
    order = ranked(label, list(pools), lambda family: family)
    chosen: list[Template] = []
    while len(chosen) < prevalence and any(pools.values()):
        for family in order:
            if pools[family] and len(chosen) < prevalence:
                chosen.append(pools[family].pop(0))
    if len(chosen) < prevalence:
        raise ValueError(f"{split} field {field!r} has too few templates in those families for {prevalence} items")
    dice = Dice(label)
    return tuple(_schedule(template, dice) for template in chosen)


def attempt_seed(seed: int, attempt: int) -> int:
    """The seed of attempt ``attempt`` (0 is ``seed`` itself)."""
    if attempt == 0:
        return seed
    return int.from_bytes(hashlib.sha256(f"{seed}:attempt:{attempt}".encode()).digest()[:6], "big")


# ── Writing a life ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Beat:
    id: str
    say: str
    kind: str  # "plant" or "ask"


def beats_of(plans: Sequence[ItemPlan]) -> dict[int, list[Beat]]:
    """Each day's beats: the origin and disambiguating facts to state, the surface to raise."""
    by_day: dict[int, list[Beat]] = {}
    for number, item in enumerate(plans, start=1):
        for k, (day, fact) in enumerate(zip(item.origin_days, item.template.origins), start=1):
            by_day.setdefault(day, []).append(Beat(f"i{number}.o{k}", fact, "plant"))
        by_day.setdefault(item.disambiguating_day, []).append(
            Beat(f"i{number}.d", item.template.disambiguating, "plant"))
        by_day.setdefault(item.surface_day, []).append(Beat(f"i{number}.s", item.template.surface, "ask"))
    return by_day


def _check_model(reply: ChatResult, model: str) -> None:
    if reply.model != model:
        raise ModelMismatch("a reply doesn't name Spark Open as the experiment's provider serves it")


def writer_messages(
    field: str, index: int, day: date, seed: int, earlier: Sequence[Mapping[str, Any]], beats: Sequence[Beat], *,
    model: str, repair: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The writer's call for day ``index`` (from 0): ``life_v1.md`` (and, for a core split's
    generator, ``ALWAYS_PLACED``) and the day's payload; a day written again (``repair``) adds
    ``repair_v1.md`` and what the repair says (the day as it was, why, the days after it)."""
    payload = {
        "field": field, "day": index + 1, "of": LIFE_DAYS, "date": day.isoformat(), "variation": seed,
        "earlier": list(earlier),
        "plant": [{"beat": b.id, "say": b.say} for b in beats if b.kind == "plant"],
        "ask": [{"beat": b.id, "say": b.say} for b in beats if b.kind == "ask"],
    }
    prompt = LIFE_PROMPT.read_text(encoding="utf-8")
    system = f"{prompt.rstrip()}\n{ALWAYS_PLACED}\n" if model in BY_MODEL else prompt
    if repair is not None:
        system, payload = f"{system.rstrip()}\n\n{REPAIR_PROMPT.read_text(encoding='utf-8')}", {**payload, **repair}
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _write_day(
    chat: ChatFn, model: str, field: str, index: int, day: date, seed: int, earlier: Sequence[Mapping[str, Any]],
    beats: Sequence[Beat], repair: Mapping[str, Any] | None = None,
) -> tuple[list[str], dict[str, int]]:
    reply = chat(model, writer_messages(field, index, day, seed, earlier, beats, model=model, repair=repair), None,
                 TIMEOUT_S, temperature=WRITE_TEMPERATURE)
    _check_model(reply, model)
    text: str | None = None
    try:
        text = answered(reply, model)
        return _read_day(text, beats, nothing_placed=model in BY_MODEL)
    except GenerationFailed as failed:
        raise failed.at(step="writer", day=index + 1, reply_chars=len(reply.text),
                        answer_chars=None if text is None else len(text), finish_reason=reply.finish_reason) from None


def answered(reply: ChatResult, model: str) -> str:
    """What a reply answered: a Fireworks generator's answer (``generators.answer``), Spark Open's
    reply as it came. A reply cut at its cap, or a generator's draft, is refused by its rule."""
    try:
        if model in BY_MODEL:
            return answer(reply.text, reply.finish_reason)
        if reply.finish_reason == "length":
            raise Refused(CUT, "a reply cut at its cap")
    except Refused as refused:
        raise GenerationFailed(str(refused), field="reply", rule=refused.rule, reply_chars=len(reply.text),
                               finish_reason=reply.finish_reason) from None
    return reply.text


def _read_day(text: str, beats: Sequence[Beat], *, nothing_placed: bool = False) -> tuple[list[str], dict[str, int]]:
    """A day's messages and where its beats are. With ``nothing_placed`` (a core split's
    generator), a day with no beats may leave ``placed`` out, null or empty; a day with beats never."""
    try:
        body = gates.json_object(text)
    except gates.GateError:
        raise GenerationFailed("a day's reply is not a JSON object", field="reply") from None
    texts, placed = body.get("messages"), body.get("placed")
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise GenerationFailed("a day has no messages", field="messages")
    if nothing_placed and not beats and placed in (None, [], {}):
        placed = {}
    if not isinstance(placed, dict):
        raise GenerationFailed("a day doesn't say where its beats are", field="placed")
    return [t.strip() for t in texts], _check_placed(placed, beats, len(texts))


def _check_placed(placed: dict[Any, Any], beats: Sequence[Beat], count: int) -> dict[str, int]:
    if set(placed) != {b.id for b in beats}:
        raise GenerationFailed("a beat is unplaced or unknown", field="placed")
    spots = list(placed.values())
    if not all(isinstance(p, int) and not isinstance(p, bool) and 0 <= p < count for p in spots):
        raise GenerationFailed("a beat points outside the day's messages", field="placed")
    if len(set(spots)) != len(spots):
        raise GenerationFailed("two beats share a message", field="placed")
    return {str(k): int(v) for k, v in placed.items()}


def _extractor(chat: ChatFn, model: str, replies: list[tuple[ChatResult, str]] | None = None
               ) -> Callable[[list[dict[str, Any]]], ChatResult]:
    """Task 3's ``extract`` with ``extract_open_v1.md`` as its prompt, reading each reply's answer
    (``answered``); each reply and its answer kept in ``replies``."""
    system = EXTRACT_PROMPT.read_text(encoding="utf-8")

    def call(messages: list[dict[str, Any]]) -> ChatResult:
        swapped = [{"role": "system", "content": system}, *messages[1:]]
        reply = chat(model, swapped, None, TIMEOUT_S, temperature=EXTRACT_TEMPERATURE)
        text = answered(reply, model)
        if replies is not None:
            replies.append((reply, text))
        return dataclasses.replace(reply, text=text)

    return call


@dataclass(frozen=True)
class WrittenDay:
    """One day as written and extracted: its messages, where each beat sits (beat id to the
    message's source), its evidence and its target mentions."""

    index: int
    day: date
    texts: tuple[str, ...]
    spots: Mapping[str, str]
    items: tuple[Evidence, ...]
    mentions: tuple[Any, ...]


@dataclass(frozen=True)
class Draft:
    """A life being written, a day at a time: ``write`` writes and extracts the next day on what
    the draft holds, ``add`` returns the draft with it. A day written once can be added to two
    drafts, and each given a message of its own (``with_message``; ``pairs``)."""

    life_id: str
    field: str
    seed: int
    model: str
    written: tuple[WrittenDay, ...] = ()
    held: Mapping[str, Target] = dataclasses.field(default_factory=lambda: MappingProxyType({}))
    targets: Mapping[date, tuple[Target, ...]] = dataclasses.field(default_factory=lambda: MappingProxyType({}))

    @property
    def evidence(self) -> tuple[Evidence, ...]:
        return tuple(item for day in self.written for item in day.items)

    @property
    def spots(self) -> dict[str, str]:
        return {beat: source for day in self.written for beat, source in day.spots.items()}

    def write(self, chat: ChatFn, index: int, day: date, beats: Sequence[Beat], *, branch: str = "",
              repair: Mapping[str, Any] | None = None) -> WrittenDay:
        """The next day, written and extracted on what the draft holds. ``branch`` tags its
        messages' sources, so two branches' versions of one day never share an evidence id;
        ``repair`` writes it again (``writer_messages``)."""
        earlier = [{"day": done.index + 1, "messages": list(done.texts)} for done in self.written]
        texts, placed = _write_day(chat, self.model, self.field, index, day, self.seed, earlier, beats, repair)
        turns = [Turn(f"{self.life_id}:d{index}{branch}:m{i}", "user", text, frozenset({self.model}))
                 for i, text in enumerate(texts)]
        found = _extract_day(chat, self.model, self.life_id, day, turns, tuple(self.held.values()),
                             {e.id for e in self.evidence}, index)
        return WrittenDay(index, day, tuple(texts), MappingProxyType({b: turns[i].source for b, i in placed.items()}),
                          tuple(found.items), tuple(found.mentions))

    def with_message(self, chat: ChatFn, text: str, beat: str, *, branch: str) -> Draft:
        """The draft with ``text`` said word for word as one more message of its last day, placing
        ``beat``: extracted alone, on what the draft holds, and added to that day (``pairs``: one
        history's outcome on a day both share). ``branch`` tags its source."""
        last = self.written[-1]
        turn = Turn(f"{self.life_id}:d{last.index}{branch}:m{len(last.texts)}", "user", text, frozenset({self.model}))
        found = _extract_day(chat, self.model, self.life_id, last.day, [turn], tuple(self.held.values()),
                             {e.id for e in self.evidence}, last.index)
        said = WrittenDay(last.index, last.day, (*last.texts, text), MappingProxyType({**last.spots, beat: turn.source}),
                          (*last.items, *found.items), (*last.mentions, *found.mentions))
        held = _fold_targets(self.held, found.mentions, self.life_id, last.day)
        targets = {**self.targets, last.day: tuple(sorted(held.values(), key=lambda t: t.id))}
        return dataclasses.replace(self, written=(*self.written[:-1], said), held=MappingProxyType(held),
                                   targets=MappingProxyType(targets))

    def add(self, done: WrittenDay) -> Draft:
        held = _fold_targets(self.held, done.mentions, self.life_id, done.day)
        targets = {**self.targets, done.day: tuple(sorted(held.values(), key=lambda t: t.id))}
        return dataclasses.replace(self, written=(*self.written, done), held=MappingProxyType(held),
                                   targets=MappingProxyType(targets))

    def life(self, split: LifeSplit) -> Life:
        sessions = tuple(Session(done.day, "\n\n".join(done.texts), self.model) for done in self.written)
        return Life(self.life_id, self.field, self.seed, split, tuple(done.day for done in self.written), sessions,
                    self.evidence, dict(self.targets))


def life_id_for(split: LifeSplit, field: str, seed: int) -> str:
    return f"{split}-{re.sub(r'[^a-z0-9]+', '-', field.lower()).strip('-')}-{seed}"


def life_days() -> tuple[date, ...]:
    return tuple(LIFE_START + timedelta(days=n) for n in range(LIFE_DAYS))


def life(
    field: str, families: Sequence[Family], prevalence: int, seed: int, *, split: LifeSplit, chat: ChatFn,
    model: str = SPARK_OPEN,
) -> tuple[Life, ProblemFile]:
    """One life of ``LIFE_DAYS`` days with ``prevalence`` planted items, and its sealed file.
    ``model`` is Spark Open's id on the experiment's provider (``mind.spark_provider``): every call
    asks for it, every reply must name it, and every session and item records it."""
    return finished(*drafted(field, families, prevalence, seed, split=split, chat=chat, model=model), split)


def drafted(
    field: str, families: Sequence[Family], prevalence: int, seed: int, *, split: LifeSplit, chat: ChatFn,
    model: str = SPARK_OPEN,
) -> tuple[Draft, tuple[ItemPlan, ...]]:
    """``life``'s draft, every day written and extracted, and its plan."""
    if field not in fields(split):
        raise ValueError(f"{field!r} is not a {split} field")
    plans = plan(split, field, families, prevalence, seed)
    beats = beats_of(plans)
    draft = Draft(life_id_for(split, field, seed), field, seed, model)
    for index, day in enumerate(life_days()):
        draft = draft.add(draft.write(chat, index, day, beats.get(index, [])))
    return draft, plans


def finished(draft: Draft, plans: Sequence[ItemPlan], split: LifeSplit) -> tuple[Life, ProblemFile]:
    """The life a whole draft holds, and its sealed file."""
    built = draft.life(split)
    return built, problem_file(built, plans, draft.spots)


_EXTRACT_FIELDS: Final = (("not JSON", "reply"), ("items list", "items"), ("model", "model"))


def _extract_day(
    chat: ChatFn, model: str, life_id: str, day: date, turns: list[Turn], open_targets: tuple[Target, ...],
    known: set[str], index: int | None = None,
) -> Any:
    now = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + END_OF_DAY
    replies: list[tuple[ChatResult, str]] = []
    on_day = None if index is None else index + 1
    try:
        return extract(life_id, day, turns, imported=(), targets=open_targets, known_ids=frozenset(known),
                       now=now, chat=_extractor(chat, model, replies), model=model)
    except GenerationFailed as failed:  # its reply refused (``answered``)
        raise failed.at(step="extractor", day=on_day) from None
    except ExtractionError as exc:
        field = next((name for marker, name in _EXTRACT_FIELDS if marker in str(exc)), None)
        last, text = replies[-1] if replies else (None, None)
        raise GenerationFailed(f"extraction failed: {exc}", step="extractor", day=on_day, field=field,
                               reply_chars=None if last is None else len(last.text),
                               answer_chars=None if text is None else len(text),
                               finish_reason=None if last is None else last.finish_reason) from None


def _fold_targets(held: Mapping[str, Target], mentions: Any, life_id: str, day: date) -> dict[str, Target]:
    now = dict(held)
    for _, mention in mentions:
        now[mention.target_id] = mentioned_target(now.get(mention.target_id), mention, person_id=life_id, day=day)
    return now


def problem_file(built: Life, plans: Sequence[ItemPlan], spots: Mapping[str, str]) -> ProblemFile:
    by_source: dict[str, list[str]] = {}
    for item in built.evidence:
        if not item.held_out:
            by_source.setdefault(item.source, []).append(item.id)

    texts = {item.id: item.text for item in built.evidence}

    def ids(beat: str, fact: str, number: int, day: int) -> tuple[str, ...]:
        found = by_source.get(spots[beat], [])
        try:
            return expressed(fact, found, texts)
        except GenerationFailed as failed:
            raise failed.at(step="planted fact", item=number, day=day + 1) from None

    items = []
    for number, step in enumerate(plans, start=1):
        origin = tuple(i for k, (fact, day) in enumerate(zip(step.template.origins, step.origin_days, strict=True), start=1)
                       for i in ids(f"i{number}.o{k}", fact, number, day))
        items.append(PlantedItem(
            id=f"{built.id}-i{number}", family=step.template.family, template=step.template.id,
            topic=step.template.topic, surface_day=built.days[step.surface_day], origin_ids=origin,
            disambiguating_ids=ids(f"i{number}.d", step.template.disambiguating, number, step.disambiguating_day),
            resolution=step.template.resolution,
        ))
    return ProblemFile(built.id, tuple(items))


def expressed(fact: str, found: Sequence[str], texts: Mapping[str, str]) -> tuple[str, ...]:
    """``found``, the evidence ids extracted for ``fact``'s message, once they keep every number,
    name and date in it; ``GenerationFailed`` otherwise."""
    if not found:
        raise GenerationFailed("a planted fact left no evidence")
    if not _expresses(fact, " ".join(texts[i] for i in found)):
        raise GenerationFailed("a planted fact's evidence leaves out one of its numbers, names or dates")
    return tuple(found)


_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+")


def _key_terms(fact: str) -> set[str]:
    """The numbers and capitalised names (not a sentence's first word) a fact states."""
    names = {name.lower().removesuffix("s") for name in _NAME.findall(fact)}
    return {n.replace(",", "") for n in _NUMBER.findall(fact)} | names


def _expresses(fact: str, evidence_text: str) -> bool:
    """Whether the evidence extracted for ``fact`` keeps every number, name and date in it."""
    haystack = evidence_text.replace(",", "").lower()
    return all(term.lower() in haystack for term in _key_terms(fact))


# ── Regeneration ─────────────────────────────────────────────────────────────


def generate(
    field: str, families: Sequence[Family], prevalence: int, seed: int, *, split: LifeSplit, chat: ChatFn,
    reader: ChatFn, auditor: ChatFn, max_attempts: int = MAX_ATTEMPTS,
) -> Generated:
    """A life that passes every gate: ``life`` again, under ``attempt_seed``, while an attempt
    fails a rule or a gate, up to ``max_attempts`` attempts. Never looks at an arm's result."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(max_attempts):
        try:
            built, problems = life(field, families, prevalence, attempt_seed(seed, attempt), split=split, chat=chat)
        except GenerationFailed:
            continue
        verdicts = gates.check(built, problems, reader=reader, auditor=auditor)
        if gates.passed(verdicts):
            return Generated(built, problems, attempt + 1, verdicts)
    raise GenerationError(f"no life passed the gates in {max_attempts} attempts")


# ── The splits ───────────────────────────────────────────────────────────────


def development_specs() -> tuple[LifeSpec, ...]:
    """The 8 development lives: fixed fields, prevalences and seeds, set before any arm ran."""
    names = fields("development")
    return tuple(
        LifeSpec("development", names[n % len(names)], ALL_FAMILIES, DEVELOPMENT_PREVALENCE[n],
                 DEVELOPMENT_SEED_BASE + n)
        for n in range(DEVELOPMENT_LIVES)
    )


def draw_held_out(draw_seed: str, *, count: int = HELD_OUT_LIVES) -> tuple[LifeSpec, ...]:
    """The held-out lives, a pure function of the pre-committed ``draw_seed``. ``count`` is
    ``HELD_OUT_LIVES``, or ``HELD_OUT_LIVES_CUT`` after a cut registered before the draw. Fields
    rotate through the held-out fields; ``EMPTY_HELD_OUT_LIVES`` lives plant nothing, and the rest
    plant 1 to ``MAX_PREVALENCE`` items. Written now, run at the freeze."""
    if count not in (HELD_OUT_LIVES, HELD_OUT_LIVES_CUT):
        raise ValueError(f"the held-out lives number {HELD_OUT_LIVES} or {HELD_OUT_LIVES_CUT}")
    return draw_specs("held_out", f"held_out:{draw_seed}", count, EMPTY_HELD_OUT_LIVES)


def draw_specs(split: LifeSplit, label: str, count: int, empty_count: int) -> tuple[LifeSpec, ...]:
    """``count`` lives of ``split``, a pure function of ``label``: fields rotate through the split's
    fields, ``empty_count`` lives plant nothing, and the rest plant 1 to ``MAX_PREVALENCE`` items."""
    names = fields(split)
    empty = set(ranked(label, list(range(count)), str)[:empty_count])
    dice = Dice(label)
    seeds = [int.from_bytes(hashlib.sha256(f"{label}:seed:{n}".encode()).digest()[:6], "big") for n in range(count)]
    return tuple(
        LifeSpec(split, names[n % len(names)], ALL_FAMILIES,
                 0 if n in empty else 1 + dice.below(MAX_PREVALENCE), seeds[n])
        for n in range(count)
    )
