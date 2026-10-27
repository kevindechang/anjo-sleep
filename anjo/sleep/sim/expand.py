"""Life expansion (Task 12a): one line of a life becomes a ``PersonFile`` on Spark Open.

``expand_life`` fills ``prompts/expand_v1.md`` with the life and its pair's shared taste items, asks
Spark Open through the ``ModelClient`` port (temperature 0, the seed given), and validates what
comes back strictly: the shared ``personas.model`` rules (counts, leaks, arrival days), plus the
rules of an expanded life (its registered line and plans stay, its pair's shared ``taste_ab`` items
appear unchanged with this life's label, new-work arrivals are spread over many days). Anything
that breaks a rule is refused with ``ExpandError``; nothing is repaired, cut down or filled in.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from anjo.sleep.lives.draw import Life, Pair, load_pairs
from anjo.sleep.personas.model import (
    MIN_HELD_OUT_PREFERENCES,
    MIN_NEW_WORK,
    NEW_WORK_FIRST_DAY,
    PLANNED_NIGHTS,
    PersonaError,
    PersonFile,
    person_file_from_dict,
    to_visible_dict,
    validate_person_file,
)
from anjo.sleep.probes.clients import ModelClient

PROMPT_PATH: Final = Path(__file__).parent / "prompts" / "expand_v1.md"
EXPAND_TEMPERATURE: Final = 0.0
HORIZON_DAYS: Final = 7  # every night sees a full week of arrivals ahead
MIN_DISTINCT_ARRIVAL_DAYS: Final = PLANNED_NIGHTS  # arrivals are spread, not clustered
VISIBLE_PREFERENCES: Final = 60  # with 15 held out, the 20% share of the registered 75


class ExpandError(ValueError):
    """The model's file broke a rule, or the life can't be expanded."""


def _pair_of(life: Life, pairs: Sequence[Pair]) -> Pair:
    for pair in pairs:
        if life in pair.lives:
            return pair
    raise ExpandError(f"{life.id} is in no pair of lives.json: its shared taste items are unknown")


def _shared_block(life: Life, pair: Pair) -> str:
    items = [
        {"id": t.id, "prompt": t.prompt, "options": list(t.options), "pick": t.pick[life.id]}
        for t in pair.taste_ab
    ]
    return "\n".join(f"  - {json.dumps(item, sort_keys=True)}" for item in items)


def _prompt(life: Life, pair: Pair, nights: int) -> str:
    values = {
        "PERSON_ID": life.id, "FIELD": life.field, "LINE": life.line, "TASTE": life.taste,
        "PLANS": "\n".join(f"- day {p.day}: {p.text}" for p in life.plans),
        "SHARED": _shared_block(life, pair),
        "VISIBLE_PREFERENCES": str(VISIBLE_PREFERENCES), "HELD_OUT_PREFERENCES": str(MIN_HELD_OUT_PREFERENCES),
        "MIN_NEW_WORK": str(MIN_NEW_WORK), "FIRST_DAY": str(NEW_WORK_FIRST_DAY),
        "LAST_DAY": str(nights + HORIZON_DAYS), "MIN_DISTINCT_DAYS": str(min(MIN_DISTINCT_ARRIVAL_DAYS, nights)),
    }
    text = PROMPT_PATH.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    if "{{" in text:
        raise ExpandError("expand_v1.md has a placeholder this module doesn't fill")
    return text


def _parse(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ExpandError("the model's answer is not one JSON object") from None
    if not isinstance(data, dict) or not isinstance(data.get("visible"), dict) or not isinstance(data.get("held_out"), dict):
        raise ExpandError("the model's answer lacks a visible part and a held-out part")
    return data


def _build(data: dict[str, Any]) -> PersonFile:
    try:
        return person_file_from_dict(data)
    except (PersonaError, TypeError, AttributeError, KeyError, ValueError) as error:
        raise ExpandError(f"the file is malformed: {error}") from None


def _check_life(pf: PersonFile, life: Life, pair: Pair) -> None:
    v, h = pf.visible, pf.held_out
    if v.person_id != life.id or v.line != life.line or v.field != life.field:
        raise ExpandError(f"the visible part is not {life.id}'s registered id, field and line")
    have = {(p.day, p.text) for p in v.plans}
    missing = [p for p in life.plans if (p.day, p.text) not in have]
    if missing:
        raise ExpandError(f"{len(missing)} of {life.id}'s registered plans are missing or changed")
    taste = {t.id: t for t in h.taste_ab}
    for shared in pair.taste_ab:
        found = taste.get(shared.id)
        if found is None or found.prompt != shared.prompt or found.options != shared.options:
            raise ExpandError(f"shared taste item {shared.id} is missing or changed")
        if found.pick != shared.pick[life.id]:
            raise ExpandError(f"shared taste item {shared.id} has the wrong label for {life.id}")
    blob = json.dumps(to_visible_dict(v), ensure_ascii=False).lower()
    leaks = [p.id for p in h.preferences if p.text.strip().lower() in blob]
    leaks += [t.id for t in h.taste_ab if t.prompt.strip().lower() in blob or any(o.strip().lower() in blob for o in t.options)]
    if leaks:
        raise ExpandError(f"held-out text is visible: {', '.join(sorted(leaks))}")


def _check_spread(pf: PersonFile, nights: int) -> None:
    days = {w.day for w in pf.held_out.new_work}
    need = min(MIN_DISTINCT_ARRIVAL_DAYS, nights)
    if len(days) < need:
        raise ExpandError(f"new-work arrivals fall on {len(days)} days, need at least {need}")


def expand_life(
    life: Life, seed: int, *, client: ModelClient, pairs: Sequence[Pair] | None = None,
    planned_nights: int = PLANNED_NIGHTS,
) -> PersonFile:
    """Expand ``life`` into a validated ``PersonFile``, or raise ``ExpandError``."""
    pair = _pair_of(life, load_pairs() if pairs is None else pairs)
    reply = client.complete(None, [{"role": "user", "content": _prompt(life, pair, planned_nights)}],
                            temperature=EXPAND_TEMPERATURE, seed=seed)
    pf = _build(_parse(reply.text))
    try:
        validate_person_file(pf, planned_nights=planned_nights)
    except PersonaError as error:
        raise ExpandError(str(error)) from None
    _check_life(pf, life, pair)
    _check_spread(pf, planned_nights)
    return pf
