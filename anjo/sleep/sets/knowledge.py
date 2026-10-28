"""A frozen forced-choice set of what a synthetic persona said about themselves: the post-hoc knowledge
measure's items (``tools.knowledge_score``, 2026-10-29).

**Built from the visible statements only:** the persona's line, project logline and constraints, tone
requests, stated views, plans and preferences (``personas.model.Persona``). Nothing held out, sealed or
registered is read: the held-out part and the probes are the morning's, and a set drawn from them would
leak them into a number every arm is compared on. No model is called.

**An item** is one statement and its counterfeits, each a whole line the model scores as a continuation:

- *a value swapped*: a count ("eighty"), a figure ("1,400"), a weekday, a month, a part of the day or a
  unit of time in the statement is replaced by another value of the same kind that the persona's own
  statements use (same case, plural and article). Up to three counterfeits, ranked by a hash;
- *a contrast swapped*: in "X, not Y" (also ", never", "instead of", "rather than", "than") the two
  phrases change places when they start with the same word and are near one length;
- *quotes swapped*: the first two quoted phrases of a statement change places.

Choice order is by a hash too, so the true line sits anywhere. The prompt is one fixed line and carries
no brief, memory or thought channel: what an adapter picks is in its weights.

**Frozen:** ``knowledge_<persona>.jsonl`` holds the set, and ``FROZEN`` its sha-256; ``load`` refuses a
file whose bytes don't match, so every arm and night is scored on the same items. To change the set,
raise ``VERSION``, write it (``python -m anjo.sleep.sets.knowledge film_founder --write``) and update
``FROZEN`` with the sha it prints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from anjo.sleep.personas.model import Persona, load_person_file
from anjo.sleep.sets.loader import SETS_DIR, SetError

VERSION: Final = 1
PROMPT: Final = "Write one line from your notes about me."
MAX_CHOICES: Final = 4
MIN_ITEMS: Final = 40
MAX_PHRASE_WORDS: Final = 8  # a contrast's phrases: longer ones are clauses, not swappable
MAX_PHRASE_GAP: Final = 2  # and their lengths differ by at most this many words
PERSONAS_DIR: Final = Path(__file__).parents[1] / "personas"
FROZEN: Final[Mapping[str, str]] = {
    "film_founder": "a7d5ffd5e0596c6653d577ace7a72ad57dfbd7a57f2581053f5841c6085479fe",
}

_UNITS: Final = ("minute", "hour", "day", "week", "month", "year")
_PARTS_OF_DAY: Final = ("morning", "afternoon", "evening", "night", "midnight", "noon", "dawn")
_WEEKDAYS: Final = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTHS: Final = ("january", "february", "march", "april", "june", "july", "august", "september", "october",
                  "november", "december")  # not May: a verb as often as a month
SMALL: Final = ("two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
TENS: Final = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_DIGITS: Final = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_COUNT: Final = rf"(?:(?:{'|'.join(TENS)})(?:-(?:{'|'.join(_DIGITS)}))?|{'|'.join(SMALL)})"
_JOINED: Final = r"(?<![\w-])({body})(?![\w-])"  # never part of a hyphenated word such as "one-page"
# The kinds a value can be, each with its pattern (the value is group 1) and whether its plural is kept.
KINDS: Final[Mapping[str, tuple[str, bool]]] = {
    "count": (_JOINED.format(body=_COUNT), False),  # "one" never: swapping it breaks the noun's number
    "figure": (r"(?<![\w.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\w]|[.,]\d)", False),
    "decade": (r"(?<![\w])(\d{4}s)(?![\w])", False),
    "weekday": (_JOINED.format(body=f"(?:{'|'.join(_WEEKDAYS)})s?"), True),
    "month": (_JOINED.format(body="|".join(_MONTHS)), False),
    "part_of_day": (_JOINED.format(body=f"(?:{'|'.join(_PARTS_OF_DAY)})s?"), True),
    "unit": (_JOINED.format(body=f"(?:{'|'.join(_UNITS)})s?") + r"(?!\s\d)", True),  # "Day 3" is a label
}
_CONTRASTS: Final = (", not ", ", never ", " instead of ", " rather than ", " than ")
_QUOTED: Final = re.compile(r"(?<![\w])(['\"])([^'\"]+?)\1(?![\w])")
_ARTICLE: Final = re.compile(r"\b(a|an|A|An)\s+$")


@dataclass(frozen=True)
class KnowledgeItem:
    """One forced choice: ``choices`` are whole lines, ``answer`` the index of the persona's own one.
    ``key`` is what tells the true line apart (the value, or the phrase that leads the contrast), which
    the scorer looks for in the simulated days to flag an item never mentioned."""

    id: str
    source: str
    kind: str
    key: str
    prompt: str
    choices: tuple[str, ...]
    answer: int

    @property
    def statement(self) -> str:
        return self.choices[self.answer]


@dataclass(frozen=True)
class KnowledgeSet:
    persona: str
    version: int
    sha: str
    items: tuple[KnowledgeItem, ...]


def persona_path(persona: str) -> Path:
    return PERSONAS_DIR / f"{_name(persona)}.json"


def set_path(persona: str) -> Path:
    return SETS_DIR / f"knowledge_{_name(persona)}.jsonl"


def _name(persona: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", persona):
        raise SetError("a persona is named in lower case, letters, digits and underscores")
    return persona


def statements(persona: Persona) -> list[tuple[str, str]]:
    """Every visible statement, with an id, in one fixed order."""
    return [
        ("line", persona.line), ("project", persona.project.logline),
        *((f"project-c{n}", text) for n, text in enumerate(persona.project.constraints, 1)),
        *((s.id, s.text) for s in persona.tone_requests), *((s.id, s.text) for s in persona.stated_views),
        *((f"plan-d{p.day}", p.text) for p in persona.plans), *((s.id, s.text) for s in persona.preferences),
    ]


def _rank(*parts: str) -> str:
    return hashlib.sha256("\x1f".join((str(VERSION), *parts)).encode()).hexdigest()


def _base(kind: str, value: str) -> str:
    """A value as the pool knows it: lower case, and without the plural its kind keeps."""
    lower = value.lower()
    plural = KINDS[kind][1] and lower.endswith("s") and lower[:-1] in (*_UNITS, *_PARTS_OF_DAY, *_WEEKDAYS)
    return lower[:-1] if plural else lower


def _fit(value: str, like: str, kind: str) -> str:
    """``value`` (a pool base) in the form of ``like``: its plural and its capital."""
    out = value + ("s" if KINDS[kind][1] and like.lower() != _base(kind, like) else "")
    if kind in {"weekday", "month"} or like[:1].isupper():
        out = out[:1].upper() + out[1:]
    return out


def _article(word: str, like: str) -> str:
    vowel = word[:1].lower() in "aeiou" or word.lower().startswith(("hour", "honest", "heir")) or word[:1] in "8"
    article = "an" if vowel else "a"
    return article.capitalize() if like[:1].isupper() else article


def _pool(texts: Sequence[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {kind: [] for kind in KINDS}
    for text in texts:
        for kind, (pattern, _) in KINDS.items():
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                base = _base(kind, match.group(1))
                if base not in found[kind]:
                    found[kind].append(base)
    return found


def _figure_class(value: str) -> tuple[bool, bool]:
    return ("," in value, "." in value)


def _distractors(kind: str, base: str, others: Sequence[str], item_id: str) -> list[str]:
    if kind == "figure":  # the same shape first (thousands with thousands), then the rest
        return sorted(others, key=lambda v: (_figure_class(v) != _figure_class(base), _rank(item_id, v)))[:MAX_CHOICES - 1]
    return sorted(others, key=lambda v: _rank(item_id, v))[:MAX_CHOICES - 1]


def _swap_value(text: str, kind: str, base: str, new: str) -> str:
    """``text`` with every ``base`` of ``kind`` replaced by ``new``, its form and article kept to fit."""
    pattern = KINDS[kind][0]
    out, last = [], 0
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        if _base(kind, match.group(1)) != base:
            continue
        word = _fit(new, match.group(1), kind)
        head = text[last:match.start(1)]
        article = _ARTICLE.search(head)
        if article:
            head = head[:article.start(1)] + _article(word, article.group(1)) + head[article.end(1):]
        out += [head, word]
        last = match.end(1)
    return "".join(out) + text[last:]


def _ordered(item_id: str, true: str, counterfeits: Sequence[str]) -> tuple[tuple[str, ...], int]:
    choices = tuple(sorted({true, *counterfeits}, key=lambda c: _rank(item_id, "order", c)))
    return choices, choices.index(true)


def _value_items(source: str, text: str, pool: Mapping[str, Sequence[str]]) -> Iterator[KnowledgeItem]:
    for kind, (pattern, _) in KINDS.items():
        bases: list[str] = []
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if (base := _base(kind, match.group(1))) not in bases:
                bases.append(base)
        for n, base in enumerate(bases, 1):
            item_id = f"{source}:{kind}:{n}"
            others = [value for value in pool[kind] if value not in bases]  # never a value the line already holds
            counterfeits = [_swap_value(text, kind, base, value) for value in _distractors(kind, base, others, item_id)]
            counterfeits = [c for c in counterfeits if c != text]
            if counterfeits:
                key = next(m.group(1) for m in re.finditer(pattern, text, flags=re.IGNORECASE) if _base(kind, m.group(1)) == base)
                choices, answer = _ordered(item_id, text, counterfeits)
                yield KnowledgeItem(item_id, source, kind, key, PROMPT, choices, answer)


def _contrast(text: str) -> tuple[str, str] | None:
    """The swapped line and the leading phrase, for a statement of the form "... X, not Y"."""
    for marker in _CONTRASTS:
        at = text.find(marker)
        if at < 0:
            continue
        rest = text[at + len(marker):]
        right = re.split(r"[.;,:]", rest, maxsplit=1)[0].strip()
        start = max(text.rfind(mark, 0, at) for mark in ".;,:") + 1
        lead = right.split()[0] if right else ""
        hits = [m.start() for m in re.finditer(rf"(?<![\w']){re.escape(lead)}(?![\w'])", text[start:at])] if lead else []
        if not hits:
            continue
        left = text[start + hits[-1]:at]
        sizes = len(left.split()), len(right.split())
        if left == right or max(sizes) > MAX_PHRASE_WORDS or abs(sizes[0] - sizes[1]) > MAX_PHRASE_GAP:
            continue
        right_at = at + len(marker) + rest.index(right)
        swapped = text[:start + hits[-1]] + right + text[at:right_at] + left + text[right_at + len(right):]
        return swapped, left
    return None


def _phrase_items(source: str, text: str) -> Iterator[KnowledgeItem]:
    contrast = _contrast(text)
    if contrast is not None:
        item_id = f"{source}:contrast:1"
        choices, answer = _ordered(item_id, text, [contrast[0]])
        yield KnowledgeItem(item_id, source, "contrast", contrast[1], PROMPT, choices, answer)
    quotes = list(_QUOTED.finditer(text))
    if len(quotes) >= 2 and quotes[0].group(2) != quotes[1].group(2):
        first, second = quotes[0], quotes[1]
        swapped = (text[:first.start(2)] + second.group(2) + text[first.end(2):second.start(2)] + first.group(2)
                   + text[second.end(2):])
        item_id = f"{source}:quote:1"
        choices, answer = _ordered(item_id, text, [swapped])
        yield KnowledgeItem(item_id, source, "quote", first.group(2), PROMPT, choices, answer)


def build(persona: Persona) -> tuple[KnowledgeItem, ...]:
    """The set, from ``persona``'s visible statements alone, in statement order."""
    found = statements(persona)
    pool = _pool([text for _, text in found])
    truths = {text for _, text in found}
    items = [item for source, text in found for item in (*_value_items(source, text, pool), *_phrase_items(source, text))]
    # A counterfeit that happens to be another visible statement would be true: such an item is left out.
    return tuple(item for item in items
                 if not any(choice in truths for n, choice in enumerate(item.choices) if n != item.answer))


def build_file(path: Path) -> tuple[KnowledgeItem, ...]:
    """``build`` over a persona file's visible part (the file is validated whole, then only that is used)."""
    return build(load_person_file(path).visible)


def _row(item: KnowledgeItem) -> dict[str, object]:
    return {"version": VERSION, "id": item.id, "source": item.source, "kind": item.kind, "key": item.key,
            "prompt": item.prompt, "choices": list(item.choices), "answer": item.answer}


def dumps(items: Sequence[KnowledgeItem]) -> str:
    return "".join(json.dumps(_row(item), sort_keys=True, ensure_ascii=False) + "\n" for item in items)


def _item(row: object, where: str) -> KnowledgeItem:
    if not isinstance(row, dict) or row.get("version") != VERSION:
        raise SetError(f"{where}: not a version {VERSION} knowledge item")
    choices, answer = row.get("choices"), row.get("answer")
    if (not isinstance(choices, list) or not 2 <= len(choices) <= MAX_CHOICES or not isinstance(answer, int)
            or not 0 <= answer < len(choices) or not all(isinstance(c, str) and c for c in choices)):
        raise SetError(f"{where}: its choices or answer are malformed")
    fields = {key: row.get(key) for key in ("id", "source", "kind", "key", "prompt")}
    if not all(isinstance(value, str) and value for value in fields.values()):
        raise SetError(f"{where}: a field is missing")
    return KnowledgeItem(id=str(fields["id"]), source=str(fields["source"]), kind=str(fields["kind"]),
                         key=str(fields["key"]), prompt=str(fields["prompt"]), choices=tuple(choices), answer=answer)


def load(persona: str, path: Path | None = None) -> KnowledgeSet:
    """The frozen set for ``persona``; a file whose sha-256 isn't ``FROZEN``'s is refused."""
    expected = FROZEN.get(persona)
    if not expected:
        raise SetError(f"no knowledge set is frozen for {persona}")
    source = path or set_path(persona)
    raw = source.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != expected:
        raise SetError(f"{source.name} has sha {sha[:12]}, not the frozen {expected[:12]}: the set changed")
    lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
    items = tuple(_item(json.loads(line), f"{source.name} line {n}") for n, line in enumerate(lines, 1))
    if len({item.id for item in items}) != len(items):
        raise SetError(f"{source.name}: an item id repeats")
    return KnowledgeSet(persona, VERSION, sha, items)


def main(argv: Sequence[str] | None = None, *, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description="Build a persona's knowledge set from its visible statements.")
    parser.add_argument("persona", help="a persona file's name, such as film_founder")
    parser.add_argument("--write", action="store_true", help="write the set file (then freeze its sha in FROZEN)")
    args = parser.parse_args(argv)
    items = build_file(persona_path(args.persona))
    text = dumps(items)
    if args.write:
        set_path(args.persona).write_text(text, encoding="utf-8")
    kinds = {kind: sum(item.kind == kind for item in items) for kind in sorted({item.kind for item in items})}
    out(json.dumps({"persona": args.persona, "version": VERSION, "items": len(items), "kinds": kinds,
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "written": args.write}))
    return 0 if len(items) >= MIN_ITEMS else 1


if __name__ == "__main__":
    raise SystemExit(main())
