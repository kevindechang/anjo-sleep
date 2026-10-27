"""The held-out lives list and the Nov 1 draw.

`lives.json` holds 20 same-field pairs with opposing taste. The draw takes one pair, then one
single life from a different pair (so its field differs). Donors follow the registered rule
`DONOR_RULE`, a pure function of the draw. The draw uses sha256, not `random`, so it is stable
across Python versions.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LIVES_PATH = Path(__file__).with_name("lives.json")
EXPECTED_PAIRS = 20
MIN_PLANS = 3
MIN_TASTE_ITEMS = 1

DONOR_RULE = (
    "The two lives of the drawn pair are each other's donor (arm 3x). "
    "The single life's donor is the first life of the drawn pair. "
    "Donors never change after the draw."
)

# Anti-project domains (spec 5.1a): health, mental health, tutoring, nutrition,
# sports coaching, politics and war. A whole-word match on the field or the line rejects a life.
_BANNED_WORDS = (
    "health", "healthcare", "medical", "medicine", "doctor", "nurse", "therapy", "therapist",
    "mental", "tutor", "tutoring", "nutrition", "nutritionist", "diet", "dietitian",
    "sports", "coaching", "politics", "political", "politician", "election", "war",
    "military", "army",
)
_BANNED = re.compile(r"\b(" + "|".join(_BANNED_WORDS) + r")\b", re.IGNORECASE)


class LivesError(ValueError):
    """lives.json broke a rule."""


@dataclass(frozen=True)
class Plan:
    day: int
    text: str


@dataclass(frozen=True)
class Life:
    id: str
    pair_id: str
    field: str
    line: str
    taste: str
    plans: tuple[Plan, ...]


@dataclass(frozen=True)
class TasteAB:
    id: str
    prompt: str
    options: tuple[str, str]
    pick: dict[str, int]


@dataclass(frozen=True)
class Pair:
    pair_id: str
    field: str
    taste_axis: str
    lives: tuple[Life, Life]
    taste_ab: tuple[TasteAB, ...]


@dataclass(frozen=True)
class Draw:
    pair: tuple[Life, Life]
    single: Life
    donors: dict[str, str]


def _pair_from(raw: dict[str, Any]) -> Pair:
    try:
        pid, field = raw["pair_id"], raw["field"]
        lives = tuple(
            Life(
                id=lf["id"],
                pair_id=pid,
                field=field,
                line=lf["line"],
                taste=lf["taste"],
                plans=tuple(Plan(int(p["day"]), p["text"]) for p in lf["plans"]),
            )
            for lf in raw["lives"]
        )
        taste = tuple(
            TasteAB(t["id"], t["prompt"], (t["options"][0], t["options"][1]), dict(t["pick"]))
            for t in raw["taste_ab"]
        )
        axis = raw["taste_axis"]
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        raise LivesError(f"malformed pair: {exc!r}") from exc
    if len(lives) != 2:
        raise LivesError(f"{pid}: a pair has exactly two lives")
    return Pair(pid, field, axis, (lives[0], lives[1]), taste)


def _check_pair(p: Pair) -> None:
    a, b = p.lives
    if a.id == b.id:
        raise LivesError(f"{p.pair_id}: duplicate life ids")
    for lf in p.lives:
        if len(lf.plans) < MIN_PLANS:
            raise LivesError(f"{lf.id}: needs at least {MIN_PLANS} dated plans")
        if _BANNED.search(p.field) or _BANNED.search(lf.line):
            raise LivesError(f"{lf.id}: falls in an excluded domain")
    if len(p.taste_ab) < MIN_TASTE_ITEMS:
        raise LivesError(f"{p.pair_id}: needs shared taste_ab items")
    for t in p.taste_ab:
        if set(t.pick) != {a.id, b.id}:
            raise LivesError(f"{t.id}: pick must name both lives")
        if any(v not in (0, 1) for v in t.pick.values()):
            raise LivesError(f"{t.id}: pick must be 0 or 1")
        if t.pick[a.id] == t.pick[b.id]:
            raise LivesError(f"{t.id}: the two lives must choose opposite labels")


def load_pairs(
    path: str | Path | None = None, *, expected_pairs: int | None = EXPECTED_PAIRS
) -> tuple[Pair, ...]:
    data = json.loads(Path(path or LIVES_PATH).read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise LivesError("unsupported lives.json version")
    pairs = tuple(_pair_from(r) for r in data["pairs"])
    if expected_pairs is not None and len(pairs) != expected_pairs:
        raise LivesError(f"expected {expected_pairs} pairs, got {len(pairs)}")
    fields = [p.field.lower() for p in pairs]
    if len(set(fields)) != len(fields):
        raise LivesError("each pair needs its own field")
    ids = [lf.id for p in pairs for lf in p.lives]
    if len(set(ids)) != len(ids):
        raise LivesError("life ids must be unique")
    for p in pairs:
        _check_pair(p)
    return pairs


def _index(seed: int, label: str, n: int) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % n


def donor_map(pair: tuple[Life, Life], single: Life) -> dict[str, str]:
    """The registered donor rule (`DONOR_RULE`)."""
    return {pair[0].id: pair[1].id, pair[1].id: pair[0].id, single.id: pair[0].id}


def draw_lives(seed: int, path: str | Path | None = None) -> Draw:
    pairs = load_pairs(path, expected_pairs=None)
    if len(pairs) < 2:
        raise LivesError("need at least two pairs to draw")
    drawn = pairs[_index(seed, "pair", len(pairs))]
    others = [p for p in pairs if p.pair_id != drawn.pair_id]
    other = others[_index(seed, "single-pair", len(others))]
    single = other.lives[_index(seed, "single-life", 2)]
    return Draw(pair=drawn.lives, single=single, donors=donor_map(drawn.lives, single))
