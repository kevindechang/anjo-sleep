"""Dailies' before and after (the live page's morning card), generated in advance per checkpoint pair.

After a night that promoted new weights, the night phase (``night.daily`` calls
``app.demo_pairs.after_night``) answers one scene, ``words.DAILY_SCENE``, twice, notes off: with the
checkpoint that served before the night (arm 3, or arm 1 when none had been promoted yet) and with
the one that serves after it. Both answers go into one JSON file labeled ``pre-generated``, as one
pair; the page serves it with no GPU and no model call. A pair already on file is kept and asks
nothing, so a rerun of the night costs nothing, and every earlier night's pair stays for its card.

Each pair records the night, both checkpoints (``None`` for the base model), the hash of the prompt
it answered and when it was generated, so a scene whose words changed since has no pair until the
file is generated again, and a pair answers only its own two checkpoints.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from anjo.sleep.demo.starters import Side, StarterError, prompt_hash, required, side_from_record, side_record
from anjo.sleep.demo.words import Scene
from anjo.sleep.types import CheckpointRef

LABEL: Final = "pre-generated"


class DailyLinesError(StarterError):
    """Dailies' line file can't be read."""


@dataclass(frozen=True)
class LinePair:
    night: str  # the date of the night whose checkpoint is ``after``
    before: str | None  # the checkpoint serving before it; None: the base model
    after: str
    scene: str
    prompt_sha256: str
    before_side: Side
    after_side: Side
    generated_at: str


@dataclass(frozen=True)
class DailyLines:
    person: str
    generated_at: str
    pairs: tuple[LinePair, ...]
    label: str = LABEL

    def find(self, before: str | None, after: str, scene: Scene) -> LinePair | None:
        """The pair answering ``scene`` (as its words read now) at exactly these two checkpoints."""
        digest = prompt_hash(scene.prompt)
        return next((p for p in self.pairs if (p.before, p.after, p.scene, p.prompt_sha256)
                     == (before, after, scene.id, digest)), None)


def generate(
    scene: Scene, before: CheckpointRef | None, after: CheckpointRef,
    answer: Callable[[CheckpointRef | None, Scene], Side], *, keep: DailyLines | None, person: str, now: datetime,
) -> DailyLines:
    """``keep`` with the night's pair added: ``scene`` answered at ``before``, then at ``after``. A pair
    ``keep`` already holds is kept, and nothing is asked."""
    stamp = now.isoformat(timespec="seconds")
    kept = () if keep is None else keep.pairs
    if keep is not None and keep.find(None if before is None else before.id, after.id, scene) is not None:
        return keep
    pair = LinePair(after.night.isoformat(), None if before is None else before.id, after.id, scene.id,
                    prompt_hash(scene.prompt), answer(before, scene), answer(after, scene), stamp)
    return DailyLines(person=person, generated_at=stamp, pairs=(*kept, pair))


def to_json(lines: DailyLines) -> str:
    return json.dumps({
        "label": lines.label, "person": lines.person, "generated_at": lines.generated_at,
        "pairs": [{"night": p.night, "before": p.before, "after": p.after, "scene": p.scene,
                   "prompt_sha256": p.prompt_sha256, "before_side": side_record(p.before_side),
                   "after_side": side_record(p.after_side), "generated_at": p.generated_at} for p in lines.pairs],
    }, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def _pair(data: Any) -> LinePair:
    if not isinstance(data, dict):
        raise DailyLinesError("a line pair is not an object")
    before = data.get("before")
    if before is not None and not isinstance(before, str):
        raise DailyLinesError("a line pair's before checkpoint is not text")
    return LinePair(required(data, "night", str), before, required(data, "after", str), required(data, "scene", str),
                    required(data, "prompt_sha256", str), side_from_record(data.get("before_side")),
                    side_from_record(data.get("after_side")), required(data, "generated_at", str))


def from_json(raw: str) -> DailyLines:
    try:
        data = json.loads(raw)
    except ValueError:
        raise DailyLinesError("the line pairs are not JSON") from None
    if not isinstance(data, dict) or data.get("label") != LABEL:
        raise DailyLinesError(f"the line pairs are not labeled {LABEL}")
    try:
        return DailyLines(person=required(data, "person", str), generated_at=required(data, "generated_at", str),
                          pairs=tuple(_pair(item) for item in required(data, "pairs", list)))
    except DailyLinesError:
        raise
    except StarterError as error:
        raise DailyLinesError(f"the line pairs: {error}") from None


def load(path: Path) -> DailyLines | None:
    """The line pairs at ``path``; None when none were generated yet."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as error:
        raise DailyLinesError(f"the line pairs can't be read ({type(error).__name__})") from None
    return from_json(raw)
