"""The starter scenes' blind pairs, generated in advance (Task 13a).

The page loads them with no GPU and no model call ("instant first", spec Task 13). The CLI
``python -m anjo.sleep.app.demo_pairs`` answers every scene once with arm 4 and once with the
demo's baseline, on the checkpoint serving at the time, and writes one JSON file labeled
``pre-generated`` with that checkpoint and its night. It runs again whenever the checkpoint changes.

Each pair records the hash of the prompt it answered, so a scene whose words changed since has no
pair until the file is generated again. The file names which arm wrote which reply; it is read on
the server and never sent to the page.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from anjo.sleep.demo.words import Scene
from anjo.sleep.types import CheckpointRef

LABEL: Final = "pre-generated"
PRODUCT: Final = "4"
BASELINES: Final = ("2C", "2F")


class StarterError(ValueError):
    """The starter pairs file can't be read."""


@dataclass(frozen=True)
class Side:
    """One side's reply, with what the reveal shows of it."""

    text: str
    prompt_tokens: int
    cost_usd: float | None


@dataclass(frozen=True)
class StarterPair:
    scene: str
    prompt_sha256: str
    product: Side
    baseline: Side


@dataclass(frozen=True)
class StarterSet:
    person: str
    checkpoint: str | None  # the adapter that answered for arm 4; None: no night promoted (the base)
    night: str | None
    baseline: str
    generated_at: str
    pairs: tuple[StarterPair, ...]
    label: str = LABEL

    def pair(self, scene: Scene) -> StarterPair | None:
        """The pair answering ``scene`` as its words read now, if there is one."""
        digest = prompt_hash(scene.prompt)
        return next((p for p in self.pairs if p.scene == scene.id and p.prompt_sha256 == digest), None)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def generate(
    scenes: Sequence[Scene], answer: Callable[[Scene], tuple[Side, Side]], *, person: str,
    checkpoint: CheckpointRef | None, baseline: str, now: datetime,
) -> StarterSet:
    """Every scene answered by ``answer``: arm 4's side, then the baseline's."""
    if baseline not in BASELINES:
        raise ValueError(f"the baseline is one of {BASELINES}")
    pairs = []
    for scene in scenes:
        product, base = answer(scene)
        pairs.append(StarterPair(scene.id, prompt_hash(scene.prompt), product, base))
    return StarterSet(
        person=person, checkpoint=None if checkpoint is None else checkpoint.id,
        night=None if checkpoint is None else checkpoint.night.isoformat(), baseline=baseline,
        generated_at=now.isoformat(timespec="seconds"), pairs=tuple(pairs),
    )


def _side_out(side: Side) -> dict[str, Any]:
    return {"text": side.text, "prompt_tokens": side.prompt_tokens, "cost_usd": side.cost_usd}


def to_json(starters: StarterSet) -> str:
    return json.dumps({
        "label": starters.label, "person": starters.person, "checkpoint": starters.checkpoint,
        "night": starters.night, "baseline": starters.baseline, "generated_at": starters.generated_at,
        "pairs": [{"scene": p.scene, "prompt_sha256": p.prompt_sha256, "product": _side_out(p.product),
                   "baseline": _side_out(p.baseline)} for p in starters.pairs],
    }, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def _get(data: Mapping[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    value = data.get(key)
    if not isinstance(value, kind) or isinstance(value, bool):
        raise StarterError(f"the starter pairs' {key} is missing or of the wrong type")
    return value


def _side(data: Any) -> Side:
    if not isinstance(data, dict):
        raise StarterError("a starter side is not an object")
    cost = data.get("cost_usd")
    if cost is not None and (not isinstance(cost, (int, float)) or isinstance(cost, bool)):
        raise StarterError("a starter side's cost is not a number")
    return Side(_get(data, "text", str), _get(data, "prompt_tokens", int), None if cost is None else float(cost))


def _pair(data: Any) -> StarterPair:
    if not isinstance(data, dict):
        raise StarterError("a starter pair is not an object")
    return StarterPair(_get(data, "scene", str), _get(data, "prompt_sha256", str),
                       _side(data.get("product")), _side(data.get("baseline")))


def from_json(raw: str) -> StarterSet:
    try:
        data = json.loads(raw)
    except ValueError:
        raise StarterError("the starter pairs are not JSON") from None
    if not isinstance(data, dict) or data.get("label") != LABEL:
        raise StarterError(f"the starter pairs are not labeled {LABEL}")
    baseline = _get(data, "baseline", str)
    if baseline not in BASELINES:
        raise StarterError("the starter pairs' baseline is not 2C or 2F")
    checkpoint, night = data.get("checkpoint"), data.get("night")
    if not all(value is None or isinstance(value, str) for value in (checkpoint, night)):
        raise StarterError("the starter pairs' checkpoint or night is not text")
    return StarterSet(
        person=_get(data, "person", str), checkpoint=checkpoint, night=night, baseline=baseline,
        generated_at=_get(data, "generated_at", str),
        pairs=tuple(_pair(item) for item in _get(data, "pairs", list)),
    )


def load(path: Path) -> StarterSet | None:
    """The starter pairs at ``path``; None when none were generated yet."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as error:
        raise StarterError(f"the starter pairs can't be read ({type(error).__name__})") from None
    return from_json(raw)
