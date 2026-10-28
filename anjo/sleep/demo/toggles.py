"""The swap toggle's pairs (Task 13a, full step; spec §5.4): her state handed to another model.

The swap measurement (Task 9, a should) answers the film student's prompts with Spark Open given
her context C and with bare Spark Open. If it ran and its pairs were saved for the page's starter
scenes, the file ``swap_pairs.json`` is read here and the toggle shows them, labeled pre-generated
and development; if no file exists, the toggle isn't offered. Nothing in the demo writes the file.

The "notes off" toggle needs no file of its own: it is the night slider's answers at her latest
checkpoint (arm 3) beside night 0's (arm 1, the base model), ``night_answers``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from anjo.sleep.demo.starters import Side, StarterError, prompt_hash, required, side_from_record
from anjo.sleep.demo.words import Scene

LABEL: Final = "pre-generated"


class SwapError(StarterError):
    """The swap pairs file can't be read."""


@dataclass(frozen=True)
class SwapPair:
    scene: str
    prompt_sha256: str
    with_state: Side  # the other model, given her context C
    bare: Side  # the same model, with no context


@dataclass(frozen=True)
class SwapSet:
    model: str
    generated_at: str
    pairs: tuple[SwapPair, ...]
    label: str = LABEL

    def pair(self, scene: Scene) -> SwapPair | None:
        digest = prompt_hash(scene.prompt)
        return next((p for p in self.pairs if p.scene == scene.id and p.prompt_sha256 == digest), None)


def _pair(data: Any) -> SwapPair:
    if not isinstance(data, dict):
        raise SwapError("a swap pair is not an object")
    return SwapPair(required(data, "scene", str), required(data, "prompt_sha256", str),
                    side_from_record(data.get("with_state")), side_from_record(data.get("bare")))


def swap_from_json(raw: str) -> SwapSet:
    try:
        data = json.loads(raw)
    except ValueError:
        raise SwapError("the swap pairs are not JSON") from None
    if not isinstance(data, dict) or data.get("label") != LABEL:
        raise SwapError(f"the swap pairs are not labeled {LABEL}")
    try:
        return SwapSet(model=required(data, "model", str), generated_at=required(data, "generated_at", str),
                       pairs=tuple(_pair(item) for item in required(data, "pairs", list)))
    except SwapError:
        raise
    except StarterError as error:
        raise SwapError(f"the swap pairs: {error}") from None


def load_swaps(path: Path) -> SwapSet | None:
    """The swap pairs at ``path``; None when the measurement saved none."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as error:
        raise SwapError(f"the swap pairs can't be read ({type(error).__name__})") from None
    return swap_from_json(raw)
