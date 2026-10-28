"""The night slider's starter answers, generated in advance per checkpoint (Task 13a, full step).

Each stop answers the starter scenes notes off: arm 3 on the stop's checkpoint, arm 1 (the base
model) at night 0. ``python -m anjo.sleep.app.demo_pairs --nights`` answers every scene once per
stop and writes one JSON file labeled ``pre-generated``; the page serves it with no GPU and no
model call. Run again after a new night, it answers only what it doesn't hold: a checkpoint it
already answered keeps its answers, and one no stop shows any more is dropped.

Each answer records its checkpoint (``None`` for the base) and the hash of the prompt it answered,
so a scene whose words changed since has no answer until the file is generated again.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from anjo.sleep.demo.starters import Side, StarterError, prompt_hash, required, side_from_record, side_record
from anjo.sleep.demo.words import Scene
from anjo.sleep.types import CheckpointRef

LABEL: Final = "pre-generated"


class NightAnswersError(StarterError):
    """The night slider's answers file can't be read."""


@dataclass(frozen=True)
class StopAnswer:
    checkpoint: str | None  # None: the base model, night 0
    scene: str
    prompt_sha256: str
    side: Side
    generated_at: str


@dataclass(frozen=True)
class NightAnswers:
    person: str
    generated_at: str
    answers: tuple[StopAnswer, ...]
    label: str = LABEL

    def find(self, checkpoint: str | None, scene: Scene) -> StopAnswer | None:
        digest = prompt_hash(scene.prompt)
        return next((a for a in self.answers
                     if a.checkpoint == checkpoint and a.scene == scene.id and a.prompt_sha256 == digest), None)

    def answer(self, checkpoint: str | None, scene: Scene) -> Side | None:
        """``checkpoint``'s answer to ``scene`` as its words read now, if there is one."""
        found = self.find(checkpoint, scene)
        return None if found is None else found.side


def generate(
    scenes: Sequence[Scene], checkpoints: Sequence[CheckpointRef | None],
    answer: Callable[[CheckpointRef | None, Scene], Side], *, keep: NightAnswers | None, person: str,
    now: datetime,
) -> NightAnswers:
    """Every scene answered at every checkpoint (``None``: the base), reusing what ``keep`` holds."""
    stamp = now.isoformat(timespec="seconds")
    out: list[StopAnswer] = []
    for ckpt in checkpoints:
        key = None if ckpt is None else ckpt.id
        for scene in scenes:
            kept = None if keep is None else keep.find(key, scene)
            out.append(kept or StopAnswer(key, scene.id, prompt_hash(scene.prompt), answer(ckpt, scene), stamp))
    return NightAnswers(person=person, generated_at=stamp, answers=tuple(out))


def to_json(answers: NightAnswers) -> str:
    return json.dumps({
        "label": answers.label, "person": answers.person, "generated_at": answers.generated_at,
        "answers": [{"checkpoint": a.checkpoint, "scene": a.scene, "prompt_sha256": a.prompt_sha256,
                     "side": side_record(a.side), "generated_at": a.generated_at} for a in answers.answers],
    }, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def _answer(data: Any) -> StopAnswer:
    if not isinstance(data, dict):
        raise NightAnswersError("a night answer is not an object")
    checkpoint = data.get("checkpoint")
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise NightAnswersError("a night answer's checkpoint is not text")
    return StopAnswer(checkpoint, required(data, "scene", str), required(data, "prompt_sha256", str),
                      side_from_record(data.get("side")), required(data, "generated_at", str))


def from_json(raw: str) -> NightAnswers:
    try:
        data = json.loads(raw)
    except ValueError:
        raise NightAnswersError("the night answers are not JSON") from None
    if not isinstance(data, dict) or data.get("label") != LABEL:
        raise NightAnswersError(f"the night answers are not labeled {LABEL}")
    try:
        return NightAnswers(
            person=required(data, "person", str), generated_at=required(data, "generated_at", str),
            answers=tuple(_answer(item) for item in required(data, "answers", list)),
        )
    except NightAnswersError:
        raise
    except StarterError as error:
        raise NightAnswersError(f"the night answers: {error}") from None


def load(path: Path) -> NightAnswers | None:
    """The night answers at ``path``; None when none were generated yet."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as error:
        raise NightAnswersError(f"the night answers can't be read ({type(error).__name__})") from None
    return from_json(raw)
