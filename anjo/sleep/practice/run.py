"""Minimal practice (Task 5, from Oct 28; Task 7 adds the teacher candidate, the critic and more
scenes).

On a training night, Spark Open plays the person in eight work-shaped scenes drawn from the
visible part of their life (``scenes_v1.md``): each a new item in their field, like a ``new_work``
probe, and cue-free (never "as we discussed"). Glimmer, the base model, answers each with context
F, and that reply is the training target. Each scene keeps the snapshot of context C from before
tonight's practice (``TrainingContext``), which becomes its training input.

**F** is the person's trainable evidence without ``practice`` items, and without homework while
hosted outputs can't train (§3), newest first, inside ``practice_f_cap`` tokens.

A scene whose message carries a cue is dropped and counted rather than trained on. Every reply is
seeded by the run seed, the person, the night and the scene, so a rerun asks the server the same.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.brief import ChatFn
from anjo.sleep.context.brief_store import union
from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.context.template import render
from anjo.sleep.practice.transcripts import PracticeTranscript
from anjo.sleep.probes.clients import ModelClient
from anjo.sleep.provenance import GLIMMER, may_train
from anjo.sleep.types import Evidence

PROMPT_PATH: Final = Path(__file__).parent / "prompts" / "scenes_v1.md"
SCENES: Final = 8
PRACTICE_F_CAP: Final = 8000  # tokens of F (§5.5), set on development against the cost ceiling
SCENE_TIMEOUT_S: Final = 180.0
SCENE_TEMPERATURE: Final = 1.0
REPLY_TEMPERATURE: Final = 1.0  # registered sampling, as for every arm
CUES: Final = (
    "as we discussed", "as discussed", "as i said", "like i said", "as i mentioned", "like i mentioned",
    "you told me", "i told you", "you said", "you mentioned", "remember when", "last time",
    "we talked about", "as before", "earlier you", "like before",
)
Snapshot = Callable[[str, str], tuple[str | None, frozenset[str] | None]]


class PracticeError(RuntimeError):
    """Spark Open's scenes couldn't be read."""


NORMAL_STOPS: Final = frozenset({"stop", "eos", "end_turn"})


@dataclass(frozen=True)
class PracticeRun:
    transcripts: tuple[PracticeTranscript, ...]
    f_ids: tuple[str, ...]
    dropped_cued: int
    dropped_cut: int = 0  # replies that ended without a normal stop (cut at max tokens): never targets
    pairs: int = 0  # full practice: scenes that kept a (best, worst) pair
    teacher_targets: int = 0  # full practice: scenes whose target is the teacher's
    critic_unreadable: int = 0  # full practice: scenes whose critic reply couldn't be read


def has_cue(message: str) -> bool:
    folded = " ".join(message.casefold().split())
    return any(cue in folded for cue in CUES)


def practice_f(
    evidence: Sequence[Evidence], *, hosted_in_training: bool, cap_tokens: int = PRACTICE_F_CAP,
) -> list[Evidence]:
    """F: trainable, not practice, not held out, no homework unless hosted outputs may train,
    newest first, inside ``cap_tokens``."""
    eligible = [
        item for item in evidence
        if item.kind != "practice" and not item.held_out
        and (hosted_in_training or item.kind != "homework")
        and may_train(item.ancestry, hosted_in_training=hosted_in_training)
    ]
    eligible.sort(key=lambda item: (item.created_at, item.id), reverse=True)
    kept: list[Evidence] = []
    used = 0
    for item in eligible:
        cost = estimate_tokens(item.text)
        if used + cost <= cap_tokens:
            kept.append(item)
            used += cost
    return kept


def _known(f: Sequence[Evidence]) -> str:
    return "\n".join(f"- ({item.kind}) {' '.join(item.text.split())}" for item in f)


def _seed(seed: int, *parts: object) -> int:
    return int(hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).hexdigest()[:8], 16)


def _scenes(raw: str, count: int) -> list[str]:
    try:
        data: Any = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
        scenes = [str(scene).strip() for scene in data["scenes"]]
    except (ValueError, KeyError, TypeError):
        raise PracticeError("Spark Open's scenes are not the JSON asked for") from None
    return [scene for scene in scenes if scene][:count]


class MinimalPractice:
    """Task 5's practice: ``run`` for the night (with counts), ``practice`` for the spec's port."""

    max_spark_calls: Final = 1  # the scenes

    def __init__(
        self, *, chat: ChatFn, client: ModelClient, snapshot: Snapshot, seed: int,
        hosted_in_training: bool = False, scenes: int = SCENES, f_cap: int = PRACTICE_F_CAP,
    ) -> None:
        self._chat = chat
        self._client = client
        self._snapshot = snapshot
        self._seed = seed
        self._hosted = hosted_in_training
        self._count = scenes
        self._f_cap = f_cap

    def practice(self, user_id: str, night: date, evidence: list[Evidence], hw: object | None) -> list[PracticeTranscript]:
        return list(self.run(user_id, night, evidence, hw, at=datetime.combine(night, datetime.min.time())).transcripts)

    def run(
        self, user_id: str, night: date, evidence: Sequence[Evidence], hw: object | None, *, at: datetime,
    ) -> PracticeRun:
        if hw is not None and not self._hosted:
            raise ValueError("homework never reaches practice while hosted outputs can't train (§3)")
        f = practice_f(evidence, hosted_in_training=self._hosted, cap_tokens=self._f_cap)
        if not f:
            return PracticeRun((), (), 0)
        prompt = render(PROMPT_PATH.read_text(encoding="utf-8"), {"KNOWN": _known(f), "COUNT": str(self._count)})
        written = self._chat(SPARK_OPEN, [{"role": "user", "content": prompt}], None, SCENE_TIMEOUT_S,
                             temperature=SCENE_TEMPERATURE)
        messages = _scenes(written.text, self._count)
        cue_free = [message for message in messages if not has_cue(message)]
        f_text = "## What you know about this person\n" + _known(f)
        f_ancestry = union(item.ancestry for item in f)
        scenes = [
            self._scene(user_id, night, index, message, f_text, f_ancestry, tuple(e.id for e in f), at)
            for index, message in enumerate(cue_free)
        ]
        transcripts = tuple(scene for scene in scenes if scene is not None)
        return PracticeRun(transcripts, tuple(item.id for item in f), len(messages) - len(cue_free),
                           dropped_cut=len(scenes) - len(transcripts))

    def _scene(
        self, user_id: str, night: date, index: int, message: str, f_text: str,
        f_ancestry: frozenset[str] | None, f_ids: tuple[str, ...], at: datetime,
    ) -> PracticeTranscript | None:
        context, c_ancestry = self._snapshot(user_id, message)  # before the reply exists
        answer = self._client.complete(
            None, [{"role": "system", "content": f_text}, {"role": "user", "content": message}],
            temperature=REPLY_TEMPERATURE, seed=_seed(self._seed, user_id, night.isoformat(), index),
        )
        if answer.finish_reason not in NORMAL_STOPS:  # cut short: a half reply never becomes a target
            return None
        reply = answer.text.strip()
        digest = hashlib.sha256(f"{user_id}\x1f{night.isoformat()}\x1f{index}".encode()).hexdigest()[:16]
        return PracticeTranscript(
            id=f"pr-{digest}", person_id=user_id, night=night, scene=index, message=message, reply=reply,
            context=context, f_ids=f_ids,
            ancestry=union((frozenset({SPARK_OPEN, GLIMMER}), f_ancestry, c_ancestry)),
            created_at=at + timedelta(seconds=index),
        )
