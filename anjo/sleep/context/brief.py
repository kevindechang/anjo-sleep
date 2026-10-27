"""The nightly brief (Task 2): a concise summary folded each night from the person's evidence.

``compile_brief`` is one fold step: the previous brief plus the day's new items become the new
brief, written by Spark Open at temperature 0 from the registered prompt ``brief_v1.md`` (never
edited: its hash is registered). Neither cost grows with the life.

Two briefs are kept, each folded from its own previous version so neither inherits the other's
text (``BriefState`` and ``fold_day``):

- the **served** brief: everything the person's state holds, homework included;
- the **training** brief: only items whose ancestry is cleared (``anjo.sleep.provenance``)
  while hosted outputs can't train.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.context.template import render
from anjo.sleep.model_api import ChatResult
from anjo.sleep.provenance import may_train
from anjo.sleep.types import Evidence

PROMPT_PATH: Final = Path(__file__).parent / "prompts" / "brief_v1.md"
TEMPERATURE: Final = 0.0
TIMEOUT_S: Final = 180.0
DEFAULT_BUDGET_TOKENS: Final = 2000
SECTIONS: Final = (
    "## The project and its constraints",
    "## Stated preferences",
    "## Corrections",
    "## What homework found, with sources",
    "## What practice showed",
)
EMPTY_BRIEF: Final = "\n\n".join(f"{heading}\nnone" for heading in SECTIONS)
_PROMPT_KIND: Final = {"lesson": "day_lesson"}  # the prompt's name for a day lesson


class BriefError(RuntimeError):
    """The model's brief broke a rule (a missing section, or over budget)."""


class ChatFn(Protocol):
    """``baseten_api.chat`` fits this."""

    def __call__(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        timeout_s: float,
        *,
        temperature: float | None = None,
    ) -> ChatResult: ...


def _default_chat() -> ChatFn:
    from anjo.sleep import baseten_api

    return baseten_api.chat


def _line(item: Evidence) -> str:
    kind = _PROMPT_KIND.get(item.kind, item.kind)
    text = item.text.replace("\n", " ")
    return f"- id={item.id} kind={kind} source={item.source}: {text}"


def _check(text: str, budget_tokens: int) -> str:
    brief = text.strip()
    positions = [brief.find(heading) for heading in SECTIONS]
    if -1 in positions or positions != sorted(positions):
        raise BriefError("the brief is missing a section or has the sections out of order")
    if estimate_tokens(brief) > budget_tokens:
        raise BriefError(f"the brief is over its {budget_tokens}-token budget")
    return brief


def _chosen(new: Sequence[Evidence], include_hosted: bool, for_training: bool) -> list[Evidence]:
    held = [item.id for item in new if item.held_out]
    if held:
        raise ValueError(f"held-out items never enter a brief: {', '.join(held)}")
    if not for_training:
        return list(new)
    return [item for item in new if may_train(item.ancestry, hosted_in_training=include_hosted)]


def compile_brief(
    prev: str | None,
    new: Sequence[Evidence],
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    include_hosted: bool = False,
    *,
    for_training: bool | None = None,
    chat: ChatFn | None = None,
) -> str:
    """Fold ``new`` into ``prev``.

    ``include_hosted`` is the prompt's flag: false keeps hosted (homework) items out, true lets
    them in. ``for_training`` (default: ``not include_hosted``) additionally drops every item
    whose ancestry isn't cleared; the training brief passes ``include_hosted=hosted_in_training``
    and the served brief passes ``include_hosted=True, for_training=False``.
    """
    training = (not include_hosted) if for_training is None else for_training
    items = _chosen(new, include_hosted, training)
    if not items:
        return prev if prev is not None else EMPTY_BRIEF
    prompt = render(PROMPT_PATH.read_text(encoding="utf-8"), {
        "PREVIOUS": prev if prev is not None else "none",
        "NEW": "\n".join(_line(item) for item in items),
        "BUDGET_TOKENS": str(budget_tokens),
        "INCLUDE_HOSTED": "true" if include_hosted else "false",
    })
    call = chat if chat is not None else _default_chat()
    result = call(
        SPARK_OPEN, [{"role": "user", "content": prompt}], None, TIMEOUT_S, temperature=TEMPERATURE,
    )
    return _check(result.text, budget_tokens)


@dataclass(frozen=True)
class BriefState:
    """The two briefs after a night. ``None`` until the first fold."""

    served: str | None
    training: str | None


def fold_day(
    state: BriefState,
    new: Sequence[Evidence],
    *,
    hosted_in_training: bool,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    chat: ChatFn | None = None,
) -> BriefState:
    """One night: the served brief from its own previous version and every new item, the
    training brief from its own previous version and the trainable ones."""
    return BriefState(
        served=compile_brief(
            state.served, new, budget_tokens, include_hosted=True, for_training=False, chat=chat,
        ),
        training=compile_brief(
            state.training, new, budget_tokens, include_hosted=hosted_in_training,
            for_training=True, chat=chat,
        ),
    )
