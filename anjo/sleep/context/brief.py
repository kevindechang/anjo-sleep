"""The nightly brief (Task 2): a concise summary folded each night from the person's evidence.

``compile_brief`` is one fold step: the previous brief plus the day's new items become the new
brief, written by Spark Open at temperature 0 from the registered prompt ``brief_v1.md`` (never
edited: its hash is registered). Neither cost grows with the life.

**Capped, and retried once.** Each call is capped (``max_tokens``) at the brief's budget plus
``REASONING_ROOM``: on the hero's first live night two brief folds cost about 5.6k output tokens a
call against a visible brief of 2,000 at most, and the longest wrote about 8.5k, so about 6.5k of
reasoning; the room is twice that (15,000 at the default budget). A call that looped at temperature
0 once wrote 32,768 tokens and the cut text failed the night; at temperature 0 a rerun loops the
same way. So a reply cut at its cap (finish reason ``length``) is a ``BriefError`` whatever its text
reads like, and a ``BriefError`` is retried once, sampled at ``RETRY_TEMPERATURE`` with the fixed
``RETRY_SEED`` (the same prompt). A second ``BriefError`` names both; the night then keeps the
briefs it had (``night/run.py``).

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
RETRY_TEMPERATURE: Final = 0.3  # a retry samples, so it leaves the loop temperature 0 fell into
RETRY_SEED: Final = 20261024  # fixed: a rerun's retry is the same call
TIMEOUT_S: Final = 180.0
DEFAULT_BUDGET_TOKENS: Final = 2000
OBSERVED_REASONING: Final = 8_500 - 2_000  # the longest live brief call's output less its visible brief
REASONING_ROOM: Final = 2 * OBSERVED_REASONING  # every Spark Open cap's room to reason (practice's too)
CUT: Final = "length"  # the finish reason of a reply that reached its max_tokens
SECTIONS: Final = (
    "## The project and its constraints",
    "## Stated preferences",
    "## Corrections",
    "## What homework found, with sources",
    "## What practice showed",
)
EMPTY_BRIEF: Final = "\n\n".join(f"{heading}\nnone" for heading in SECTIONS)
_PROMPT_KIND: Final = {"lesson": "day_lesson"}  # the prompt's name for a day lesson


def max_tokens_for(budget_tokens: int) -> int:
    """A brief call's cap: the brief it may write, plus twice the most reasoning seen."""
    return budget_tokens + REASONING_ROOM


MAX_TOKENS: Final = max_tokens_for(DEFAULT_BUDGET_TOKENS)


class BriefError(RuntimeError):
    """The model's brief broke a rule (a missing section, over budget, or cut at its cap)."""


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


class BriefChat(Protocol):
    """The brief's ``ChatFn``: its reply capped (``max_tokens``), its retry seeded (``seed``).
    ``baseten_api.chat`` and the night's metered wrap of it fit this."""

    def __call__(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        timeout_s: float,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> ChatResult: ...


def _default_chat() -> BriefChat:
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


def _written(result: ChatResult, budget_tokens: int, cap: int) -> str:
    """The brief in ``result``. A reply cut at its cap is no brief, whatever its text reads like."""
    if result.finish_reason == CUT:
        raise BriefError(f"the brief ran to its {cap}-token cap")
    return _check(result.text, budget_tokens)


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
    chat: BriefChat | None = None,
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
    messages, cap = [{"role": "user", "content": prompt}], max_tokens_for(budget_tokens)
    try:
        return _written(call(SPARK_OPEN, messages, None, TIMEOUT_S, temperature=TEMPERATURE, max_tokens=cap),
                        budget_tokens, cap)
    except BriefError as first:  # at temperature 0 a rerun breaks the same way: retry once, sampled
        retry = call(SPARK_OPEN, messages, None, TIMEOUT_S, temperature=RETRY_TEMPERATURE, max_tokens=cap,
                     seed=RETRY_SEED)
        try:
            return _written(retry, budget_tokens, cap)
        except BriefError as second:
            raise BriefError(f"{first}; retried: {second}") from second


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
    chat: BriefChat | None = None,
) -> BriefState:
    """One night: the served brief from its own previous version and every new item, the
    training brief from its own previous version and the trainable ones. Either one's
    ``BriefError`` stops the fold, so a night keeps both briefs it had, never one of each."""
    return BriefState(
        served=compile_brief(
            state.served, new, budget_tokens, include_hosted=True, for_training=False, chat=chat,
        ),
        training=compile_brief(
            state.training, new, budget_tokens, include_hosted=hosted_in_training,
            for_training=True, chat=chat,
        ),
    )
