"""Which objective the nights train (Kevin, 2026-10-28): ``ANJO_SLEEP_OBJECTIVE`` is ``sft_dpo`` (the
default: SFT, then a DPO pass on the night's (best, worst) practice pairs) or ``sft``. Anything else is
refused at start. The night command, the accelerated runs and the synthetic example build their recipe here.
MIND's thinker (``thinker.py``, Task 17b) never reads it and keeps ``LoraCfg()``'s own ``sft``.

A night whose practice kept no usable pair trains ``sft`` for that night (``lora.objective_used``).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from anjo.sleep.train.lora import LoraCfg, Objective

OBJECTIVE_ENV: Final = "ANJO_SLEEP_OBJECTIVE"
OBJECTIVES: Final[tuple[Objective, ...]] = ("sft_dpo", "sft")
DEFAULT_OBJECTIVE: Final[Objective] = "sft_dpo"


class ObjectiveError(ValueError):
    """``ANJO_SLEEP_OBJECTIVE`` names no objective."""


def objective_name(environ: Mapping[str, str]) -> Objective:
    """The objective ``environ`` names: ``sft_dpo`` when unset or blank."""
    chosen = environ.get(OBJECTIVE_ENV, "").strip() or DEFAULT_OBJECTIVE
    if chosen not in OBJECTIVES:
        raise ObjectiveError(f"{OBJECTIVE_ENV} is {' or '.join(OBJECTIVES)}")
    return chosen


def night_cfg(environ: Mapping[str, str]) -> LoraCfg:
    """The nights' recipe: the default one with the setting's objective."""
    return LoraCfg(objective=objective_name(environ))
