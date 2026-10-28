"""Which objective the nights train (Kevin, 2026-10-28): ``ANJO_SLEEP_OBJECTIVE`` is ``sft_dpo`` (the
default: SFT, then a DPO pass on the night's (best, worst) practice pairs), ``sft``, or ``sft_dpo_seed``
(2026-10-29: the sleep paradigm's one loss, ``sleep_loss.py``). Anything else is refused at start. The
night command, the accelerated runs and the synthetic example build their recipe here. MIND's thinker
(``thinker.py``, Task 17b) never reads it and keeps ``LoraCfg()``'s own ``sft``.

A night whose practice kept no usable pair trains ``sft`` for that night (``lora.objective_used``).

Two more switches ride on ``sft_dpo_seed`` and are refused with any other objective:

- ``ANJO_SLEEP_ADAPTER_FOLD=1``: each night trains a fresh slot beside the person's frozen long-term
  adapter, then folds both into one plain LoRA of rank ``FOLD_RANK`` (the server's cap), so capacity
  stays bounded across nights;
- ``ANJO_SLEEP_DREAMS=1``: the night adds Glimmer's dreams from MIND's dream states (``night/dreams.py``).

``ANJO_SLEEP_MATCH_MINUTES`` is the live comparison's equal-budget control and nothing else: today's
objective keeps training SFT until the job has used that many minutes. Every real night leaves it unset.
Each switch off (unset, blank or ``0``) is today's night.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

from anjo.sleep.night.dreams import Terms
from anjo.sleep.serving.config import MAX_LORA_RANK
from anjo.sleep.train.lora import LoraCfg, LoraCfgError, Objective

OBJECTIVE_ENV: Final = "ANJO_SLEEP_OBJECTIVE"
FOLD_ENV: Final = "ANJO_SLEEP_ADAPTER_FOLD"
DREAMS_ENV: Final = "ANJO_SLEEP_DREAMS"
MATCH_ENV: Final = "ANJO_SLEEP_MATCH_MINUTES"
OBJECTIVES: Final[tuple[Objective, ...]] = ("sft_dpo", "sft", "sft_dpo_seed")
DEFAULT_OBJECTIVE: Final[Objective] = "sft_dpo"
SEED_OBJECTIVE: Final[Objective] = "sft_dpo_seed"
FOLD_RANK: Final = MAX_LORA_RANK  # the long-term adapter: the most the server takes


class ObjectiveError(ValueError):
    """``ANJO_SLEEP_OBJECTIVE`` names no objective, or a switch can't run with it."""


def objective_name(environ: Mapping[str, str]) -> Objective:
    """The objective ``environ`` names: ``sft_dpo`` when unset or blank."""
    chosen = environ.get(OBJECTIVE_ENV, "").strip() or DEFAULT_OBJECTIVE
    if chosen not in OBJECTIVES:
        raise ObjectiveError(f"{OBJECTIVE_ENV} is {' or '.join(OBJECTIVES)}")
    return chosen


def _on(environ: Mapping[str, str], name: str) -> bool:
    value = environ.get(name, "").strip()
    if value not in ("", "0", "1"):
        raise ObjectiveError(f"{name} is 1 (on) or 0 (off)")
    return value == "1"


def _match_minutes(environ: Mapping[str, str]) -> float | None:
    raw = environ.get(MATCH_ENV, "").strip()
    if not raw:
        return None
    try:
        minutes = float(raw)
    except ValueError:
        raise ObjectiveError(f"{MATCH_ENV} is a number of minutes") from None
    if not math.isfinite(minutes):
        raise ObjectiveError(f"{MATCH_ENV} is a number of minutes")
    return minutes


def night_cfg(environ: Mapping[str, str]) -> LoraCfg:
    """The nights' recipe: the default one with the settings' objective and switches."""
    chosen = objective_name(environ)
    fold, dreams = _on(environ, FOLD_ENV), _on(environ, DREAMS_ENV)
    if (fold or dreams) and chosen != SEED_OBJECTIVE:
        raise ObjectiveError(f"{FOLD_ENV} and {DREAMS_ENV} run only with {OBJECTIVE_ENV}={SEED_OBJECTIVE}")
    try:
        return LoraCfg(objective=chosen, fold_rank=FOLD_RANK if fold else None, dreams=dreams,
                       match_minutes=_match_minutes(environ))
    except LoraCfgError as error:
        raise ObjectiveError(str(error)) from None


def night_terms(environ: Mapping[str, str]) -> Terms:
    """Which extra rows the night builds for the recipe ``environ`` names: none unless ``sft_dpo_seed``."""
    cfg = night_cfg(environ)
    return Terms(seed=cfg.objective == SEED_OBJECTIVE, dreams=cfg.dreams)
