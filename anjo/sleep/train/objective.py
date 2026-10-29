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

``ANJO_SLEEP_PROTECT`` (``anchors``, ``random`` or ``off``) rides on the fold too, and is refused without it:
tonight's slot trains off the forgetting anchors' activation basis, or a random basis of the same shape
(the ablation), and the fold keeps the projection (``sleep_protect.py``). The night ships the anchor rows.

``ANJO_SLEEP_CAPACITY=1`` (2026-10-29, the knowledge ceiling's capacity point on a night) trains at the
most rank the server takes (``MAX_LORA_RANK``) and ``CAPACITY_PASSES`` passes over the night's rows, one
row a step, in place of the step plan; the rates, rows and DPO pass stay the recipe's. With any objective
(``sft_dpo_seed`` carries it to the loss as ``terms.passes``; the others in the host's step plan), never
with the fold, whose long-term adapter is at that rank already.

Each switch off (unset, blank or ``0``) is today's night. The live comparison's matched control is no
switch: it is set per run (``sim.accelerated --match-file``), so no host setting can pad a real night.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Final

from anjo.sleep.night.dreams import Terms
from anjo.sleep.serving.config import MAX_LORA_RANK
from anjo.sleep.train.lora import PROTECTIONS, LoraCfg, LoraCfgError, Objective, Protection

OBJECTIVE_ENV: Final = "ANJO_SLEEP_OBJECTIVE"
FOLD_ENV: Final = "ANJO_SLEEP_ADAPTER_FOLD"
DREAMS_ENV: Final = "ANJO_SLEEP_DREAMS"
PROTECT_ENV: Final = "ANJO_SLEEP_PROTECT"
CAPACITY_ENV: Final = "ANJO_SLEEP_CAPACITY"
CAPACITY_PASSES: Final = 3
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


def _protection(environ: Mapping[str, str]) -> Protection:
    value = environ.get(PROTECT_ENV, "").strip() or "off"
    for known in PROTECTIONS:
        if value == known:
            return known
    raise ObjectiveError(f"{PROTECT_ENV} is {' or '.join(PROTECTIONS)}")


def night_cfg(environ: Mapping[str, str]) -> LoraCfg:
    """The nights' recipe: the default one with the settings' objective and switches."""
    chosen = objective_name(environ)
    fold, dreams, protect = _on(environ, FOLD_ENV), _on(environ, DREAMS_ENV), _protection(environ)
    if (fold or dreams) and chosen != SEED_OBJECTIVE:
        raise ObjectiveError(f"{FOLD_ENV} and {DREAMS_ENV} run only with {OBJECTIVE_ENV}={SEED_OBJECTIVE}")
    if protect != "off" and not fold:
        raise ObjectiveError(f"{PROTECT_ENV} runs only with {OBJECTIVE_ENV}={SEED_OBJECTIVE} and {FOLD_ENV}=1: "
                             "the fold keeps tonight's slot apart from the long-term adapter")
    capacity = _on(environ, CAPACITY_ENV)
    if capacity and fold:
        raise ObjectiveError(f"{CAPACITY_ENV} runs without {FOLD_ENV}: the fold's adapter is at the server's rank already")
    try:
        cfg = LoraCfg(objective=chosen, fold_rank=FOLD_RANK if fold else None, dreams=dreams, protect=protect)
        return replace(cfg, r=MAX_LORA_RANK, passes=CAPACITY_PASSES) if capacity else cfg
    except LoraCfgError as error:
        raise ObjectiveError(str(error)) from None


def night_terms(environ: Mapping[str, str]) -> Terms:
    """Which extra rows the night builds for the recipe ``environ`` names: none unless ``sft_dpo_seed``."""
    cfg = night_cfg(environ)
    return Terms(seed=cfg.objective == SEED_OBJECTIVE, dreams=cfg.dreams, protect=cfg.protect != "off")
