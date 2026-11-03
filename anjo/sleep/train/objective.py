"""Which objective the nights train (Kevin, 2026-10-28): ``ANJO_SLEEP_OBJECTIVE`` is ``sft_dpo_seed``
(the default since 2026-11-01: the sleep paradigm's one loss, ``sleep_loss.py``), ``sft_dpo`` (SFT,
then a DPO pass on the night's (best, worst) practice pairs) or ``sft``. Anything else is refused at
start. The night command, the accelerated runs and the synthetic example build their recipe here.
MIND's thinker (``thinker.py``, Task 17b) never reads it and keeps ``LoraCfg()``'s own ``sft``.

An ``sft_dpo`` night whose practice kept no usable pair trains ``sft`` for that night (``lora.objective_used``).

The fold, dreams and day dreams ride on ``sft_dpo_seed``: an explicit request is refused with any other
objective, and day dreams left unset stay quietly off there:

- ``ANJO_SLEEP_ADAPTER_FOLD=1``: each night trains a fresh slot beside the person's frozen long-term
  adapter, then folds both into one plain LoRA of rank ``FOLD_RANK`` (the server's cap), so capacity
  stays bounded across nights;
- ``ANJO_SLEEP_DREAMS=1``: the night adds Glimmer's dreams from MIND's dream states (``night/dreams.py``);
- ``ANJO_SLEEP_DAY_DREAMS`` (on since 2026-11-01): the night adds earlier practice rows (at most
  ``SEED_REPLAY_MAX``), and her Glimmer's own day dreams from the consolidated day: the messages the
  person might send next and her replies to them (``night/day_dreams.py``). The training job keeps a
  dream group only if it lowers the held-out seeding KL without dropping the anchor score
  (``sleep_loss.py``). Allowed with capacity, which is where their volume pays. Under another
  objective it stays quietly off; an explicit ``1`` there is refused, as before.

- ``ANJO_SLEEP_SDFT=1``: self-distillation (Shenfeld, Damani, Hübotter and Agrawal, "Self-Distillation
  Enables Continual Learning", arXiv 2601.19897): queries built on the host from the seeding rows' own
  messages, bare (``tone``, as a tone_hold probe is) or asking for a choice (``taste``), whose teacher
  reads the consolidated day before the query and whose student reads the query alone, scored on the
  student's own samples (``train/terms.py``, ``sleep_loss.py``). Never in a matched control.

``ANJO_SLEEP_PROTECT`` (``anchors``, ``random`` or ``off``) rides on the fold too, and is refused without it:
tonight's slot trains off the forgetting anchors' activation basis, or a random basis of the same shape
(the ablation), and the fold keeps the projection (``sleep_protect.py``). The night ships the anchor rows.

``ANJO_SLEEP_CAPACITY`` (on since 2026-11-01; the knowledge ceiling's capacity point on a night) trains
at the most rank the server takes (``MAX_LORA_RANK``) and ``CAPACITY_PASSES`` passes over the night's
rows, one row a step, in place of the step plan; the rates, rows and DPO pass stay the recipe's.
``ANJO_SLEEP_CAPACITY=0`` is today's step plan. With any objective (``sft_dpo_seed`` carries it to the
loss as ``terms.passes``; the others in the host's step plan), never with the fold, whose long-term
adapter is at that rank already.

Each of the fold and dreams switches off (unset, blank or ``0``) leaves the night without that part;
``ANJO_SLEEP_CAPACITY=0`` turns capacity off; ``ANJO_SLEEP_DAY_DREAMS=0`` turns day dreams off (unset
they follow the objective, quietly off under another loss). The live comparison's matched control is
``ANJO_SLEEP_OBJECTIVE=sft_dpo`` (today's objective before 2026-11-01); its arms set their switches
per run (``tools/sleep_objective_live.py``), and its clock match is a per-run ``--match-file`` to
``sim.accelerated``, so no host setting can pad a real night.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Final

from anjo.sleep.night.day_dreams import SWITCH_ENV
from anjo.sleep.night.dreams import Terms
from anjo.sleep.serving.config import MAX_LORA_RANK
from anjo.sleep.train.lora import PROTECTIONS, LoraCfg, LoraCfgError, Objective, Protection

OBJECTIVE_ENV: Final = "ANJO_SLEEP_OBJECTIVE"
FOLD_ENV: Final = "ANJO_SLEEP_ADAPTER_FOLD"
DREAMS_ENV: Final = "ANJO_SLEEP_DREAMS"
DAY_DREAMS_ENV: Final = SWITCH_ENV  # the app wires her day dreams by the same name (``night/day_dreams``)
PROTECT_ENV: Final = "ANJO_SLEEP_PROTECT"
CAPACITY_ENV: Final = "ANJO_SLEEP_CAPACITY"
SDFT_ENV: Final = "ANJO_SLEEP_SDFT"
CAPACITY_PASSES: Final = 3
OBJECTIVES: Final[tuple[Objective, ...]] = ("sft_dpo", "sft", "sft_dpo_seed")
DEFAULT_OBJECTIVE: Final[Objective] = "sft_dpo_seed"
SEED_OBJECTIVE: Final[Objective] = "sft_dpo_seed"
FOLD_RANK: Final = MAX_LORA_RANK  # the long-term adapter: the most the server takes


class ObjectiveError(ValueError):
    """``ANJO_SLEEP_OBJECTIVE`` names no objective, or a switch can't run with it."""


def objective_name(environ: Mapping[str, str]) -> Objective:
    """The objective ``environ`` names: ``sft_dpo_seed`` when unset or blank."""
    chosen = environ.get(OBJECTIVE_ENV, "").strip() or DEFAULT_OBJECTIVE
    if chosen not in OBJECTIVES:
        raise ObjectiveError(f"{OBJECTIVE_ENV} is {' or '.join(OBJECTIVES)}")
    return chosen


def _switch(environ: Mapping[str, str], name: str, *, default: bool) -> tuple[bool, bool]:
    """The switch's value, and whether ``environ`` set it: ``default`` when unset or blank; any word
    but ``0`` or ``1`` is refused."""
    value = environ.get(name, "").strip()
    if value not in ("", "0", "1"):
        raise ObjectiveError(f"{name} is 1 (on) or 0 (off)")
    return (default if not value else value == "1"), bool(value)


def _on(environ: Mapping[str, str], name: str) -> bool:
    return _switch(environ, name, default=False)[0]


def _protection(environ: Mapping[str, str]) -> Protection:
    value = environ.get(PROTECT_ENV, "").strip() or "off"
    for known in PROTECTIONS:
        if value == known:
            return known
    raise ObjectiveError(f"{PROTECT_ENV} is {' or '.join(PROTECTIONS)}")


def night_cfg(environ: Mapping[str, str]) -> LoraCfg:
    """The nights' recipe: the default one with the settings' objective and switches."""
    chosen = objective_name(environ)
    fold, dreams = _on(environ, FOLD_ENV), _on(environ, DREAMS_ENV)
    day_requested, day_set = _switch(environ, DAY_DREAMS_ENV, default=True)
    protect = _protection(environ)
    if (fold or dreams or (day_requested and day_set)) and chosen != SEED_OBJECTIVE:
        raise ObjectiveError(f"{FOLD_ENV}, {DREAMS_ENV} and {DAY_DREAMS_ENV} run only with "
                             f"{OBJECTIVE_ENV}={SEED_OBJECTIVE}")
    day_dreams = day_requested and chosen == SEED_OBJECTIVE
    if protect != "off" and not fold:
        raise ObjectiveError(f"{PROTECT_ENV} runs only with {OBJECTIVE_ENV}={SEED_OBJECTIVE} and {FOLD_ENV}=1: "
                             "the fold keeps tonight's slot apart from the long-term adapter")
    capacity = _switch(environ, CAPACITY_ENV, default=True)[0]
    if capacity and fold:
        raise ObjectiveError(f"{CAPACITY_ENV} runs without {FOLD_ENV}: the fold's adapter is at the server's rank already")
    sdft = _on(environ, SDFT_ENV)
    if sdft and chosen != SEED_OBJECTIVE:
        raise ObjectiveError(f"{SDFT_ENV} runs only with {OBJECTIVE_ENV}={SEED_OBJECTIVE}: its KL is the seeding term's")
    try:
        cfg = LoraCfg(objective=chosen, fold_rank=FOLD_RANK if fold else None, dreams=dreams, day_dreams=day_dreams,
                      protect=protect, sdft=sdft)
        return replace(cfg, r=MAX_LORA_RANK, passes=CAPACITY_PASSES) if capacity else cfg
    except LoraCfgError as error:
        raise ObjectiveError(str(error)) from None


def night_terms(environ: Mapping[str, str]) -> Terms:
    """Which extra rows the night builds for the recipe ``environ`` names: none unless ``sft_dpo_seed``."""
    cfg = night_cfg(environ)
    return Terms(seed=cfg.objective == SEED_OBJECTIVE, dreams=cfg.dreams, protect=cfg.protect != "off",
                 day_dreams=cfg.day_dreams, sdft=cfg.sdft)
