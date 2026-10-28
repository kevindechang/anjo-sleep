"""Training replay's two thinker adapters (Task 17b): thinker-1 and thinker-shuffled.

- **The recipe** is Task 5's (``LoraCfg``, ``train_lora_with_receipt``): QLoRA on Glimmer from a
  cold start, with identical settings for both. Both datasets hold one whole trajectory from each
  of the same start states, so their rows and tokens may differ; one step plan serves both
  (``shared_steps``: the larger set's accumulation and step count, sent as ``max_steps``), so each
  takes the same updates and sees every row at least once. The plan records each set's rows and
  tokens. No trajectory is ever cut: the dataset drops a start state whose rows are too long by the
  host's estimate, and the script, counting with the real tokenizer, stops before the model loads
  if any row would still be dropped (``keep_every_row``), so only the VM's boot is spent.
- **Colab only, never Modal:** ``train_thinkers`` takes a ``ColabRunner``, which ``colab_runner``
  builds to train replay's one synthetic owner (``REPLAY_OWNER``) and nobody else, in MIND's own
  sessions (``anjo-mind-…``), so a Sleep night's sweep never stops one. The runner holds
  each job's worst case on the ``colab_cu`` pool before its VM starts; ``train_thinkers`` first
  checks the pool can hold both jobs, so it never trains one and stops on the other.
- **Rendering:** each row is rendered at the server's reasoning strength on the date it was
  served on (its ``current_date``: the branch's night); the run's own date, and the adapter's
  ``night``, is the latest of them (``render_night``). Each adapter records that rendering and its
  ``CheckpointRef`` carries it, so the server never serves it under another (``served.py``).
- **Dry run first:** ``plan`` is pure, and a dry run (the default) prints it (rows, tokens, steps,
  estimated A100 minutes and compute units, and the hold beside what the pool has left) and
  touches neither the runner nor a hold.

The estimates are the spec's: about 1-2M tokens in 30-60 minutes on one A100-80GB
(``A100_TOKENS_PER_MINUTE``), plus ``SETUP_MINUTES`` for the VM, the installs and the base
weights. Compute units are minutes at the runner's A100 rate (``A100_CU_PER_HOUR``).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

from anjo.sleep.budget import TRAINING_TIMEOUT_S, Ledger
from anjo.sleep.mind.replay.dataset import ReplayData
from anjo.sleep.stops import BudgetStop
from anjo.sleep.train.dataset import Dataset
from anjo.sleep.train.lora import (
    AdapterPublisher,
    LoraCfg,
    StepPlan,
    TrainError,
    fresh_adapter_name,
    plan_steps,
    train_lora_with_receipt,
)
from anjo.sleep.train.runners.colab import A100_CU_PER_HOUR, MIND_SESSION_PREFIX, POOL, ColabCli, ColabRunner
from anjo.sleep.types import CheckpointRef

REPLAY_OWNER: Final = "apd-replay"  # the synthetic owner every replay adapter trains under
LABELS: Final = ("thinker-1", "thinker-shuffled")
REPLAY_NIGHT: Final = 0  # the ``n<night>`` of a replay adapter's name: no person's night
A100_TOKENS_PER_MINUTE: Final = 33_000.0
SETUP_MINUTES: Final = 15.0


def is_replay_owner(person_id: str) -> bool:
    return person_id == REPLAY_OWNER


def colab_runner(cli: ColabCli, ledger: Ledger, *, clock: Callable[[], float] = time.monotonic) -> ColabRunner:
    """The Colab runner for replay: it trains ``REPLAY_OWNER`` only, never a real person, in
    sessions named ``anjo-mind-…``."""
    return ColabRunner(cli=cli, ledger=ledger, is_synthetic=is_replay_owner, clock=clock,
                       session_prefix=MIND_SESSION_PREFIX)


@dataclass(frozen=True)
class AdapterPlan:
    label: str
    examples: int
    tokens: int
    steps: int
    epochs: int
    accumulation: int
    minutes: float
    compute_units: float


@dataclass(frozen=True)
class ThinkerPlan:
    """Both adapters' plans, the worst case the runner holds for each job, and what the pool has
    left (``None`` when no ledger was read)."""

    adapters: tuple[AdapterPlan, ...]
    hold_cu: float
    remaining_cu: float | None

    @property
    def compute_units(self) -> float:
        return sum(adapter.compute_units for adapter in self.adapters)


def _starts(data: Dataset) -> frozenset[str]:
    return frozenset(example.id.rsplit(":b", 1)[0] for example in data.examples)


def shared_steps(data: ReplayData, cfg: LoraCfg, scale: int = 1) -> StepPlan:
    """The one step plan both adapters train with: the larger set's, so the smaller one's rows
    are each seen at least once too. Refuses sets that don't draw on the same start states."""
    if not data.thinker1.examples or _starts(data.thinker1) != _starts(data.shuffled):
        raise TrainError("thinker-1 and thinker-shuffled each hold one trajectory from the same start states")
    return plan_steps(scale * max(len(data.thinker1.examples), len(data.shuffled.examples)), cfg)


def _adapter_plan(label: str, data: Dataset, steps: StepPlan, scale: int = 1) -> AdapterPlan:
    seen = steps.steps * steps.accumulation * data.tokens / len(data.examples)  # tokens over every update
    minutes = SETUP_MINUTES + seen / A100_TOKENS_PER_MINUTE
    return AdapterPlan(
        label=label, examples=scale * len(data.examples), tokens=scale * data.tokens, steps=steps.steps,
        epochs=steps.epochs, accumulation=steps.accumulation, minutes=minutes,
        compute_units=minutes / 60.0 * A100_CU_PER_HOUR,
    )


def plan(
    data: ReplayData, cfg: LoraCfg = LoraCfg(), *, timeout_s: float = TRAINING_TIMEOUT_S,
    remaining_cu: float | None = None, scale: int = 1,
) -> ThinkerPlan:
    """What training both adapters would take, on one shared step plan. ``scale`` sizes it as if
    each set held that many times its rows at the same tokens a row (APD's timing measures one
    development life and projects all of them)."""
    if scale < 1:
        raise TrainError("a plan's scale is at least 1")
    steps = shared_steps(data, cfg, scale)
    adapters = tuple(_adapter_plan(label, d, steps, scale)
                     for label, d in zip(LABELS, (data.thinker1, data.shuffled), strict=True))
    return ThinkerPlan(adapters=adapters, hold_cu=timeout_s / 3600.0 * A100_CU_PER_HOUR, remaining_cu=remaining_cu)


def describe(found: ThinkerPlan) -> str:
    """The dry run's printout."""
    lines = ["Replay training plan (Task 5's QLoRA recipe on one Colab A100-80GB each, cold start, "
             "the same settings and the same number of updates for both):"]
    for a in found.adapters:
        lines.append(
            f"- {a.label}: {a.examples} rows, about {a.tokens:,} tokens; {a.steps} optimizer steps "
            f"(max_steps {a.steps}, accumulation {a.accumulation}); about {a.minutes:.0f} A100 minutes, "
            f"{a.compute_units:.1f} compute units"
        )
    left = "unknown" if found.remaining_cu is None else f"{found.remaining_cu:.1f}"
    lines.append(
        f"Estimated cost: {found.compute_units:.1f} compute units in all. The runner holds "
        f"{found.hold_cu:.1f} per job on {POOL} before it starts ({left} left)."
    )
    return "\n".join(lines)


def _check_room(found: ThinkerPlan, timeout_s: float) -> None:
    if any(a.minutes * 60.0 > timeout_s for a in found.adapters):
        raise TrainError("an adapter's estimate doesn't fit the training timeout; refused before any VM")
    room = found.remaining_cu or 0.0
    if room < found.hold_cu * len(found.adapters):
        raise BudgetStop(f"{POOL} can't hold both replay jobs ({room:.1f} left)")


def render_night(data: ReplayData) -> date:
    """The latest date a row was served on: the training run's own date. Every row carries one."""
    days = [example.current_date for d in (data.thinker1, data.shuffled) for example in d.examples]
    if not days or any(day is None for day in days):
        raise TrainError("every replay row carries the date it was served on")
    return max(day for day in days if day is not None)


def _used(adapter_cache: Path) -> frozenset[str]:
    return frozenset(p.name for p in adapter_cache.iterdir()) if adapter_cache.is_dir() else frozenset()


def train_thinkers(
    data: ReplayData, runner: ColabRunner, ledger: Ledger, *, workdir: Path, adapter_cache: Path,
    publish: AdapterPublisher, revision: str | None, cfg: LoraCfg = LoraCfg(), dry_run: bool = True,
    timeout_s: float = TRAINING_TIMEOUT_S, out: Callable[[str], None] = print,
) -> tuple[ThinkerPlan, tuple[CheckpointRef, ...]]:
    """Print the plan; unless ``dry_run``, then train thinker-1 and thinker-shuffled on Colab."""
    found = plan(data, cfg, timeout_s=timeout_s, remaining_cu=ledger.remaining(POOL))
    out(describe(found))
    if dry_run:
        return found, ()
    if not isinstance(runner, ColabRunner):
        raise TrainError("replay's adapters train on the Colab runner only")
    _check_room(found, timeout_s)
    used: frozenset[str] = _used(adapter_cache)
    night, steps = render_night(data), shared_steps(data, cfg)
    refs: tuple[CheckpointRef, ...] = ()
    for label, dataset in zip(LABELS, (data.thinker1, data.shuffled), strict=True):
        name = fresh_adapter_name(f"{REPLAY_OWNER}:{label}", REPLAY_NIGHT, used)
        ref, _ = train_lora_with_receipt(
            None, dataset, cfg, runner, person_id=REPLAY_OWNER, night=night, name=name,
            workdir=workdir / label, adapter_cache=adapter_cache, publish=publish, revision=revision,
            timeout_s=timeout_s, steps=steps, keep_every_row=True,
        )
        used, refs = used | {name}, (*refs, ref)
    return found, refs
