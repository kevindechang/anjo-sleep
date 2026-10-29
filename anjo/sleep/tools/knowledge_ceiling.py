"""The knowledge measure's oracle ceiling: one adapter trained with plain SFT on the very statements the
frozen set's items are made from, then scored on the set like a night (``tools.knowledge_score``).
**CEILING: contamination by design, never a result.** It answers one question: handed the answers, can
this pipeline (the LoRA, its step plan, the rendering and the serving path) store the facts at all?

    python -m anjo.sleep.tools.knowledge_ceiling                          # the plan: nothing is held or sent
    python -m anjo.sleep.tools.knowledge_ceiling --point capacity --go    # train it (paid: one Modal job)
    python -m anjo.sleep.tools.knowledge_score --ceiling capacity=<adapter> --go   # then score it

**The rows:** one per statement behind the set's items (each item's true choice is its statement, word
for word), after the set's own prompt alone (no system turn, as the scorer asks it), trained on the
gates' ``EVAL_DAY``, the date every scored prompt carries. So it is a verbatim, same-prompt ceiling:
it measures storage, not generalisation.

**The points:** ``recipe`` is the nights' own recipe (``LoraCfg()``: rank 16, its rate, its step plan;
cold, SFT); ``capacity`` is the most the server takes (rank 32) and ``CAPACITY_PASSES`` passes over the
rows, one row a step.

**Spend:** one call on the night's Modal runner (``modal_job``, an A100-80GB) under the synthetic person
``CEILING_OWNER``..., holding ``CEILING_TIMEOUT_S`` at Modal's rate first and settling at what ran. It
takes the night lock, so it is refused while a night runs. The adapter lands on the adapters volume,
where the server reads it; it is nobody's, and the run prints its removal. Never prints a statement.
"""
from __future__ import annotations

import argparse
import os
import secrets
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from anjo.sleep.budget import Bound, BudgetError, default_ledger
from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.gates import EVAL_DAY
from anjo.sleep.night.sweep import NightBusy, swept_night
from anjo.sleep.serving import config
from anjo.sleep.sets import knowledge
from anjo.sleep.sets.knowledge import KnowledgeSet
from anjo.sleep.sets.loader import SetError
from anjo.sleep.stops import BudgetStop
from anjo.sleep.train import modal_job
from anjo.sleep.train.dataset import ROLE_OVERHEAD_TOKENS, Dataset, Example
from anjo.sleep.train.lora import LoraCfg, TrainError, adapter_name, plan_steps, train_lora_with_receipt
from anjo.sleep.train.runners import choice, colab
from anjo.sleep.train.runners import modal as modal_runner

CEILING: Final = "CEILING: contamination by design (trained on the set's own statements), never a result"
CEILING_OWNER: Final = "ceiling-knowledge-"  # the ceiling's synthetic person, a fresh one each run
CEILING_TIMEOUT_S: Final = 3600.0  # the job's hard timeout, held in full before it starts
CAPACITY_PASSES: Final = 3
POINTS: Final = ("recipe", "capacity")


class CeilingError(ValueError):
    """The ceiling can't be built as asked."""


def ceiling_dataset(the_set: KnowledgeSet) -> Dataset:
    """One row per statement behind ``the_set``'s items, in the set's order: its prompt, then its line."""
    lines: dict[str, tuple[str, str]] = {}
    for item in the_set.items:
        if lines.setdefault(item.source, (item.prompt, item.statement)) != (item.prompt, item.statement):
            raise CeilingError(f"statement {item.source} has two true lines or prompts in the set")
    examples = tuple(
        Example(id=f"ceiling-{n}", prompt=(("user", prompt),), completion=line, context_dropped=False,
                tokens=estimate_tokens(prompt) + estimate_tokens(line) + 2 * ROLE_OVERHEAD_TOKENS, new=True)
        for n, (prompt, line) in enumerate(lines.values(), 1)
    )
    return Dataset(examples=examples, tokens=sum(e.tokens for e in examples), dropped={})


def point_cfg(point: str, rows: int) -> LoraCfg:
    """The recipe a point trains under: the nights' own, or the most the server takes, several passes."""
    if point == "recipe":
        return LoraCfg(objective="sft")
    if point == "capacity":
        return LoraCfg(objective="sft", r=config.MAX_LORA_RANK, min_optimizer_steps=CAPACITY_PASSES * rows)
    raise CeilingError(f"a point is one of {', '.join(POINTS)}")


def _plan(the_set: KnowledgeSet, data: Dataset, point: str, cfg: LoraCfg, held: float, left: float,
          revision: str | None) -> list[str]:
    steps = plan_steps(len(data.examples), cfg)
    return [
        CEILING,
        f"set: {the_set.persona} v{the_set.version}, {len(the_set.items)} items from {len(data.examples)} statements, "
        f"sha {the_set.sha[:12]}: one row per statement, the set's prompt alone, dated {EVAL_DAY.isoformat()}",
        f"point: {point}, rank {cfg.r}, rate {cfg.lr:g}, cold SFT, accumulation {steps.accumulation} x "
        f"{steps.epochs} epochs = {steps.steps} steps",
        f"runner: modal ({modal_job.APP_NAME}, one {modal_job.GPU}), Glimmer at revision "
        f"{revision or f'(unset: {config.REVISION_ENV})'}",
        f"hold: ${held:.2f} on {modal_runner.POOL} (the {CEILING_TIMEOUT_S / 3600:g} h hard timeout at "
        f"${modal_runner.A100_USD_PER_HOUR:.2f} an hour), ${left:.2f} left; settled at what ran",
        f"then: python -m anjo.sleep.tools.knowledge_score --persona {the_set.persona} --ceiling {point}=<adapter> --go",
    ]


def _landed(folder: Path, name: str) -> None:
    """Nothing to upload: the Modal job landed the adapter on the adapters volume, where the server reads it."""


def main(
    argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
    out: Callable[[str], None] = print, modal_calls: Callable[[], modal_runner.ModalCalls] = modal_runner.SdkCalls,
) -> int:
    """The plan, and with ``--go`` the paid job. Exit 0 trained (or planned), 1 failed, 2 refused."""
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(prog="python -m anjo.sleep.tools.knowledge_ceiling", description=CEILING)
    parser.add_argument("--persona", default="film_founder", help="the persona whose frozen set it is trained on")
    parser.add_argument("--point", choices=POINTS, default="recipe", help="the nights' recipe, or rank 32 and more passes")
    parser.add_argument("--go", action="store_true", help="train (paid); without it, the plan")
    args = parser.parse_args(argv)
    try:
        the_set = knowledge.load(args.persona)
        ledger = default_ledger(env)
    except (SetError, BudgetError) as error:
        out(f"ceiling refused: {error}")
        return 2
    data = ceiling_dataset(the_set)
    cfg = point_cfg(args.point, len(data.examples))
    revision = env.get(config.REVISION_ENV, "").strip() or None
    held = Bound(modal_runner.A100_USD_PER_HOUR, CEILING_TIMEOUT_S).amount
    for line in _plan(the_set, data, args.point, cfg, held, ledger.remaining(modal_runner.POOL), revision):
        out(line)
    if not args.go:
        out("nothing was held, started or spent; --go trains, holding the amount above first")
        return 0
    if revision is None:
        out(f"ceiling refused: {config.REVISION_ENV} names Glimmer's pinned commit")
        return 2
    person = f"{CEILING_OWNER}{args.point}-{secrets.token_hex(6)}"
    name = adapter_name(person, 1, 0)
    chosen = choice.build("modal", ledger=ledger, calls_dir=ledger.path.parent / modal_runner.CALLS_DIR,
                          is_synthetic=lambda person_id: person_id.startswith(CEILING_OWNER),
                          colab_cli=colab.ColabCli, modal_calls=modal_calls)
    try:
        with swept_night(ledger, chosen.sweep), tempfile.TemporaryDirectory(prefix="anjo-ceiling-") as root:
            _, receipt = train_lora_with_receipt(
                None, data, cfg, chosen.runner, person_id=person, night=EVAL_DAY, name=name,
                workdir=Path(root) / "work", adapter_cache=Path(root) / "adapters", publish=_landed,
                revision=revision, timeout_s=CEILING_TIMEOUT_S,
            )
    except (TrainError, modal_runner.ModalTrainError, BudgetStop, NightBusy, TimeoutError) as error:
        out(f"ceiling failed: {error}")
        return 1
    out(f"trained {name} on {receipt.gpu}: {receipt.steps} steps, {receipt.epochs} epochs on {receipt.examples} "
        f"statements, {receipt.dropped_truncated} dropped, {receipt.gpu_minutes:.1f} GPU min; {CEILING}")
    out(f"score it: python -m anjo.sleep.tools.knowledge_score --persona {the_set.persona} "
        f"--ceiling {args.point}={name} --go")
    out(f"then remove it: modal volume rm {config.ADAPTERS_VOLUME} /{name} --recursive")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
