"""Expand the drawn final lives into runnable persona files (Task 12b step 4), on Spark Open.

    python -m anjo.sleep.sim.expand_cli [--personas-dir DIR] [--dry-run]

It reads the ledger's one ``final_draw`` entry (``lives.draw_cli.registered_draw``) and runs
``expand_life`` for each of its three lives, then writes ``<personas dir>/<life id>.json``: the
folder the simulator, the night and the judge brief read personas from (``anjo/sleep/personas``),
as ``{"visible": ..., "held_out": ...}``, and checks the file loads (``load_person_file``).

**Registered parameters** (this file is hashed at the freeze, ``prereg.register_freeze``):
``EXPAND_SEED``, the seed of attempt 1 (attempt *k* asks seed ``EXPAND_SEED + k - 1``, so a retry
samples again); ``PLANNED_NIGHTS``, the horizon every loader validates new-work arrival days against
(days 2 to 15); at most ``EXPAND_ATTEMPTS`` attempts per life, counted in the ledger across runs.

**Refusal.** ``expand_life`` refuses a file that breaks a rule and never repairs one. Each refusal
is a ``final_expand_refused`` entry (the sha256 of the model's reply, the life, the attempt, the
seed and the reason), and the next attempt runs. A life refused ``EXPAND_ATTEMPTS`` times is not
written: the run goes on to the other lives and exits 3 naming it. The draw stands: no life is
redrawn or replaced, and the README says which life could not be expanded. A failed call (the
provider down) is not an attempt: it stops the run, and the rerun starts the life again.

**Written once.** A life whose file exists and matches its ``final_persona`` entry is done and
skipped; a file that exists without a matching entry stops the run before any call. Each written
file is a ``final_persona`` entry (the file's sha256; its note: life, attempt, seed, horizon, and the
canonical hashes of the visible and held-out parts).

**Cost.** Each call holds its worst case (``SPARK_MAX_IN_TOKENS`` in, ``EXPAND_MAX_OUT_TOKENS`` out,
the reply capped there) on ``baseten_sleep_usd`` (or the pool the selected Spark provider maps it to,
``budget.selected_pools``) and settles the tokens it metered (``lives.prep``). ``--dry-run`` checks the draw
and the files and calls nothing.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from anjo.sleep import baseten_api
from anjo.sleep.budget import BudgetError, Ledger, default_ledger
from anjo.sleep.lives import prep
from anjo.sleep.lives.draw import Life, load_pairs
from anjo.sleep.lives.draw_cli import DrawRefused, registered_draw
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.holds import SPARK_MAX_IN_TOKENS, held, usd
from anjo.sleep.personas.model import (
    PLANNED_NIGHTS,
    canonical_json,
    load_person_file,
    to_held_out_dict,
    to_visible_dict,
)
from anjo.sleep.prereg import ledger
from anjo.sleep.sim.expand import ExpandError, expand_life
from anjo.sleep.stops import BudgetStop

PERSONAS_DIR: Final = Path(__file__).resolve().parents[1] / "personas"
EXPAND_SEED: Final = 20261101
EXPAND_ATTEMPTS: Final = 3
EXPAND_MAX_OUT_TOKENS: Final = 32_768  # a whole life file: 75 preferences, 25+ new work, 15 taste items
EXPAND_TIMEOUT_S: Final = 900.0
SPARK_POOL: Final = ("baseten_sleep_usd",)
PERSONA_KIND: Final = "final_persona"
REFUSED_KIND: Final = "final_expand_refused"
REFUSED_EXIT: Final = 3


class ExpandRunRefused(RuntimeError):
    """A check before any call failed; nothing was called or written."""


class SparkOpenClient:
    """The ``ModelClient`` ``expand_life`` asks: Spark Open through ``baseten_api.chat`` (which sends
    it to the selected Spark provider), the base model only, the reply capped at
    ``EXPAND_MAX_OUT_TOKENS``. It keeps the last reply's text and the tokens metered so far."""

    def __init__(self, chat: Callable[..., ChatResult] = baseten_api.chat) -> None:
        self._chat = chat
        self.tokens: tuple[int, int] = (0, 0)
        self.last_text = ""

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float,
                 seed: int | None = None) -> ChatResult:
        if adapter is not None:
            raise ValueError("life expansion runs on the base Spark Open model, never an adapter")
        self.last_text = ""
        result = self._chat(baseten_api.SPARK_OPEN, messages, None, EXPAND_TIMEOUT_S, temperature=temperature,
                            max_tokens=EXPAND_MAX_OUT_TOKENS, seed=seed)
        self.tokens = (self.tokens[0] + int(result.input_tokens), self.tokens[1] + int(result.output_tokens))
        self.last_text = result.text
        return result


@dataclass(frozen=True)
class LifeOutcome:
    life: str
    status: str  # "written", "done" (already written), "refused", "planned"
    attempts: int = 0
    sha256: str | None = None


def _file_sha(path: Path) -> str | None:
    return ledger.sha256_file(path) if path.exists() else None


def _plan(lives: Sequence[str], personas_dir: Path, entries: Sequence[ledger.LedgerEntry]) -> dict[str, str]:
    """Each life's state before any call: ``done``, ``todo`` or ``exhausted``; a stray file refuses."""
    state = {}
    for life in lives:
        found = _file_sha(personas_dir / f"{life}.json")
        if found is not None:
            if found != prep.registered_sha(entries, PERSONA_KIND, life):
                raise ExpandRunRefused(f"{life}.json exists without a matching {PERSONA_KIND} entry")
            state[life] = "done"
        elif prep.attempts_used(entries, REFUSED_KIND, life) >= EXPAND_ATTEMPTS:
            state[life] = "exhausted"
        else:
            state[life] = "todo"
    return state


def _write(life: Life, pf: Any, personas_dir: Path, attempt: int, seed: int) -> str:
    target = personas_dir / f"{life.id}.json"
    body = {"visible": to_visible_dict(pf.visible), "held_out": to_held_out_dict(pf.held_out)}
    pending = target.with_name(f".{target.name}.tmp")
    pending.write_text(prep.canonical(body) + "\n", encoding="utf-8")
    load_person_file(pending)  # the runtime loaders read it at the default horizon: it must load there
    pending.replace(target)
    sha = ledger.sha256_file(target)
    ledger.register(PERSONA_KIND, sha, prep.canonical({
        "life": life.id, "attempt": attempt, "seed": seed, "planned_nights": PLANNED_NIGHTS,
        "visible_sha256": ledger.sha256_bytes(canonical_json(to_visible_dict(pf.visible))),
        "held_out_sha256": ledger.sha256_bytes(canonical_json(to_held_out_dict(pf.held_out))),
    }))
    return sha


def _expand_one(life: Life, client: SparkOpenClient, personas_dir: Path, budget: Ledger | None,
                prices: tuple[float, float]) -> LifeOutcome:
    used = prep.attempts_used(ledger.read_entries(), REFUSED_KIND, life.id)
    worst = prep.tokens_usd(prices, SPARK_MAX_IN_TOKENS, EXPAND_MAX_OUT_TOKENS)
    for attempt in range(used + 1, EXPAND_ATTEMPTS + 1):
        seed = EXPAND_SEED + attempt - 1
        before = client.tokens

        def spent(before: tuple[int, int] = before) -> float:
            return prep.tokens_usd(prices, client.tokens[0] - before[0], client.tokens[1] - before[1])

        try:
            with held(budget, f"{prep.HOLD_PREFIX}expand:{life.id}:s{seed}", SPARK_POOL, usd(worst), spent):
                pf = expand_life(life, seed, client=client, planned_nights=PLANNED_NIGHTS)
        except ExpandError as refused:
            ledger.register(REFUSED_KIND, ledger.sha256_bytes(client.last_text.encode()), prep.canonical(
                {"life": life.id, "attempt": attempt, "seed": seed, "reason": str(refused)}))
            continue
        return LifeOutcome(life.id, "written", attempt, _write(life, pf, personas_dir, attempt, seed))
    return LifeOutcome(life.id, "refused", EXPAND_ATTEMPTS)


def run(*, client: SparkOpenClient, personas_dir: Path = PERSONAS_DIR, budget: Ledger | None,
        environ: Mapping[str, str] | None = None, dry_run: bool = False) -> list[LifeOutcome]:
    """Expand every drawn life not yet written. Raises ``ExpandRunRefused`` before any call."""
    env = os.environ if environ is None else environ
    try:
        draw = registered_draw()
    except DrawRefused as error:
        raise ExpandRunRefused(str(error)) from None
    lives = {lf.id: lf for p in load_pairs() for lf in p.lives}
    prices = prep.spark_prices(env)
    with prep.swept(None if dry_run else budget):
        state = _plan(draw.lives, personas_dir, ledger.read_entries())
        out = []
        for life_id in draw.lives:
            if state[life_id] == "done":
                out.append(LifeOutcome(life_id, "done", sha256=_file_sha(personas_dir / f"{life_id}.json")))
            elif state[life_id] == "exhausted":
                out.append(LifeOutcome(life_id, "refused", EXPAND_ATTEMPTS))
            elif dry_run:
                out.append(LifeOutcome(life_id, "planned"))
            else:
                out.append(_expand_one(lives[life_id], client, personas_dir, budget, prices))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Expand the drawn final lives into persona files, on Spark Open.")
    ap.add_argument("--personas-dir", type=Path, default=PERSONAS_DIR)
    ap.add_argument("--dry-run", action="store_true", help="check the draw and the files; call nothing")
    args = ap.parse_args(argv)
    try:
        budget = None if args.dry_run else default_ledger()
        outcomes = run(client=SparkOpenClient(), personas_dir=args.personas_dir, budget=budget, dry_run=args.dry_run)
    except (ExpandRunRefused, ledger.LedgerError, BudgetError, BudgetStop, ValueError, OSError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    for o in outcomes:
        print(prep.canonical({"life": o.life, "status": o.status, "attempts": o.attempts, "sha256": o.sha256}))
    return REFUSED_EXIT if any(o.status == "refused" for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
