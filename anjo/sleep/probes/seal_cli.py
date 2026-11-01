"""Write and seal the drawn final lives' probes (Task 12b step 4), on Muse Spark.

    python -m anjo.sleep.probes.seal_cli [--personas-dir DIR] [--sealed-dir DIR] [--dry-run]

For each life of the ledger's one ``final_draw`` entry it reads the expanded persona file (which
must match its ``final_persona`` entry), runs ``generate_probes`` (Muse Spark on the Meta API, the
final test's probe writer, temperature 0) with the registered ``PROBE_SEED``, and seals them with
``seal_probes`` into ``ANJO_SLEEP_SEALED_DIR`` (or ``--sealed-dir``) as ``<life id>.jsonl``: the file
the night and the morning load for that persona. Each sealed file is a ``final_sealed`` entry (the
file's sha256; its note: life, seed, probe counts per kind).

**Never overwritten.** A sealed file that exists and matches its ``final_sealed`` entry is done and
skipped; one that exists without a matching entry stops the run before any call.

**Refusal.** ``generate_probes`` refuses output that breaks a rule (``ProbeGenError``). Each refusal
is a ``final_probes_refused`` entry, and the life is tried again on the next run, at most
``PROBE_ATTEMPTS`` times in all, counted in the ledger. A life out of attempts exits 3 and is not
sealed; no life is redrawn. A failed call is not an attempt: it stops the run.

**Cost.** A life's four calls hold their worst case (``SPARK_MAX_IN_TOKENS`` in and
``PROBE_MAX_OUT_TOKENS`` out each, the reply capped there) on ``meta_usd`` and settle the tokens they
metered, priced at ``ANJO_SLEEP_PRICES``' Spark lines as the morning's judge is (``lives.prep``).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from anjo.sleep import meta_api
from anjo.sleep.budget import BudgetError, Ledger, default_ledger
from anjo.sleep.lives import prep
from anjo.sleep.lives.draw_cli import DrawRefused, registered_draw
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.holds import SPARK_MAX_IN_TOKENS, held, usd
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.prereg import ledger
from anjo.sleep.probes.generate import WRITTEN_KINDS, ProbeGenError, generate_probes, seal_probes
from anjo.sleep.sim.expand_cli import PERSONA_KIND, PERSONAS_DIR
from anjo.sleep.stops import BudgetStop

SEALED_ENV: Final = "ANJO_SLEEP_SEALED_DIR"
PROBE_MODEL: Final = "muse-spark"
PROBE_SEED: Final = 20261101
PROBE_ATTEMPTS: Final = 3
PROBE_CALLS: Final = len(WRITTEN_KINDS) + 1  # the written kinds, then new_work; taste_ab needs no call
PROBE_MAX_OUT_TOKENS: Final = 16_384
META_POOL: Final = ("meta_usd",)
SEALED_KIND: Final = "final_sealed"
REFUSED_KIND: Final = "final_probes_refused"
REFUSED_EXIT: Final = 3


class SealRunRefused(RuntimeError):
    """A check before any call failed; nothing was called or written."""


class MeteredChat:
    """``generate_probes``' chat: Muse Spark on the Meta API, each reply capped at
    ``PROBE_MAX_OUT_TOKENS``, the tokens metered for the hold."""

    def __init__(self, chat: Callable[..., ChatResult] = meta_api.chat) -> None:
        self._chat = chat
        self.tokens: tuple[int, int] = (0, 0)

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                 timeout_s: float, *, temperature: float | None = None) -> ChatResult:
        result = self._chat(model, messages, tools, timeout_s, temperature=temperature,
                            max_tokens=PROBE_MAX_OUT_TOKENS)
        self.tokens = (self.tokens[0] + int(result.input_tokens), self.tokens[1] + int(result.output_tokens))
        return result


@dataclass(frozen=True)
class SealOutcome:
    life: str
    status: str  # "sealed", "done" (already sealed), "refused", "planned"
    sha256: str | None = None


def _plan(lives: Sequence[str], personas_dir: Path, sealed_dir: Path,
          entries: Sequence[ledger.LedgerEntry]) -> dict[str, str]:
    state = {}
    for life in lives:
        persona = personas_dir / f"{life}.json"
        if not persona.exists() or ledger.sha256_file(persona) != prep.registered_sha(entries, PERSONA_KIND, life):
            raise SealRunRefused(f"{life}.json is missing or doesn't match its {PERSONA_KIND} entry")
        sealed = sealed_dir / f"{life}.jsonl"
        if sealed.exists():
            if ledger.sha256_file(sealed) != prep.registered_sha(entries, SEALED_KIND, life):
                raise SealRunRefused(f"{sealed.name} exists without a matching {SEALED_KIND} entry")
            state[life] = "done"
        elif prep.attempts_used(entries, REFUSED_KIND, life) >= PROBE_ATTEMPTS:
            state[life] = "exhausted"
        else:
            state[life] = "todo"
    return state


def _seal_one(life: str, chat: MeteredChat, personas_dir: Path, sealed_dir: Path, budget: Ledger | None,
              prices: tuple[float, float]) -> SealOutcome:
    pf = load_person_file(personas_dir / f"{life}.json")
    worst = PROBE_CALLS * prep.tokens_usd(prices, SPARK_MAX_IN_TOKENS, PROBE_MAX_OUT_TOKENS)
    before = chat.tokens

    def spent() -> float:
        return prep.tokens_usd(prices, chat.tokens[0] - before[0], chat.tokens[1] - before[1])

    try:
        with held(budget, f"{prep.HOLD_PREFIX}probes:{life}:s{PROBE_SEED}", META_POOL, usd(worst), spent):
            probes = generate_probes(pf, PROBE_SEED, chat=chat, model=PROBE_MODEL)
    except ProbeGenError as refused:
        ledger.register(REFUSED_KIND, ledger.sha256_bytes(str(refused).encode()), prep.canonical(
            {"life": life, "seed": PROBE_SEED, "reason": str(refused)}))
        return SealOutcome(life, "refused")
    sha = seal_probes(probes, sealed_dir)
    counts: dict[str, int] = {}
    for p in probes:
        counts[p.kind] = counts.get(p.kind, 0) + 1
    ledger.register(SEALED_KIND, sha, prep.canonical({"life": life, "seed": PROBE_SEED, "counts": counts}))
    return SealOutcome(life, "sealed", sha)


def run(*, chat: MeteredChat, budget: Ledger | None, personas_dir: Path = PERSONAS_DIR,
        sealed_dir: Path | None = None, environ: Mapping[str, str] | None = None,
        dry_run: bool = False) -> list[SealOutcome]:
    """Seal every drawn life not yet sealed. Raises ``SealRunRefused`` before any call."""
    env = os.environ if environ is None else environ
    named = env.get(SEALED_ENV, "").strip()
    if sealed_dir is None and not named:
        raise SealRunRefused(f"{SEALED_ENV} is not set and no --sealed-dir was given")
    folder = sealed_dir if sealed_dir is not None else Path(named)
    try:
        draw = registered_draw()
    except DrawRefused as error:
        raise SealRunRefused(str(error)) from None
    prices = prep.spark_prices(env, provider=False)
    with prep.swept(None if dry_run else budget):
        state = _plan(draw.lives, personas_dir, folder, ledger.read_entries())
        out = []
        for life in draw.lives:
            if state[life] == "done":
                out.append(SealOutcome(life, "done", ledger.sha256_file(folder / f"{life}.jsonl")))
            elif state[life] == "exhausted":
                out.append(SealOutcome(life, "refused"))
            elif dry_run:
                out.append(SealOutcome(life, "planned"))
            else:
                out.append(_seal_one(life, chat, personas_dir, folder, budget, prices))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write and seal the drawn final lives' probes, on Muse Spark.")
    ap.add_argument("--personas-dir", type=Path, default=PERSONAS_DIR)
    ap.add_argument("--sealed-dir", type=Path, default=None, help=f"default: ${SEALED_ENV}")
    ap.add_argument("--dry-run", action="store_true", help="check the draw and the files; call nothing")
    args = ap.parse_args(argv)
    try:
        budget = None if args.dry_run else default_ledger()
        outcomes = run(chat=MeteredChat(), budget=budget, personas_dir=args.personas_dir,
                       sealed_dir=args.sealed_dir, dry_run=args.dry_run)
    except (SealRunRefused, ledger.LedgerError, BudgetError, BudgetStop, ValueError, OSError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    for o in outcomes:
        print(prep.canonical({"life": o.life, "status": o.status, "sha256": o.sha256}))
    return REFUSED_EXIT if any(o.status == "refused" for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
