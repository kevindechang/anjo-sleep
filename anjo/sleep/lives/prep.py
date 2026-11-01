"""What the final lives' paid preparation shares (expansion, probes): holds, prices and the ledger's record.

Each paid call holds its worst case on the budget ledger before it is sent and settles the tokens it
metered (``night.holds.held``), under the night lock, so no night starts while it runs. A run killed
hard leaves its holds open: the next run settles every open ``final_prep:`` hold at its full bound
first, the honest charge for work that may have run (as the night's sweep does for its own kinds).

Tokens are priced at ``ANJO_SLEEP_PRICES``' Spark lines (``spark_in,spark_out,train,serve`` per
million), or the selected Spark provider's own price when it lists one, as the night prices them.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Final

from anjo.sleep import spark_provider
from anjo.sleep.budget import Ledger
from anjo.sleep.night.sweep import night_lock
from anjo.sleep.prereg import ledger as prereg

HOLD_PREFIX: Final = "final_prep:"
PRICES_ENV: Final = "ANJO_SLEEP_PRICES"
SPARK_SETTING: Final = spark_provider.SETTING  # the Spark Open provider's setting, for the freeze's record


def spark_prices(environ: Mapping[str, str], *, provider: bool = True) -> tuple[float, float]:
    """Spark's dollars per million tokens in and out. ``provider``: the selected Spark Open
    provider's listed price wins (Spark Open's calls); off for Muse Spark on the Meta API."""
    parts = environ.get(PRICES_ENV, "").split(",")
    try:
        found = tuple(float(p) for p in parts)
    except ValueError:
        found = ()
    if len(found) != 4 or any(p < 0 for p in found):
        raise ValueError(f"{PRICES_ENV} holds four prices (Spark in and out per million, the GPUs' hours)")
    listed = spark_provider.from_env(environ).price if provider else None
    return listed if listed is not None else (found[0], found[1])


def tokens_usd(prices: tuple[float, float], tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * prices[0] + tokens_out * prices[1]) / 1e6


@contextmanager
def swept(budget: Ledger | None) -> Iterator[int]:
    """The night lock, after any open ``final_prep:`` hold is settled at its full bound. Yields how
    many were. Without a budget ledger (tests) nothing is locked or held."""
    if budget is None:
        yield 0
        return
    with night_lock(budget):
        stale = [h for h in budget.open_holds() if h.job.startswith(HOLD_PREFIX)]
        for h in stale:
            h.settle(h.bound.amount)
        yield len(stale)


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def notes(entries: Sequence[prereg.LedgerEntry], kind: str) -> list[tuple[prereg.LedgerEntry, dict[str, Any]]]:
    """Each ``kind`` entry with its note read as JSON (a note that isn't JSON is skipped)."""
    out = []
    for e in entries:
        if e.kind != kind:
            continue
        try:
            body = json.loads(e.note)
        except json.JSONDecodeError:
            continue
        if isinstance(body, dict):
            out.append((e, body))
    return out


def attempts_used(entries: Sequence[prereg.LedgerEntry], kind: str, life: str) -> int:
    """How many refused attempts the ledger records for ``life`` under ``kind``."""
    return sum(1 for _, body in notes(entries, kind) if body.get("life") == life)


def registered_sha(entries: Sequence[prereg.LedgerEntry], kind: str, life: str) -> str | None:
    """The sha256 the last ``kind`` entry for ``life`` registered, or None."""
    found = [e.sha256 for e, body in notes(entries, kind) if body.get("life") == life]
    return found[-1] if found else None
