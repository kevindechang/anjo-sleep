"""The panel's record of the experiment (Task 13a, full step): the results chart, the Task 16 result
and the pre-registration ledger's hashes, ready to send.

- ``chart_view``: every arm's score per dimension and night (``reports.results``), development and
  final apart and labeled, never pooled; each night names its probe set (``probes``), and the page
  draws one line a set. The development nights are the film student's own; the
  final ones are the held-out lives', scored after the freeze, so until then the final side is
  ``None`` and the page says "not yet".
- ``task16_view``: MIND's report (``reports.mind.render``), its held-out section line by line, the
  pooled line first. Absent until the final run writes it.
- ``ledger_view``: the ledger's entries, newest first, with whether the chain verifies here and
  whether its head is the one last pushed to the public ``prereg`` branch.

Every free text passes the held-out filter, as in ``panel``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from anjo.sleep.demo.heldout import HeldOutText
from anjo.sleep.prereg.ledger import LedgerEntry, LedgerError, verify_entries
from anjo.sleep.reports.results import ARMS, Night, dimensions

LEDGER_SHOWN: Final = 200  # the newest entries sent; the count says how many there are
HELD_OUT_HEADING: Final = "## Held-out lives"
POOLED: Final = "Pooled"


def _life(nights: Sequence[Night], dims: Sequence[str], name: str | None) -> dict[str, Any]:
    dated = sorted(nights, key=lambda n: n.night)
    return {"name": name, "nights": [
        {"night": number, "date": n.night, "status": n.status, "probes": n.probes,
         "scores": {arm: {d: n.arms[arm][d].score for d in dims if d in n.arms[arm]} for arm in ARMS if arm in n.arms}}
        for number, n in enumerate(dated, start=1)]}


def chart_view(development: Sequence[Night], final_lives: Sequence[Sequence[Night]], *, name: str) -> dict[str, Any]:
    """``development``: the demo person's nights; ``final_lives``: each held-out life's nights. Only
    nights carrying the matching label are drawn, so the two kinds never mix."""
    dev = [n for n in development if n.label == "development"]
    lives = [final for final in ([n for n in life if n.label == "final"] for life in final_lives) if final]
    everything = [*dev, *(n for life in lives for n in life)]
    if not everything:
        return {"state": "none", "dimensions": [], "arms": [], "development": None, "final": None}
    dims = dimensions(everything)
    arms = [arm for arm in ARMS if any(arm in n.arms for n in everything)]
    return {
        "state": "ok", "dimensions": list(dims), "arms": arms,
        "development": {"label": "development", "lives": [_life(dev, dims, name)]} if dev else None,
        "final": {"label": "final", "lives": [_life(life, dims, None) for life in lives]} if lives else None,
    }


def _bullet(line: str) -> str | None:
    stripped = line.strip()
    return stripped[2:].strip() if stripped.startswith("- ") else None


def task16_view(text: str | None, held: HeldOutText) -> dict[str, Any]:
    """MIND's report: the notes before its held-out section, then that section's lines."""
    if text is None:
        return {"state": "none", "headline": None, "notes": [], "lines": []}
    head, _, rest = text.partition(HELD_OUT_HEADING)
    section = rest.split("\n## ", 1)[0]
    notes = [line.strip() for line in head.splitlines() if line.strip() and not line.startswith("#")]
    bullets = [b for b in map(_bullet, section.splitlines()) if b]
    pooled = next((b for b in bullets if b.startswith(POOLED)), None)
    return {
        "state": "ok" if bullets else "none", "headline": held.item(pooled),
        "notes": [held.item(note) for note in notes], "lines": [held.item(b) for b in bullets if b != pooled],
    }


def ledger_view(entries: Sequence[LedgerEntry], *, pushed: str | None, held: HeldOutText) -> dict[str, Any]:
    """The ledger's newest entries, and whether its chain holds and matches the pushed head."""
    if not entries:
        return {"state": "none", "count": 0, "verified": None, "in_sync": None, "head": None, "entries": []}
    try:
        verify_entries(entries)
        verified = True
    except LedgerError:
        verified = False
    head = entries[-1].entry_hash
    return {
        "state": "ok", "count": len(entries), "verified": verified, "head": head,
        "in_sync": None if pushed is None else pushed == head,
        "entries": [{"seq": e.seq, "ts": e.ts, "kind": e.kind, "sha256": e.sha256, "note": held.item(e.note),
                     "entry_hash": e.entry_hash} for e in reversed(entries[-LEDGER_SHOWN:])],
    }
