"""The ``IntentionSource`` over Kevin's intentions ledger (Task 7a), read-only.

It reads the open entries the person hasn't kept out of chat (``intentions.open_visible``) and
turns each into a ``Commitment``. A synthetic person has no ledger and gets ``()``. A stamp the
ledger can't read is no date: an undated entry is never due, so the chooser never acts on it.
"""
from __future__ import annotations

from datetime import datetime

from anjo.core import intentions
from anjo.sleep.app.cleared_sessions import is_synthetic
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.types import Commitment


class LedgerIntentions:
    def intentions(self, user_id: str, as_of: datetime) -> tuple[Commitment, ...]:
        require_sleep_host()
        if as_of.tzinfo is None:
            raise ValueError("as_of must carry a zone: it is the night's clock")
        if is_synthetic(user_id):
            return ()
        return tuple(_commitment(record, as_of) for record in intentions.open_visible(user_id))


def _commitment(record: dict[str, object], as_of: datetime) -> Commitment:
    return Commitment(
        id=str(record["id"]),
        label=str(record.get("text") or ""),
        check_after=_check_after(record, as_of),
    )


def _check_after(record: dict[str, object], as_of: datetime) -> datetime | None:
    stamp = record.get("check_after")
    if not isinstance(stamp, str) or not stamp:
        return None
    deadline = record.get("deadline")
    try:
        instant: datetime = intentions.check_after_instant(
            stamp, as_of, deadline if isinstance(deadline, dict) else None,
        )
    except ValueError:
        return None  # not a stamp: an undated entry is never due
    return instant
