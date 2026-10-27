"""The held note's delivery (Task 5), through Anjo's existing outreach.

- ``OutreachNotifier.queue`` holds the note as the person's pending outreach
  (``outreach.queue_outreach``), lock-screen private, for their next open.
- ``OutreachNotifier.push`` announces it only for a person who turned notifications on and has a
  device to reach, through the existing ``send_outreach_push`` and all its gates. The note's source
  isn't one the push may quote, so a lock screen only ever shows the generic copy.
- ``no_delivery(user_id)`` holds pending outreach back while a cleared session opens: the note
  waits for the next uncleared open (``outreach.DELIVERY_HOLDS``, checked by ``held``).
- ``passes_reply_check`` is Anjo's reply-side crisis check, run on each note's final text
  (sources and held-back line included); a note that trips it is declined, never queued.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_HELD: ContextVar[frozenset[str]] = ContextVar("sleep_delivery_held", default=frozenset())


def held(user_id: str) -> bool:
    """True while ``user_id``'s cleared session is opening."""
    return user_id in _HELD.get()


@contextmanager
def no_delivery(user_id: str) -> Iterator[None]:
    token = _HELD.set(_HELD.get() | {user_id})
    try:
        yield
    finally:
        _HELD.reset(token)


class OutreachNotifier:
    """The ``Notifier`` port over ``anjo.core.outreach`` and ``anjo.core.notifications``."""

    def queue(self, user_id: str, message: str, source: str) -> None:
        from anjo.core.outreach import queue_outreach

        queue_outreach(user_id, message, source)

    def push(self, user_id: str, message: str) -> bool:
        from anjo.core.notifications import ensure_preferences, send_outreach_push

        preferences = ensure_preferences(user_id)
        if not preferences["enabled"] or preferences["token_count"] <= 0:
            return False
        return bool(send_outreach_push(user_id))


def passes_reply_check(text: str) -> bool:
    """True when ``text`` passes Anjo's reply-side crisis check."""
    from anjo.core.crisis_reply import reply_crisis_reason

    return reply_crisis_reason(text) is None
