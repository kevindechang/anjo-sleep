"""Harness learning: the person's ways of working with them, one vetted edit a night.

The loop is on unless ``ANJO_SLEEP_HARNESS=0``. Off, the night writes no strategy and the morning
asks no ``2H`` arm, so every file it writes is today's. On, a night may add, replace or remove one
line of the person's strategy list after a contrastive probe promotes it past a noise bar
(``harness.loop``); the morning's ``2H`` arm reads the list and is reported beside the registered
arms, never instead of one.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Final

HARNESS_ENV: Final = "ANJO_SLEEP_HARNESS"
# A strategy list is a few ways of working, not a second brief. The store enforces both limits,
# and the loop tells the proposer the same numbers.
STRATEGY_MAX: Final = 8
STRATEGY_WORDS: Final = 40


def harness_on(environ: Mapping[str, str] | None = None) -> bool:
    """The switch: on when unset, blank or ``1``; the exact value ``0`` is off. Any other value is
    off too, as it always was: the switch refuses nothing, it only reads the one word."""
    return (os.environ if environ is None else environ).get(HARNESS_ENV, "").strip() in ("", "1")
