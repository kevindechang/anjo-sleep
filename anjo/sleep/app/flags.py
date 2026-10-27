"""Anjo Sleep's switches (spec §12.1).

Every sleep code path checks the host flag first. With ``ANJO_SLEEP_HOST`` unset, the app
behaves byte for byte as on main: no sleep code runs and no export changes.
"""
from __future__ import annotations

import os
from typing import Final

HOST_ENV: Final = "ANJO_SLEEP_HOST"


class SleepHostOff(RuntimeError):
    """A sleep code path ran on a host without ``ANJO_SLEEP_HOST=1``."""


def sleep_host() -> bool:
    return os.environ.get(HOST_ENV) == "1"


def require_sleep_host() -> None:
    if not sleep_host():
        raise SleepHostOff(f"Anjo Sleep runs only with {HOST_ENV}=1")


# ── Per person (Task 4) ───────────────────────────────────────────────────────

CAPABILITY: Final = "sleep"


def sleep_enabled_for(user_id: str) -> bool:
    """Whether this person's turns and nights are Anjo Sleep's: the host flag first, then the
    ``sleep`` capability's switch and cohort (``ANJO_SLEEP_ENABLED``,
    ``ANJO_SLEEP_FORCED_USER_IDS``, ``ANJO_SLEEP_ROLLOUT_PCT``). Off the host the capability is
    never declared, so forced ids alone flag nobody and the app's manifest is unchanged."""
    if not sleep_host():
        return False
    from anjo.core.capabilities import capability_enabled_for, declare_sleep_capability

    declare_sleep_capability()
    return bool(capability_enabled_for(CAPABILITY, user_id))
