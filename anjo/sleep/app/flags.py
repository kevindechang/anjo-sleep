"""Anjo Sleep's switches (spec §12.1).

Every sleep code path checks the host flag first. With ``ANJO_SLEEP_HOST`` unset, the app
behaves byte for byte as on main: no sleep code runs and no export changes.
"""
from __future__ import annotations

import os
from pathlib import Path
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
    apply_tombstones()
    from anjo.core.capabilities import capability_enabled_for, declare_sleep_capability

    declare_sleep_capability()
    return bool(capability_enabled_for(CAPABILITY, user_id))


FORCED_ENV: Final = "ANJO_SLEEP_FORCED_USER_IDS"


def apply_tombstones() -> None:
    """Take every deleted person out of this process's forced-id list (Task 9). Called at startup
    and before every flag read, so the list is always the setting minus the tombstones."""
    forced = os.environ.get(FORCED_ENV, "")
    if not forced:
        return
    from anjo.core import paths
    from anjo.sleep.ownership.tombstone import Tombstones

    kept = Tombstones(Path(paths.DATA_ROOT) / "sleep" / "deleted").without(forced)
    if kept != forced:
        os.environ[FORCED_ENV] = kept
