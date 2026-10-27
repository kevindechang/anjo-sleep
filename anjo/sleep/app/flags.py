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
