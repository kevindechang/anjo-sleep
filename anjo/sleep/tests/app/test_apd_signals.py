# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""SIGTERM and SIGHUP end an APD command as Ctrl-C does, through its cleanup (Task 17 review), so
its Colab session is stopped and its holds settled. Private import: the handler is the CLI's own."""
from __future__ import annotations

import os
import signal
import time

import pytest

from anjo.sleep.mind.apd import __main__ as apd_main


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_a_terminating_signal_runs_every_finally_and_restores_the_handler(sig: signal.Signals) -> None:
    before = signal.getsignal(sig)
    cleaned: list[str] = []
    with pytest.raises(SystemExit) as stopped, apd_main._signals_exit():
        try:
            os.kill(os.getpid(), sig)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:  # the handler runs in this thread, between bytecodes
                time.sleep(0.01)
        finally:
            cleaned.append("session stopped, holds settled")
    assert cleaned and stopped.value.code == 128 + sig
    assert signal.getsignal(sig) is before
