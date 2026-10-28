# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""A live APD command keeps a Mac awake for as long as it runs, and no longer (Task 17 review).
Private import: ``_awake`` is the CLI's own; the keeper here is a harmless stand-in process."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from anjo.sleep.mind.apd import __main__ as apd_main


def test_a_live_command_keeps_the_mac_awake_tied_to_its_own_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    started: list[list[str]] = []
    keepers: list[subprocess.Popen[bytes]] = []

    def start(argv: list[str]) -> subprocess.Popen[bytes]:
        started.append(argv)
        keepers.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]))
        return keepers[-1]

    with apd_main._awake(live=True, start=start):
        assert started == [["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())]]
        assert keepers[0].poll() is None
    assert keepers[0].wait(10) is not None  # stopped on the way out
    with apd_main._awake(live=False, start=start):
        pass
    assert len(started) == 1  # the fakes keep nothing awake
