# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""SIGTERM and SIGHUP end an APD command as Ctrl-C does, through its cleanup (Task 17 review), so
its Colab session is stopped and its holds settled. Private import: the handler is the CLI's own."""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path

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


def test_sigterm_inside_a_session_stops_it_and_deletes_its_throwaway_key(tmp_path: Path) -> None:
    from anjo.sleep.mind.apd import colab_vllm, fakes
    from anjo.sleep.train.runners.colab import ColabCli

    made: list[Path] = []

    def keygen(path: Path) -> None:
        fakes.fake_keygen(path)
        made.append(path)

    vm = fakes.FakeColab()
    with pytest.raises(SystemExit), apd_main._signals_exit():
        with colab_vllm.serve(ColabCli(Path("/fake/colab"), run=vm), None, fakes.SETTINGS, timeout_s=fakes.SERVE_S,
                              tunnels=fakes.fake_tunnels(vm), clock=fakes.Ticks(), wait=lambda _session: True,
                              keygen=keygen, key_dir=tmp_path / "ssh"):
            assert made[-1].is_file()
            os.kill(os.getpid(), signal.SIGTERM)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                time.sleep(0.01)
    key = made[-1]
    assert not key.exists() and not key.with_name(f"{key.name}.pub").exists() and not key.parent.exists()
    assert vm.commands()[-1] == "stop" and vm.tunnel_closed_before_stop
