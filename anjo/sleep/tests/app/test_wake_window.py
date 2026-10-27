# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The keep-warm window reads the app's data dir as the app does. It needs private Anjo
(``anjo.core.paths``), so it lives here with the other adapter tests, which the public repo does
not collect."""
from __future__ import annotations

from pathlib import Path

import pytest

from anjo.sleep.serving import client, wake
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.tests.test_wake import TOKEN, URL


@pytest.fixture(autouse=True)
def on_modal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_BACKEND", "modal")


def test_a_keep_warm_window_needs_the_apps_data_dir_and_reads_it_as_the_app_does(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Run from a shell without the host's environment file, a window would write where the page
    never looks, and on Modal the page would call Glimmer asleep through a paid window."""
    from anjo.core import paths

    monkeypatch.setenv(wake.URL_ENV, URL)
    monkeypatch.setenv(wake.TOKEN_ENV, TOKEN)
    monkeypatch.delenv(wake.DATA_DIR_ENV, raising=False)
    woke: list[float] = []

    def waking(url: str, token: str, timeout_s: float) -> WakeResult:
        woke.append(timeout_s)
        return WakeResult(1.0, True)

    monkeypatch.setattr(client, "wake", waking)
    with pytest.raises(SystemExit) as refused:
        wake.main(["--keep-warm-minutes", "30"])
    assert refused.value.code == 2 and woke == []  # refused before anything woke
    assert wake.main(["--wake"]) == 0 and len(woke) == 1  # a wake alone writes no window
    monkeypatch.setenv(wake.DATA_DIR_ENV, str(tmp_path / "data" / ".." / "data"))
    assert wake.window_file() == wake.window_path(paths._resolve_data_root())
