# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""A live APD command reads only the real prereg ledger, verified up to the head pushed to the
public ``prereg`` branch (Task 17 review). Private import: ``_entries`` is the CLI's own; the
fakes keep the data dir's stand-in."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from anjo.sleep.mind.apd import __main__ as apd_main
from anjo.sleep.mind.apd import pilot
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.prereg.ledger import DATA_DIR_ENV, LedgerError, head_path, ledger_path, register


def _register(path: Path, kind: str) -> str:
    return register(kind, "0" * 64, kind, path=path).entry_hash


def test_live_reads_only_the_pushed_prereg_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "env"))
    data, args = DataDir(tmp_path / "data"), argparse.Namespace(prereg=None)
    with pytest.raises(LedgerError, match="missing|never pushed"):
        apd_main._entries(args, data, fake=False)
    head = _register(ledger_path(), pilot.DRAW_SEED_KIND)
    with pytest.raises(LedgerError, match="never pushed"):
        apd_main._entries(args, data, fake=False)
    head_path().write_text(head)
    assert [entry.kind for entry in apd_main._entries(args, data, fake=False)] == [pilot.DRAW_SEED_KIND]
    _register(ledger_path(), pilot.FREEZE_KIND)  # appended, never pushed
    with pytest.raises(LedgerError, match="does not match"):
        apd_main._entries(args, data, fake=False)
    _register(data.root / "prereg" / "ledger.jsonl", pilot.FREEZE_KIND)
    assert [entry.kind for entry in apd_main._entries(args, data, fake=True)] == [pilot.FREEZE_KIND]
