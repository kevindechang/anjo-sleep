# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The JSON Lines files under the evidence store and the target log (Task 3): one writer at a
time across processes, and a new file's directory entry is synced."""
from __future__ import annotations

import fcntl
import os
import stat
import threading
from pathlib import Path

import pytest

from anjo.sleep.evidence.jsonl import PLAIN, append_records, read_records

FIRST = b'{"n":1}\n'
OTHER_WRITER = b'{"n":"other"}\n'


def test_an_append_waits_for_another_writers_line(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_bytes(FIRST)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, OTHER_WRITER[:5])  # another writer, mid-line
        appender = threading.Thread(target=append_records, args=(path, PLAIN, [{"n": 2}]))
        appender.start()
        appender.join(0.3)
        assert appender.is_alive()  # it waits for the lock instead of cutting the line
        os.write(fd, OTHER_WRITER[5:])
    finally:
        os.close(fd)  # releases the lock
    appender.join()

    assert read_records(path, PLAIN) == [{"n": 1}, {"n": "other"}, {"n": 2}]


def test_creating_a_file_syncs_its_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synced_folders: list[int] = []
    real_fsync = os.fsync

    def spy(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced_folders.append(os.fstat(fd).st_ino)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    path = tmp_path / "sleep" / "log.jsonl"

    append_records(path, PLAIN, [{"n": 1}])
    assert path.parent.stat().st_ino in synced_folders  # the file's entry
    assert tmp_path.stat().st_ino in synced_folders  # the new folder's entry

    synced_folders.clear()
    append_records(path, PLAIN, [{"n": 2}])
    assert synced_folders == []
