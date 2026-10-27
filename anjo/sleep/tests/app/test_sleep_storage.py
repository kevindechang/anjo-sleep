# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The person's sleep files (Task 3): encrypted at rest, in the account export on the sleep host,
and erased with the account or a reset."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.types import Evidence, Result, Target

DAY = date(2026, 10, 28)
NOW = datetime(2026, 10, 29, 2, 0, tzinfo=timezone.utc)
STAMP = "2026-10-29T02:00:00+00:00"


def _uid() -> str:
    from anjo.core.db import get_db

    row = get_db().execute("SELECT user_id FROM users WHERE username = ?", ("testuser",)).fetchone()
    return str(row["user_id"])


def _evidence(user_id: str, text: str = "They prefer long takes.") -> Evidence:
    return Evidence(
        id="e-1", person_id=user_id, day=DAY, kind="preference", text=text, source="message:1",
        held_out=False, created_at=NOW, model=SPARK_OPEN, ancestry=frozenset({SPARK_OPEN}),
    )


def _target(user_id: str, label: str = "the festival cut") -> Target:
    return Target(
        id="t-1", person_id=user_id, kind="commitment", label=label,
        check_after=NOW + timedelta(days=3), status="open", revision=1, acted_revision=0,
        confidence=None, last_mentioned=DAY,
    )


def _seed(user_id: str) -> None:
    from anjo.sleep.app.storage import evidence_store, target_store

    evidence_store(user_id).add([_evidence(user_id)])
    targets = target_store(user_id)
    targets.upsert(_target(user_id))
    targets.add_result(Result(id="r-1", target_id="t-1", homework_id="hw-1", verified_sources=1,
                              created_at=NOW, consumed=False))


def _sleep_dir(user_id: str) -> Path:
    from anjo.sleep.app.storage import sleep_dir

    return sleep_dir(user_id)


def _export(user_id: str, scope: str = "full") -> dict[str, Any]:
    from anjo.core.export_manifest import build_export

    exported: dict[str, Any] = build_export(
        user_id, account={"user_id": user_id}, exported_at=STAMP, scope=scope,
    )
    return exported


def test_store_is_encrypted_at_rest() -> None:
    from anjo.sleep.app import cleared_sessions
    from anjo.sleep.app.storage import evidence_store, target_store

    user_id = "synthetic-film-student"
    evidence_store(user_id).add([_evidence(user_id, text="SECRET-PREFERENCE about takes")])
    target_store(user_id).upsert(_target(user_id, label="SECRET-PLAN for the festival"))
    cleared_sessions.register_synthetic(user_id, at=NOW)
    cleared_sessions.mark_cleared(user_id, "SECRET-SESSION", at=NOW)

    files = sorted(_sleep_dir(user_id).iterdir())
    assert [path.name for path in files] == ["cleared_sessions.jsonl", "evidence.jsonl", "targets.jsonl"]
    for path in files:
        raw = path.read_bytes()
        assert b"SECRET" not in raw and b"takes" not in raw and b"festival" not in raw
        assert all(line.startswith(b"enc1:") for line in raw.splitlines())
        assert path.stat().st_mode & 0o077 == 0
    assert evidence_store(user_id).for_day(DAY)[0].text == "SECRET-PREFERENCE about takes"
    assert target_store(user_id).open_targets(NOW)[0].label == "SECRET-PLAN for the festival"


def test_a_plaintext_line_is_refused() -> None:
    from anjo.sleep.app.storage import evidence_store
    from anjo.sleep.evidence.jsonl import LogError

    path = _sleep_dir("person-real") / "evidence.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"id": "e-plain"}\n')
    with pytest.raises(LogError, match="line 1"):
        evidence_store("person-real").for_day(DAY)


def test_export_includes_evidence(auth_client: Any) -> None:
    from anjo.core.export_manifest import check_inventory

    user_id = _uid()
    _seed(user_id)
    check_inventory(user_id)

    exported = _export(user_id)

    sleep = exported["sleep"]
    assert [item["id"] for item in sleep["evidence"]] == ["e-1"]
    assert sleep["evidence"][0]["text"] == "They prefer long takes."
    assert [section["key"] for section in exported["manifest"]["sections"] if section["key"] == "sleep"] == ["sleep"]
    memory_only = _export(user_id, scope="memory")
    assert "sleep" not in memory_only
    excluded = {tuple(entry["covers"]) for entry in memory_only["manifest"]["sections"]
                if entry["status"] == "excluded"}
    assert ("sleep/**",) in excluded


def test_targets_exported_and_erased(auth_client: Any) -> None:
    from anjo.core.erasure import factory_reset_user
    from anjo.sleep.app.storage import target_store

    user_id = _uid()
    _seed(user_id)

    sleep = _export(user_id)["sleep"]
    assert [target["id"] for target in sleep["targets"]] == ["t-1"]
    assert [result["id"] for result in sleep["results"]] == ["r-1"]
    assert sleep["expectations"] == [] and sleep["cleared_sessions"] == []

    factory_reset_user(user_id)

    assert not _sleep_dir(user_id).exists()
    assert target_store(user_id).open_targets(NOW) == []


def test_account_delete_removes_evidence(auth_client: Any) -> None:
    from anjo.core.erasure import delete_account_data
    from anjo.core.user_data_guard import UserDataUnavailable
    from anjo.sleep.app.storage import evidence_store

    user_id = _uid()
    _seed(user_id)
    assert (_sleep_dir(user_id) / "evidence.jsonl").exists()

    delete_account_data(user_id)

    assert not _sleep_dir(user_id).exists()
    with pytest.raises(UserDataUnavailable):
        evidence_store(user_id).add([_evidence(user_id)])
    assert not _sleep_dir(user_id).exists()


def test_export_without_the_sleep_host_is_unchanged(auth_client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.core.export_inventory import SECTIONS, active_sections
    from anjo.core.export_manifest import InventoryError, check_inventory

    user_id = _uid()
    monkeypatch.delenv("ANJO_SLEEP_HOST")

    assert active_sections() == SECTIONS
    exported = _export(user_id)
    assert "sleep" not in exported
    assert all("sleep/**" not in entry["covers"] for entry in exported["manifest"]["sections"])

    # Sleep files off the host fail the inventory: they are never dropped silently.
    monkeypatch.setenv("ANJO_SLEEP_HOST", "1")
    _seed(user_id)
    monkeypatch.delenv("ANJO_SLEEP_HOST")
    with pytest.raises(InventoryError, match="sleep/evidence.jsonl"):
        check_inventory(user_id)


def test_export_refuses_a_sleep_file_it_does_not_know(auth_client: Any) -> None:
    user_id = _uid()
    _seed(user_id)
    (_sleep_dir(user_id) / "notes.jsonl").write_text("enc1:x\n")

    with pytest.raises(ValueError, match="notes.jsonl"):
        _export(user_id)


def test_the_stores_need_the_sleep_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.app import cleared_sessions
    from anjo.sleep.app.flags import SleepHostOff
    from anjo.sleep.app.storage import evidence_store, target_store

    monkeypatch.delenv("ANJO_SLEEP_HOST")
    for call in (
        lambda: evidence_store("person-real"),
        lambda: target_store("person-real"),
        lambda: cleared_sessions.register_synthetic("person-real", at=NOW),
    ):
        with pytest.raises(SleepHostOff):
            call()
