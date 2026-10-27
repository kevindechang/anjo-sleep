# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Her workspace on the person's sleep files (Task 17): Task 3's encrypted JSON Lines, folded and
replayed to the same digest, in the account export, and erased with the account or a reset."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pytest

from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry import types as it

NIGHT = datetime(2026, 10, 29, 2, 0, tzinfo=timezone.utc)
STAMP = "2026-10-29T02:00:00+00:00"


def _uid() -> str:
    from anjo.core.db import get_db

    row = get_db().execute("SELECT user_id FROM users WHERE username = ?", ("testuser",)).fetchone()
    return str(row["user_id"])


def _events() -> list[it.WorkspaceEvent]:
    """A night that opens one question on two items from different days."""
    refs = (
        it.EvidenceRef(id="e-d2-1", day=date(2026, 10, 27), created_at=datetime(2026, 10, 27, 12, tzinfo=timezone.utc)),
        it.EvidenceRef(id="e-d5-1", day=date(2026, 10, 28), created_at=datetime(2026, 10, 28, 12, tzinfo=timezone.utc)),
    )
    entries: list[Any] = [
        it.RunStart(date(2026, 10, 28), NIGHT, "notice", it.FULL_RULES, "priority", None, it.Budget(), False),
        it.Retrieved(None, "new_evidence", "", refs),
        it.NoticeOp("contradiction", "SECRET-SUBJECT: scene 4's lens against the kitchen", ("t-1",),
                    ("e-d2-1", "e-d5-1")),
    ]
    ws, events = it.Workspace(), []
    for entry in entries:
        result = kernel.apply(ws, it.Step(ws.seq, NIGHT, entry, None, None, None, 0))
        assert isinstance(result, it.Applied)
        ws = result.ws
        events.append(result.event)
    return events


def _export(user_id: str) -> dict[str, Any]:
    from anjo.core.export_manifest import build_export

    exported: dict[str, Any] = build_export(user_id, account={"user_id": user_id}, exported_at=STAMP)
    return exported


def test_workspace_is_encrypted_and_replays_to_its_digest() -> None:
    from anjo.sleep.app.storage import WORKSPACE_FILE, sleep_dir, workspace_log

    user_id = "synthetic-film-student"
    log = workspace_log(user_id)
    for event in _events():
        log.append(event)
    raw = (sleep_dir(user_id) / WORKSPACE_FILE).read_bytes()
    assert b"SECRET" not in raw and all(line.startswith(b"enc1:") for line in raw.splitlines())
    reread = workspace_log(user_id)
    assert reread.events() == tuple(_events()) and reread.workspace().questions[0].id == "q2"
    assert reread.digest() == reread.replay_digest() == _events()[-1].after


def test_workspace_exported_and_erased(auth_client: Any) -> None:
    from anjo.core.erasure import factory_reset_user
    from anjo.core.export_manifest import check_inventory
    from anjo.sleep.app.storage import sleep_dir, workspace_log

    user_id = _uid()
    log = workspace_log(user_id)
    for event in _events():
        log.append(event)
    check_inventory(user_id)

    workspace = _export(user_id)["sleep"]["workspace"]
    assert [event["step"]["seq"] for event in workspace["events"]] == [0, 1, 2]
    assert workspace["state"]["questions"][0]["subject"].startswith("SECRET-SUBJECT")

    factory_reset_user(user_id)

    assert not sleep_dir(user_id).exists()
    assert workspace_log(user_id).events() == ()


def test_account_delete_erases_the_workspace(auth_client: Any) -> None:
    from anjo.core.erasure import delete_account_data
    from anjo.core.user_data_guard import UserDataUnavailable
    from anjo.sleep.app.storage import WORKSPACE_FILE, sleep_dir, workspace_log

    user_id = _uid()
    workspace_log(user_id).append(_events()[0])
    assert (sleep_dir(user_id) / WORKSPACE_FILE).exists()

    delete_account_data(user_id)

    assert not sleep_dir(user_id).exists()
    with pytest.raises(UserDataUnavailable):
        workspace_log(user_id).append(_events()[0])
