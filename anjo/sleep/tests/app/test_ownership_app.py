# mypy: disallow-untyped-decorators=False, disallow-untyped-calls=False
# (pyproject's follow_imports = "skip" hides pytest's own types)
"""Undo, delete and what changed on the sleep host (Task 9): the routes (same user only, absent off the
host), a delete over the real encrypted stores (MIND's workspace log included), and account erasure's
hook. The remote stores are fakes; no Colab CLI or bucket is reached."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.tests.ownership_fakes import FakeS3
from anjo.sleep.types import CheckpointRef, GateResult, NightResult

NIGHT = date(2026, 10, 28)
SECOND = date(2026, 10, 29)
ALICE, BOB = "person-alice", "person-bob"
SECRET = "SECRET-REASON-THAT-NAMES-A-CASE"


@pytest.fixture(autouse=True)
def no_colab(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No Colab CLI on this machine: a delete here never runs one."""
    from anjo.sleep.train.runners import colab

    monkeypatch.setattr(colab, "COLAB_BIN", tmp_path / "no-colab")


@pytest.fixture(autouse=True)
def capabilities_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.core import capabilities

    monkeypatch.setattr(capabilities, "CAPABILITY_ENV_VARS", dict(capabilities.CAPABILITY_ENV_VARS))
    monkeypatch.setattr(capabilities, "CAPABILITY_COHORTS", dict(capabilities.CAPABILITY_COHORTS))
    monkeypatch.setattr(capabilities, "_UNSWEPT_CAPABILITIES", set(capabilities._UNSWEPT_CAPABILITIES))
    monkeypatch.setenv("ANJO_SLEEP_ENABLED", "1")
    monkeypatch.setenv("ANJO_SLEEP_FORCED_USER_IDS", f"{ALICE},{BOB}")


def ckpt(uid: str, night: date, name: str, parent: str | None = None) -> CheckpointRef:
    return CheckpointRef(name, uid, night, f"modal-volume://anjo-sleep-adapters/{name}", parent)


def _night(uid: str, night: date, status: str, serving: CheckpointRef | None) -> None:
    """A finished night on disk, as the night command leaves it; one gate carries a failure reason."""
    from anjo.sleep.app.ownership import night_logs
    from anjo.sleep.night.nightlog import NightLog, night_folder

    log = NightLog(night_folder(night_logs(uid), night))
    log.step("choose", "ok", uses="new", system=None, action="revisit_thread", target_id="t-1")
    log.step("note", "ok", uses="existing", system="outreach", outcome="held")
    gate = GateResult("collapse", status == "promoted", 0.0, 0.0, None if status == "promoted" else SECRET)
    log.finish(NightResult(status="promoted" if status == "promoted" else "rejected", failed_at=None, checkpoint=serving if status == "promoted" else None,
                           gates={"collapse": gate}, choices=(), cost_usd=0.0, cost_parts={}, log_path=log.path),
               {"night": night.isoformat(), "night_end": night.isoformat()})


def _seed(uid: str) -> tuple[CheckpointRef, CheckpointRef]:
    from anjo.sleep.app.storage import checkpoint_registry

    registry = checkpoint_registry()
    first, second = ckpt(uid, NIGHT, "p-aaaaaaaaaaaa-n1"), ckpt(uid, SECOND, "p-aaaaaaaaaaaa-n2", "p-aaaaaaaaaaaa-n1")
    for item in (first, second):
        registry.record_trained(uid, item.id, item.night)
        registry.promote(uid, item)
    _night(uid, NIGHT, "promoted", first)
    _night(uid, SECOND, "rejected", None)
    return first, second


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from anjo.dashboard.auth import get_current_user_id
    from anjo.sleep.app.api import router

    who = {"user": ALICE}
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_current_user_id] = lambda: who["user"]
    with TestClient(app) as client:
        client.who = who
        yield client


# ── The routes ────────────────────────────────────────────────────────────────

ROUTES = (("get", "/api/sleep/nights"), ("get", "/api/sleep/nights/2026-10-28"),
          ("get", "/api/sleep/what-changed"), ("post", "/api/sleep/undo"))


def test_routes_absent_without_sleep_host(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(ALICE)
    monkeypatch.delenv("ANJO_SLEEP_HOST")
    for method, path in ROUTES:
        assert getattr(api, method)(path).status_code == 404, path  # even for a flagged person's own session

    script = (
        "import json\n"
        "from anjo.dashboard.app import app\n"
        "print(json.dumps(sorted({r.path for r in app.routes if '/sleep' in getattr(r, 'path', '')})))\n"
    )
    env = {key: value for key, value in os.environ.items() if key != "ANJO_SLEEP_HOST"}
    run = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True,
                         timeout=300, check=False)
    assert run.returncode == 0, run.stderr[-2000:]
    assert json.loads(run.stdout.strip().splitlines()[-1]) == []  # the app mounts none of them off the host


def test_routes_refuse_other_users(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = _seed(ALICE)

    api.who["user"] = BOB  # flagged, with nothing of their own
    assert api.get("/api/sleep/nights", params={"user_id": ALICE}).json() == []  # a user id in the request is ignored
    assert api.get("/api/sleep/nights/2026-10-28").status_code == 404
    assert api.get("/api/sleep/what-changed", params={"since": "2026-10-01", "user_id": ALICE}).json() == []
    assert api.post("/api/sleep/undo", params={"user_id": ALICE}).status_code == 409
    from anjo.sleep.app.storage import checkpoint_registry

    assert checkpoint_registry().current(ALICE) == second  # Bob's undo never touched Alice

    monkeypatch.setenv("ANJO_SLEEP_FORCED_USER_IDS", ALICE)  # Bob is not on the sleep cohort
    for method, path in ROUTES:
        assert getattr(api, method)(path).status_code == 403, path

    api.who["user"] = ALICE
    nights = api.get("/api/sleep/nights").json()
    assert [(n["night"], n["status"]) for n in nights] == [("2026-10-28", "promoted"), ("2026-10-29", "rejected")]
    assert nights[0]["checkpoint"] == first.id and nights[1]["gates"] == [{"gate": "collapse", "passed": False}]
    assert nights[0]["action"] == "revisit_thread" and nights[0]["note"] == "held"
    assert api.get("/api/sleep/nights/2026-10-29").json()["status"] == "rejected"
    assert api.get("/api/sleep/nights/2026-10-30").status_code == 404
    assert api.get("/api/sleep/nights/not-a-date").status_code == 422


def test_no_route_returns_a_failure_reason_or_conversation_text(api: Any) -> None:
    _seed(ALICE)
    bodies = [api.get("/api/sleep/nights").text, api.get("/api/sleep/nights/2026-10-29").text,
              api.get("/api/sleep/what-changed", params={"since": "2026-10-01"}).text]
    assert all(SECRET not in body for body in bodies)
    lines = api.get("/api/sleep/what-changed", params={"since": "2026-10-01"}).json()
    assert [(c["night"], c["kind"]) for c in lines][:2] == [("2026-10-28", "promoted"), ("2026-10-28", "studied")]
    assert any(c["kind"] == "rejected" and "collapse" in c["line"] for c in lines)


def test_undo_route_restores_the_previous_version_once(api: Any) -> None:
    from anjo.sleep.app.storage import checkpoint_registry

    first, _ = _seed(ALICE)
    done = api.post("/api/sleep/undo")
    assert done.status_code == 200 and done.json() == {"checkpoint": first.id, "night": "2026-10-28"}
    assert checkpoint_registry().current(ALICE) == first
    assert api.post("/api/sleep/undo").status_code == 409  # nothing earlier


# ── delete_all over the real stores ───────────────────────────────────────────


def _events() -> list[it.WorkspaceEvent]:
    ws = it.Workspace()
    entries: list[Any] = [
        it.RunStart(NIGHT, datetime(2026, 10, 28, 2, tzinfo=timezone.utc), "notice", it.FULL_RULES, "priority",
                    None, it.Budget(), False),
    ]
    events = []
    for entry in entries:
        result = kernel.apply(ws, it.Step(ws.seq, datetime(2026, 10, 28, 2, tzinfo=timezone.utc), entry, None, None, None, 0))
        assert isinstance(result, it.Applied)
        ws = result.ws
        events.append(result.event)
    return events


def _lay_down(uid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Everything a person has on disk: encrypted sleep files, registry, logs, scores, sealed probes, training copies."""
    from anjo.sleep.app.ownership import person_results
    from anjo.sleep.app.storage import SLEEP_DIR, data_root, evidence_store, sleep_dir, workspace_log
    from anjo.sleep.tests.night_fakes import own

    _seed(uid)
    evidence_store(uid).add([own(uid, NIGHT, 1)])
    for event in _events():
        workspace_log(uid).append(event)
    sealed = tmp_path / "sealed"
    (sealed / uid / "2026-10-28").mkdir(parents=True)
    (sealed / uid / "2026-10-28" / "probes.jsonl").write_text("{}")
    monkeypatch.setenv("ANJO_SLEEP_SEALED_DIR", str(sealed))
    training = data_root() / SLEEP_DIR / "training"
    for part in ("adapters", "work"):
        (training / part / "p-aaaaaaaaaaaa-n1").mkdir(parents=True)
        (training / part / "p-aaaaaaaaaaaa-n1" / "file").write_text("x")
        (training / part / "p-bbbbbbbbbbbb-n1").mkdir(parents=True)  # somebody else's
    (person_results(uid) / "scores").mkdir(parents=True)
    (person_results(uid) / "scores" / "night-01.json").write_text("{}")
    return {"sleep": sleep_dir(uid), "results": person_results(uid), "sealed": sealed / uid,
            "registry": data_root() / SLEEP_DIR / "checkpoints" / uid, "training": training / "adapters"}


def test_delete_all_clears_the_mind_workspace_log(auth_client: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.app.ownership import cancel_flag, delete_all
    from anjo.sleep.app.storage import WORKSPACE_FILE, workspace_log

    uid = "person-mind"
    monkeypatch.setenv("ANJO_SLEEP_FORCED_USER_IDS", f"{uid},{ALICE}")
    where = _lay_down(uid, tmp_path, monkeypatch)
    assert (where["sleep"] / WORKSPACE_FILE).exists() and workspace_log(uid).events()
    before = cancel_flag().epoch(uid)

    report = delete_all(uid)

    assert report.cancelled and cancel_flag().epoch(uid) == before + 1
    assert dict(report.erased)["mind workspace"] == 1
    assert workspace_log(uid).events() == () and not (where["sleep"] / WORKSPACE_FILE).exists()
    for name in ("sleep", "results", "sealed", "registry"):
        assert not where[name].exists(), name
    assert sorted(p.name for p in where["training"].iterdir()) == ["p-bbbbbbbbbbbb-n1"]  # only hers went
    assert os.environ["ANJO_SLEEP_FORCED_USER_IDS"] == ALICE  # the forced-id entry too


# ── Account erasure's hook ────────────────────────────────────────────────────


@pytest.fixture
def hooked(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from types import MappingProxyType

    from anjo.core.anjo_model import client, hosts
    from anjo.sleep.app.register import register_hooks, unregister_hooks

    factories = dict(hosts._FACTORIES)
    monkeypatch.setattr(hosts, "_FACTORIES", factories)
    monkeypatch.setattr(hosts, "RUNTIMES", MappingProxyType(factories))
    monkeypatch.setattr(client, "OFF_VOCABULARY_HOSTS", set(client.OFF_VOCABULARY_HOSTS))
    register_hooks()
    yield
    unregister_hooks()


def test_account_erasure_also_removes_remote_adapters_and_the_rest(
    auth_client: Any, hooked: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anjo.core.erasure import delete_account_data
    from anjo.sleep.app.ownership import cancel_flag
    from anjo.sleep.serving.sagemaker import invoke

    uid = ALICE
    where = _lay_down(uid, tmp_path, monkeypatch)
    bucket = FakeS3({"adapters/p-aaaaaaaaaaaa-n1/adapter_config.json", "adapters/p-aaaaaaaaaaaa-n2/adapter_config.json",
                     "adapters/p-bbbbbbbbbbbb-n1/adapter_config.json"})
    monkeypatch.setenv("ANJO_SLEEP_S3_BUCKET", "anjo-sleep-sagemaker-test")
    monkeypatch.setattr(invoke, "make_client", lambda *args, **kwargs: bucket)

    delete_account_data(uid)

    assert bucket.keys == {"adapters/p-bbbbbbbbbbbb-n1/adapter_config.json"}
    for name in ("sleep", "results", "sealed", "registry"):
        assert not where[name].exists(), name
    assert cancel_flag().epoch(uid) == 1


def test_account_erasure_of_a_person_with_no_sleep_data_touches_nothing(
    auth_client: Any, hooked: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anjo.core.erasure import delete_account_data
    from anjo.sleep.app.ownership import cancel_flag
    from anjo.sleep.serving.sagemaker import invoke

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a person with no sleep data reaches no store")

    monkeypatch.setenv("ANJO_SLEEP_S3_BUCKET", "anjo-sleep-sagemaker-test")
    monkeypatch.setattr(invoke, "make_client", never)
    delete_account_data("person-nothing")
    assert cancel_flag().epoch("person-nothing") == 0
