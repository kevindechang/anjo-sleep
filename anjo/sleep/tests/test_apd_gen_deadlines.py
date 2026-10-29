"""Generation deadlines cover admission, transport construction and result collection."""
from __future__ import annotations

import dataclasses
import fcntl
import threading
import time
import types
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import httpx
import pytest

from anjo.sleep import baseten_api, fireworks_api, meta_api
from anjo.sleep.mind.apd import backend, gen_batch, generate, retry
from anjo.sleep.mind.apd.limiter import Limiter
from anjo.sleep.model_api import ModelApiError


def test_limiter_admission_has_a_deadline() -> None:
    limit = Limiter(1)
    limit.acquire()
    try:
        with pytest.raises(ModelApiError, match="limiter.*timed out"):
            limit.acquire(timeout_s=0.01)
    finally:
        limit.release()


def test_budget_room_has_a_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "ROOM_WAIT_S", 0.01)
    flight = backend.Flight()
    since = flight.board()
    with pytest.raises(ModelApiError, match="budget room.*timed out"):
        flight.room(since, None)
    flight.land(held=False)


def test_results_have_a_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gen_batch, "WORKER_TIMEOUT_S", 0.01)
    with pytest.raises(gen_batch.CallFailed, match="worker.*timed out"):
        list(gen_batch._results([Future()]))


def test_stuck_writer_fails_attempts_and_parallel_run_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    released = threading.Event()
    entered: list[float] = []

    def stuck(*args: Any, **kwargs: Any) -> Any:
        entered.append(args[3])
        released.wait()  # deliberately ignores the model timeout; released in cleanup
        raise AssertionError("late reply must never be consumed")

    monkeypatch.setattr(generate, "TIMEOUT_S", 0.01)
    monkeypatch.setattr(backend, "CALL_GRACE_S", 0.01)
    back = dataclasses.replace(backend.fake_backend(), life_chat=stuck)
    start = time.monotonic()
    try:
        with gen_batch.side_by_side(back, [lambda b: gen_batch.generate_life(generate.development_specs()[0], b)], 1) as results:
            with pytest.raises(generate.GenerationError, match="timed out"):
                list(results)
        assert len(entered) == generate.MAX_ATTEMPTS * retry.ATTEMPTS
        assert time.monotonic() - start < 1.0
    finally:
        released.set()


def test_vertex_constructor_sets_a_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    sdk = types.SimpleNamespace(Client=lambda **kw: captured.update(kw))
    real = backend.importlib.import_module
    monkeypatch.setattr(backend.importlib, "import_module", lambda name: sdk if name == "google.genai" else real(name))
    backend._vertex.cache_clear()
    try:
        backend._vertex()
        assert captured["http_options"].timeout == int(backend.GEMINI_JUDGE_TIMEOUT_S * 1000)
    finally:
        backend._vertex.cache_clear()


@pytest.mark.parametrize("api,model,key", [
    (fireworks_api.generate, fireworks_api.DEEPSEEK_V4P1_FLASH, fireworks_api.KEY_ENV),
    (fireworks_api.chat, fireworks_api.SPARK_OPEN, fireworks_api.KEY_ENV),
    (baseten_api.chat, baseten_api.SPARK_OPEN, baseten_api.KEY_ENV),
    (meta_api.chat, "muse-spark", meta_api.KEY_ENV),
])
def test_http_clients_set_all_timeout_phases(api: Any, model: str, key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(key, "fake-test-key")
    seen: list[dict[str, Any]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"model": model, "choices": [{"message": {"content": "ok"}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    api(model, [{"role": "user", "content": "test"}], None, 0.25, transport=httpx.MockTransport(reply))
    assert seen == [{"connect": 0.25, "read": 0.25, "write": 0.25, "pool": 0.25}]


def test_file_lock_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep import waits

    monkeypatch.setattr(waits, "LOCK_TIMEOUT_S", 0.01)
    path = tmp_path / "lock"
    with path.open("w") as first, path.open("w") as second:
        fcntl.flock(first, fcntl.LOCK_EX)
        with pytest.raises(ModelApiError, match="file lock.*timed out"):
            waits.file_lock(second)


def test_thread_lock_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep import waits

    monkeypatch.setattr(waits, "LOCK_TIMEOUT_S", 0.01)
    lock = threading.Lock()
    with lock, pytest.raises(ModelApiError, match="lock.*timed out"), waits.locked(lock):
        pytest.fail("a held lock cannot be entered")


def test_worker_deadline_halts_a_real_call_and_drains_before_unlock(monkeypatch: pytest.MonkeyPatch) -> None:
    release, entered = threading.Event(), threading.Event()
    monkeypatch.setattr(gen_batch, "WORKER_TIMEOUT_S", 0.03)
    monkeypatch.setattr(backend, "CALL_POLL_S", 0.005)
    done = threading.Event()

    def stuck(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        release.wait()
        raise AssertionError("unconsumed late reply")

    def work(b: backend.Backend) -> Any:
        try:
            return backend.bounded(stuck, b.halt)("fake", [], None, 300.0)
        finally:
            done.set()

    try:
        with pytest.raises(gen_batch.CallFailed, match="worker.*timed out"):
            with gen_batch.side_by_side(backend.fake_backend(), [work], 1) as results:
                assert entered.wait(0.5)
                list(results)
        assert done.is_set()  # cannot release the owning context while the worker runs
    finally:
        release.set()


def test_ceiling_lock_timeout_ends_one_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    checked: list[int] = []

    def check() -> None:
        checked.append(1)
        if len(checked) == 1:
            raise ModelApiError("a file lock timed out after 0.01s")

    b = dataclasses.replace(backend.fake_backend(), ceiling=types.SimpleNamespace(check=check))
    made = gen_batch.generate_life(generate.development_specs()[0], b)
    assert made.attempts == 2 and len(checked) == 2

