# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The Modal training runner (Task 5): the worst case is held on ``modal_usd`` before the call starts,
a call that may still run is cancelled in a ``finally`` (on failure, timeout and interrupt too), a
failed cancel settles the whole hold, only the job's own inputs go up, real people other than the
ones named are refused, and a killed run's or a deleted person's call is cancelled. Modal is a fake:
no container, no network."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS, Bound, Ledger
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.serving import config, framing
from anjo.sleep.stops import BudgetStop
from anjo.sleep.train import colab_train, modal_job
from anjo.sleep.train.lora import DATASET_FILE, JOB, VOLUME_URI, adapter_name
from anjo.sleep.train.runners.colab import RealPersonRefused
from anjo.sleep.train.runners.modal import (
    A100_USD_PER_HOUR,
    POOL,
    CallRecords,
    ModalRunner,
    ModalTrainError,
    RemoteJob,
    SpawnRefused,
    cancel_leftover_calls,
    hold_base,
    stop_person_calls,
)

TIMEOUT_S = 7200.0
NAME = adapter_name("p1", 1, 0)
RENDERING = framing.rendering("low")


def finished(job: RemoteJob) -> dict[str, Any]:
    """What the container returns for ``job``: the receipt and the adapter's two files, landed."""
    meta = {"peft_type": "LORA", "base_model_name_or_path": config.MODEL, "r": 16, framing.RENDERING_KEY: RENDERING}
    return {"adapter": job.name, "receipt": {"steps": 9, "epochs": 1, "train_minutes": 4.0, "examples": 3,
                                             "dropped_truncated": 0, "gpu": "NVIDIA A100-SXM4-80GB"},
            "files": {"adapter_config.json": json.dumps(meta).encode(), "adapter_model.safetensors": b"lora"}}


class FakeCalls:
    """Modal's calls. ``fail`` is where one goes wrong: ``spawn`` (it may have started), ``refused``
    (Modal refused it before it started), ``remote`` (the job failed), ``timeout``, ``interrupt``
    (Ctrl-C while waiting) or ``cancel``. ``on_spawn`` sees each job as it is spawned."""

    def __init__(self, fail: str | None = None, on_spawn: Callable[[RemoteJob], None] | None = None) -> None:
        self.fail = fail
        self.on_spawn = on_spawn
        self.spawned: list[RemoteJob] = []
        self.waited: list[tuple[str, float]] = []
        self.cancelled: list[str] = []

    def spawn(self, job: RemoteJob) -> str:
        if self.on_spawn is not None:
            self.on_spawn(job)
        if self.fail == "spawn":
            raise ConnectionError("the connection dropped after the request went out")
        if self.fail == "refused":
            raise SpawnRefused("the training app isn't deployed")
        self.spawned.append(job)
        return f"fc-{len(self.spawned):08d}"

    def wait(self, call_id: str, timeout_s: float) -> Mapping[str, object]:
        self.waited.append((call_id, timeout_s))
        if self.fail == "remote":
            raise RuntimeError("the training script exited with 1")
        if self.fail == "timeout":
            raise TimeoutError("past the job's hard timeout")
        if self.fail == "interrupt":
            raise KeyboardInterrupt
        return finished(self.spawned[-1])

    def cancel(self, call_id: str) -> None:
        if self.fail == "cancel":
            raise ConnectionError("Modal didn't answer")
        self.cancelled.append(call_id)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 900.0  # every reading is 15 minutes on
        return self.now


def _inputs(tmp_path: Path, person: str = "synthetic-1", prev: Path | None = None) -> dict[str, Any]:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / DATASET_FILE).write_text('{"id": "x"}\n')
    return {"person_id": person, "adapter_name": NAME, "night": "2026-10-28", "dataset_path": str(work / DATASET_FILE),
            "workdir": str(work), "config": {"r": 16, "revision": "a" * 40, "rendering": RENDERING},
            "prev_adapter_dir": None if prev is None else str(prev)}


def _runner(calls: FakeCalls, ledger: Ledger, records: CallRecords, *, real: frozenset[str] = frozenset()) -> ModalRunner:
    return ModalRunner(calls=calls, ledger=ledger, records=records, is_synthetic=lambda uid: uid.startswith("synthetic"),
                       allow_real=real, clock=Clock())


def _events(ledger: Ledger) -> list[dict[str, Any]]:
    return read_records(ledger.path, PLAIN)


@pytest.mark.parametrize("fail", [None, "remote", "timeout", "interrupt"])
def test_modal_runner_holds_before_start_and_cancels_in_finally(tmp_path: Path, fail: str | None) -> None:
    ledger, records = Ledger(tmp_path / "budget.jsonl"), CallRecords(tmp_path / "calls")
    held_at_spawn: list[int] = []
    calls = FakeCalls(fail, on_spawn=lambda job: held_at_spawn.append(len(ledger.open_holds())))
    runner = _runner(calls, ledger, records)
    if fail is None:
        outputs = runner.run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
        assert outputs["steps"] == 9 and outputs["gpu_minutes"] == pytest.approx(15.0)
        assert outputs["landed"] == f"{VOLUME_URI}/{NAME}" and outputs["compute_units"] == 0.0
        out = Path(str(outputs["adapter_dir"]))
        assert sorted(p.name for p in out.iterdir()) == sorted((*config.ADAPTER_FILES, colab_train.RECEIPT))
        assert calls.cancelled == []  # a call that finished is not cancelled
    else:
        with pytest.raises((RuntimeError, TimeoutError, KeyboardInterrupt)):
            runner.run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
        assert calls.cancelled == ["fc-00000001"]  # a call that may still run is cancelled
    assert held_at_spawn == [1]  # the hold was taken before the call started
    reserve, settle = _events(ledger)
    assert reserve["op"] == "reserve" and reserve["pools"] == [POOL] and reserve["timeout_s"] == TIMEOUT_S
    assert reserve["rate_per_hour"] == A100_USD_PER_HOUR == 2.50  # Modal's A100-80GB list price, 2026-10-27
    assert reserve["amount"] == pytest.approx(5.0)
    assert settle["op"] == "settle" and settle["amount"] == pytest.approx(0.25 * A100_USD_PER_HOUR)  # measured
    assert records.items() == []  # the call's record is gone once it ended or was cancelled
    [(call_id, waited)] = calls.waited
    assert call_id == "fc-00000001" and 0 < waited <= TIMEOUT_S  # waited under the hard timeout


def test_a_failed_cancel_settles_the_whole_hold_and_says_so(tmp_path: Path) -> None:
    ledger, records = Ledger(tmp_path / "budget.jsonl"), CallRecords(tmp_path / "calls")
    calls = FakeCalls("cancel")
    calls.wait = lambda call_id, timeout_s: (_ for _ in ()).throw(TimeoutError("late"))  # type: ignore[method-assign]
    with pytest.raises(ModalTrainError, match="by hand"):
        _runner(calls, ledger, records).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert ledger.used(POOL) == pytest.approx(Bound(A100_USD_PER_HOUR, TIMEOUT_S).amount)
    assert ledger.open_holds() == ()
    assert records.items() == [(NAME, "fc-00000001")]  # kept, so the next night's sweep cancels it


def test_a_spawn_that_may_have_started_settles_the_whole_hold_and_a_refused_one_only_its_time(tmp_path: Path) -> None:
    ledger, records = Ledger(tmp_path / "budget.jsonl"), CallRecords(tmp_path / "calls")
    with pytest.raises(ConnectionError):
        _runner(FakeCalls("spawn"), ledger, records).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert ledger.used(POOL) == pytest.approx(5.0)  # no id to cancel by: the worst case is the honest charge
    other = Ledger(tmp_path / "other.jsonl")
    with pytest.raises(SpawnRefused):
        _runner(FakeCalls("refused"), other, records).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert other.used(POOL) == pytest.approx(0.25 * A100_USD_PER_HOUR) and other.open_holds() == ()


def test_no_room_in_modal_usd_stops_before_any_call(tmp_path: Path) -> None:
    calls = FakeCalls()
    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "modal_usd": 4.0})
    with pytest.raises(BudgetStop):
        _runner(calls, ledger, CallRecords(tmp_path / "calls")).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert calls.spawned == []


def test_modal_runner_refuses_other_real_people(tmp_path: Path) -> None:
    ledger, records = Ledger(tmp_path / "budget.jsonl"), CallRecords(tmp_path / "calls")
    calls = FakeCalls()
    runner = _runner(calls, ledger, records, real=frozenset({"kevin-dev"}))
    with pytest.raises(RealPersonRefused):
        runner.run(JOB, _inputs(tmp_path, person="someone-else"), timeout_s=TIMEOUT_S)
    with pytest.raises(RealPersonRefused):  # nobody named: every real person is refused, as on Colab
        _runner(calls, ledger, records).run(JOB, _inputs(tmp_path, person="kevin-dev"), timeout_s=TIMEOUT_S)
    assert calls.spawned == [] and _events(ledger) == []
    runner.run(JOB, _inputs(tmp_path, person="kevin-dev"), timeout_s=TIMEOUT_S)  # Kevin's own night (§12.1)
    runner.run(JOB, {**_inputs(tmp_path, person="synthetic-2"), "adapter_name": adapter_name("p2", 1, 0)},
               timeout_s=TIMEOUT_S)
    assert len(calls.spawned) == 2


def test_only_the_jobs_inputs_go_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_not-a-real-token")
    monkeypatch.setenv("ANJO_SECRET", "not-a-real-secret-value")
    prev = tmp_path / "prev"
    prev.mkdir()
    for file in config.ADAPTER_FILES:
        (prev / file).write_bytes(f"prev {file}".encode())
    calls = FakeCalls()
    _runner(calls, Ledger(tmp_path / "budget.jsonl"), CallRecords(tmp_path / "calls")).run(
        JOB, _inputs(tmp_path, prev=prev), timeout_s=TIMEOUT_S)
    [job] = calls.spawned
    assert job.dataset == '{"id": "x"}\n' and job.name == NAME
    assert json.loads(job.config_json) == _inputs(tmp_path)["config"]
    assert dict(job.prev) == {file: f"prev {file}".encode() for file in config.ADAPTER_FILES}
    assert job.exec_s == modal_job.exec_seconds(TIMEOUT_S) < TIMEOUT_S
    sent = repr(job)
    assert "hf_not-a-real-token" not in sent and "not-a-real-secret-value" not in sent
    bad = _inputs(tmp_path)
    bad["dataset_path"] = str(tmp_path / ".env")
    with pytest.raises(ModalTrainError, match="dataset"):
        _runner(FakeCalls(), Ledger(tmp_path / "other.jsonl"), CallRecords(tmp_path / "c2")).run(
            JOB, bad, timeout_s=TIMEOUT_S)


def test_a_timeout_too_short_to_train_and_another_job_are_refused_before_any_hold(tmp_path: Path) -> None:
    ledger, calls = Ledger(tmp_path / "budget.jsonl"), FakeCalls()
    runner = _runner(calls, ledger, CallRecords(tmp_path / "calls"))
    with pytest.raises(ModalTrainError):
        runner.run(JOB, _inputs(tmp_path), timeout_s=600.0)
    with pytest.raises(ModalTrainError):
        runner.run("gates", _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert calls.spawned == [] and _events(ledger) == []
    assert hold_base("2026-10-28", "synthetic-1") == "modal_train:2026-10-28:synthetic-1"


def test_a_job_that_returns_no_adapter_fails_after_settling(tmp_path: Path) -> None:
    ledger, calls = Ledger(tmp_path / "budget.jsonl"), FakeCalls()
    calls.wait = lambda call_id, timeout_s: {"adapter": NAME, "receipt": {}, "files": {}}  # type: ignore[method-assign]
    with pytest.raises(ModalTrainError, match="adapter_config.json"):
        _runner(calls, ledger, CallRecords(tmp_path / "calls")).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert [e["op"] for e in _events(ledger)] == ["reserve", "settle"]


def test_the_sweep_cancels_every_call_a_killed_run_left(tmp_path: Path) -> None:
    records = CallRecords(tmp_path / "calls")
    assert not records.used() and cancel_leftover_calls(records, FakeCalls()) == (0, 0)
    records.add("p-aaaaaaaaaaaa-n3", "fc-00000007")
    records.add("p-bbbbbbbbbbbb-n1", "fc-00000008")
    calls = FakeCalls()
    assert cancel_leftover_calls(records, calls) == (2, 0)
    assert calls.cancelled == ["fc-00000007", "fc-00000008"] and records.items() == [] and records.used()
    records.add("p-aaaaaaaaaaaa-n4", "fc-00000009")
    with pytest.raises(ModalTrainError, match="by hand"):
        cancel_leftover_calls(records, FakeCalls("cancel"))
    assert records.items() == [("p-aaaaaaaaaaaa-n4", "fc-00000009")]


def test_stop_person_calls_cancels_only_theirs(tmp_path: Path) -> None:
    records = CallRecords(tmp_path / "calls")
    records.add("p-aaaaaaaaaaaa-n3", "fc-00000007")
    records.add("p-bbbbbbbbbbbb-n1", "fc-00000008")  # somebody else's night
    calls = FakeCalls()
    assert stop_person_calls(records, "p1", {"p-aaaaaaaaaaaa-n2", "p-aaaaaaaaaaaa-n3"}, calls=calls) == 1
    assert calls.cancelled == ["fc-00000007"] and records.items() == [("p-bbbbbbbbbbbb-n1", "fc-00000008")]
    assert stop_person_calls(CallRecords(tmp_path / "none"), "p1", {"p-aaaaaaaaaaaa-n3"}) == 0  # no client needed


def test_call_records_hold_names_and_ids_only(tmp_path: Path) -> None:
    records = CallRecords(tmp_path / "calls")
    for bad_name, bad_id in (("kevin@example.com", "fc-00000001"), ("p-aaaaaaaaaaaa-n3", "../../etc"),
                             ("p-aaaaaaaaaaaa-n3", "")):
        with pytest.raises(ValueError):
            records.add(bad_name, bad_id)
    records.add("p-aaaaaaaaaaaa-n3", "fc-01ABCdef")
    assert json.loads((tmp_path / "calls" / "p-aaaaaaaaaaaa-n3.json").read_text()) == {"call": "fc-01ABCdef"}
