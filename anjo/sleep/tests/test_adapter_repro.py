# mypy: disallow-untyped-decorators=False
"""Offline receipts for the plan-first adapter diagnostic; no real server."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any

import httpx
import pytest

from anjo.sleep.budget import POOLS, Ledger
from anjo.sleep.serving import config
from anjo.sleep.serving.client import ServingError, TransportFailed, VllmClient, WakeResult
from anjo.sleep.tests.test_serving_client import FakeVllm
from anjo.sleep.tools import adapter_repro as repro

X, Y = "p-aaaaaaaa-n1", "p-bbbbbbbb-n2"
ARGS = ["--x", X, "--y", Y]
QUIET = 'vllm:num_requests_running{model_name="base"} 0\nvllm:num_requests_waiting 0\n'


class FakeClient:
    rendering = "test"

    def __init__(self, *, unsupported: bool = False) -> None:
        self.events: list[Any] = []
        self.unsupported = unsupported
        self.observer: Callable[[str], AbstractContextManager[None]] = lambda _: nullcontext()

    def on(self, day: Any) -> FakeClient:
        return self

    def reusing_tokens(self) -> FakeClient:
        return self

    def once(self) -> FakeClient:
        return self

    def observing_requests(self, observer: Callable[[str], AbstractContextManager[None]]) -> FakeClient:
        self.observer = observer
        return self

    def metrics(self) -> str:
        self.events.append("metrics")
        return QUIET

    def continuation_logprobs(self, adapter: Any, messages: Any, continuation: str,
                              *, cache_salt: str | None = None) -> Sequence[float]:
        with self.observer("/v1/completions"):
            self.events.append((None if adapter is None else adapter.id, messages, continuation, cache_salt))
            if cache_salt and self.unsupported:
                raise ServingError("/v1/completions returned HTTP 400")
        return (-1.0,)

    def reload_adapter(self, name: str) -> None:
        with self.observer("/v1/unload_lora_adapter"):
            self.events.append(("reload", name))
        with self.observer("/v1/load_lora_adapter"):
            self.events.append(("load", name))


def never(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("the plan must not touch live dependencies")


def run(fake: FakeClient, **kwargs: Any) -> list[dict[str, Any]]:
    lines: list[str] = []
    assert repro.main([*ARGS, "--go"], environ={}, client=fake, wake=lambda: WakeResult(0.0, True),
                      out=lines.append, **{"ledger": None, **kwargs}) == 0
    return [json.loads(line) for line in lines if line.startswith("{")]


def test_plan_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repro.backend, "ModalServing", never)
    monkeypatch.setattr(repro, "default_ledger", never)
    lines: list[str] = []
    assert repro.main(ARGS, environ={}, client=never, wake=never, ledger=never, out=lines.append) == 0
    assert "330" in lines[0] and "660" in lines[0] and "nothing is sent" in lines[0]
    assert repro.estimate_s(repro.choices(repro.knowledge.load("film_founder"))) < 15 * 60


def test_sequence_and_every_request_is_bracketed_by_metrics() -> None:
    fake = FakeClient()
    records = run(fake)
    passes = [r for r in records if "label" in r]
    assert [r["label"] for r in passes] == ["A1", "A2", "A3", "B1", "B2", "B3",
                                           "C.base", "C.X1", "C.Y", "C.X2", "D.X"]
    assert [r["adapter"] for r in passes] == ["X"] * 6 + ["base", "X", "Y", "X", "X"]
    calls = [e for e in fake.events if isinstance(e, tuple) and len(e) == 4]
    assert len(calls) == 330
    assert calls[:30] == calls[30:60] == calls[60:90]
    salts = [calls[n][3] for n in (90, 120, 150)]
    assert len(set(salts)) == 3 and all(salts)
    for start in (90, 120, 150):
        assert len({c[3] for c in calls[start:start + 30]}) == 1
    assert all(c[3] is None for c in calls[:90] + calls[180:])
    for n, event in enumerate(fake.events):
        if isinstance(event, tuple) and len(event) == 4:
            assert fake.events[n - 1] == fake.events[n + 1] == "metrics"
    reload_at = fake.events.index(("reload", X))
    assert sum(isinstance(e, tuple) and len(e) == 4 for e in fake.events[:reload_at]) == 300
    assert [len(r["in_flight"]) for r in passes] == [30] * 10 + [32]
    assert all(not r["other_in_flight"] for r in passes)
    assert len({r["logprobs_sha256"] for r in passes}) == 1
    assert all(r["max_logprob_diff"] == 0 and r["flips"] == 0 for r in passes)
    assert records[-1]["verdict"] == "noise"
    assert "Write one line" not in json.dumps(records)


def test_unsupported_salt_skips_all_b_and_still_reads_after_error() -> None:
    fake = FakeClient(unsupported=True)
    records = run(fake)
    assert {"cache_salt": "unsupported"} in records
    assert [r["label"] for r in records if "label" in r] == ["A1", "A2", "A3", "C.base",
                                                            "C.X1", "C.Y", "C.X2", "D.X"]
    assert sum(isinstance(e, tuple) and len(e) == 4 for e in fake.events) == 241
    rejected = next(n for n, event in enumerate(fake.events)
                    if isinstance(event, tuple) and len(event) == 4 and event[3])
    assert fake.events[rejected - 1] == fake.events[rejected + 1] == "metrics"


def test_metrics_sums_workers_and_keeps_optional_lora_gauge() -> None:
    result = repro.parse_metrics(QUIET + 'vllm:num_requests_running{engine="1"} 2\n'
                                 'vllm:lora_requests_info{running_lora_adapters="x,y"} 1\n')
    assert result["running"] == 2 and result["waiting"] == 0
    assert result["lora"][0]["value"] == 1
    for bad in ("", QUIET.replace(" 0", " NaN"), QUIET.replace(" 0", " -1")):
        with pytest.raises(ServingError):
            repro.parse_metrics(bad)


def result(label: str, scores: tuple[float, ...] = (0.0, -1.0), busy: bool = False) -> repro.Pass:
    readings = [{"before": {"running": int(busy), "waiting": 0},
                 "after": {"running": 0, "waiting": 0}} for _ in scores]
    return repro.Pass(label, "X", scores, readings)


@pytest.mark.parametrize(("passes", "expected"), [
    ([result("A1"), result("A2", (0.5, -1)), result("A3"),
      result("B1", (0.7, -1)), result("B2", (0.7, -1)), result("B3", (0.7, -1))], "H1"),
    ([result("A1"), result("A2", (0.5, -1), True)], "H2"),
    ([result("A1"), result("A2"), result("A3"), result("C.X1"),
      result("C.X2", (0.5, -1)), result("D.X")], "H3"),
    ([result("A1"), result("A2", (0.009, -1))], "noise"),
    ([result("A1"), result("A2", (0.01, -1))], "inconclusive"),
    ([result("A1"), result("A2", (0.5, -1)), result("C.X1"),
      result("C.X2", (0.5, -1)), result("D.X")], "inconclusive"),
    ([result("A1"), result("A2", (0, -0.5)),
      repro.Pass("A3", "X", (0, -0.5), [{"before": {"running": 1, "waiting": 0},
                                         "after": {"running": 0, "waiting": 0}},
                                        {"before": {"running": 0, "waiting": 0},
                                         "after": {"running": 0, "waiting": 0}}])], "inconclusive"),
])
def test_verdict(passes: list[repro.Pass], expected: str) -> None:
    assert repro.verdict(passes) == expected


def test_salt_is_per_request_and_default_payload_is_unchanged() -> None:
    fake = FakeVllm()
    client = VllmClient("https://serve.example", "tok", transport=httpx.MockTransport(fake)).reusing_tokens()
    for salt in (None, "pass-1", "pass-2", None):
        client.continuation_logprobs(None, [{"role": "user", "content": "q"}], " A", cache_salt=salt)
    bodies = [b for path, b, _ in fake.requests if path == "/v1/completions"]
    assert [b.get("cache_salt") for b in bodies] == [None, "pass-1", "pass-2", None]
    assert "cache_salt" not in bodies[0] and bodies[0] == bodies[3]
    assert all("cache_salt" not in b for path, b, _ in fake.requests if path == "/tokenize")


def test_http_metrics_and_explicit_reload_forget_local_loaded_set() -> None:
    events: list[tuple[str, str]] = []

    def server(request: httpx.Request) -> httpx.Response:
        events.append((request.method, request.url.path))
        assert request.headers["Authorization"] == "Bearer tok"
        if request.url.path == "/metrics":
            return httpx.Response(200, text=QUIET)
        if request.url.path.endswith("lora_adapter"):
            assert json.loads(request.content)["lora_name"] == X
            return httpx.Response(200, text="Success")
        return FakeVllm()(request)

    client = VllmClient("https://serve.example", "tok", transport=httpx.MockTransport(server),
                        adapters_dir=config.ADAPTERS_DIR).reusing_tokens()
    observed: list[str] = []

    @contextmanager
    def observer(path: str) -> Iterator[None]:
        assert client.metrics() == QUIET
        try:
            yield
        finally:
            assert client.metrics() == QUIET
            observed.append(path)

    client = client.observing_requests(observer)
    from anjo.sleep.gates import EVAL_DAY
    from anjo.sleep.types import CheckpointRef

    ref = CheckpointRef(X, "test", EVAL_DAY, "unused", None, rendering=client.rendering)
    client.continuation_logprobs(ref, [{"role": "user", "content": "q"}], " A")
    client.reload_adapter(X)
    client.continuation_logprobs(ref, [{"role": "user", "content": "q"}], " A")
    assert [path for _, path in events if path.endswith("lora_adapter")] == [
        "/v1/load_lora_adapter", "/v1/unload_lora_adapter", "/v1/load_lora_adapter"]
    assert observed == ["/v1/load_lora_adapter", "/tokenize", "/tokenize", "/tokenize", "/v1/completions",
                        "/v1/unload_lora_adapter", "/v1/load_lora_adapter", "/v1/completions"]
    assert [method for method, _ in events] == ["GET", "POST", "GET"] * len(observed)


def test_settles_hold_when_wake_fails(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl", caps={**dict.fromkeys(POOLS, 0.0), "modal_usd": 5.0})
    assert repro.main([*ARGS, "--go"], environ={}, client=never, wake=lambda: WakeResult(1.0, False),
                      ledger=ledger, out=lambda _: None) == 1
    assert ledger.open_holds() == ()


def test_deadline_stops_inside_a_pass_and_never_calls_reload() -> None:
    fake = FakeClient()
    ticks = iter(float(n * 100) for n in range(100))
    records: list[str] = []
    assert repro.main([*ARGS, "--go"], environ={}, client=fake, wake=lambda: WakeResult(0, True),
                      ledger=None, clock=lambda: next(ticks), out=records.append) == 2
    assert not any(isinstance(e, tuple) and e[0] == "reload" for e in fake.events)
    assert json.loads(records[-1])["verdict"] == "incomplete"


def test_flips_count_changed_winners_and_exclude_incomplete_items() -> None:
    selected = (repro.Choice("full", 0, 2, "q", "a"), repro.Choice("full", 1, 2, "q", "b"),
                repro.Choice("partial", 0, 3, "q", "c"), repro.Choice("partial", 1, 3, "q", "d"))
    first = result("A1", (0, -1, 0, -1))
    got = result("A2", (-1, 0, -1, 0))
    assert repro.pass_line(got, first, selected)["flips"] == 1
    assert repro.pass_line(first, first, selected)["flips"] == 0


def test_non_salt_errors_are_not_skipped_and_hold_settles(tmp_path: Path) -> None:
    class Broken(FakeClient):
        def metrics(self) -> str:
            raise ServingError("/metrics returned HTTP 401")

    ledger = Ledger(tmp_path / "ledger.jsonl", caps={**dict.fromkeys(POOLS, 0.0), "modal_usd": 5.0})
    with pytest.raises(ServingError, match="401"):
        run(Broken(), ledger=ledger)
    assert ledger.open_holds() == ()


def test_night_lock_refuses_before_any_request(tmp_path: Path) -> None:
    import fcntl

    from anjo.sleep.night.sweep import LOCK_NAME

    ledger = Ledger(tmp_path / "ledger.jsonl", caps={**dict.fromkeys(POOLS, 0.0), "modal_usd": 5.0})
    with (tmp_path / LOCK_NAME).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert repro.main([*ARGS, "--go"], environ={}, client=never, wake=never,
                          ledger=ledger, out=lambda _: None) == 2
    assert ledger.open_holds() == ()


@pytest.mark.parametrize("edge", ["before", "after"])
def test_observer_transport_failure_never_replays_a_post(edge: str) -> None:
    fake = FakeVllm()

    @contextmanager
    def observer(path: str) -> Iterator[None]:
        if path == "/v1/completions" and edge == "before":
            raise TransportFailed("metrics failed before")
        yield
        if path == "/v1/completions" and edge == "after":
            raise TransportFailed("metrics failed after")

    client = VllmClient("https://serve.example", "tok", transport=httpx.MockTransport(fake),
                        sleep=lambda _: None).observing_requests(observer)
    try:
        assert client.attempts > 1
        with pytest.raises(TransportFailed, match="metrics failed"):
            client.continuation_logprobs(None, [{"role": "user", "content": "q"}], " A")
        assert sum(path == "/v1/completions" for path, _, _ in fake.requests) == (edge == "after")
        assert client.retries() == 0
    finally:
        client.close()
