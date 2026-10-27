# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Glimmer on SageMaker behind the serving port: the backend switch, the wake and sleep around every
phase, the adapter upload that never overwrites, the timeouts on every AWS call, and the IAM
documents. Every AWS client is a stand-in checked against botocore's own service model, offline."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import anjo.sleep
from anjo.sleep.serving import backend, client, config, wake
from anjo.sleep.serving.backend import BACKEND_ENV, MODAL_TOKEN_ENV, MODAL_URL_ENV, ModalServing
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.serving.config import ServeConfigError
from anjo.sleep.serving.sagemaker import settings
from anjo.sleep.serving.sagemaker.serving import SageMakerServing
from anjo.sleep.tests.aws_fakes import Body, EventBody, FakeAws, client_error

SLEEP_ROOT = Path(anjo.sleep.__file__).resolve().parent
AWS_DOCS = SLEEP_ROOT / "deploy" / "aws"
URL = "https://glimmer.example.test"
TOKEN = "-".join(("fake", "bearer", "value"))  # built at runtime: no key-shaped literal for the secret scan
BUCKET = "anjo-sleep-sagemaker-test"  # the operator's policy reaches only buckets named *sagemaker*
ADAPTER = "p-1a2b3c4d-n3"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Component:
    """The inference component's copy count, as SageMaker's control plane reports and sets it."""

    def __init__(self, desired: int, order: list[str] | None = None) -> None:
        self.desired = desired
        self.order = order if order is not None else []
        self.updates: list[dict[str, Any]] = []

    def describe(self, **params: Any) -> dict[str, Any]:
        self.order.append(f"describe:{self.desired}")
        return {"InferenceComponentName": params["InferenceComponentName"], "InferenceComponentStatus": "InService",
                "RuntimeConfig": {"DesiredCopyCount": self.desired, "CurrentCopyCount": self.desired}}

    def update(self, **params: Any) -> dict[str, Any]:
        self.updates.append(params)
        self.desired = params["DesiredRuntimeConfig"]["CopyCount"]
        self.order.append(f"set:{self.desired}")
        return {"InferenceComponentArn": f"arn:aws:sagemaker:{settings.REGION}:{'0' * 12}:inference-component/x"}

    def control(self) -> FakeAws:
        return FakeAws("sagemaker", describe_inference_component=self.describe,
                       update_inference_component_runtime_config=self.update)


def _endpoint(clock: Clock, ready_at: float, order: list[str]) -> FakeAws:
    """The endpoint: no capacity (SageMaker's 400) until the copy is up, then vLLM's model list."""

    def invoke(**params: Any) -> dict[str, Any]:
        order.append("poll")
        clock.now += 1.0  # each poll takes a second
        if clock.now < ready_at:
            raise client_error("ValidationError", 400)
        return {"Body": Body(b'{"object": "list", "data": []}'), "ContentType": "application/json"}

    return FakeAws("sagemaker-runtime", invoke_endpoint=invoke)


def _sagemaker(component: Component, endpoint: FakeAws | None = None, clock: Clock | None = None,
               **clients: Any) -> SageMakerServing:
    clock = clock or Clock()
    return SageMakerServing(bucket=BUCKET, control=component.control(), ping_runtime=endpoint,
                            clock=clock, sleep=clock.sleep, **clients)


# ── The backend switch ───────────────────────────────────────────────────────


def test_backend_env_selects_sagemaker_or_modal(monkeypatch: pytest.MonkeyPatch) -> None:
    assert backend.backend_name({}) == "sagemaker"  # the default on the dev host
    assert backend.backend_name({BACKEND_ENV: "modal"}) == "modal"
    assert backend.backend_name({BACKEND_ENV: " sagemaker "}) == "sagemaker"
    with pytest.raises(ServeConfigError, match="sagemaker or modal"):
        backend.backend_name({BACKEND_ENV: "baseten"})

    chosen = backend.from_env({BACKEND_ENV: "sagemaker", settings.BUCKET_ENV: BUCKET})
    assert isinstance(chosen, SageMakerServing) and chosen.name == "sagemaker"
    assert isinstance(backend.from_env({}), SageMakerServing)  # no bucket yet: only an upload needs it
    with pytest.raises(ServeConfigError, match="anjo-sleep-sagemaker-"):
        backend.from_env({settings.BUCKET_ENV: "anjo-sleep-adapters"})  # outside what the operator can write

    modal = backend.from_env({BACKEND_ENV: "modal", MODAL_URL_ENV: URL, MODAL_TOKEN_ENV: TOKEN})
    assert isinstance(modal, ModalServing) and modal.name == "modal"
    with pytest.raises(ServeConfigError, match=f"{MODAL_URL_ENV} is not set"):
        backend.from_env({BACKEND_ENV: "modal", MODAL_TOKEN_ENV: TOKEN})

    # Modal keeps working as it did: its wake is the bearer-token poll, its client the https server,
    # and sleeping is Modal's own idle window, so sleep_now calls nothing.
    asked: list[tuple[str, str, float]] = []

    def woken(url: str, token: str, timeout_s: float) -> WakeResult:
        asked.append((url, token, timeout_s))
        return WakeResult(3.0, True)

    monkeypatch.setattr(client, "wake", woken)
    assert modal.wake() == WakeResult(3.0, True) and asked == [(URL, TOKEN, client.WAKE_TOTAL_TIMEOUT_S)]
    modal.sleep_now()
    assert isinstance(modal.client(), client.VllmClient)

    monkeypatch.setenv(BACKEND_ENV, "modal")
    monkeypatch.setenv(MODAL_URL_ENV, URL)
    monkeypatch.setenv(MODAL_TOKEN_ENV, TOKEN)
    assert backend.from_env().name == "modal"  # the process environment when no mapping is given
    assert wake.URL_ENV == MODAL_URL_ENV and wake.TOKEN_ENV == MODAL_TOKEN_ENV  # one name for each


# ── Waking and sleeping ──────────────────────────────────────────────────────


def test_sagemaker_wake_sets_copies_then_polls_and_is_bounded() -> None:
    assert settings.WAKE_TOTAL_TIMEOUT_S == 1200.0  # a cold start provisions the instance first
    order: list[str] = []
    clock = Clock()
    component = Component(0, order)
    endpoint = _endpoint(clock, ready_at=300.0, order=order)
    done = _sagemaker(component, endpoint, clock).wake()

    assert done.ok and 300.0 <= done.seconds < 330.0
    assert order[:3] == ["describe:0", "set:1", "poll"]  # one copy first, then the polls
    assert component.updates == [{"InferenceComponentName": settings.COMPONENT, "DesiredRuntimeConfig": {"CopyCount": 1}}]
    polls = endpoint.named("invoke_endpoint")
    assert len(polls) > 3 and all(
        (p["EndpointName"], p["InferenceComponentName"]) == (settings.ENDPOINT, settings.COMPONENT) for p in polls)
    assert {json.dumps(json.loads(p["Body"]), sort_keys=True) for p in polls} == {
        json.dumps({"body": {}, "path": "/v1/models", "stream": False}, sort_keys=True)}
    assert clock.sleeps == sorted(clock.sleeps) and max(clock.sleeps) <= client.WAKE_BACKOFF_MAX_S

    # Already up: nothing is set, one poll answers.
    order.clear()
    awake = Component(1, order)
    assert _sagemaker(awake, _endpoint(Clock(), 0.0, order)).wake().ok and order == ["describe:1", "poll"]

    # Never up: it stops at the hard total and says so.
    clock = Clock()
    never = _sagemaker(Component(0), _endpoint(clock, 10**9, []), clock).wake()
    assert not never.ok and settings.WAKE_TOTAL_TIMEOUT_S <= never.seconds <= settings.WAKE_TOTAL_TIMEOUT_S + 60


def test_sagemaker_wake_puts_back_a_copy_the_idle_net_took_and_gives_up_after_a_few() -> None:
    order: list[str] = []
    clock = Clock()
    component = Component(0, order)

    def invoke(**params: Any) -> dict[str, Any]:
        clock.now += 1.0
        if order.count("set:1") == 1 and component.desired == 1:
            component.desired = 0  # the AWS-side idle net fires once, mid-wake
            order.append("net:0")
        if clock.now < 200.0:
            raise client_error("ValidationError", 400)
        return {"Body": Body(b"{}"), "ContentType": "application/json"}

    endpoint = FakeAws("sagemaker-runtime", invoke_endpoint=invoke)
    assert _sagemaker(component, endpoint, clock).wake().ok
    assert order.count("set:1") == 2 and order.index("net:0") < len(order) - order[::-1].index("set:1") - 1

    stubborn = Component(0)
    stubborn.update = lambda **params: {"InferenceComponentArn": "x"}  # type: ignore[method-assign]  # never sticks
    clock = Clock()
    gave_up = _sagemaker(stubborn, _endpoint(clock, 10**9, []), clock).wake()
    assert not gave_up.ok and gave_up.seconds < settings.WAKE_TOTAL_TIMEOUT_S


def test_sleep_now_sets_no_copies_and_a_ping_needs_the_model_list() -> None:
    order: list[str] = []
    component = Component(1, order)
    serving = _sagemaker(component, _endpoint(Clock(), 0.0, order))
    assert serving.ping() is True
    serving.sleep_now()
    assert component.desired == 0 and order[-1] == "set:0"
    order.clear()
    serving.sleep_now()  # already asleep: nothing is set again
    assert order == ["describe:0"]
    assert _sagemaker(Component(0), _endpoint(Clock(), 10**9, [])).ping() is False


def _phases_world(tmp_path: Path) -> tuple[Any, Any]:
    from anjo.sleep.tests.night_fakes import NIGHT, World

    world = World(tmp_path / "night")
    world.plan("p1", NIGHT)
    return world, NIGHT


class Boom(RuntimeError):
    pass


def _boom(*_args: Any, **_kwargs: Any) -> Any:
    raise Boom("a step broke")


def _day(sleeper: Any) -> None:
    from anjo.sleep.personas.model import load_person_file
    from anjo.sleep.sim.day import run_day

    class Broken:
        def send(self, session_id: str, text: str, *, cleared: bool) -> str:
            raise Boom("the dev API broke")

    class Model:
        def complete(self, system: str, messages: list[dict[str, str]], *, temperature: float, seed: int) -> str:
            return "A line."

    class Replies:
        def of_reply(self, session_id: str) -> frozenset[str] | None:
            return None

    person = load_person_file(SLEEP_ROOT / "personas" / "founder.json").visible
    run_day(person, 1, 7, model=Model(), transport=Broken(), replies=Replies(),
            wake=lambda: WakeResult(1.0, True), sleep=sleeper)


def _night(sleeper: Any, tmp_path: Path) -> None:
    from anjo.sleep.night.run import run_night_batch_with

    world, night = _phases_world(tmp_path)
    run_night_batch_with(world.deps(wake=lambda: WakeResult(1.0, True), sleep=sleeper, evidence=_boom),
                         {"p1": night})


def _morning(sleeper: Any, tmp_path: Path) -> None:
    from anjo.sleep.tests.test_morning import Rig

    rig = Rig(tmp_path)
    rig.morning = replace(rig.morning, wake=lambda: WakeResult(1.0, True), sleep=sleeper,
                          judge=type("J", (), {"verdict": _boom})())
    rig.run()


class FakeServing:
    name = "fake"

    def __init__(self, order: list[str]) -> None:
        self.order = order

    def wake(self) -> WakeResult:
        self.order.append("wake")
        return WakeResult(1.0, True)

    def ping(self) -> bool:
        self.order.append("ping")
        raise Boom("the endpoint broke mid-window")

    def sleep_now(self) -> None:
        self.order.append("sleep")


@pytest.mark.parametrize("phase", ["day", "night", "morning", "keep_warm"])
def test_every_phase_puts_glimmer_back_to_sleep_even_on_failure(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    def sleeper() -> None:
        order.append("sleep")

    with pytest.raises(Boom):
        if phase == "day":
            _day(sleeper)
        elif phase == "night":
            _night(sleeper, tmp_path)
        elif phase == "morning":
            _morning(sleeper, tmp_path)
        else:
            monkeypatch.setattr(backend, "from_env", lambda environ=None: FakeServing(order))
            monkeypatch.setenv(wake.DATA_DIR_ENV, str(tmp_path))
            wake.main(["--keep-warm-minutes", "5"])
    assert order.count("sleep") == 1 and order[-1] == "sleep"


def test_a_phase_that_ends_well_also_sleeps_and_nested_phases_sleep_once_at_the_end(tmp_path: Path) -> None:
    from anjo.sleep.night import daily
    from anjo.sleep.night.run import run_night_batch_with
    from anjo.sleep.tests.test_morning import Rig
    from anjo.sleep.types import NightResult

    order: list[str] = []

    def sleeper() -> None:
        order.append("sleep")

    world, night = _phases_world(tmp_path)
    result = run_night_batch_with(world.deps(wake=lambda: WakeResult(1.0, True), sleep=sleeper), {"p1": night})["p1"]
    assert result.status == "promoted" and order == ["sleep"]

    # The daily night phase runs the night and then the morning in one process: Glimmer stays up
    # between them and goes to sleep once, after the morning.
    order.clear()
    rig = Rig(tmp_path / "morning")
    rig.morning = replace(rig.morning, sleep=sleeper)

    def nights(**kw: Any) -> dict[str, NightResult]:
        with backend.phase(sleeper):
            order.append("night")
        return {"acct": NightResult("nothing_new", None, None, {}, (), 0.0, {}, tmp_path / "log.jsonl")}

    def score(*args: Any, **kw: Any) -> None:
        order.append("morning")
        rig.run()

    markers = tmp_path / "daily"
    daily.write_marker(markers, "founder", date(2026, 10, 29), 1,
                       __import__("anjo.sleep.sim.day", fromlist=["DayRun"]).DayRun(
                           (datetime(2026, 10, 29, tzinfo=timezone.utc),) * 2, 4, 0, ()))
    daily.run_night_phase("acct", "founder", datetime(2026, 10, 30, 2, 0, tzinfo=daily.TAIPEI), markers=markers,
                          nights=nights, morning=lambda *a: None, score=score)
    assert order == ["night", "morning", "sleep"]


def test_keep_warm_on_sagemaker_pings_through_the_backend_and_sleeps_at_the_end(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    order: list[str] = []
    monkeypatch.setenv(wake.DATA_DIR_ENV, str(tmp_path))

    class Healthy(FakeServing):
        def ping(self) -> bool:
            self.order.append("ping")
            return True

    monkeypatch.setattr(backend, "from_env", lambda environ=None: Healthy(order))
    monkeypatch.setattr("anjo.sleep.serving.wake.time.sleep", lambda seconds: None)
    ticks = iter(range(0, 10**6, 60))
    monkeypatch.setattr("anjo.sleep.serving.wake.time.monotonic", lambda: float(next(ticks)))
    assert wake.main(["--keep-warm-minutes", "5"]) == 0
    assert order[0] == "wake" and order[-1] == "sleep" and order.count("ping") >= 2
    assert '"keep_warm_minutes": 5.0' in capsys.readouterr().out

    # A wake on its own leaves Glimmer up for the demo, unless the wake failed.
    order.clear()
    assert wake.main(["--wake"]) == 0 and order == ["wake"]

    class Cold(Healthy):
        def wake(self) -> WakeResult:
            self.order.append("wake")
            return WakeResult(1200.0, False)

    order.clear()
    monkeypatch.setattr(backend, "from_env", lambda environ=None: Cold(order))
    assert wake.main(["--wake"]) == 1 and order == ["wake", "sleep"]


# ── Adapters ─────────────────────────────────────────────────────────────────


def _adapter(tmp_path: Path, **meta: Any) -> Path:
    folder = tmp_path / ADAPTER
    folder.mkdir(parents=True, exist_ok=True)
    body = {"peft_type": "LORA", "base_model_name_or_path": config.MODEL, "r": 16, **meta}
    (folder / "adapter_config.json").write_text(json.dumps(body), encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(b"\0" * 64)
    return folder


class Bucket:
    def __init__(self, keys: set[str] | None = None) -> None:
        self.keys = set(keys or ())
        self.lost_race = False

    def listing(self, **params: Any) -> dict[str, Any]:
        found = sorted(k for k in self.keys if k.startswith(params["Prefix"]))
        return {"KeyCount": len(found), "Contents": [{"Key": k} for k in found]}

    def put(self, **params: Any) -> dict[str, Any]:
        if params["Key"] in self.keys or self.lost_race:
            raise client_error("PreconditionFailed", 412, "PutObject")
        self.keys.add(params["Key"])
        return {"ETag": '"x"'}

    def fake(self) -> FakeAws:
        return FakeAws("s3", list_objects_v2=self.listing, put_object=self.put)


def test_adapter_upload_refuses_existing_name(tmp_path: Path) -> None:
    folder = _adapter(tmp_path)
    taken = Bucket({f"adapters/{ADAPTER}/adapter_config.json"})
    s3 = taken.fake()
    with pytest.raises(ServeConfigError, match="already"):
        _sagemaker(Component(0), s3=s3).upload_adapter(folder, ADAPTER)
    assert s3.named("put_object") == []  # checked first: nothing is written

    fresh = Bucket()
    s3 = fresh.fake()
    _sagemaker(Component(0), s3=s3).upload_adapter(folder, ADAPTER)
    puts = s3.named("put_object")
    assert [p["Key"] for p in puts] == [f"adapters/{ADAPTER}/adapter_model.safetensors",
                                        f"adapters/{ADAPTER}/adapter_config.json"]  # the config last: it marks whole
    assert all(p["IfNoneMatch"] == "*" and p["Bucket"] == BUCKET for p in puts)
    (listed,) = s3.named("list_objects_v2")
    assert listed["Prefix"] == f"adapters/{ADAPTER}/"

    racing = Bucket()
    racing.lost_race = True  # another upload of the same name landed between the check and the put
    with pytest.raises(ServeConfigError, match="already"):
        _sagemaker(Component(0), s3=racing.fake()).upload_adapter(folder, ADAPTER)

    bad = _adapter(tmp_path / "bad", r=64)
    s3 = Bucket().fake()
    with pytest.raises(ServeConfigError, match="rank"):
        _sagemaker(Component(0), s3=s3).upload_adapter(bad, ADAPTER)
    assert s3.calls == []
    with pytest.raises(ServeConfigError, match=settings.BUCKET_ENV):
        SageMakerServing(bucket=None, s3=Bucket().fake()).upload_adapter(folder, ADAPTER)


def test_the_night_uploads_through_the_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from anjo.sleep.night import __main__ as night_command

    seen: list[tuple[Path, str]] = []

    class Uploads:
        def upload_adapter(self, folder: Path, name: str) -> None:
            seen.append((folder, name))

    monkeypatch.setattr(backend, "from_env", lambda environ=None: Uploads())
    night_command._publish(tmp_path, ADAPTER)
    assert seen == [(tmp_path, ADAPTER)]


# ── Timeouts ─────────────────────────────────────────────────────────────────


def test_invoke_calls_have_timeouts() -> None:
    serving = SageMakerServing(bucket=BUCKET)
    made = {"control": serving._control(), "runtime": serving._runtime(), "ping": serving._ping_runtime(),
            "s3": serving._s3()}
    for name, aws in made.items():
        cfg = aws.meta.config
        assert 0 < cfg.connect_timeout <= 10 and 0 < cfg.read_timeout <= client.REQUEST_TIMEOUT_S, name
        assert aws.meta.region_name == settings.REGION == "us-east-1"
        assert cfg.retries["total_max_attempts"] <= 3, name
    assert made["runtime"].meta.config.read_timeout == client.REQUEST_TIMEOUT_S
    assert made["runtime"].meta.config.retries["total_max_attempts"] == 1  # a retry would double the wait
    assert made["ping"].meta.config.read_timeout == client.WAKE_POLL_TIMEOUT_S

    # The night's client speaks the same vLLM API through the endpoint's envelope, on the stream API.
    calls: list[dict[str, Any]] = []

    def invoke(**params: Any) -> dict[str, Any]:
        calls.append(json.loads(params["Body"]))
        return {"Body": EventBody([b'{"tokens": [1,', b' 2]}']), "ContentType": "application/json"}

    night = SageMakerServing(bucket=BUCKET, runtime=FakeAws(
        "sagemaker-runtime", invoke_endpoint_with_response_stream=invoke)).client()
    assert night._tokens({"model": config.MODEL, "prompt": "x"}) == [1, 2]
    assert calls == [{"path": "/tokenize", "body": {"model": config.MODEL, "prompt": "x"}, "stream": True}]

    def refused(**params: Any) -> dict[str, Any]:
        raise client_error("ModelError", 424, "InvokeEndpointWithResponseStream", OriginalStatusCode=503)

    broken = SageMakerServing(bucket=BUCKET, runtime=FakeAws(
        "sagemaker-runtime", invoke_endpoint_with_response_stream=refused)).client()
    with pytest.raises(client.ServingError, match="HTTP 503") as raised:
        broken.complete(None, [{"role": "user", "content": "private words"}], temperature=1.0)
    assert "private" not in str(raised.value)


def _sse(*events: dict[str, Any] | str) -> bytes:
    return b"".join(b"data: " + (e if isinstance(e, str) else json.dumps(e)).encode() + b"\n\n" for e in events)


def test_long_generation_uses_the_stream_api() -> None:
    # InvokeEndpoint gives up after 60 s; a night's long reply streams, so only the read timeout
    # bounds it. The text comes back as one completion, cut anywhere on the way.
    calls: list[dict[str, Any]] = []
    sent = _sse(
        {"model": config.MODEL, "choices": [{"index": 0, "text": ""}]},
        {"model": config.MODEL, "choices": [{"index": 0, "text": "Good "}]},
        {"model": config.MODEL, "choices": [{"index": 0, "text": "night \u2014 sleep", "finish_reason": "stop"}]},
        {"model": config.MODEL, "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 4}},
        "[DONE]")
    bodies: list[EventBody] = []

    def stream(**params: Any) -> dict[str, Any]:
        calls.append(json.loads(params["Body"]))
        if calls[-1]["path"] == "/tokenize":  # the chat's ids, then the reply's header
            return {"Body": EventBody([b'{"tokens": [1, 2]}']), "ContentType": "application/json"}
        bodies.append(EventBody(sent[i:i + 7] for i in range(0, len(sent), 7)))  # splits lines and a UTF-8 char
        return {"Body": bodies[-1], "ContentType": "text/event-stream"}

    only_streams = FakeAws("sagemaker-runtime", invoke_endpoint_with_response_stream=stream)
    night = SageMakerServing(bucket=BUCKET, runtime=only_streams).client()
    result = night.complete(None, [{"role": "user", "content": "hi"}], temperature=1.0)
    assert (result.text, result.finish_reason, result.model) == ("Good night \u2014 sleep", "stop", config.MODEL)
    assert (result.input_tokens, result.output_tokens) == (12, 4)
    assert [call["path"] for call in calls] == ["/tokenize", "/tokenize", "/v1/completions"]
    call = calls[-1]
    assert call["stream"] is True and call["body"]["prompt"] == [1, 2, 1, 2]
    assert call["body"]["stream"] is True and call["body"]["stream_options"] == {"include_usage": True}
    assert "stream" not in calls[0]["body"] and bodies[-1].closed  # token ids come back as JSON

    # An error event, a stream that ends early or drops, all fail the call, never a word of it.
    for broken in (_sse({"error": {"message": "private words", "code": 500}}),
                   _sse({"model": config.MODEL, "choices": [{"index": 0, "text": "private"}]})):
        sent = broken
        with pytest.raises(client.ServingError) as raised:
            night.complete(None, [{"role": "user", "content": "hi"}], temperature=1.0)
        assert "private" not in str(raised.value)

    def dropped(**params: Any) -> dict[str, Any]:
        return {"Body": EventBody([b"data: {"], end=client_error("ModelStreamError", 400)), "ContentType": "x"}

    cut = SageMakerServing(bucket=BUCKET, runtime=FakeAws(
        "sagemaker-runtime", invoke_endpoint_with_response_stream=dropped)).client()
    with pytest.raises(client.ServingError, match="failed in transport"):
        cut.complete(None, [{"role": "user", "content": "hi"}], temperature=1.0)


# ── No keys, least privilege ─────────────────────────────────────────────────

_KEY_ID = re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16}\b")
_ACCOUNT_IN_ARN = re.compile(r"arn:aws:[a-z0-9-]+:[a-z0-9-]*:[0-9]{12}:")
# Built at runtime, so this file holds none of the names it looks for.
_KEY_NAMES = tuple(prefix + rest for prefix in ("AWS_", "aws_")
                   for rest in ("SECRET_ACCESS_KEY", "ACCESS_KEY_ID", "SESSION_TOKEN",
                                "secret_access_key", "access_key_id", "session_token")
                   if rest.isupper() == prefix.isupper())


def _key_findings(text: str) -> list[str]:
    found = [f"key id {m.start()}" for m in _KEY_ID.finditer(text)]
    found += [f"key name {name}" for name in _KEY_NAMES if name in text]
    found += [f"account id {m.start()}" for m in _ACCOUNT_IN_ARN.finditer(text)]
    return found


def test_no_aws_key_in_repo_or_env_example() -> None:
    assert _key_findings("AK" + "IA" + "Q" * 16) and _key_findings("AWS_" + "SECRET_ACCESS_KEY=")  # it would see one
    texts = [p for p in SLEEP_ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and p.suffix in {".py", ".md", ".json", ".toml", ".txt", ".ini", ""}]
    example = SLEEP_ROOT.parents[1] / ".env.example"  # the private repo's; the public repo has none
    texts += [example] if example.is_file() else []
    findings = {str(p.relative_to(SLEEP_ROOT.parents[1])): _key_findings(p.read_text(encoding="utf-8", errors="replace"))
                for p in texts}
    assert {path: hits for path, hits in findings.items() if hits} == {}
    runbook = (SLEEP_ROOT / "deploy" / "dev_host.md").read_text(encoding="utf-8")
    for name in ("AWS_ROLE_ARN", "AWS_WEB_IDENTITY_TOKEN_FILE", "ANJO_SLEEP_BACKEND", settings.BUCKET_ENV):
        assert f"`{name}`" in runbook  # the keyless route is the one the runbook gives


def _statements(name: str) -> list[dict[str, Any]]:
    doc = json.loads((AWS_DOCS / name).read_text(encoding="utf-8"))
    assert doc["Version"] == "2012-10-17"
    statements: list[dict[str, Any]] = doc["Statement"]
    for statement in statements:
        assert statement["Effect"] == "Allow" and statement.get("Sid")
        assert not {"NotAction", "NotResource", "NotPrincipal"} & set(statement)
    return statements


def _listed(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def test_iam_runtime_policy_is_least_privilege() -> None:
    assert {p.name for p in AWS_DOCS.glob("*.json")} == {
        "devhost-trust.json", "devhost-permissions.json",
        "sagemaker-exec-trust.json", "sagemaker-exec-permissions.json"}  # one trust and one permissions per role
    runtime = _statements("devhost-permissions.json")
    by_sid = {s["Sid"]: s for s in runtime}
    actions = {a for s in runtime for a in _listed(s["Action"])}
    assert actions == {
        "sagemaker:InvokeEndpoint", "sagemaker:InvokeEndpointWithResponseStream",
        "sagemaker:DescribeInferenceComponent", "sagemaker:UpdateInferenceComponentRuntimeConfig",
        "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket",
    }
    endpoint = f"arn:aws:sagemaker:us-east-1:<ACCOUNT_ID>:endpoint/{settings.ENDPOINT}"
    component = f"arn:aws:sagemaker:us-east-1:<ACCOUNT_ID>:inference-component/{settings.COMPONENT}"
    bucket = f"arn:aws:s3:::{settings.BUCKET_PLACEHOLDER}"
    assert _listed(by_sid["InvokeOurEndpointOnly"]["Resource"]) == [endpoint, component]
    assert _listed(by_sid["WakeAndSleepOurComponentOnly"]["Resource"]) == [component]
    assert _listed(by_sid["AdaptersOnly"]["Resource"]) == [f"{bucket}/adapters/*"]
    listing = by_sid["ListAdaptersOnly"]
    assert _listed(listing["Resource"]) == [bucket]
    assert listing["Condition"] == {"StringLike": {"s3:prefix": ["adapters/*"]}}

    (assume,) = _statements("devhost-trust.json")
    from anjo.sleep.serving.sagemaker import token_refresh

    assert assume["Principal"] == {"Federated": "accounts.google.com"}  # Google's built-in provider
    assert assume["Action"] == "sts:AssumeRoleWithWebIdentity"
    assert assume["Condition"] == {"StringEquals": {
        "accounts.google.com:sub": "<VM_SERVICE_ACCOUNT_UNIQUE_ID>",
        "accounts.google.com:aud": "<VM_SERVICE_ACCOUNT_UNIQUE_ID>",
        "accounts.google.com:oaud": token_refresh.AUDIENCE,
    }}
    (service,) = _statements("sagemaker-exec-trust.json")
    assert service["Principal"] == {"Service": "sagemaker.amazonaws.com"} and service["Action"] == "sts:AssumeRole"
    execution = {a for s in _statements("sagemaker-exec-permissions.json") for a in _listed(s["Action"])}
    assert not {"s3:PutObject", "s3:DeleteObject", "sagemaker:InvokeEndpoint"} & execution  # the container only reads
    assert settings.RUNTIME_ROLE == "anjo-sleep-devhost" and settings.EXECUTION_ROLE == "anjo-sleep-sagemaker-exec"

    readme = (AWS_DOCS / "README.md").read_text(encoding="utf-8")
    for path in sorted(AWS_DOCS.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"[0-9]{12}", text), path.name  # placeholders only, never an account id
        assert f"`{path.name}`" in readme, path.name  # each file says which role it belongs to
        for statement in _statements(path.name):
            assert not any("*" in action for action in _listed(statement.get("Action", []))), path.name
            resources = _listed(statement.get("Resource", []))
            for resource in resources:
                if resource == "*":
                    continue
                assert "<ACCOUNT_ID>" in resource or settings.BUCKET_PLACEHOLDER in resource, resource
                assert ":us-east-1:" in resource or resource.startswith("arn:aws:s3:::"), resource
            if "*" in resources:  # only where AWS takes no narrower resource, each explained
                assert len(resources) == 1 and f"`{statement['Sid']}`" in readme, statement["Sid"]
