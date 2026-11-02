# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The judges' warm trainer (2026-11-02): its own Modal app, loaded once per container, plain SFT only,
landing adapters exactly as the cold job does; on the host, the warm runner first and the cold one when
it can't, and the keep-warm calls. ``modal`` is a fake and the training a stand-in: no container, no
GPU, no network."""
from __future__ import annotations

import importlib
import json
import sys
import threading
import types
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import Ledger
from anjo.sleep.serving import config, framing
from anjo.sleep.tests.test_modal_runner import FakeCalls, _inputs
from anjo.sleep.train import colab_train, modal_job, warm_job
from anjo.sleep.train.lora import LoraCfg, StepPlan, adapter_name, training_config
from anjo.sleep.train.runners import warm
from anjo.sleep.train.runners.modal import CallRecords, ModalRunner, RemoteJob, SpawnRefused

PIN = "c" * 40
NAME = adapter_name("p1", 1, 0)
APP_MODULE = "anjo.sleep.train.modal_warm_app"


def _recipe(**extra: Any) -> dict[str, Any]:
    recipe = training_config(LoraCfg(), 3, warm=False, revision=PIN, night=date(2026, 11, 3),
                             steps=StepPlan(accumulation=1, epochs=1, steps_per_epoch=8))
    return {**recipe, **extra}


# ── the container's side ─────────────────────────────────────────────────────


def test_the_scale_down_window_is_90_s_unless_the_deploy_says_otherwise() -> None:
    assert warm_job.scaledown_seconds({}) == 90
    assert warm_job.scaledown_seconds({warm_job.SCALEDOWN_ENV: "120"}) == 120
    for bad in ("5", "abc", "601"):
        with pytest.raises(warm_job.WarmJobError):
            warm_job.scaledown_seconds({warm_job.SCALEDOWN_ENV: bad})
    assert warm_job.PING_S < warm_job.DEFAULT_SCALEDOWN_S  # a ping lands before the container scales down


def test_it_runs_plain_sft_only() -> None:
    assert warm_job.refusal(_recipe()) is None
    assert warm_job.refusal(_recipe(objective="sft_dpo")) is not None
    assert warm_job.refusal(_recipe(channel={"k": 1})) is not None
    assert warm_job.refusal(_recipe(terms={"fold_rank": 1})) is not None
    assert warm_job.refusal(_recipe(match_minutes=4.0)) is not None


def test_it_loads_glimmer_as_the_training_script_does() -> None:
    assert dict(warm_job.QUANTIZATION) == colab_train.build_settings(_recipe())["quantization"]


def test_its_own_app_carries_the_cold_files_and_its_own() -> None:
    remote = [path for _, path in warm_job.WARM_FILES]
    assert remote[: len(modal_job.TRAIN_FILES)] == [path for _, path in modal_job.TRAIN_FILES]
    assert any(path.endswith("warm_job.py") for path in remote) and any(path.endswith("modal_warm_app.py") for path in remote)
    assert warm_job.files_digest() != modal_job.files_digest()  # the cold app's digest is untouched
    assert warm_job.APP_NAME not in (modal_job.APP_NAME, "anjo-sleep-serve", "anjo-sleep-serve-det")


class FakeTrain:
    """Stands in for the loaded model's training: writes what save_pretrained and the receipt would."""

    def __init__(self, rendering: str) -> None:
        self.rendering = rendering
        self.seen: list[tuple[str, Path | None]] = []

    def __call__(self, cfg: dict[str, Any], data: Path, prev: Path | None, out: Path) -> dict[str, Any]:
        self.seen.append((data.read_text(), prev))
        meta = {"peft_type": "LORA", "base_model_name_or_path": config.MODEL, "r": 16,
                framing.RENDERING_KEY: self.rendering}
        (out / "adapter_config.json").write_text(json.dumps(meta))
        (out / "adapter_model.safetensors").write_bytes(b"lora weights")
        (out / "README.md").write_text("model card")  # never lands
        return {"steps": 8, "examples": 3, "rendering": self.rendering}


def _volumes(tmp_path: Path) -> tuple[Path, Path]:
    weights, adapters = tmp_path / "weights", tmp_path / "adapters"
    (weights / config.MODEL / PIN).mkdir(parents=True)
    (weights / config.MODEL / PIN / "config.json").write_text("{}")
    adapters.mkdir()
    return weights, adapters


def _call(tmp_path: Path, recipe: Mapping[str, Any], train: FakeTrain, prev: Mapping[str, bytes] | None = None) -> dict[str, Any]:
    weights, adapters = tmp_path / "weights", tmp_path / "adapters"
    return warm_job.train_warm_on_volume(
        dataset='{"id": "x"}\n', config_json=json.dumps(recipe), prev=prev or {}, name=NAME, pinned=PIN,
        weights_root=weights, adapters_root=adapters, train=train, train_files=warm_job.WARM_FILES)


def test_a_warm_call_lands_the_adapter_checked_with_its_rendering(tmp_path: Path) -> None:
    _, adapters = _volumes(tmp_path)
    recipe = _recipe()
    train = FakeTrain(recipe["rendering"])
    prev = {file: f"prev {file}".encode() for file in config.ADAPTER_FILES}
    done = _call(tmp_path, recipe, train, prev)
    landed = adapters / NAME
    assert sorted(p.name for p in landed.iterdir()) == sorted(config.ADAPTER_FILES)
    config.check_adapter(landed, NAME)
    assert done["adapter"] == NAME and done["receipt"]["steps"] == 8
    assert done[modal_job.DIGEST_KEY] == warm_job.files_digest()
    [(data, warm_dir)] = train.seen
    assert data == '{"id": "x"}\n' and warm_dir is not None
    with pytest.raises(warm_job.WarmJobError, match="already on the volume"):
        _call(tmp_path, recipe, train)


def test_a_warm_call_refuses_before_training(tmp_path: Path) -> None:
    _volumes(tmp_path)
    train = FakeTrain(_recipe()["rendering"])
    with pytest.raises(warm_job.WarmJobError, match="plain SFT"):
        _call(tmp_path, _recipe(objective="sft_dpo", dpo_beta=0.1, dpo_learning_rate=1e-6), train)
    with pytest.raises(modal_job.ModalJobError, match="pinned to"):
        _call(tmp_path, _recipe(revision="d" * 40), train)
    assert train.seen == []


def test_an_adapter_without_its_rendering_never_lands(tmp_path: Path) -> None:
    _, adapters = _volumes(tmp_path)
    with pytest.raises(warm_job.WarmJobError, match="rendering"):
        _call(tmp_path, _recipe(), FakeTrain("other/low"))
    assert not (adapters / NAME).exists()


class _Volume:
    def __init__(self, name: str, read_only: bool = False, **kwargs: Any) -> None:
        self.name, self.ro = name, read_only
        self.reloads = self.commits = 0

    def read_only(self) -> _Volume:
        return _Volume(self.name, True)

    def reload(self) -> None:
        self.reloads += 1

    def commit(self) -> None:
        self.commits += 1


class _Chain:
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: self


def _load_app(monkeypatch: pytest.MonkeyPatch, env: Mapping[str, str] | None = None) -> tuple[Any, dict[str, Any], Any]:
    declared: dict[str, Any] = {}
    stopped: list[bool] = []

    def cls(**options: Any) -> Any:
        declared.update(options)
        return lambda klass: klass

    fake = types.SimpleNamespace(
        Image=types.SimpleNamespace(debian_slim=lambda **kw: _Chain()),
        App=lambda *a, **kw: types.SimpleNamespace(cls=cls, name=a[0] if a else kw.get("name")),
        Volume=types.SimpleNamespace(from_name=lambda name, **kw: _Volume(name, **kw)),
        Secret=types.SimpleNamespace(from_name=lambda name: pytest.fail(f"the warm app names a secret: {name}")),
        enter=lambda: (lambda fn: fn), method=lambda: (lambda fn: fn),
        experimental=types.SimpleNamespace(stop_fetching_inputs=lambda: stopped.append(True)),
    )
    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name, *a: fake if name == "modal" else real(name, *a))
    monkeypatch.delitem(sys.modules, APP_MODULE, raising=False)
    monkeypatch.setenv(config.REVISION_ENV, PIN)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    return importlib.import_module(APP_MODULE), declared, stopped


def test_the_warm_app_is_one_gpu_container_with_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    app, declared, _ = _load_app(monkeypatch, {warm_job.SCALEDOWN_ENV: "120"})
    assert app.app.name == warm_job.APP_NAME
    assert declared["gpu"] == modal_job.GPU and declared["secrets"] == []
    assert declared["max_containers"] == 1 and declared["scaledown_window"] == 120
    assert declared["timeout"] == modal_job.FUNCTION_TIMEOUT_S
    volumes = declared["volumes"]
    assert volumes[config.WEIGHTS_DIR].ro and not volumes[config.ADAPTERS_DIR].ro


def test_a_call_that_leaves_the_base_dirty_retires_the_container(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app, _, stopped = _load_app(monkeypatch)
    trainer = app.WarmTrainer()
    trainer.loaded = warm_job.Loaded(model=None, tokenizer=None, seconds=80.0)
    calls: list[str] = []

    def dirty(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["name"])
        trainer.loaded.clean = False
        raise warm_job.WarmJobError("the adapter didn't come out")

    monkeypatch.setattr(warm_job, "train_warm_on_volume", dirty)
    with pytest.raises(warm_job.WarmJobError):
        trainer.train("{}", "{}", {}, NAME)
    assert calls == [NAME] and stopped == [True]
    assert trainer.warm() == {"loaded_s": 80.0, "clean": False}


# ── the host's side ──────────────────────────────────────────────────────────


def test_the_trainer_and_ping_settings() -> None:
    assert warm.trainer_mode({}) == "warm" and warm.trainer_mode({warm.TRAINER_ENV: "cold"}) == "cold"
    assert warm.ping_seconds({}) == warm_job.PING_S and warm.ping_seconds({warm.PING_ENV: "30"}) == 30.0
    with pytest.raises(warm.WarmChoiceError):
        warm.trainer_mode({warm.TRAINER_ENV: "hot"})
    with pytest.raises(warm.WarmChoiceError):
        warm.ping_seconds({warm.PING_ENV: "1"})


class Cold:
    def __init__(self) -> None:
        self.ran: list[str] = []

    def run(self, job: str, inputs: Mapping[str, object], *, timeout_s: float) -> Mapping[str, object]:
        self.ran.append(str(inputs["adapter_name"]))
        return {"steps": 8, "adapter_dir": "out"}


def _warm_first(tmp_path: Path, calls: FakeCalls, *, mode: str = "warm") -> tuple[warm.WarmFirstRunner, Cold, Ledger]:
    ledger, cold = Ledger(tmp_path / "budget.jsonl"), Cold()
    runner = ModalRunner(calls=calls, ledger=ledger, records=CallRecords(tmp_path / "calls"),
                         is_synthetic=lambda uid: False, allow_real={"judge-1"}, clock=calls.clock,
                         digest=warm_job.files_digest, app_name=warm_job.APP_NAME)
    return warm.WarmFirstRunner(warm=runner, cold=cold, mode=mode), cold, ledger


def _judge_inputs(tmp_path: Path, **recipe: Any) -> dict[str, Any]:
    inputs = _inputs(tmp_path, person="judge-1")
    return {**inputs, "config": _recipe(**recipe)}


def test_a_plain_sft_night_trains_warm_and_says_so(tmp_path: Path) -> None:
    calls = FakeCalls()
    runner, cold, _ = _warm_first(tmp_path, calls)
    done = runner.run("train_lora", _judge_inputs(tmp_path), timeout_s=7200.0)
    assert done["runner"] == warm.WARM_RUNNER and "fallback_reason" not in done
    assert len(calls.spawned) == 1 and cold.ran == []


@pytest.mark.parametrize(("why", "mode", "fail", "recipe"), [
    ("refused recipe", "warm", None, {"objective": "sft_dpo"}),
    ("cold mode", "cold", None, {}),
    ("not deployed", "warm", "refused", {}),
])
def test_anything_the_warm_app_cant_take_trains_cold(
    tmp_path: Path, why: str, mode: str, fail: str | None, recipe: dict[str, Any],
) -> None:
    runner, cold, _ = _warm_first(tmp_path, FakeCalls(fail=fail), mode=mode)
    done = runner.run("train_lora", _judge_inputs(tmp_path, **recipe), timeout_s=7200.0)
    assert done["runner"] == warm.COLD_RUNNER and done["fallback_reason"], why
    assert cold.ran == [NAME]


def test_a_warm_call_that_started_and_failed_fails_the_training(tmp_path: Path) -> None:
    runner, cold, _ = _warm_first(tmp_path, FakeCalls(fail="remote"))
    with pytest.raises(RuntimeError):
        runner.run("train_lora", _judge_inputs(tmp_path), timeout_s=7200.0)
    assert cold.ran == []  # it may have spent: no second call


def test_a_warm_app_deployed_with_other_files_fails_the_training(tmp_path: Path) -> None:
    calls = FakeCalls()
    runner, _, _ = _warm_first(tmp_path, calls)
    original = calls.wait
    calls.wait = lambda call_id, timeout_s: {**original(call_id, timeout_s), modal_job.DIGEST_KEY: "0" * 64}  # type: ignore[method-assign]
    with pytest.raises(Exception, match=warm_job.APP_NAME):
        runner.run("train_lora", _judge_inputs(tmp_path), timeout_s=7200.0)


def test_the_warmer_pings_until_stopped() -> None:
    pinged = threading.Event()
    count = {"n": 0}

    def ping() -> bool:
        count["n"] += 1
        pinged.set()
        return True

    warmer = warm.Warmer(ping, ping_s=10.0)
    warmer.start()
    warmer.start()  # once
    assert pinged.wait(5.0)
    warmer.stop()
    assert not warmer.running and count["n"] == 1


class _Spawned:
    def __init__(self, log: list[Any], name: str, fail: type[BaseException] | None = None) -> None:
        self.log, self.name, self.fail = log, name, fail

    def spawn(self, *args: Any) -> Any:
        if self.fail is not None:
            raise self.fail("no")
        self.log.append((self.name, args))
        return types.SimpleNamespace(object_id="fc-00000001")


def _fake_modal(monkeypatch: pytest.MonkeyPatch, log: list[Any], fail: type[BaseException] | None = None) -> Any:
    class Error(Exception):
        pass

    class NotFoundError(Error):
        pass

    trainer = types.SimpleNamespace(train=_Spawned(log, "train", fail), warm=_Spawned(log, "warm", fail))

    def from_name(app: str, name: str) -> Any:
        log.append(("cls", app, name))
        return lambda: trainer

    fake = types.SimpleNamespace(
        exception=types.SimpleNamespace(Error=Error, NotFoundError=NotFoundError, InvalidError=NotFoundError,
                                        AuthError=NotFoundError),
        Cls=types.SimpleNamespace(from_name=from_name),
    )
    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name, *a: fake if name == "modal" else real(name, *a))
    return fake


def test_warm_calls_spawn_the_class_method_with_the_jobs_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[Any] = []
    _fake_modal(monkeypatch, log)
    job = RemoteJob(dataset="d", config_json="{}", prev={"a": b"1"}, name=NAME, exec_s=1800.0)
    assert warm.WarmCalls().spawn(job) == "fc-00000001"
    assert ("cls", warm_job.APP_NAME, warm_job.CLASS) in log
    assert ("train", ("d", "{}", {"a": b"1"}, NAME)) in log
    assert warm.WarmCalls().warm() is True and log[-1] == ("warm", ())


def test_a_warm_app_modal_refuses_is_spawn_refused_and_a_failed_ping_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[Any] = []
    fake = _fake_modal(monkeypatch, log, fail=None)
    fake.Cls = types.SimpleNamespace(from_name=lambda app, name: (_ for _ in ()).throw(fake.exception.NotFoundError("x")))
    job = RemoteJob(dataset="d", config_json="{}", prev={}, name=NAME, exec_s=1800.0)
    with pytest.raises(SpawnRefused):
        warm.WarmCalls().spawn(job)
    assert warm.WarmCalls().warm() is False

