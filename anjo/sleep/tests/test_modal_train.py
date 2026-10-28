# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The Modal training app (Task 5): its image installs the training script's own pins, it holds no
key, the base loads from the weights volume at the pinned revision only, and the adapter lands on the
adapters volume under its name only after the upload's checks and with its rendering. ``modal`` is a
fake and the training script a stand-in: no container, no GPU, no network."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import TRAINING_TIMEOUT_S
from anjo.sleep.serving import config, framing
from anjo.sleep.train import colab_train, modal_job
from anjo.sleep.train.lora import adapter_name
from anjo.sleep.train.runners.colab import ColabRunner

PIN = "c" * 40
NAME = adapter_name("p1", 1, 0)
RENDERING = framing.rendering("low")
APP_MODULE = "anjo.sleep.train.modal_app"


class _Chain:
    """Stands in for modal's image builder: records every call and returns itself."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> _Chain:
            self.calls.append((name, args, kwargs))
            return self

        return call


class _Volume:
    def __init__(self, name: str, read_only: bool = False, **kwargs: Any) -> None:
        self.name, self.ro, self.kwargs = name, read_only, kwargs

    def read_only(self) -> _Volume:
        return _Volume(self.name, True, **self.kwargs)


def _load_app(monkeypatch: pytest.MonkeyPatch, revision: str | None = PIN) -> tuple[Any, _Chain, dict[str, Any]]:
    """Import the app with a fake ``modal``; returns the module, the image's calls and the function's options."""
    image, declared = _Chain(), {}

    def function(**options: Any) -> Any:
        declared.update(options)
        return lambda fn: fn

    fake = types.SimpleNamespace(
        Image=types.SimpleNamespace(debian_slim=lambda **kw: image),
        App=lambda *a, **kw: types.SimpleNamespace(function=function),
        Volume=types.SimpleNamespace(from_name=lambda name, **kw: _Volume(name, **kw)),
        Secret=types.SimpleNamespace(from_name=lambda name: pytest.fail(f"the training app names a secret: {name}")),
    )
    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name, *a: fake if name == "modal" else real(name, *a))
    monkeypatch.delitem(sys.modules, APP_MODULE, raising=False)
    if revision is None:
        monkeypatch.delenv(config.REVISION_ENV, raising=False)
    else:
        monkeypatch.setenv(config.REVISION_ENV, revision)
    return importlib.import_module(APP_MODULE), image, declared


def test_modal_train_image_pins_match_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    _, image, _ = _load_app(monkeypatch)
    installs = [args for name, args, _ in image.calls if name in ("pip_install", "uv_pip_install")]
    assert installs == [(colab_train.TORCH, *colab_train.TRAIN_PACKAGES)]  # the one list, nothing else
    assert (ColabRunner.__init__.__kwdefaults__ or {})["packages"] is colab_train.TRAIN_PACKAGES  # Colab installs the same
    files = {args[1]: Path(args[0]) for name, args, _ in image.calls if name == "add_local_file"}
    assert files[modal_job.REMOTE_SCRIPT] == Path(colab_train.__file__)  # the script the Colab VM runs
    assert modal_job.REMOTE_SCRIPT == "/root/anjo/sleep/train/colab_train.py"


def test_the_app_trains_on_one_a100_with_weights_read_only_and_adapters_read_write(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, declared = _load_app(monkeypatch)
    assert declared["gpu"] == modal_job.GPU == config.BF16_A100.gpu == "A100-80GB"
    assert declared["timeout"] == modal_job.FUNCTION_TIMEOUT_S == TRAINING_TIMEOUT_S  # §12.1's training cap
    assert 0 < declared["startup_timeout"] == modal_job.STARTUP_TIMEOUT_S
    volumes = declared["volumes"]
    assert set(volumes) == {config.WEIGHTS_DIR, config.ADAPTERS_DIR}
    assert volumes[config.WEIGHTS_DIR].name == config.WEIGHTS_VOLUME and volumes[config.WEIGHTS_DIR].ro
    assert volumes[config.ADAPTERS_DIR].name == config.ADAPTERS_VOLUME and not volumes[config.ADAPTERS_DIR].ro
    assert modal_job.APP_NAME == "anjo-sleep-train" and modal_job.FUNCTION == "train"


def test_no_deploy_without_a_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(config.ServeConfigError, match=config.REVISION_ENV):
        _load_app(monkeypatch, revision=None)


class Script:
    """Stands in for ``colab_train.py`` in the container: records its command line and environment,
    and the warm start's files it was handed, then writes what the script writes (``adapter`` overrides
    the adapter's config)."""

    def __init__(self, adapter: Mapping[str, Any] | None = None, extra: str | None = None, code: int = 0) -> None:
        self.adapter = adapter
        self.extra = extra
        self.code = code
        self.seen: list[tuple[list[str], dict[str, str], float]] = []
        self.prev: dict[str, bytes] = {}

    def __call__(self, args: Sequence[str], *, env: Mapping[str, str], timeout: float, **kw: Any) -> Any:
        self.seen.append((list(args), dict(env), timeout))
        if "ANJO_TRAIN_PREV" in env:
            self.prev = {p.name: p.read_bytes() for p in Path(env["ANJO_TRAIN_PREV"]).iterdir()}
        out = Path(env["ANJO_TRAIN_OUT"])
        out.mkdir(parents=True, exist_ok=True)
        meta = {"peft_type": "LORA", "base_model_name_or_path": config.MODEL, "r": 16,
                framing.RENDERING_KEY: RENDERING, **(self.adapter or {})}
        (out / "adapter_config.json").write_text(json.dumps(meta))
        (out / "adapter_model.safetensors").write_bytes(b"lora weights")
        (out / colab_train.RECEIPT).write_text(json.dumps({"steps": 9, "examples": 3, "rendering": RENDERING}))
        (out / "README.md").write_text("PEFT's model card")  # save_pretrained writes one; it never lands
        if self.extra:
            (out / self.extra).write_text("x")
        return subprocess.CompletedProcess(list(args), self.code, None, None)


def _volumes(tmp_path: Path, revision: str = PIN) -> tuple[Path, Path]:
    weights, adapters = tmp_path / "weights", tmp_path / "adapters"
    (weights / config.MODEL / revision).mkdir(parents=True)
    (weights / config.MODEL / revision / "config.json").write_text("{}")
    adapters.mkdir()
    return weights, adapters


def _job(tmp_path: Path, script: Script, *, revision: object = PIN, name: str = NAME,
         prev: Mapping[str, bytes] | None = None, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    weights, adapters = tmp_path / "weights", tmp_path / "adapters"
    recipe = {"r": 16, "revision": revision, "rendering": RENDERING}
    return modal_job.train_on_volume(
        dataset='{"id": "x"}\n', config_json=json.dumps(recipe), prev=prev or {}, name=name, exec_s=3600.0,
        pinned=PIN, weights_root=weights, adapters_root=adapters, run=script, environ=environ or {"PATH": "/usr/bin"},
    )


def test_modal_adapter_lands_checked_with_its_rendering(tmp_path: Path) -> None:
    weights, adapters = _volumes(tmp_path)
    script = Script()
    prev = {file: f"prev {file}".encode() for file in config.ADAPTER_FILES}
    done = _job(tmp_path, script, prev=prev)
    landed = adapters / NAME
    assert sorted(p.name for p in landed.iterdir()) == sorted(config.ADAPTER_FILES)  # nothing else lands
    config.check_adapter(landed, NAME)  # the checks an upload passes
    assert json.loads((landed / "adapter_config.json").read_text())[framing.RENDERING_KEY] == RENDERING
    assert done["adapter"] == NAME and done["receipt"]["steps"] == 9
    assert done["files"] == {file: (landed / file).read_bytes() for file in config.ADAPTER_FILES}
    [(args, env, timeout)] = script.seen
    assert args[-1] == modal_job.REMOTE_SCRIPT and timeout == 3600.0
    assert env[modal_job.BASE_ENV] == str(weights / config.MODEL / PIN)  # the base comes off the volume
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"  # nothing downloads
    assert script.prev == prev  # the warm start's two files, as sent

    with pytest.raises(modal_job.ModalJobError, match="already on the volume"):  # a name never changes
        _job(tmp_path, Script())


@pytest.mark.parametrize(("adapter", "extra", "match"), [
    ({framing.RENDERING_KEY: framing.rendering("high")}, None, "rendering"),
    ({framing.RENDERING_KEY: None}, None, "rendering"),
    ({"base_model_name_or_path": "/weights/meta-models/Muse-Glimmer-30B/" + PIN}, None, "base model"),
    ({"r": 64}, None, "rank"),
])
def test_an_adapter_that_fails_the_uploads_checks_never_lands(
    tmp_path: Path, adapter: Mapping[str, Any], extra: str | None, match: str,
) -> None:
    _, adapters = _volumes(tmp_path)
    with pytest.raises((modal_job.ModalJobError, config.ServeConfigError), match=match):
        _job(tmp_path, Script(adapter, extra))
    assert list(adapters.iterdir()) == []


@pytest.mark.parametrize(("revision", "match"), [
    (None, "pinned revision"), ("", "pinned revision"), ("main", "pinned revision"),
    ("d" * 40, "isn't the one"),
])
def test_an_unpinned_or_mismatched_revision_is_refused_before_training(tmp_path: Path, revision: object, match: str) -> None:
    _, adapters = _volumes(tmp_path)
    _volumes(tmp_path / "other", "d" * 40)
    script = Script()
    with pytest.raises(modal_job.ModalJobError, match=match):
        _job(tmp_path, script, revision=revision)
    assert script.seen == [] and list(adapters.iterdir()) == []


def test_a_revision_missing_from_the_volume_and_a_failed_script_are_refused(tmp_path: Path) -> None:
    (tmp_path / "weights").mkdir()
    (tmp_path / "adapters").mkdir()
    with pytest.raises(modal_job.ModalJobError, match="download_weights"):
        _job(tmp_path, Script())
    _volumes(tmp_path / "ok")
    with pytest.raises(modal_job.ModalJobError, match="exited"):
        _job(tmp_path / "ok", Script(code=1))
    assert list((tmp_path / "ok" / "adapters").iterdir()) == []


def test_modal_train_sends_no_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. The app names no secret, and its image's environment holds the revision pin and offline flags only.
    _, image, declared = _load_app(monkeypatch)
    assert not declared.get("secrets")
    [image_env] = [args[0] for name, args, _ in image.calls if name == "env"]
    assert image_env == modal_job.image_env(PIN) == {config.REVISION_ENV: PIN, **modal_job.OFFLINE}
    # 2. The container passes the training script nothing that looks like a key, nor Anjo's or Modal's settings.
    _volumes(tmp_path)
    script = Script()
    _job(tmp_path, script, environ={"PATH": "/usr/bin", "HF_TOKEN": "hf_x", "ANJO_SECRET": "s", "OPENAI_API_KEY": "k",
                                    "MODAL_IDENTITY_TOKEN": "m", "CLOUD_SECRET_ACCESS_KEY": "a", "PASSWORD": "p",
                                    "CUDA_VISIBLE_DEVICES": "0"})
    [(_, env, _)] = script.seen
    assert env["PATH"] == "/usr/bin" and env["CUDA_VISIBLE_DEVICES"] == "0"
    for name in ("HF_TOKEN", "ANJO_SECRET", "OPENAI_API_KEY", "MODAL_IDENTITY_TOKEN", "CLOUD_SECRET_ACCESS_KEY", "PASSWORD"):
        assert name not in env
    assert {name for name in env if name.startswith("ANJO_")} == {
        "ANJO_TRAIN_CONFIG", "ANJO_TRAIN_DATASET", "ANJO_TRAIN_OUT", modal_job.BASE_ENV}
    # 3. The runner's call carries the job's inputs and nothing read from the environment (test_modal_runner).
    assert [field for field in modal_job.JOB_FIELDS] == ["dataset", "config_json", "prev", "name", "exec_s"]


def test_the_script_loads_a_local_base_and_records_glimmer_as_the_adapters_base(tmp_path: Path) -> None:
    assert colab_train.BASE_ENV == modal_job.BASE_ENV
    base = tmp_path / "glimmer"
    assert colab_train.base_model({}) is None  # Colab: Glimmer by name, at the recipe's revision
    with pytest.raises(ValueError, match="folder"):
        colab_train.base_model({colab_train.BASE_ENV: str(base)})
    base.mkdir()
    assert colab_train.base_model({colab_train.BASE_ENV: str(base)}) == str(base)
    with pytest.raises(ValueError, match="smoke"):
        colab_train.base_model({colab_train.BASE_ENV: str(base), colab_train.SMOKE_ENV: str(base)})
    out = tmp_path / "out"
    out.mkdir()
    (out / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": str(base)}))
    colab_train.record_base_model(out)
    assert json.loads((out / "adapter_config.json").read_text())["base_model_name_or_path"] == config.MODEL
    assert colab_train.MODEL == config.MODEL


def test_the_jobs_setup_fits_inside_the_hard_timeout() -> None:
    exec_s = modal_job.exec_seconds(TRAINING_TIMEOUT_S)
    assert modal_job.MIN_EXEC_S <= exec_s
    assert exec_s + modal_job.STARTUP_TIMEOUT_S + modal_job.RETURN_S + modal_job.CANCEL_S == TRAINING_TIMEOUT_S
    for short in (0.0, 600.0, float("nan"), float("inf")):
        with pytest.raises(modal_job.ModalJobError):
            modal_job.exec_seconds(short)
