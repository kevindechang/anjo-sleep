# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Glimmer's serving setup (Task 4): the vLLM command at each precision, the adapter check, and the
Modal app's wiring, read through a stand-in ``modal`` module so nothing is deployed or called."""
from __future__ import annotations

import importlib
import json
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.serving import config
from anjo.sleep.serving.config import ServeConfig, ServeConfigError

REVISION = "0123456789abcdef0123456789abcdef01234567"
ADAPTER = "p-1a2b3c4d-n3"


def _settings(**changes: Any) -> ServeConfig:
    base = {
        "precision": config.SERVING, "revision": REVISION, "tool_parser": None,
        "gpu_snapshot": False, "vllm_version": config.DEFAULT_VLLM_VERSION, "hf_secret": None,
    }
    return ServeConfig(**{**base, **changes})


def test_glimmer_serves_at_fp8_on_one_l40s_with_lora_on() -> None:
    assert config.SERVING is config.FP8_L40S
    command = config.vllm_command(_settings())

    assert command[:3] == ["vllm", "serve", f"/weights/{config.MODEL}/{REVISION}"]
    assert command[command.index("--quantization") + 1] == "fp8"
    assert command[command.index("--served-model-name") + 1] == config.MODEL
    assert "--enable-lora" in command
    assert "--language-model-only" in command  # no LoRA on the vision tower: its dummy-image profile run crashes
    sizes = {flag: command[command.index(flag) + 1] for flag in ("--max-lora-rank", "--max-loras", "--max-cpu-loras")}
    assert sizes == {"--max-lora-rank": "32", "--max-loras": "8", "--max-cpu-loras": "16"}
    assert command[command.index("--reasoning-parser") + 1] == "muse_glimmer"  # content is the to=user channel
    assert "--enable-auto-tool-choice" not in command  # no parser: turns run without tools
    assert not any("KEY" in part or "TOKEN" in part for part in command)
    env = config.vllm_env()
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"  # no nvcc in the image for FlashInfer's JIT
    assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "True"  # the client loads each adapter by name
    # vLLM's filesystem resolver never loaded an adapter (seen live): nothing may look as if it does.
    assert "VLLM_PLUGINS" not in env and "VLLM_LORA_RESOLVER_CACHE_DIR" not in env


def test_the_bf16_fallback_is_one_line_away_and_quantizes_nothing() -> None:
    command = config.vllm_command(_settings(precision=config.BF16_A100))

    assert "--quantization" not in command
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert config.BF16_A100.gpu == "A100-80GB"


def test_a_tool_call_parser_turns_tool_calls_on() -> None:
    command = config.vllm_command(_settings(tool_parser="muse_glimmer"))  # vLLM 0.28.0's, for ATEM calls

    assert command[-3:] == ["--enable-auto-tool-choice", "--tool-call-parser", "muse_glimmer"]


def test_every_call_is_rendered_at_one_reasoning_strength_low_unless_set() -> None:
    assert config.reasoning_strength({}) == "low" == config.DEFAULT_REASONING_STRENGTH
    assert config.reasoning_strength({config.REASONING_STRENGTH_ENV: " xhigh "}) == "xhigh"
    for wrong in ("max", "LOW", "low."):
        with pytest.raises(ServeConfigError, match=config.REASONING_STRENGTH_ENV):
            config.reasoning_strength({config.REASONING_STRENGTH_ENV: wrong})


def test_settings_come_from_the_environment_and_are_checked() -> None:
    read = ServeConfig.from_env({
        config.REVISION_ENV: REVISION, config.TOOL_PARSER_ENV: "llama3_json", config.SNAPSHOT_ENV: "1",
        config.HF_SECRET_ENV: "anjo-sleep-hf",
    })
    assert (read.revision, read.tool_parser, read.gpu_snapshot) == (REVISION, "llama3_json", True)
    assert read.hf_secret == "anjo-sleep-hf"
    assert (ServeConfig.from_env({}).revision, ServeConfig.from_env({}).hf_secret) == (None, None)
    with pytest.raises(ServeConfigError, match="secret"):
        ServeConfig.from_env({config.HF_SECRET_ENV: "HF_TOKEN=abc"})
    with pytest.raises(ServeConfigError, match="commit"):
        ServeConfig.from_env({config.REVISION_ENV: "main"})
    with pytest.raises(ServeConfigError, match="parser"):
        ServeConfig.from_env({config.TOOL_PARSER_ENV: "a b"})
    with pytest.raises(ServeConfigError, match=config.REVISION_ENV):
        config.vllm_command(_settings(revision=None))


def _adapter(tmp_path: Path, **meta: Any) -> Path:
    folder = tmp_path / ADAPTER
    folder.mkdir()
    settings = {"peft_type": "LORA", "base_model_name_or_path": config.MODEL, "r": 16, **meta}
    (folder / "adapter_config.json").write_text(json.dumps(settings), encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(b"\0" * 64)
    return folder


def test_an_adapter_is_checked_before_it_is_uploaded(tmp_path: Path) -> None:
    config.check_adapter(_adapter(tmp_path), ADAPTER)

    for name in ("adapter", "p-XYZ-n1", "p-1a2b3c4d-n"):
        with pytest.raises(ServeConfigError, match="named"):
            config.check_adapter_name(name)


@pytest.mark.parametrize(
    ("meta", "match"),
    [({"r": 128}, "rank"), ({"base_model_name_or_path": "other/model"}, "base model"),
     ({"peft_type": "IA3"}, "LoRA")],
)
def test_an_adapter_the_server_cant_load_is_refused(
    tmp_path: Path, meta: dict[str, Any], match: str,
) -> None:
    with pytest.raises(ServeConfigError, match=match):
        config.check_adapter(_adapter(tmp_path, **meta), ADAPTER)


def test_an_adapter_folder_holds_only_the_peft_files(tmp_path: Path) -> None:
    folder = _adapter(tmp_path)
    (folder / "notes.txt").write_text("private", encoding="utf-8")

    with pytest.raises(ServeConfigError, match="exactly"):
        config.check_adapter(folder, ADAPTER)


# ── The Modal app, through a stand-in modal module ────────────────────────────


class _Recorder:
    """Records each decorator's arguments and hands the decorated object back unchanged."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.uploads: list[tuple[str, str]] = []
        self.on_volume: set[str] = set()
        self.files: dict[str, bytes] = {}  # what the adapters volume's API reads back, by path
        self.reads: list[str] = []

    def decorator(self, name: str) -> Callable[..., Callable[[Any], Any]]:
        def make(*_args: Any, **kwargs: Any) -> Callable[[Any], Any]:
            def apply(target: Any) -> Any:
                self.calls.append((name, {**kwargs, "target": getattr(target, "__name__", "")}))
                return target

            return apply

        return make

    def by(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for call, kwargs in self.calls if call == name]


def _fake_modal(recorder: _Recorder) -> types.ModuleType:
    module = types.ModuleType("modal")

    class Image:
        @staticmethod
        def debian_slim(**_kwargs: Any) -> Image:
            return Image()

        def pip_install(self, *packages: str) -> Image:
            recorder.calls.append(("pip_install", {"packages": packages}))
            return self

        def env(self, values: dict[str, str]) -> Image:
            recorder.calls.append(("env", {"values": values}))
            return self

        def add_local_file(self, local: Path, remote: str) -> Image:
            recorder.calls.append(("add_local_file", {"local": str(local), "remote": remote}))
            return self

    class Batch:
        def __enter__(self) -> Batch:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def put_directory(self, local: str, remote: str) -> None:
            recorder.uploads.append((local, remote))
            recorder.on_volume.add(remote.strip("/"))

    class Volume:
        @staticmethod
        def from_name(name: str, **_kwargs: Any) -> Volume:
            return Volume()

        def listdir(self, path: str, *, recursive: bool = False) -> list[Any]:
            assert (path, recursive) == ("/", False)
            return [types.SimpleNamespace(path=name) for name in sorted(recorder.on_volume)]

        def batch_upload(self, force: bool = False) -> Batch:
            recorder.calls.append(("batch_upload", {"force": force}))
            return Batch()

        def read_file(self, path: str) -> Iterator[bytes]:
            recorder.reads.append(path)
            if path not in recorder.files:
                raise FileNotFoundError(path)
            data = recorder.files[path]
            yield from (data[:7], data[7:])  # in chunks, as the API streams it

    class App:
        def __init__(self, name: str, **_kwargs: Any) -> None:
            recorder.calls.append(("app", {"name": name}))
            self.function = recorder.decorator("function")
            self.cls = recorder.decorator("cls")
            self.local_entrypoint = recorder.decorator("local_entrypoint")

    module.Image = Image  # type: ignore[attr-defined]
    module.Volume = Volume  # type: ignore[attr-defined]
    module.App = App  # type: ignore[attr-defined]
    module.Secret = types.SimpleNamespace(from_name=lambda name: name)  # type: ignore[attr-defined]
    module.concurrent = recorder.decorator("concurrent")  # type: ignore[attr-defined]
    module.enter = recorder.decorator("enter")  # type: ignore[attr-defined]
    module.exit = recorder.decorator("exit")  # type: ignore[attr-defined]
    module.web_server = recorder.decorator("web_server")  # type: ignore[attr-defined]
    module.method = recorder.decorator("method")  # type: ignore[attr-defined]
    return module


@pytest.fixture
def modal_app(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[types.ModuleType, _Recorder]]:
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "modal", _fake_modal(recorder))
    monkeypatch.setenv(config.REVISION_ENV, REVISION)
    monkeypatch.delitem(sys.modules, "anjo.sleep.serving.modal_app", raising=False)
    module = importlib.import_module("anjo.sleep.serving.modal_app")
    yield module, recorder
    sys.modules.pop("anjo.sleep.serving.modal_app", None)


def test_the_modal_app_serves_on_the_configured_gpu_and_scales_to_zero(
    modal_app: tuple[types.ModuleType, _Recorder],
) -> None:
    _module, recorder = modal_app

    assert recorder.by("app") == [{"name": "anjo-sleep-serve"}]
    (serve,) = recorder.by("cls")
    assert (serve["gpu"], serve["scaledown_window"], serve["timeout"]) == ("L40S", 600, 3 * 3600)
    assert serve["max_containers"] == config.MAX_CONTAINERS == 1  # every arm shares one GPU: one wake, one container
    assert "enable_memory_snapshot" not in serve  # the GPU snapshot is behind its flag
    assert {config.WEIGHTS_DIR, config.ADAPTERS_DIR} == set(serve["volumes"])
    (download,) = recorder.by("function")
    assert download["timeout"] == config.DOWNLOAD_TIMEOUT_S and download["target"] == "download_weights"
    (web,) = recorder.by("web_server")
    assert (web["port"], web["startup_timeout"]) == (config.PORT, config.STARTUP_TIMEOUT_S)
    (env,) = recorder.by("env")
    assert env["values"][config.REVISION_ENV] == REVISION
    assert env["values"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert not any("KEY" in name or "TOKEN" in name for name in env["values"])
    (packages,) = recorder.by("pip_install")
    assert f"vllm=={config.DEFAULT_VLLM_VERSION}" in packages["packages"]
    copied = {call["remote"] for call in recorder.by("add_local_file")}
    assert copied == {
        "/root/anjo/sleep/__init__.py", "/root/anjo/sleep/serving/__init__.py",
        "/root/anjo/sleep/serving/config.py",
    }


def test_prompt_embeds_are_served_only_when_their_flag_is_set(
    modal_app: tuple[types.ModuleType, _Recorder],
) -> None:
    module, recorder = modal_app
    off = ServeConfig.from_env({config.REVISION_ENV: REVISION})
    assert off.prompt_embeds is False and module.SETTINGS.prompt_embeds is False
    assert config.vllm_command(off) == config.vllm_command(_settings())  # off: today's command, flag for flag
    assert "--enable-prompt-embeds" not in config.vllm_command(off)
    assert config.PROMPT_EMBEDS_ENV not in recorder.by("env")[0]["values"]
    on = ServeConfig.from_env({config.REVISION_ENV: REVISION, config.PROMPT_EMBEDS_ENV: "1"})
    assert config.vllm_command(on) == [*config.vllm_command(off), "--enable-prompt-embeds"]
    assert "--language-model-only" in config.vllm_command(on)  # still text only, LoRA still on
    assert module._settings_env(on)[config.PROMPT_EMBEDS_ENV] == "1"  # the container reads the same setting


def test_serve_never_reloads_its_adapters_volume(modal_app: tuple[types.ModuleType, _Recorder]) -> None:
    """On night 1c (2026-10-28) the server stopped answering before the night had sent any adapter
    load, and the stall was in the container's Modal process, where the adapters volume reload thread
    ran. ``Serve`` never reloads the volume: its mount is the volume as committed when the container
    started, and an adapter committed later reaches local disk through ``stage``."""
    module, recorder = modal_app
    assert [call["target"] for call in recorder.by("enter")] == ["start"]  # no refresher, nothing periodic
    assert ".reload(" not in Path(str(module.__file__)).read_text(encoding="utf-8")
    assert [call["target"] for call in recorder.by("method")] == ["stage"]


def _on_volume(recorder: _Recorder, folder: Path) -> None:
    for file in config.ADAPTER_FILES:
        recorder.files[f"/{folder.name}/{file}"] = (folder / file).read_bytes()


def test_a_new_adapter_is_staged_from_the_volume_onto_local_disk(
    modal_app: tuple[types.ModuleType, _Recorder], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``stage`` reads the adapter through the volume's API, never its mount, checks it as an upload
    is checked, and moves it into place whole, so vLLM loads it from local disk."""
    module, recorder = modal_app
    _on_volume(recorder, _adapter(tmp_path))
    monkeypatch.setattr(config, "STAGED_ADAPTERS_DIR", str(tmp_path / "staged"))

    assert module.Serve().stage(ADAPTER) is True
    staged = tmp_path / "staged" / ADAPTER
    assert sorted(path.name for path in staged.iterdir()) == sorted(config.ADAPTER_FILES)
    assert (staged / "adapter_model.safetensors").read_bytes() == recorder.files[f"/{ADAPTER}/adapter_model.safetensors"]
    read = len(recorder.reads)
    assert module.Serve().stage(ADAPTER) is True and len(recorder.reads) == read  # staged already: not read again

    assert module.Serve().stage("p-bbbbbbbbbbbb-n2") is False  # not on the volume
    assert sorted(path.name for path in (tmp_path / "staged").iterdir()) == [ADAPTER]  # nothing half-written left


def test_staging_refuses_a_bad_name_or_an_adapter_the_server_cant_load(
    modal_app: tuple[types.ModuleType, _Recorder], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, recorder = modal_app
    _on_volume(recorder, _adapter(tmp_path, r=128))
    monkeypatch.setattr(config, "STAGED_ADAPTERS_DIR", str(tmp_path / "staged"))

    with pytest.raises(ServeConfigError, match="named"):
        module.Serve().stage("../weights")
    with pytest.raises(ServeConfigError, match="rank"):
        module.Serve().stage(ADAPTER)
    assert not (tmp_path / "staged").exists() or list((tmp_path / "staged").iterdir()) == []
    assert recorder.reads == [f"/{ADAPTER}/{file}" for file in config.ADAPTER_FILES]  # the bad name read nothing


def test_the_gpu_snapshot_is_tried_only_when_its_flag_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "modal", _fake_modal(recorder))
    monkeypatch.setenv(config.REVISION_ENV, REVISION)
    monkeypatch.setenv(config.SNAPSHOT_ENV, "1")
    monkeypatch.delitem(sys.modules, "anjo.sleep.serving.modal_app", raising=False)
    try:
        importlib.import_module("anjo.sleep.serving.modal_app")
    finally:
        sys.modules.pop("anjo.sleep.serving.modal_app", None)

    (serve,) = recorder.by("cls")
    assert serve["enable_memory_snapshot"] is True
    assert serve["experimental_options"] == {"enable_gpu_snapshot": True}
    assert recorder.by("enter")[0]["snap"] is True


def test_an_adapter_is_put_on_the_volume_under_its_name(
    modal_app: tuple[types.ModuleType, _Recorder], tmp_path: Path,
) -> None:
    module, recorder = modal_app
    folder = _adapter(tmp_path)

    module.put_adapter(folder, ADAPTER)
    assert recorder.uploads == [(str(folder), f"/{ADAPTER}")]

    assert recorder.by("batch_upload") == [{"force": False}]  # Modal itself refuses to overwrite too

    (folder / "extra.bin").write_bytes(b"\0")
    with pytest.raises(ServeConfigError):
        module.put_adapter(folder, "p-1a2b3c4d-n4")
    assert len(recorder.uploads) == 1  # a refused folder never reaches the volume


def test_a_pinned_nights_adapter_can_never_change(
    modal_app: tuple[types.ModuleType, _Recorder], tmp_path: Path,
) -> None:
    module, recorder = modal_app
    folder = _adapter(tmp_path)
    module.put_adapter(folder, ADAPTER)

    with pytest.raises(ServeConfigError, match="already on the volume"):
        module.put_adapter(folder, ADAPTER)
    assert len(recorder.uploads) == 1


def _import_app(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder) -> types.ModuleType:
    monkeypatch.setitem(sys.modules, "modal", _fake_modal(recorder))
    monkeypatch.delitem(sys.modules, "anjo.sleep.serving.modal_app", raising=False)
    try:
        return importlib.import_module("anjo.sleep.serving.modal_app")
    finally:
        sys.modules.pop("anjo.sleep.serving.modal_app", None)


def test_the_app_refuses_to_load_without_a_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checked at import, so ``modal deploy`` fails before anything is built, not at the first wake."""
    monkeypatch.delenv(config.REVISION_ENV, raising=False)
    recorder = _Recorder()

    with pytest.raises(ServeConfigError, match=config.REVISION_ENV):
        _import_app(monkeypatch, recorder)
    assert recorder.by("cls") == [] and recorder.by("function") == []


@pytest.mark.parametrize("hf_secret", [None, "anjo-sleep-hf"])
def test_each_function_gets_only_its_own_secret(
    monkeypatch: pytest.MonkeyPatch, hf_secret: str | None,
) -> None:
    """Serving holds only VLLM_API_KEY's secret. Downloading needs none (the repo isn't gated), or
    only the Hugging Face one the deploy names."""
    monkeypatch.setenv(config.REVISION_ENV, REVISION)
    if hf_secret is None:
        monkeypatch.delenv(config.HF_SECRET_ENV, raising=False)
    else:
        monkeypatch.setenv(config.HF_SECRET_ENV, hf_secret)
    recorder = _Recorder()
    _import_app(monkeypatch, recorder)

    (serve,) = recorder.by("cls")
    (download,) = recorder.by("function")
    assert serve["secrets"] == [config.SECRET_NAME]
    assert download["secrets"] == ([] if hf_secret is None else [hf_secret])
