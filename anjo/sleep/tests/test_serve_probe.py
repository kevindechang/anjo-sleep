# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The live serve probe (``tools.serve_probe``): without ``--go`` it prints its plan and sends nothing;
with it, it times each step against the server and prints only step names, outcomes and milliseconds,
never the URL, the token or a reply. Fakes only: no network."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from anjo.sleep.serving import config
from anjo.sleep.serving.backend import MODAL_TOKEN_ENV, MODAL_URL_ENV
from anjo.sleep.tests.test_adapter_load import ADAPTER, LiveVllm, Stage, Waits
from anjo.sleep.tools import serve_probe

URL = "https://serve.example"
TOKEN = "secret-token-1234"
ENV = {MODAL_URL_ENV: URL, MODAL_TOKEN_ENV: TOKEN}


def _run(argv: list[str], fake: Any, capsys: pytest.CaptureFixture[str], *, stage: Any = None,
         environ: dict[str, str] | None = None) -> tuple[int, list[dict[str, Any]], str]:
    waits = Waits()
    code = serve_probe.main(argv, environ=ENV if environ is None else environ, transport=httpx.MockTransport(fake),
                            stage=stage, clock=waits.clock, sleep=waits.sleep)
    out = capsys.readouterr().out
    return code, [json.loads(line) for line in out.splitlines() if line.startswith("{")], out


def test_without_go_it_prints_the_plan_and_sends_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    fake = LiveVllm()
    code, lines, out = _run(["--adapter", ADAPTER.id], fake, capsys, environ={})  # no env needed for the plan
    assert code == 0 and lines == [] and fake.paths == [] and fake.probes == []
    assert "--go" in out and "/v1/load_lora_adapter" in out and config.STAGED_ADAPTERS_DIR in out


def test_go_times_each_step_of_the_base_model(capsys: pytest.CaptureFixture[str]) -> None:
    fake = LiveVllm()
    code, lines, out = _run(["--go"], fake, capsys)
    assert code == 0
    assert [line["step"] for line in lines] == ["wake", "models", "tokenize", "complete", "done"]
    assert all(line["ok"] for line in lines) and all(isinstance(line["ms"], int) for line in lines[:-1])
    assert fake.paths == ["/tokenize", "/v1/completions"] and len(fake.probes) == 2
    assert URL not in out and TOKEN not in out and "hello" not in out  # never the endpoint, the token or a reply


def test_an_adapter_the_mount_cant_show_is_staged_then_probed(capsys: pytest.CaptureFixture[str]) -> None:
    fake = LiveVllm(on_disk=set())
    stage = Stage(fake, committed={ADAPTER.id})
    code, lines, _ = _run(["--adapter", ADAPTER.id, "--go"], fake, capsys, stage=stage)
    assert code == 0 and stage.asked == [ADAPTER.id]
    assert [line["step"] for line in lines] == [
        "wake", "models", "tokenize", "complete", "load", "stage", "load_staged",
        "models", "tokenize", "complete", "done"]
    assert [line["ok"] for line in lines if line["step"] == "load"] == [False]  # a 404 on the mount: not an error
    assert all("error" not in line for line in lines)
    assert [load["lora_path"] for load in fake.loads] == [
        f"{config.ADAPTERS_DIR}/{ADAPTER.id}", f"{config.STAGED_ADAPTERS_DIR}/{ADAPTER.id}"]


def test_an_adapter_on_the_mount_is_never_staged(capsys: pytest.CaptureFixture[str]) -> None:
    fake = LiveVllm()
    stage = Stage(fake, committed={ADAPTER.id})
    code, lines, _ = _run(["--adapter", ADAPTER.id, "--go"], fake, capsys, stage=stage)
    assert code == 0 and stage.asked == [] and "stage" not in [line["step"] for line in lines]


def test_a_wedged_server_reports_each_step_and_fails(capsys: pytest.CaptureFixture[str]) -> None:
    def wedged(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})  # the API server answers; nothing past it does
        raise httpx.ReadTimeout("no answer", request=request)

    code, lines, out = _run(["--go"], wedged, capsys)
    assert code == 1 and lines[-1] == {"step": "done", "ok": False}
    assert [(line["step"], line["ok"]) for line in lines[:4]] == [
        ("wake", True), ("models", True), ("tokenize", False), ("complete", False)]
    assert lines[2]["error"] == "/tokenize failed in transport (ReadTimeout)" and TOKEN not in out


def test_a_server_that_never_wakes_stops_at_the_wake(capsys: pytest.CaptureFixture[str]) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    code, lines, _ = _run(["--go"], down, capsys)
    assert code == 1 and [line["step"] for line in lines] == ["wake", "done"] and not lines[0]["ok"]


def test_it_refuses_a_bad_adapter_name_or_a_missing_setting(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as refused:
        _run(["--adapter", "../weights", "--go"], LiveVllm(), capsys)
    assert refused.value.code == 2
    with pytest.raises(SystemExit, match=MODAL_TOKEN_ENV) as missing:
        _run(["--go"], LiveVllm(), capsys, environ={MODAL_URL_ENV: URL})
    assert URL not in str(missing.value)
