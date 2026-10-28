# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Glimmer on Colab vLLM behind a tunnel (the APD runner): held before it starts, stopped in a
``finally``, health-checked before any arm, and nothing but vLLM and its own key on the VM. The
Colab CLI, the tunnel and Glimmer are fakes: no VM, no network."""
from __future__ import annotations

import contextlib
import functools
import json
import subprocess
from contextlib import AbstractContextManager
from pathlib import Path

import pytest

from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.mind.apd import colab_vllm, fakes
from anjo.sleep.mind.apd.colab_vllm import ColabSession, NotReady, serve
from anjo.sleep.serving import config, framing
from anjo.sleep.serving.config import ServeConfig
from anjo.sleep.stops import BudgetStop
from anjo.sleep.train.runners.colab import A100_CU_PER_HOUR, ColabCli, ColabError

TIMEOUT_S = 3 * 3600.0
SETTINGS = ServeConfig(precision=config.SERVING, revision="a" * 40, tool_parser=None, gpu_snapshot=False,
                       vllm_version=config.DEFAULT_VLLM_VERSION)
KEYS = {"ANJO_BASETEN_API_KEY": "baseten-secret-1", "ANJO_META_API_KEY": "meta-secret-2",
        "GOOGLE_API_KEY": "google-secret-3", "GEMINI_API_KEY": "gemini-secret-4",
        "ANJO_FIREWORKS_API_KEY": "fireworks-secret-5"}


def _serve(vm: fakes.FakeColab, ledger: Ledger | None) -> AbstractContextManager[ColabSession]:
    wait = functools.partial(colab_vllm.ready, clock=fakes.Ticks(), sleep=lambda _s: None)
    return serve(ColabCli(Path("/fake/colab"), run=vm), ledger, SETTINGS, timeout_s=TIMEOUT_S,
                 tunnels=fakes.fake_tunnels(vm), clock=fakes.Ticks(), wait=wait, keygen=fakes.fake_keygen)


class _Proc:
    """An ``ssh`` child that never exits on its own."""

    killed = terminated = 0

    def poll(self) -> None:
        return None

    def kill(self) -> None:
        _Proc.killed += 1

    def terminate(self) -> None:
        _Proc.terminated += 1

    def wait(self, timeout: float) -> int:
        return 0


def test_colab_vllm_holds_before_start_and_stops_in_finally(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    vm = fakes.FakeColab(ledger=ledger)
    with _serve(vm, ledger) as session:
        assert isinstance(session, ColabSession) and session.live
        assert vm.held_at_new == pytest.approx(TIMEOUT_S / 3600 * A100_CU_PER_HOUR)  # held before `new`
        assert session.client().base_url == session.base_url
    assert not session.live and vm.commands()[-1] == "stop" and vm.tunnel_closed_before_stop
    assert ledger.open_holds() == () and 0 < ledger.used("colab_cu") < TIMEOUT_S / 3600 * A100_CU_PER_HOUR
    with pytest.raises(ColabError, match="isn't serving"):
        session.client()

    failing = fakes.FakeColab()  # a failure inside the run: the VM is stopped anyway
    with pytest.raises(RuntimeError, match="arm broke"):
        with _serve(failing, None):
            raise RuntimeError("arm broke")
    assert failing.commands()[-1] == "stop" and failing.tunnel_closed_before_stop

    ledger2 = Ledger(tmp_path / "b2.jsonl", caps=CAPS)
    sick = fakes.FakeColab(ledger=ledger2, healthy=False)  # the health check fails: nothing yields
    with pytest.raises(NotReady):
        with _serve(sick, ledger2):
            raise AssertionError("no arm starts on a server that isn't up")
    assert sick.commands()[-1] == "stop" and ledger2.open_holds() == ()


def test_a_failed_stop_settles_the_whole_hold_and_raises(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    vm = fakes.FakeColab(ledger=ledger, fail="stop")
    with pytest.raises(ColabError, match="by hand"):
        with _serve(vm, ledger):
            pass
    assert ledger.used("colab_cu") == pytest.approx(TIMEOUT_S / 3600 * A100_CU_PER_HOUR)
    timed_out = fakes.FakeColab(ledger=Ledger(tmp_path / "b2.jsonl", caps=CAPS), fail="new", timeout=True)
    with pytest.raises(subprocess.TimeoutExpired):
        with _serve(timed_out, timed_out.ledger):
            pass
    assert timed_out.commands() == ["new", "stop"]  # a `new` that timed out may have made a VM
    assert timed_out.ledger is not None and timed_out.ledger.used("colab_cu") == pytest.approx(
        TIMEOUT_S / 3600 * A100_CU_PER_HOUR)


def test_no_room_stops_before_anything_starts(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "colab_cu": 5.0})
    vm = fakes.FakeColab(ledger=ledger)
    with pytest.raises(BudgetStop):
        with _serve(vm, ledger):
            pass
    assert vm.calls == []


def test_colab_vllm_never_uploads_or_passes_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in KEYS.items():
        monkeypatch.setenv(name, value)
    keys: list[str] = []
    for _ in range(2):
        vm = fakes.FakeColab()
        with _serve(vm, None) as session:
            keys.append(session.key)
        argv = [part for args, _ in vm.calls for part in args]
        assert not any(value in part for value in KEYS.values() for part in argv)
        assert not any(session.key in part for part in argv)  # its own key in no command line either
        uploads = [args for args, _ in vm.calls if args[3] == "upload"]
        assert [args[-1] for args in uploads] == [colab_vllm.KEY_FILE] and vm.key_modes == [0o600]
        assert not Path(uploads[0][-2]).exists()  # the local copy is gone
        assert ".env" not in " ".join(argv)
        execs = [args for args, _ in vm.calls if args[3] == "exec"]
        assert execs and all(args[args.index("-f") + 1] == str(colab_vllm.SCRIPT) for args in execs)
        names = {flag.split("=", 1)[0] for args in execs for flag in args[args.index("--timeout") + 3::2]}
        assert names <= {"ANJO_COLAB_STEP", "ANJO_GLIMMER_MODEL", "ANJO_GLIMMER_REVISION", "ANJO_GLIMMER_DIR",
                         "ANJO_VLLM_COMMAND", "ANJO_VLLM_ENV", "ANJO_VLLM_SECONDS", "ANJO_VLLM_KEY_FILE"}
        start = vm.env_of("start")
        assert start["ANJO_VLLM_KEY_FILE"] == colab_vllm.KEY_FILE and vm.keys == {}  # the start step took the key
        command = json.loads(start["ANJO_VLLM_COMMAND"])
        assert command == config.vllm_command(SETTINGS, model_path=f"{colab_vllm.WEIGHTS_DIR}/{'a' * 40}",
                                              host="127.0.0.1", port=config.PORT)
        assert "--language-model-only" in command
        assert session.key not in " ".join(command)  # vLLM reads its key from its environment
        assert vm.installed == [f"vllm=={SETTINGS.vllm_version}"]
    assert keys[0] != keys[1] and all(len(key) >= 32 for key in keys)


def test_the_tunnel_is_one_local_forward_through_the_colab_proxy(tmp_path: Path) -> None:
    identity = tmp_path / "id_ed25519"
    argv = colab_vllm.ssh_command("anjo-mind-apd-1a2b3c4d", 41234, 8000, tmp_path / "k", identity)
    assert argv[:6] == ["ssh", "-F", "/dev/null", "-N", "-L", "127.0.0.1:41234:127.0.0.1:8000"]
    proxy = next(opt for opt in argv if opt.startswith("ProxyCommand="))
    assert f"ssh --proxy-mode -s anjo-mind-apd-1a2b3c4d -i {identity}" in proxy
    assert "--rm" not in proxy and "--gpu" not in proxy
    options = {opt.split("=", 1)[0]: opt.split("=", 1)[1] for opt in argv[argv.index("-o") + 1::2] if "=" in opt}
    assert argv[argv.index("-i") + 1] == str(identity) and options["IdentitiesOnly"] == "yes"
    assert {name: options[name] for name in ("IdentityAgent", "ForwardAgent", "ControlMaster", "ControlPath",
                                             "ClearAllForwardings", "ExitOnForwardFailure", "ServerAliveInterval")} == {
        "IdentityAgent": "none", "ForwardAgent": "no", "ControlMaster": "no", "ControlPath": "none",
        "ClearAllForwardings": "no", "ExitOnForwardFailure": "yes", "ServerAliveInterval": "15"}
    assert argv.count("-L") == 1 and "-R" not in argv and "-D" not in argv  # the one forward, nothing else
    killed = _Proc.killed
    tunnel = colab_vllm.SshTunnel(argv, 0.05, popen=lambda _: _Proc())  # type: ignore[arg-type, return-value]
    import time
    time.sleep(0.2)
    assert _Proc.killed == killed + 1  # the hard timeout fired
    tunnel.close()
    with pytest.raises(ValueError, match="finite"):
        colab_vllm.SshTunnel(argv, float("inf"), popen=lambda _: _Proc())  # type: ignore[arg-type, return-value]


def test_each_session_tunnels_with_a_throwaway_key_made_before_new_and_deleted_after(tmp_path: Path) -> None:
    """No key of the user's: a fresh ed25519 pair per session, made before ``new`` in a 0700 folder
    of the run's own, passed to ``ssh -i`` and the proxy's ``-i`` only, and gone after the session
    however it ended. It never reaches a CLI call: the one upload is vLLM's key."""
    made: list[Path] = []
    started: list[list[str]] = []

    def keygen(path: Path) -> None:
        fakes.fake_keygen(path)
        made.append(path)

    class Watching(fakes.FakeColab):
        def _new(self, argv: list[str], session: str) -> str:
            key = made[-1]
            assert key.is_file() and key.stat().st_mode & 0o777 == 0o600
            assert key.parent.stat().st_mode & 0o777 == 0o700
            return super()._new(argv, session)

    def popen(argv: list[str]) -> _Proc:
        started.append(argv)
        return _Proc()

    opener = colab_vllm.ssh_tunnel(tmp_path / "ssh", popen=popen)  # type: ignore[arg-type]
    for vm, broke in ((Watching(), None), (Watching(), "arm broke"), (Watching(fail="new"), "colab new failed")):
        with pytest.raises((RuntimeError, ColabError)) if broke else contextlib.nullcontext():
            with serve(ColabCli(Path("/fake/colab"), run=vm), None, SETTINGS, timeout_s=TIMEOUT_S, tunnels=opener,
                       clock=fakes.Ticks(), wait=lambda _session: True, keygen=keygen, key_dir=tmp_path / "ssh"):
                argv = started[-1]
                proxy = next(opt for opt in argv if opt.startswith("ProxyCommand="))
                assert argv[argv.index("-i") + 1] == str(made[-1]) and proxy.endswith(f" -i {made[-1]}")
                if broke:
                    raise RuntimeError(broke)
        key = made[-1]
        assert not key.exists() and not key.with_name(f"{key.name}.pub").exists() and not key.parent.exists()
        assert not any(str(key.parent) in part for args, _ in vm.calls for part in args)
        assert [args[-1] for args, _ in vm.calls if args[3] == "upload"] == ([colab_vllm.KEY_FILE] if vm.fail is None
                                                                               else [])
        assert vm.commands()[-1] == "stop"
    assert len(set(made)) == 3 and len(started) == 2  # a fresh key per session; no tunnel when `new` failed


def test_preflight_refuses_before_new_when_a_tool_is_missing(tmp_path: Path) -> None:
    binary = tmp_path / "colab"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    colab_vllm.preflight(binary, which=lambda tool: f"/usr/bin/{tool}")
    with pytest.raises(colab_vllm.PreflightError, match="ssh-keygen"):
        colab_vllm.preflight(binary, which=lambda tool: None if tool == "ssh-keygen" else f"/usr/bin/{tool}")
    with pytest.raises(colab_vllm.PreflightError, match="colab"):
        colab_vllm.preflight(tmp_path / "missing", which=lambda tool: f"/usr/bin/{tool}")

    def no_keygen(path: Path) -> None:
        raise colab_vllm.PreflightError("ssh-keygen isn't on this machine")

    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    vm = fakes.FakeColab(ledger=ledger)
    with pytest.raises(colab_vllm.PreflightError):
        with serve(ColabCli(Path("/fake/colab"), run=vm), ledger, SETTINGS, timeout_s=TIMEOUT_S,
                   tunnels=fakes.fake_tunnels(vm), clock=fakes.Ticks(), keygen=no_keygen, key_dir=tmp_path / "ssh"):
            raise AssertionError("nothing serves without a key")
    assert vm.calls == [] and ledger.open_holds() == () and ledger.used("colab_cu") == 0  # nothing held or started
    assert list((tmp_path / "ssh").iterdir()) == []


def test_the_real_keygen_makes_a_bare_ed25519_pair(tmp_path: Path) -> None:
    import shutil
    if shutil.which("ssh-keygen") is None:
        pytest.skip("no ssh-keygen here")
    with colab_vllm.session_key(tmp_path, colab_vllm.ssh_keygen) as key:
        public = key.with_name(f"{key.name}.pub").read_text(encoding="utf-8").split()
        assert public[0] == "ssh-ed25519" and len(public) == 2  # no comment: no user or host name
        assert key.stat().st_mode & 0o777 == 0o600 and "PRIVATE KEY" in key.read_text(encoding="utf-8")
    assert not key.parent.exists()


def test_adapters_upload_only_checked_and_under_the_servers_rendering(tmp_path: Path) -> None:
    vm = fakes.FakeColab()
    good = fakes.adapter_folder(tmp_path / "p-0123456789ab-n0", framing.rendering(config.reasoning_strength()))
    other = fakes.adapter_folder(tmp_path / "p-ba9876543210-n0", "other/low")
    with _serve(vm, None) as session:
        session.upload_adapter(good, good.name)
        with pytest.raises(ColabError, match="rendering"):
            session.upload_adapter(other, other.name)
    uploads = [args for args, _ in vm.calls if args[3] == "upload" and args[-1] != colab_vllm.KEY_FILE]
    assert [Path(args[-1]).name for args in uploads] == list(config.ADAPTER_FILES)
    assert all(args[-1].startswith(f"{colab_vllm.ADAPTERS_DIR}/{good.name}/") for args in uploads)
    assert vm.adapters == {good.name}


def test_a_step_that_never_says_it_finished_fails_the_serve() -> None:
    class Silent(fakes.FakeColab):
        def _exec(self, argv: list[str], session: str) -> str:
            super()._exec(argv, session)
            return "Traceback (most recent call last): ...\n"  # the kernel raised; the CLI exited 0

    vm = Silent()
    with pytest.raises(ColabError, match="download step"):
        with _serve(vm, None):
            raise AssertionError("nothing serves after a failed step")
    assert vm.commands()[-1] == "stop"


def test_the_start_step_takes_the_key_from_its_file_and_deletes_it(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("colab_vllm_start", colab_vllm.SCRIPT)
    assert spec is not None and spec.loader is not None
    start = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(start)
    key_file = tmp_path / ".vllm_key"
    key_file.write_text("k" * 40 + "\n")
    assert start._take_key(str(key_file)) == "k" * 40 and not key_file.exists()
    assert start.STEP_OK == colab_vllm.STEP_OK


def test_loopback_requests_never_take_a_proxy_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real loopback server stands in for the tunnel's end (no custom transport, so httpx would
    honour the environment's proxies); the proxy named is a dead port."""
    import http.server
    import threading

    from anjo.sleep.serving.client import HttpPoster

    class Vllm(http.server.BaseHTTPRequestHandler):
        def _answer(self) -> None:
            body = json.dumps({"data": [{"id": config.MODEL}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _answer

        def log_message(self, *_args: object) -> None:
            return None

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Vllm)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    dead = colab_vllm.free_port()
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, f"http://127.0.0.1:{dead}")  # would carry the bearer token elsewhere
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    try:
        port = server.server_address[1]
        session = ColabSession("anjo-mind-apd-1a2b3c4d", port, "k" * 40, ColabCli(Path("/fake/colab"), run=fakes.FakeColab()))
        assert colab_vllm.ready(session, timeout_s=5, clock=fakes.Ticks(1.0), sleep=lambda _s: None)
        assert HttpPoster(f"http://127.0.0.1:{port}", "k" * 40).post("/v1/models", {})["data"]
    finally:
        server.shutdown()


def test_a_sessions_deadline_and_compute_units_run_on_the_wall_clock() -> None:
    """A Mac asleep stops ``time.monotonic`` while the VM bills on, so the session's deadline and
    its settled compute units read wall time."""
    import dataclasses
    import inspect
    import time

    from anjo.sleep.mind.apd import backend as backends

    assert inspect.signature(serve).parameters["clock"].default is time.time
    assert next(f for f in dataclasses.fields(backends.Backend) if f.name == "clock").default is time.time
