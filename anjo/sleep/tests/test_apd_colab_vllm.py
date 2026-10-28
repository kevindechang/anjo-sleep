# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Glimmer on Colab vLLM behind a tunnel (the APD runner): held before it starts, stopped in a
``finally``, health-checked before any arm, and nothing but vLLM and its own key on the VM. The
Colab CLI, the tunnel and Glimmer are fakes: no VM, no network."""
from __future__ import annotations

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
        "GOOGLE_API_KEY": "google-secret-3", "GEMINI_API_KEY": "gemini-secret-4"}


def _serve(vm: fakes.FakeColab, ledger: Ledger | None) -> AbstractContextManager[ColabSession]:
    wait = functools.partial(colab_vllm.ready, clock=fakes.Ticks(), sleep=lambda _s: None)
    return serve(ColabCli(Path("/fake/colab"), run=vm), ledger, SETTINGS, timeout_s=TIMEOUT_S,
                 tunnels=fakes.fake_tunnels(vm), clock=fakes.Ticks(), wait=wait)


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
        assert ".env" not in " ".join(argv) and "upload" not in vm.commands()
        execs = [args for args, _ in vm.calls if args[3] == "exec"]
        assert execs and all(args[args.index("-f") + 1] == str(colab_vllm.SCRIPT) for args in execs)
        names = {flag.split("=", 1)[0] for args in execs for flag in args[args.index("--timeout") + 3::2]}
        assert names <= {"ANJO_COLAB_STEP", "ANJO_GLIMMER_MODEL", "ANJO_GLIMMER_REVISION", "ANJO_GLIMMER_DIR",
                         "ANJO_VLLM_COMMAND", "ANJO_VLLM_ENV", "ANJO_VLLM_SECONDS", "VLLM_API_KEY"}
        start = vm.env_of("start")
        assert start["VLLM_API_KEY"] == session.key
        command = json.loads(start["ANJO_VLLM_COMMAND"])
        assert command == config.vllm_command(SETTINGS, model_path=f"{colab_vllm.WEIGHTS_DIR}/{'a' * 40}",
                                              host="127.0.0.1", port=config.PORT)
        assert session.key not in " ".join(command)  # vLLM reads its key from its environment
        assert vm.installed == [f"vllm=={SETTINGS.vllm_version}"]
    assert keys[0] != keys[1] and all(len(key) >= 32 for key in keys)


def test_the_tunnel_is_one_local_forward_through_the_colab_proxy(tmp_path: Path) -> None:
    argv = colab_vllm.ssh_command("anjo-mind-apd-1a2b3c4d", 41234, 8000, tmp_path / "k")
    assert argv[:4] == ["ssh", "-N", "-L", "127.0.0.1:41234:127.0.0.1:8000"]
    proxy = next(opt for opt in argv if opt.startswith("ProxyCommand="))
    assert "ssh --proxy-mode -s anjo-mind-apd-1a2b3c4d" in proxy and "--rm" not in proxy and "--gpu" not in proxy

    class Proc:
        killed = terminated = 0

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            Proc.killed += 1

        def terminate(self) -> None:
            Proc.terminated += 1

        def wait(self, timeout: float) -> int:
            return 0

    tunnel = colab_vllm.SshTunnel(argv, 0.05, popen=lambda _: Proc())  # type: ignore[arg-type, return-value]
    import time
    time.sleep(0.2)
    assert Proc.killed == 1  # the hard timeout fired
    tunnel.close()
    with pytest.raises(ValueError, match="finite"):
        colab_vllm.SshTunnel(argv, float("inf"), popen=lambda _: Proc())  # type: ignore[arg-type, return-value]


def test_adapters_upload_only_checked_and_under_the_servers_rendering(tmp_path: Path) -> None:
    vm = fakes.FakeColab()
    good = fakes.adapter_folder(tmp_path / "p-0123456789ab-n0", framing.rendering(config.reasoning_strength()))
    other = fakes.adapter_folder(tmp_path / "p-ba9876543210-n0", "other/low")
    with _serve(vm, None) as session:
        session.upload_adapter(good, good.name)
        with pytest.raises(ColabError, match="rendering"):
            session.upload_adapter(other, other.name)
    uploads = [args for args, _ in vm.calls if args[3] == "upload"]
    assert [Path(args[-1]).name for args in uploads] == list(config.ADAPTER_FILES)
    assert all(args[-1].startswith(f"{colab_vllm.ADAPTERS_DIR}/{good.name}/") for args in uploads)
    assert vm.adapters == {good.name}
