"""Glimmer on Colab vLLM behind an SSH tunnel: the one server every APD arm and every replay call
uses (Tasks 17 and 17b; spec: "APD runs Glimmer on Colab vLLM at one precision for every arm").

``serve`` is a context manager around one Colab A100 session:

1. **Hold first.** The session's worst case, its hard timeout at the A100 rate, is held on
   ``colab_cu`` before anything starts, as the Colab runner holds a training job. With no room the
   run stops (``BudgetStop``) and nothing starts.
2. **Only vLLM runs on the VM.** ``new`` an A100 session named ``anjo-mind-apd-<random>`` (MIND's
   prefix: Sleep's night sweep never stops it), ``install`` the pinned vLLM
   (``ServeConfig.vllm_version``), download the pinned Glimmer revision, then start ``vllm serve``
   exactly as ``serving.config.vllm_command`` builds it (the same revision, precision and LoRA
   settings, so the 17b adapters load too), bound to the VM's loopback, under the session's hard
   timeout. The VM-side steps are ``colab_vllm_start.py``; everything it needs goes in the kernel's
   environment. The only secret that leaves this machine is the session's own API key, drawn here
   at random for this session (``new_key``). No ``.env``, and no Baseten, Meta or Google key, is
   ever uploaded or passed.
3. **Reached through a tunnel.** ``ssh -N -L 127.0.0.1:<local>:127.0.0.1:8000`` with the Colab CLI as
   its ProxyCommand (``colab ssh --proxy-mode -s <session>``). The tunnel has a hard timeout of its
   own (a timer kills it at the session's deadline), and every CLI call runs under one.
4. **Health-checked** (``ready``): ``/v1/models`` answers with Glimmer through the tunnel, with the
   key, before ``serve`` yields, so no arm starts on a server that isn't up.
5. **Stopped in a ``finally``**, on success, on failure and when the health check fails: the
   tunnel is closed first (its ProxyCommand would otherwise make a new session if asked again),
   then the session is stopped and the hold settled at the time it ran. A stop that fails settles
   the whole hold and raises, so a VM that may still run never goes unnoticed.

The yielded ``ColabSession`` is the only server replay and the APD arms accept
(``replay.spend.colab_client``): it is live only inside ``serve``, and its client points at its own
tunnel's loopback end. Adapters go up to its LoRA directory by ``upload_adapter``, checked first.

A run killed hard runs no ``finally``: the APD command's own startup sweep stops every
``anjo-mind-`` session left over (``train.runners.colab.stop_leftover_sessions``), and vLLM on the
VM exits at the session's deadline whatever happens here.
"""
from __future__ import annotations

import json
import math
import secrets
import shlex
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Final, Protocol

import httpx

from anjo.sleep.budget import Bound, Ledger, Reservation
from anjo.sleep.night.holds import hold
from anjo.sleep.serving import config, framing
from anjo.sleep.serving.client import VllmClient
from anjo.sleep.serving.config import ServeConfig
from anjo.sleep.train.runners.colab import (
    A100_CU_PER_HOUR,
    AUTH,
    COLAB_BIN,
    GPU,
    INSTALL_TIMEOUT_S,
    MIND_SESSION_PREFIX,
    NEW_TIMEOUT_S,
    POOL,
    STOP_TIMEOUT_S,
    UPLOAD_TIMEOUT_S,
    ColabCli,
    ColabError,
)

SCRIPT: Final = Path(__file__).with_name("colab_vllm_start.py")
NAME: Final = "colab"  # the backend's name, as ``serving.backend`` names its own
LOOPBACK: Final = "127.0.0.1"
ROOT: Final = "/content"
WEIGHTS_DIR: Final = f"{ROOT}/glimmer"
ADAPTERS_DIR: Final = f"{ROOT}/adapters"
SSH: Final = "ssh"
SSH_USER: Final = "root"
SESSION_LABEL: Final = "apd-"
DOWNLOAD_TIMEOUT_S: Final = config.DOWNLOAD_TIMEOUT_S
START_TIMEOUT_S: Final = 120.0
MKDIR_TIMEOUT_S: Final = 60.0
READY_TIMEOUT_S: Final = config.STARTUP_TIMEOUT_S  # the 30B model loads well inside it
READY_POLL_S: Final = 10.0
READY_BACKOFF_S: Final = 5.0
TUNNEL_CLOSE_S: Final = 10.0
EXEC_MARGIN_S: Final = 60.0  # the CLI's own wait past the kernel's timeout
MIN_SERVE_S: Final = 1800.0  # what a session must have left to serve once it is up
MAX_SESSION_S: Final = 12 * 3600.0  # inside a Colab Pro runtime's own limit
SETUP_S: Final = (NEW_TIMEOUT_S + INSTALL_TIMEOUT_S + DOWNLOAD_TIMEOUT_S + START_TIMEOUT_S + 2 * EXEC_MARGIN_S
                  + READY_TIMEOUT_S + STOP_TIMEOUT_S)
_PACKAGE: Final = "vllm"


class NotReady(ColabError):
    """vLLM never answered through the tunnel inside ``READY_TIMEOUT_S``."""


class Tunnel(Protocol):
    """An open local forward to the VM's vLLM port. ``transport`` is ``None`` for a real tunnel
    (requests go over the socket); a fake carries the transport its fake server answers on."""

    @property
    def transport(self) -> httpx.BaseTransport | None: ...

    def close(self) -> None: ...


TunnelOpener = Callable[[str, int, int, float], Tunnel]  # session, local port, remote port, hard timeout


def new_key() -> str:
    """A random API key for one session, drawn here and known only to this run and that VM."""
    return secrets.token_urlsafe(32)


def free_port() -> int:
    """A free port on this machine's loopback, for the tunnel's local end."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, 0))
        return int(probe.getsockname()[1])


def session_name(token: str | None = None) -> str:
    """``anjo-mind-apd-<8 hex>``: MIND's prefix and a random tail, never an id."""
    return f"{MIND_SESSION_PREFIX}{SESSION_LABEL}{token or secrets.token_hex(4)}"


# ── The tunnel ────────────────────────────────────────────────────────────────


def ssh_command(session: str, local_port: int, remote_port: int, known_hosts: Path,
                binary: Path = COLAB_BIN) -> list[str]:
    """``ssh`` holding one local forward to the VM's loopback, through the Colab CLI's proxy.
    Each session's host key is accepted once into its own file, then held to."""
    proxy = shlex.join([str(binary), *AUTH, "ssh", "--proxy-mode", "-s", session])
    return [
        SSH, "-N", "-L", f"{LOOPBACK}:{local_port}:{LOOPBACK}:{remote_port}",
        "-o", f"ProxyCommand={proxy}", "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4",
        "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known_hosts}",
        f"{SSH_USER}@{session}",
    ]


class SshTunnel:
    """The ``ssh`` forward as a child process, killed at ``timeout_s`` by a timer if nothing
    closed it first."""

    transport: httpx.BaseTransport | None = None

    def __init__(self, command: Sequence[str], timeout_s: float,
                 popen: Callable[[list[str]], subprocess.Popen[bytes]] | None = None) -> None:
        if not (0 < timeout_s < math.inf):
            raise ValueError("a tunnel has a finite timeout")
        start = popen or (lambda argv: subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                                         stderr=subprocess.DEVNULL))
        self._process = start(list(command))
        self._timer = threading.Timer(timeout_s, self._process.kill)
        self._timer.daemon = True
        self._timer.start()

    def close(self) -> None:
        self._timer.cancel()
        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=TUNNEL_CLOSE_S)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=TUNNEL_CLOSE_S)


def ssh_tunnel(known_hosts_dir: Path) -> TunnelOpener:
    """The real opener: one ``SshTunnel`` per session, its host keys in their own file."""

    def open_tunnel(session: str, local_port: int, remote_port: int, timeout_s: float) -> Tunnel:
        known_hosts_dir.mkdir(parents=True, exist_ok=True)
        return SshTunnel(ssh_command(session, local_port, remote_port, known_hosts_dir / f"{session}.known_hosts"),
                         timeout_s)

    return open_tunnel


# ── The session handle ────────────────────────────────────────────────────────


@dataclass
class _Live:
    up: bool = False


@dataclass(frozen=True)
class ColabSession:
    """One live Glimmer server on Colab, as ``serve`` yields it: replay's ``GlimmerServer`` and
    the APD arms' server. ``session`` is the Colab session's name; the server answers on this
    machine's loopback at ``local_port``, with ``key``."""

    session: str
    local_port: int
    key: str = field(repr=False)
    cli: ColabCli = field(repr=False, compare=False)
    transport: httpx.BaseTransport | None = field(default=None, repr=False, compare=False)
    _state: _Live = field(default_factory=_Live, repr=False, compare=False)
    name: ClassVar[str] = NAME

    @property
    def base_url(self) -> str:
        return f"http://{LOOPBACK}:{self.local_port}"

    @property
    def live(self) -> bool:
        return self._state.up

    def client(self) -> VllmClient:
        """A client of this session's own tunnel; refused once the session has ended."""
        if not self.live:
            raise ColabError(f"session {self.session} isn't serving")
        return VllmClient(self.base_url, self.key, transport=self.transport)

    def upload_adapter(self, folder: Path, name: str) -> None:
        """Put one adapter into the server's LoRA directory, where vLLM's resolver loads it by
        name: a checked LoRA of Glimmer, recorded under the rendering this server serves."""
        config.check_adapter(folder, name)
        if _rendering(folder) != self.client().rendering:
            raise ColabError(f"adapter {name} wasn't trained under the rendering this server serves")
        remote = f"{ADAPTERS_DIR}/{name}"
        _exec(self.cli, self.session, MKDIR_TIMEOUT_S, ANJO_COLAB_STEP="mkdir", ANJO_COLAB_DIR=remote)
        for file in config.ADAPTER_FILES:
            self.cli.call("upload", "-s", self.session, str(folder / file), f"{remote}/{file}",
                          timeout_s=UPLOAD_TIMEOUT_S)


def _rendering(folder: Path) -> object:
    try:
        meta = json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta.get(framing.RENDERING_KEY) if isinstance(meta, dict) else None


def ready(session: ColabSession, *, timeout_s: float = READY_TIMEOUT_S, clock: Callable[[], float] = time.monotonic,
          sleep: Callable[[float], None] = time.sleep) -> bool:
    """Poll ``/v1/models`` through the tunnel, with the key, until it names Glimmer, for at most
    ``timeout_s``. A failed poll is ``False`` for that poll, never an exception."""
    started = clock()
    while clock() - started < timeout_s:
        try:
            with httpx.Client(base_url=session.base_url, headers={"Authorization": f"Bearer {session.key}"},
                              transport=session.transport, timeout=READY_POLL_S) as http:
                answer = http.get("/v1/models")
            served = [str(model.get("id")) for model in answer.json().get("data", [])] if answer.status_code == 200 else []
        except (httpx.HTTPError, ValueError, AttributeError):
            served = []
        if config.MODEL in served:
            return True
        sleep(min(READY_BACKOFF_S, max(0.0, timeout_s - (clock() - started))))
    return False


# ── Serving ───────────────────────────────────────────────────────────────────


def _exec(cli: ColabCli, session: str, timeout_s: float, **env: str) -> str:
    flags = [part for key, value in env.items() for part in ("--env", f"{key}={value}")]
    return cli.call("exec", "-s", session, "-f", str(SCRIPT), "--timeout", str(timeout_s), *flags,
                    timeout_s=timeout_s + EXEC_MARGIN_S)


def server_env() -> dict[str, str]:
    """vLLM's environment on the VM: the night's, with Colab's folders, and no download plugin."""
    return {**config.vllm_env(), "VLLM_LORA_RESOLVER_CACHE_DIR": ADAPTERS_DIR, "HF_HOME": f"{ROOT}/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "0"}


def server_command(settings: ServeConfig) -> list[str]:
    """``vllm serve`` as ``serving.config`` builds it, on the pinned weights, on the VM's loopback."""
    return config.vllm_command(settings, model_path=f"{WEIGHTS_DIR}/{settings.pinned_revision()}", host=LOOPBACK,
                               port=config.PORT)


def _boot(cli: ColabCli, session: str, settings: ServeConfig, key: str, serve_s: float, vm: _Vm) -> None:
    vm.state = "maybe"  # a `new` that times out may still have made the VM
    try:
        cli.call("new", "-s", session, "--gpu", GPU, "--high-mem", timeout_s=NEW_TIMEOUT_S)
    except (ColabError, OSError):
        vm.state = "none"
        raise
    vm.state = "up"
    cli.call("install", "-s", session, f"{_PACKAGE}=={settings.vllm_version}", timeout_s=INSTALL_TIMEOUT_S)
    revision = settings.pinned_revision()
    _exec(cli, session, DOWNLOAD_TIMEOUT_S, ANJO_COLAB_STEP="download", ANJO_GLIMMER_MODEL=config.MODEL,
          ANJO_GLIMMER_REVISION=revision, ANJO_GLIMMER_DIR=f"{WEIGHTS_DIR}/{revision}")
    _exec(cli, session, START_TIMEOUT_S, ANJO_COLAB_STEP="start",
          ANJO_VLLM_COMMAND=json.dumps(server_command(settings)), ANJO_VLLM_ENV=json.dumps(server_env()),
          ANJO_VLLM_SECONDS=str(int(serve_s)), VLLM_API_KEY=key)


@dataclass
class _Vm:
    state: str = "none"  # none, maybe (a `new` that may have made one) or up


def _finish(cli: ColabCli, session: str, vm: _Vm, reservation: Reservation | None, elapsed: float) -> None:
    """Stop the VM, then settle what it ran for; a failed stop settles the whole hold and raises."""
    try:
        cli.call("stop", "-s", session, timeout_s=STOP_TIMEOUT_S)
    except (ColabError, subprocess.TimeoutExpired, OSError) as error:
        if vm.state != "none":
            if reservation is not None:
                reservation.settle(reservation.bound.amount)
            raise ColabError(f"stopping {session} failed; stop it by hand before anything else") from error
    if reservation is None:
        return
    reservation.settle(reservation.bound.amount if vm.state == "maybe" else elapsed / 3600.0 * A100_CU_PER_HOUR)


def check_timeout(timeout_s: float) -> None:
    if not (SETUP_S + MIN_SERVE_S <= timeout_s <= MAX_SESSION_S):
        raise ValueError(f"a serving session's hard timeout is {SETUP_S + MIN_SERVE_S:.0f} to {MAX_SESSION_S:.0f} s")


def hold_amount(timeout_s: float) -> float:
    """The compute units ``serve`` holds for a session with this hard timeout."""
    return Bound(A100_CU_PER_HOUR, timeout_s).amount


@contextmanager
def serve(
    cli: ColabCli, ledger: Ledger | None, settings: ServeConfig, *, timeout_s: float, tunnels: TunnelOpener,
    clock: Callable[[], float] = time.monotonic, key: Callable[[], str] = new_key,
    port: Callable[[], int] = free_port, session: str | None = None,
    wait: Callable[[ColabSession], bool] = ready,
) -> Iterator[ColabSession]:
    """One Glimmer server on a Colab A100, held, health-checked, and stopped on the way out."""
    check_timeout(timeout_s)
    settings.pinned_revision()  # no server starts unpinned
    name = session or session_name()
    reservation = None if ledger is None else hold(ledger, f"colab_serve:{name}", (POOL,), Bound(A100_CU_PER_HOUR, timeout_s))
    started, vm = clock(), _Vm()
    tunnel: Tunnel | None = None
    handle: ColabSession | None = None
    try:
        secret = key()
        _boot(cli, name, settings, secret, timeout_s - (clock() - started), vm)
        local = port()
        tunnel = tunnels(name, local, config.PORT, timeout_s - (clock() - started))
        handle = ColabSession(name, local, secret, cli, tunnel.transport)
        if not wait(handle):
            raise NotReady(f"Glimmer on {name} didn't answer through the tunnel in time")
        handle._state.up = True
        yield handle
    finally:
        if handle is not None:
            handle._state.up = False
        try:
            if tunnel is not None:
                tunnel.close()
        finally:
            _finish(cli, name, vm, reservation, max(0.0, clock() - started))
