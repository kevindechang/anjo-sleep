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
   timeout. The VM-side steps are ``colab_vllm_start.py``; what it needs goes in the kernel's
   environment, and each step prints ``STEP_OK`` when it finished, which ``_exec`` checks. The only
   secret that leaves this machine is the session's own API key, drawn here at random for this
   session (``new_key``). It goes up as a file of its own (``KEY_FILE``, written 0600 here), which
   the start step reads into vLLM's environment and deletes: never in a command line, where
   ``ps`` would show it. No ``.env``, and no Baseten, Fireworks, Meta or Google key, is ever
   uploaded or passed.
3. **Reached through a tunnel.** ``ssh -N -L 127.0.0.1:<local>:127.0.0.1:8000`` with the Colab CLI as
   its ProxyCommand (``colab ssh --proxy-mode -s <session>``). The tunnel has a hard timeout of its
   own (a timer kills it at the session's deadline), and every CLI call runs under one. ``ssh``
   reads no config file and uses no agent. Its one identity is a throwaway ed25519 pair with no
   passphrase, made for this session before anything is held or started (``session_key``, in a
   0700 folder of the run's own, the key 0600) and deleted in the outermost ``finally``: it is
   passed to ``ssh -i`` and to the proxy's ``-i`` (the CLI sends the VM its public half) and
   nowhere else, never uploaded or logged. Nothing is forwarded but the one port, and no shared
   connection is opened or reused. ``preflight`` checks the CLI, ``ssh`` and ``ssh-keygen`` are
   here before a command serves.
4. **Health-checked** (``ready``): ``/v1/models`` answers with Glimmer through the tunnel, with the
   key, before ``serve`` yields, so no arm starts on a server that isn't up.
5. **Stopped in a ``finally``**, on success, on failure and when the health check fails: the
   tunnel is closed first (its ProxyCommand would otherwise make a new session if asked again),
   then the session is stopped and the hold settled at the time it ran. A stop that fails settles
   the whole hold and raises, so a VM that may still run never goes unnoticed.

The yielded ``ColabSession`` is the only server replay and the APD arms accept
(``replay.spend.colab_client``): it is live only inside ``serve``, and its client points at its own
tunnel's loopback end, a transient failure tried again (``retry``). Adapters go up to its LoRA
directory by ``upload_adapter``, checked first.

A run killed hard runs no ``finally``: the next APD command's sweep, under MIND's lock
(``mind.sweep``), stops every ``anjo-mind-`` session left over, and vLLM on the VM exits at the
session's deadline whatever happens here. The session's deadline and its compute units are read on
the wall clock, which runs on while a Mac sleeps, as the VM's bill does. Before each run,
``check_room`` refuses one the session can't finish (``SessionEnding``).
"""
from __future__ import annotations

import json
import math
import os
import secrets
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Final, Protocol

import httpx

from anjo.sleep.budget import Bound, Ledger, Reservation
from anjo.sleep.mind.apd.retry import RetryTransport
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
KEY_FILE: Final = f"{ROOT}/.vllm_key"  # the start step reads it into vLLM's environment and deletes it
STEP_OK: Final = "ANJO_STEP_OK"  # what colab_vllm_start.py prints when a step finished
SSH: Final = "ssh"
SSH_USER: Final = "root"
SSH_KEYGEN: Final = "ssh-keygen"
SESSION_KEY: Final = "id_ed25519"  # the throwaway key's file name in its session's own folder
KEYGEN_TIMEOUT_S: Final = 30.0
SESSION_LABEL: Final = "apd-"
DOWNLOAD_TIMEOUT_S: Final = config.DOWNLOAD_TIMEOUT_S
START_TIMEOUT_S: Final = 120.0
MKDIR_TIMEOUT_S: Final = 60.0
READY_TIMEOUT_S: Final = config.STARTUP_TIMEOUT_S  # the 30B model loads well inside it
READY_POLL_S: Final = 10.0
READY_BACKOFF_S: Final = 5.0
CHECK_TIMEOUT_S: Final = 120.0  # how long a check before each run waits for vLLM to answer
TUNNEL_CLOSE_S: Final = 10.0
EXEC_MARGIN_S: Final = 60.0  # the CLI's own wait past the kernel's timeout
MIN_SERVE_S: Final = 1800.0  # what a session must have left to serve once it is up
MAX_SESSION_S: Final = 12 * 3600.0  # inside a Colab Pro runtime's own limit
SETUP_S: Final = (NEW_TIMEOUT_S + INSTALL_TIMEOUT_S + DOWNLOAD_TIMEOUT_S + START_TIMEOUT_S + 2 * EXEC_MARGIN_S
                  + READY_TIMEOUT_S + STOP_TIMEOUT_S)
_PACKAGE: Final = "vllm"


class NotReady(ColabError):
    """vLLM never answered through the tunnel inside ``READY_TIMEOUT_S``."""


class PreflightError(ColabError):
    """A tool a session needs isn't on this machine: refused before anything is held or started."""


class SessionEnding(ColabError):
    """The session can't carry another run: it ended, its tunnel dropped, vLLM stopped answering,
    or too little of its time is left. The batch stops cleanly; what finished is kept."""


class Tunnel(Protocol):
    """An open local forward to the VM's vLLM port. ``transport`` is ``None`` for a real tunnel
    (requests go over the socket); a fake carries the transport its fake server answers on."""

    @property
    def transport(self) -> httpx.BaseTransport | None: ...

    def close(self) -> None: ...

    def alive(self) -> bool: ...


TunnelOpener = Callable[[str, int, int, float, Path], Tunnel]  # session, ports, hard timeout, identity
KeyGen = Callable[[Path], None]  # makes a key pair with no passphrase at ``path`` and ``path.pub``


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


def preflight(binary: Path = COLAB_BIN, *, which: Callable[[str], str | None] = shutil.which) -> None:
    """What a serving command needs on this machine, checked before anything is held: the Colab
    CLI, ``ssh`` and ``ssh-keygen``."""
    missing = [] if binary.is_file() and os.access(binary, os.X_OK) else [f"the colab CLI ({binary})"]
    missing += [tool for tool in (SSH, SSH_KEYGEN) if which(tool) is None]
    if missing:
        raise PreflightError(f"a Colab session needs {', '.join(missing)} on this machine")


def ssh_keygen(path: Path) -> None:
    """A fresh ed25519 pair at ``path`` and ``path.pub``: no passphrase, and no comment, so no
    user or host name rides on the public half."""
    tool = shutil.which(SSH_KEYGEN)
    if tool is None:
        raise PreflightError(f"{SSH_KEYGEN} isn't on this machine")
    try:
        subprocess.run([tool, "-q", "-t", "ed25519", "-N", "", "-C", "", "-f", str(path)], check=True,
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=KEYGEN_TIMEOUT_S)
    except (subprocess.SubprocessError, OSError) as error:
        raise ColabError("ssh-keygen couldn't make the session's key") from error


@contextmanager
def session_key(parent: Path | None, keygen: KeyGen = ssh_keygen) -> Iterator[Path]:
    """One session's throwaway SSH key, in a 0700 folder of its own under ``parent`` (the run's
    private ``ssh`` folder; the user's temporary folder when ``None``), the key 0600. Both halves
    and the folder are deleted on the way out, however the session ended."""
    if parent is not None:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="apd-key-", dir=parent))  # 0700
    key = folder / SESSION_KEY
    try:
        keygen(key)
        key.chmod(0o600)
        yield key
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def ssh_command(session: str, local_port: int, remote_port: int, known_hosts: Path, identity: Path,
                binary: Path = COLAB_BIN) -> list[str]:
    """``ssh`` holding one local forward to the VM's loopback, through the Colab CLI's proxy, with
    no config file, no agent, and ``identity`` only. Each session's host key is accepted once into
    its own file, then held to."""
    proxy = shlex.join([str(binary), *AUTH, "ssh", "--proxy-mode", "-s", session, "-i", str(identity)])
    options = (
        f"ProxyCommand={proxy}", "ExitOnForwardFailure=yes", "BatchMode=yes",
        "ServerAliveInterval=15", "ServerAliveCountMax=4",
        "IdentitiesOnly=yes", "IdentityAgent=none", "ForwardAgent=no", "ForwardX11=no",
        "ClearAllForwardings=no", "PermitLocalCommand=no", "ControlMaster=no", "ControlPath=none",
        "StrictHostKeyChecking=accept-new", f"UserKnownHostsFile={known_hosts}", "GlobalKnownHostsFile=/dev/null",
    )
    return [SSH, "-F", "/dev/null", "-N", "-L", f"{LOOPBACK}:{local_port}:{LOOPBACK}:{remote_port}",
            "-i", str(identity), *(part for option in options for part in ("-o", option)), f"{SSH_USER}@{session}"]


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

    def alive(self) -> bool:
        return self._process.poll() is None

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


def ssh_tunnel(known_hosts_dir: Path,
               popen: Callable[[list[str]], subprocess.Popen[bytes]] | None = None) -> TunnelOpener:
    """The real opener: one ``SshTunnel`` per session, on that session's own key, its host keys in
    their own file."""

    def open_tunnel(session: str, local_port: int, remote_port: int, timeout_s: float, identity: Path) -> Tunnel:
        known_hosts_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        return SshTunnel(ssh_command(session, local_port, remote_port, known_hosts_dir / f"{session}.known_hosts",
                                     identity), timeout_s, popen)

    return open_tunnel


# ── The session handle ────────────────────────────────────────────────────────


@dataclass
class _Live:
    up: bool = False
    deadline: float = math.inf  # on ``clock``, when the session must be stopping
    tunnel: Tunnel | None = None
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep


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

    def check_room(self, need_s: float) -> None:
        """Refuse (``SessionEnding``) a run this session can't finish: it ended, its tunnel is
        down, vLLM doesn't answer, or less than ``need_s`` is left before its deadline."""
        state = self._state
        if not state.up:
            raise SessionEnding(f"session {self.session} isn't serving")
        if state.tunnel is not None and not state.tunnel.alive():
            raise SessionEnding(f"the tunnel to {self.session} dropped")
        if not ready(self, timeout_s=CHECK_TIMEOUT_S, clock=state.clock, sleep=state.sleep):
            raise SessionEnding(f"Glimmer on {self.session} stopped answering")
        left = state.deadline - state.clock()
        if left < need_s:
            raise SessionEnding(f"{self.session} has {max(0.0, left) / 60:.0f} min of its time left, less than "
                                f"the {need_s / 60:.0f} a whole life-arm needs")

    def client(self) -> VllmClient:
        """A client of this session's own tunnel, a transient failure tried again
        (``retry.RetryTransport``); refused once the session has ended."""
        if not self.live:
            raise ColabError(f"session {self.session} isn't serving")
        transport = RetryTransport(self.transport or httpx.HTTPTransport(), sleep=self._state.sleep)
        return VllmClient(self.base_url, self.key, transport=transport)

    def upload_adapter(self, folder: Path, name: str) -> None:
        """Put one adapter into the server's LoRA directory, where vLLM's resolver loads it by
        name: a checked LoRA of Glimmer, recorded under the rendering this server serves."""
        config.check_adapter(folder, name)
        if adapter_rendering(folder) != self.client().rendering:
            raise ColabError(f"adapter {name} wasn't trained under the rendering this server serves")
        remote = f"{ADAPTERS_DIR}/{name}"
        _exec(self.cli, self.session, MKDIR_TIMEOUT_S, ANJO_COLAB_STEP="mkdir", ANJO_COLAB_DIR=remote)
        for file in config.ADAPTER_FILES:
            self.cli.call("upload", "-s", self.session, str(folder / file), f"{remote}/{file}",
                          timeout_s=UPLOAD_TIMEOUT_S)


def attach(session: str, local_port: int, key: str, *, transport: httpx.BaseTransport | None = None) -> ColabSession:
    """A handle on a session this run's own ``serve`` holds open in another process: kill-resume's
    children (``pilot.kill_setup``), which inherit its tunnel's port and key from the parent's
    environment. The parent holds, health-checks and stops the session; the handle is live for as
    long as the child runs."""
    if not session.startswith(MIND_SESSION_PREFIX) or not 0 < local_port < 65536:
        raise ColabError("only a MIND session's own loopback tunnel can be attached")
    handle = ColabSession(session, local_port, key, ColabCli(), transport)
    handle._state.up = True
    return handle


def adapter_rendering(folder: Path) -> object:
    """The rendering an adapter folder records it was trained under, or ``None``."""
    try:
        meta = json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta.get(framing.RENDERING_KEY) if isinstance(meta, dict) else None


def ready(session: ColabSession, *, timeout_s: float = READY_TIMEOUT_S, clock: Callable[[], float] = time.monotonic,
          sleep: Callable[[float], None] = time.sleep) -> bool:
    """Poll ``/v1/models`` through the tunnel, with the key, until it names Glimmer, for at most
    ``timeout_s``. A failed poll is ``False`` for that poll, never an exception. The loopback poll
    never takes a proxy from the environment."""
    started = clock()
    while clock() - started < timeout_s:
        try:
            with httpx.Client(base_url=session.base_url, headers={"Authorization": f"Bearer {session.key}"},
                              transport=session.transport, timeout=READY_POLL_S, trust_env=False) as http:
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
    """One step of ``colab_vllm_start.py``; a step that didn't print its ``STEP_OK`` line failed."""
    flags = [part for key, value in env.items() for part in ("--env", f"{key}={value}")]
    out = cli.call("exec", "-s", session, "-f", str(SCRIPT), "--timeout", str(timeout_s), *flags,
                   timeout_s=timeout_s + EXEC_MARGIN_S)
    if f"{STEP_OK} {env['ANJO_COLAB_STEP']}" not in out.splitlines():
        raise ColabError(f"the {env['ANJO_COLAB_STEP']} step on {session} didn't finish")
    return out


def _send_key(cli: ColabCli, session: str, key: str) -> None:
    """The session's key up to ``KEY_FILE``, from a file only this user can read, gone after."""
    with tempfile.TemporaryDirectory(prefix="apd-key-") as folder:
        path = Path(folder) / "vllm_key"
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as file:
            file.write(key)
        cli.call("upload", "-s", session, str(path), KEY_FILE, timeout_s=UPLOAD_TIMEOUT_S)


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
    _send_key(cli, session, key)
    _exec(cli, session, START_TIMEOUT_S, ANJO_COLAB_STEP="start",
          ANJO_VLLM_COMMAND=json.dumps(server_command(settings)), ANJO_VLLM_ENV=json.dumps(server_env()),
          ANJO_VLLM_SECONDS=str(int(serve_s)), ANJO_VLLM_KEY_FILE=KEY_FILE)


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
    clock: Callable[[], float] = time.time, key: Callable[[], str] = new_key,
    port: Callable[[], int] = free_port, session: str | None = None,
    wait: Callable[[ColabSession], bool] = ready, sleep: Callable[[float], None] = time.sleep,
    keygen: KeyGen = ssh_keygen, key_dir: Path | None = None,
) -> Iterator[ColabSession]:
    """One Glimmer server on a Colab A100 on a throwaway key, held, health-checked, and stopped on
    the way out; the key is deleted last."""
    check_timeout(timeout_s)
    settings.pinned_revision()  # no server starts unpinned
    name = session or session_name()
    with session_key(key_dir, keygen) as identity:  # before anything is held or started
        reservation = None if ledger is None else hold(ledger, f"colab_serve:{name}", (POOL,),
                                                       Bound(A100_CU_PER_HOUR, timeout_s))
        started, vm = clock(), _Vm()
        tunnel: Tunnel | None = None
        handle: ColabSession | None = None
        try:
            secret = key()
            _boot(cli, name, settings, secret, timeout_s - (clock() - started), vm)
            local = port()
            tunnel = tunnels(name, local, config.PORT, timeout_s - (clock() - started), identity)
            handle = ColabSession(name, local, secret, cli, tunnel.transport)
            if not wait(handle):
                raise NotReady(f"Glimmer on {name} didn't answer through the tunnel in time")
            handle._state.deadline = started + timeout_s - STOP_TIMEOUT_S
            handle._state.tunnel, handle._state.clock, handle._state.sleep = tunnel, clock, sleep
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
