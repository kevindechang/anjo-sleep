"""Which server Glimmer runs on: SageMaker (the default) or Modal (the fallback), from
``ANJO_SLEEP_BACKEND``.

Both answer the same port, ``Serving``: ``wake`` (poll until vLLM answers, under a hard total),
``ping`` (one cheap request), ``sleep_now`` (stop paying now), ``client`` (the night's ``VllmClient``)
and ``upload_adapter`` (one checked adapter, never over one that exists). On Modal, ``sleep_now``
does nothing: Modal scales to zero by itself after ``config.SCALEDOWN_S`` idle. On SageMaker it sets
the copy count to 0. Modal's client stages an adapter its container's mount can't show
(``stage_on_modal``), since the container never reloads the adapters volume.

**Every phase ends asleep.** A simulated day, a night, a morning run and a keep-warm window each
run inside ``phase``, which calls the phase's ``sleep`` on the way out, after an error too. Phases
nest: the daily night runs the night and then the morning in one process, so Glimmer stays up
between them and goes to sleep once, when the outermost phase ends.

The SageMaker side imports boto3, so it is imported only when chosen.
"""
from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol

from anjo.sleep.serving import client
from anjo.sleep.serving.client import ServingError, VllmClient, WakeResult
from anjo.sleep.serving.config import ADAPTERS_DIR, APP_NAME, STAGED_ADAPTERS_DIR, ServeConfigError

if TYPE_CHECKING:
    from anjo.sleep.serving.sagemaker.serving import SageMakerServing

BACKEND_ENV: Final = "ANJO_SLEEP_BACKEND"
BACKENDS: Final = ("sagemaker", "modal")
DEFAULT_BACKEND: Final = "sagemaker"
MODAL_URL_ENV: Final = "ANJO_SLEEP_MODAL_URL"
MODAL_TOKEN_ENV: Final = "ANJO_SLEEP_MODAL_TOKEN"
STAGE_TIMEOUT_S: Final = 120.0  # one adapter copied from the volume onto the container's disk, queue included

Sleep = Callable[[], None]


class Serving(Protocol):
    @property
    def name(self) -> str: ...

    def wake(self) -> WakeResult: ...

    def ping(self) -> bool: ...

    def sleep_now(self) -> None: ...

    def client(self) -> VllmClient: ...

    def upload_adapter(self, folder: Path, name: str) -> None: ...


@dataclass(frozen=True)
class ModalServing:
    """Glimmer on Modal, as it ran before SageMaker: the bearer-token poll, the https server, and
    Modal's own idle window for sleeping."""

    url: str
    token: str = field(repr=False)
    name: ClassVar[str] = "modal"

    def wake(self) -> WakeResult:
        return client.wake(self.url, self.token, client.WAKE_TOTAL_TIMEOUT_S)

    def ping(self) -> bool:
        return client.ping(self.url, self.token)

    def sleep_now(self) -> None:
        """Nothing: Modal scales the container down after ``config.SCALEDOWN_S`` idle."""

    def client(self) -> VllmClient:
        """A client that loads each adapter from the adapters volume before its first use, staging one
        committed since the container started."""
        return VllmClient(self.url, self.token, adapters_dir=ADAPTERS_DIR, stage=stage_on_modal)

    def upload_adapter(self, folder: Path, name: str) -> None:
        from anjo.sleep.serving.modal_app import put_adapter  # needs the modal package: imported only to upload

        put_adapter(folder, name)


def stage_on_modal(name: str) -> str | None:
    """Copy adapter ``name`` onto the serving container's own disk with the deployed app's
    ``Serve.stage`` (``modal_app``), and name the folder it lands in; None when the adapters volume has
    no such adapter. Imports the modal package, as training does, so only a client that stages needs
    it. Any failure, a wait past ``STAGE_TIMEOUT_S`` included, is ``ServingError`` naming its type only."""
    try:
        modal: Any = importlib.import_module("modal")
        staged = modal.Cls.from_name(APP_NAME, "Serve")().stage.spawn(name).get(timeout=STAGE_TIMEOUT_S)
    except Exception as error:
        raise ServingError(f"staging an adapter on Modal failed ({type(error).__name__})") from None
    return STAGED_ADAPTERS_DIR if staged else None


def backend_name(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    chosen = env.get(BACKEND_ENV, "").strip() or DEFAULT_BACKEND
    if chosen not in BACKENDS:
        raise ServeConfigError(f"{BACKEND_ENV} is sagemaker or modal")
    return chosen


def _set(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ServeConfigError(f"{name} is not set")
    return value


def from_env(environ: Mapping[str, str] | None = None) -> Serving:
    """The chosen backend, from the process environment unless a mapping is given."""
    env = os.environ if environ is None else environ
    if backend_name(env) == "modal":
        return ModalServing(_set(env, MODAL_URL_ENV), _set(env, MODAL_TOKEN_ENV))
    return _sagemaker(env)


def _sagemaker(env: Mapping[str, str]) -> SageMakerServing:
    from anjo.sleep.serving.sagemaker import settings
    from anjo.sleep.serving.sagemaker.serving import SageMakerServing

    bucket = env.get(settings.BUCKET_ENV, "").strip() or None
    if bucket is not None:
        try:
            settings.check_bucket(bucket)
        except ValueError as error:
            raise ServeConfigError(f"{settings.BUCKET_ENV}: {error}") from None
    return SageMakerServing(bucket=bucket)


class _Phases(threading.local):
    def __init__(self) -> None:
        self.depth = 0
        self.pending: list[Sleep] = []


_phases = _Phases()


@contextmanager
def phase(sleep: Sleep | None = None) -> Iterator[None]:
    """Run a phase, then put Glimmer to sleep with ``sleep`` (None: a phase that only groups the
    ones inside it). Inside another phase, the sleep waits for the outermost one to end. Each sleep
    runs once, even when another raises; an error from one is raised, never swallowed."""
    _phases.depth += 1
    if sleep is not None and sleep not in _phases.pending:
        _phases.pending.append(sleep)
    try:
        yield
    finally:
        _phases.depth -= 1
        if _phases.depth == 0:
            due, _phases.pending = _phases.pending, []
            with ExitStack() as sleeps:
                for each in reversed(due):  # ExitStack runs them last in, first out
                    sleeps.callback(each)
