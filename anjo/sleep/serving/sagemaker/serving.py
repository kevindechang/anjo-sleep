"""Glimmer on SageMaker behind the serving port (``serving.backend.Serving``): the dev host's side.

- ``wake`` sets the inference component's copy count to 1 if it is 0, then polls ``/v1/models``
  through the endpoint until vLLM answers, under the hard total ``settings.WAKE_TOTAL_TIMEOUT_S``.
  If the copy count drops back to 0 mid-wake (the idle alarm), it is set again, a few times at most.
- ``sleep_now`` sets the copy count to 0, so the instance scales away. Every phase calls it when it
  ends (``backend.phase``), well or not.
- ``client`` is the night's ``VllmClient`` over the endpoint's envelope.
- ``upload_adapter`` puts one checked adapter folder under ``adapters/<name>/`` in the bucket, never
  over one already there.

The AWS clients are made on first use, each with its timeouts set (``invoke.make_client``); tests
pass stand-ins.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from anjo.sleep.serving import client as vllm_client
from anjo.sleep.serving import config
from anjo.sleep.serving.client import ServingError, VllmClient, WakeResult
from anjo.sleep.serving.config import ServeConfigError
from anjo.sleep.serving.sagemaker import settings
from anjo.sleep.serving.sagemaker.invoke import (
    CONTROL_ATTEMPTS,
    CONTROL_READ_TIMEOUT_S,
    REFUSED,
    InvokeError,
    Invoker,
    SageMakerPoster,
    make_client,
)

_BUSY = frozenset({"Creating", "Updating"})
_TAKEN = frozenset({"PreconditionFailed", "ConditionalRequestConflict"})


class SageMakerServing:
    """One inference component on one endpoint. ``bucket`` is the adapters bucket (only an upload
    needs it); ``clock`` and ``sleep`` drive the waits."""

    name = "sagemaker"

    def __init__(
        self, *, bucket: str | None, control: Any = None, runtime: Any = None, ping_runtime: Any = None,
        s3: Any = None, clock: Callable[[], float] | None = None, sleep: Callable[[float], None] | None = None,
        endpoint: str = settings.ENDPOINT, component: str = settings.COMPONENT,
    ) -> None:
        self._bucket = bucket
        self._clients: dict[str, Any] = {
            k: v for k, v in (("control", control), ("runtime", runtime), ("ping", ping_runtime), ("s3", s3))
            if v is not None
        }
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._endpoint, self._component = endpoint, component

    # ── the clients, each with its timeouts ──────────────────────────────────

    def _made(self, key: str, make: Callable[[], Any]) -> Any:
        if key not in self._clients:
            self._clients[key] = make()
        return self._clients[key]

    def _control(self) -> Any:
        return self._made("control", lambda: make_client(
            "sagemaker", read_s=CONTROL_READ_TIMEOUT_S, attempts=CONTROL_ATTEMPTS))

    def _runtime(self) -> Any:  # one try: a retried 300 s request would double the wait
        return self._made("runtime", lambda: make_client(
            "sagemaker-runtime", read_s=vllm_client.REQUEST_TIMEOUT_S, attempts=1))

    def _ping_runtime(self) -> Any:
        return self._made("ping", lambda: make_client(
            "sagemaker-runtime", read_s=vllm_client.WAKE_POLL_TIMEOUT_S, attempts=1))

    def _s3(self) -> Any:
        return self._made("s3", lambda: make_client("s3", read_s=CONTROL_READ_TIMEOUT_S, attempts=CONTROL_ATTEMPTS))

    # ── the copy count ───────────────────────────────────────────────────────

    def _describe(self) -> dict[str, Any]:
        found: dict[str, Any] = self._control().describe_inference_component(InferenceComponentName=self._component)
        return found

    @staticmethod
    def _desired(found: dict[str, Any]) -> int:
        return int(found.get("RuntimeConfig", {}).get("DesiredCopyCount", 0))

    def _set_copies(self, count: int) -> None:
        self._control().update_inference_component_runtime_config(
            InferenceComponentName=self._component, DesiredRuntimeConfig={"CopyCount": count})

    # ── the port ─────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """One ``GET /v1/models`` through the endpoint: True only when vLLM answers. No capacity yet
        (SageMaker's 400, read as a 503) or a container still loading is False; a call the role may not make raises."""
        invoker = Invoker(self._ping_runtime(), self._endpoint, self._component)
        try:
            vllm_client.json_object("/v1/models", invoker.call("/v1/models", {}))
        except InvokeError as error:
            if error.status in REFUSED:
                raise ServingError(f"the dev host's role may not invoke the endpoint ({error.code})") from None
            return False
        except ServingError:
            return False  # a reply that isn't vLLM's yet
        return True

    def wake(self) -> WakeResult:
        clock, started = self._clock, self._clock()
        deadline = started + settings.WAKE_TOTAL_TIMEOUT_S
        if self._desired(self._describe()) == 0:
            self._set_copies(1)
        backoff, reasserts = vllm_client.WAKE_BACKOFF_START_S, 0
        while True:
            if self.ping():
                return WakeResult(clock() - started, True)
            left = deadline - clock()
            if left <= 0:
                return WakeResult(clock() - started, False)
            self._sleep(min(backoff, left))
            backoff = min(backoff * 2, vllm_client.WAKE_BACKOFF_MAX_S)
            if self._desired(self._describe()) == 0:  # the idle alarm took the copy back
                if reasserts >= settings.WAKE_MAX_REASSERTS:
                    return WakeResult(clock() - started, False)
                reasserts += 1
                self._set_copies(1)

    def sleep_now(self) -> None:
        """Set the copy count to 0 unless it is 0 already. An update in progress (a wake still
        provisioning) is waited out, up to ``settings.SLEEP_WAIT_S``; then the call is made anyway,
        and an error is raised, never swallowed (the idle alarm is the net behind it)."""
        waited = 0.0
        found = self._describe()
        while found.get("InferenceComponentStatus") in _BUSY and waited < settings.SLEEP_WAIT_S:
            self._sleep(settings.SLEEP_POLL_S)
            waited += settings.SLEEP_POLL_S
            found = self._describe()
        if self._desired(found) != 0:
            self._set_copies(0)

    def client(self) -> VllmClient:
        return VllmClient.over(SageMakerPoster(Invoker(self._runtime(), self._endpoint, self._component)))

    def upload_adapter(self, folder: Path, name: str) -> None:
        """Checked first, then refused if the name is taken, then written with ``IfNoneMatch``, so
        two uploads racing can't overwrite each other. The config goes last: it marks the folder whole."""
        config.check_adapter(folder, name)
        if not self._bucket:
            raise ServeConfigError(f"set {settings.BUCKET_ENV} to the adapters bucket")
        s3, prefix = self._s3(), f"{settings.ADAPTERS_PREFIX}{name}/"
        if s3.list_objects_v2(Bucket=self._bucket, Prefix=prefix, MaxKeys=1).get("KeyCount", 0):
            raise ServeConfigError(f"adapter {name} is already in the bucket; a night's adapter never changes")
        for file in reversed(config.ADAPTER_FILES):  # the weights, then the config
            with (folder / file).open("rb") as body:
                try:
                    s3.put_object(Bucket=self._bucket, Key=prefix + file, Body=body, IfNoneMatch="*")
                except ClientError as error:
                    taken = error.response.get("Error", {}).get("Code") in _TAKEN
                    if taken or error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 412:
                        raise ServeConfigError(f"adapter {name} is already in the bucket; a night's adapter never changes") from None
                    raise
