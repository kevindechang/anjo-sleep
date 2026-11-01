"""The simulator's dev-API transport on the sleep host (Task 5).

``DevApiTransport`` opens the synthetic person's session as cleared (``POST /api/sleep/dev/sessions``)
and posts each line to the existing chat route (``POST /api/chat/<id>/message``), reading the
reply from its stream. A turn the server can't answer (an HTTP error or an ``error`` event, as
when Glimmer is down for a cleared session) raises ``TurnFailed``: the day ends short.

``simulate`` is what ``python -m anjo.sleep.night --day`` calls: one day per persona account,
with ``NewestReply`` so each line Anjo answers carries its recorded ancestry. The day's Spark Open
calls hold their worst case on ``baseten_sleep_usd`` first and settle the tokens they metered,
under the night lock (``night_lock``), so a night start never sweeps a live day's hold.
Glimmer is woken first (``serving.backend``); if it can't be, the day does not start. The day
ends with Glimmer put back to sleep.

Before any of that, the token is checked against the day's account: ``GET /api/billing/config``, an
authenticated read with no side effects, echoes the caller's user id. A token that isn't ``user_id``'s,
or can't be checked, refuses the day (``TokenRefused``): no hold, no wake, no line posted as anyone.

**Several lives in one process** (the final test's batch, ``sim.accelerated --user A --user B ...``):
``ANJO_SLEEP_SIM_TOKEN_FILES`` maps each account to the file holding its own token
(``account=path,account=path``, the files ``obj_accounts.py`` writes), so each day posts as its own
account (``sim_token``). Paths only: no token sits in the environment or a command line. An account
the map doesn't name uses ``ANJO_SLEEP_SIM_TOKEN``, as one life always has. No message names a
token, an account or a token file. The final batch checks every life's token against its account
(``check_day``) before any life's first day, so one bad token can't stop the run after another life's
day posted.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

import httpx

from anjo.sleep.app.ancestry import NewestReply
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.storage import SLEEP_DIR, data_root
from anjo.sleep.budget import LEDGER_NAME, Ledger
from anjo.sleep.night.holds import held, spark_usd, spark_worst_usd, usd
from anjo.sleep.night.sweep import night_lock
from anjo.sleep.serving import backend
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.sim.day import DayRun, SparkSim, TurnFailed, run_day
from anjo.sleep.sim.persona_sim import MAX_TURNS

DEV_URL_ENV: Final = "ANJO_SLEEP_DEV_URL"  # the sleep host's own address, e.g. http://127.0.0.1:8000
SIM_TOKEN_ENV: Final = "ANJO_SLEEP_SIM_TOKEN"  # a signed-in token for the synthetic account
SIM_TOKEN_FILES_ENV: Final = "ANJO_SLEEP_SIM_TOKEN_FILES"  # account=path,...: each account's own token file
TURN_TIMEOUT_S: Final = 180.0
DEFAULT_RETRY_AFTER_S: Final = 1.0  # a busy refusal without a usable ``Retry-After``
MIN_RETRY_AFTER_S: Final = 0.1  # so the total wait always grows and the loop ends
MAX_RETRY_AFTER_S: Final = 5.0  # one wait never exceeds this, whatever the server asks
MAX_BUSY_WAIT_S: Final = 30.0  # all the waiting one turn does before it fails as busy
WHOAMI: Final = "/api/billing/config"  # an authenticated read that echoes the caller's user id


class TokenRefused(RuntimeError):
    """The sim token isn't the day's account's, or couldn't be checked: the day doesn't start.
    Its message names neither the token nor any account."""


def reply_text(stream: str) -> str:
    """The reply in an SSE body: its ``token`` events in order. An ``error`` event fails the turn."""
    text: list[str] = []
    for block in stream.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        event = fields.get("event")
        if event == "error":
            raise TurnFailed("the chat route sent an error event", reason="error_event")
        if event == "token":
            text.append(str(json.loads(fields.get("data", "{}")).get("text", "")))
    reply = "".join(text).strip()
    if not reply:
        raise TurnFailed("the chat route sent no reply", reason="no_reply")
    return reply


BUSY: Final = "busy: a response is already in progress"
STALE: Final = "stale session"
BUSY_CODE: Final = "busy"  # the reason a turn that stayed busy past its bound carries


def _body(response: httpx.Response) -> dict[str, object]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _refusal(response: httpx.Response) -> str:
    """What a 409 is, never its words: the busy lease, a keyed refusal's ``code``, or a stale session."""
    body = _body(response)
    if body.get("retry") is True:
        return BUSY
    code = body.get("code")
    return code if isinstance(code, str) and code.isidentifier() else STALE


def _retry_after(response: httpx.Response) -> float:
    try:
        asked = float(response.headers.get("Retry-After", DEFAULT_RETRY_AFTER_S))
    except ValueError:
        asked = DEFAULT_RETRY_AFTER_S
    return min(max(asked, MIN_RETRY_AFTER_S), MAX_RETRY_AFTER_S) if asked == asked else DEFAULT_RETRY_AFTER_S


class DevApiTransport:
    """``persona_sim.ChatTransport`` over the dev API, every session cleared."""

    def __init__(self, base_url: str, token: str, *, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {token}"},
                                  transport=transport, timeout=TURN_TIMEOUT_S)
        self._session: str | None = None
        self._sleep = sleep

    def _post(self, path: str, body: dict[str, object]) -> httpx.Response:
        waited = 0.0
        while True:
            response = self._post_once(path, body)
            if response.status_code != 409:
                break
            refusal = _refusal(response)
            if refusal != BUSY:
                raise TurnFailed(f"{path} returned HTTP 409 ({refusal})", reason=refusal)
            wait = _retry_after(response)
            if waited + wait > MAX_BUSY_WAIT_S:
                raise TurnFailed(f"{path} returned HTTP 409 ({BUSY}) for {waited:g}s", reason=BUSY_CODE)
            self._sleep(wait)
            waited += wait
        if response.is_error:
            raise TurnFailed(f"{path} returned HTTP {response.status_code}", reason=f"http_{response.status_code}")
        return response

    def _post_once(self, path: str, body: dict[str, object]) -> httpx.Response:
        try:
            return self._http.post(path, json=body)
        except httpx.HTTPError as error:
            raise TurnFailed(f"{path} failed in transport ({type(error).__name__})", reason=type(error).__name__) from None

    def check_account(self, user_id: str) -> None:
        """Refuse a token that isn't ``user_id``'s before any line is posted; fail closed."""
        try:
            response = self._http.get(WHOAMI)
        except httpx.HTTPError as error:
            raise TokenRefused(f"the sim token couldn't be checked ({type(error).__name__})") from None
        if response.is_error:
            raise TokenRefused(f"the sim token couldn't be checked: {WHOAMI} returned HTTP {response.status_code}")
        if _body(response).get("user_id") != user_id:
            raise TokenRefused("the sim token belongs to another account than the day's: no line is posted")

    def close(self) -> None:
        self._http.close()

    def send(self, session_id: str, text: str, *, cleared: bool) -> str:
        if not cleared:
            raise ValueError("the simulator's sessions are cleared")
        if self._session is None:
            self._session = str(self._post("/api/sleep/dev/sessions", {"cleared": True}).json()["session_id"])
        return reply_text(self._post(f"/api/chat/{self._session}/message", {"text": text}).text)


def _token_files(text: str) -> dict[str, str]:
    """``account=path,account=path`` as a map; anything else is refused, quoting none of it."""
    named: dict[str, str] = {}
    for part in (p.strip() for p in text.split(",") if p.strip()):
        account, _, path = part.partition("=")
        if not account.strip() or not path.strip() or account.strip() in named:
            raise TokenRefused(f"{SIM_TOKEN_FILES_ENV} is account=path pairs, comma-separated, each account once")
        named[account.strip()] = path.strip()
    return named


def sim_token(user_id: str) -> str:
    """The token ``user_id``'s day posts with: the file ``ANJO_SLEEP_SIM_TOKEN_FILES`` names for the
    account, else ``ANJO_SLEEP_SIM_TOKEN`` (one life's run, unchanged). ``check_account`` still checks it."""
    files = os.environ.get(SIM_TOKEN_FILES_ENV, "").strip()
    path = _token_files(files).get(user_id) if files else None
    if path is None:
        return os.environ[SIM_TOKEN_ENV]
    try:
        token = Path(path).read_text(encoding="utf-8").strip()
    except OSError as error:
        raise TokenRefused(f"the day's account's sim token file can't be read ({type(error).__name__})") from None
    if not token:
        raise TokenRefused("the day's account's sim token file is empty")
    return token


def check_day(user_id: str, *, transport: httpx.BaseTransport | None = None) -> None:
    """``user_id``'s sim token, read and checked against its account as its day would (``TokenRefused``:
    a token file missing, empty or another account's), posting nothing and holding nothing."""
    require_sleep_host()
    checking = DevApiTransport(os.environ[DEV_URL_ENV], sim_token(user_id), transport=transport)
    try:
        checking.check_account(user_id)
    finally:
        checking.close()


def simulate(user_id: str, persona: str, day_index: int, seed: int) -> DayRun:
    require_sleep_host()
    transport = DevApiTransport(os.environ[DEV_URL_ENV], sim_token(user_id))
    transport.check_account(user_id)  # before the hold and the wake: a wrong pairing posts nothing
    from anjo.sleep.app.night import PERSONAS, prices

    person = load_person_file(PERSONAS / f"{persona}.json").visible
    model, price = SparkSim(), prices()
    ledger = Ledger(data_root() / SLEEP_DIR / LEDGER_NAME)
    with night_lock(ledger), held(ledger, f"sim:{user_id}:d{day_index}:s{seed}", ("baseten_sleep_usd",),
                                  usd(spark_worst_usd(price, MAX_TURNS)), lambda: spark_usd(price, *model.tokens)):
        serving = backend.from_env()  # SageMaker, or Modal (ANJO_SLEEP_BACKEND)
        return run_day(person, day_index, seed, model=model, transport=transport, replies=NewestReply(user_id),
                       wake=serving.wake, sleep=serving.sleep_now)
