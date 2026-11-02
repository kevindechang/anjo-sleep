"""The paid Gemini client for the cognition sprint's Gemini rivals (``--set client=...:complete``).

Only the lead runs it, on synthetic lives only (the cog worlds make no real person), and only
under a run-level hold on ``vertex_mind_usd`` that the operator script takes before the run and
settles after it. Inside the run the spend is capped here, across every worker process: each call
first checks, under a file lock on the run's spend file, that what is spent plus this call's worst
case stays under the cap, and adds what it really cost when it returns. A call that would pass the
cap raises ``SpendCapReached``, which fails the run rather than scoring a half-answered life.

Settings come from the environment, so the eval CLI's worker processes inherit them:
``COG_VERTEX_SPEND`` (the spend file), ``COG_VERTEX_CAP_USD`` (the cap) and, optionally,
``COG_VERTEX_MODEL``. Each call is logged to ``<spend file>.calls.jsonl``: tokens in and out, the
finish reason and the seconds, never the prompt or the reply.
"""
from __future__ import annotations

import fcntl
import functools
import importlib
import json
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator

MODEL_ID: Final = "gemini-3.8-flash"  # the APD reader's and the second judge's model (verified 2026-10-27)
PRICE_IN_PER_M: Final = 0.75  # list, 2026-10-28 (mind.apd.readers.GEMINI_READER)
PRICE_OUT_PER_M: Final = 3.75
LOCATION: Final = "global"
THINKING: Final = "low"  # the floor 3.8 accepts on Vertex
MAX_OUTPUT_TOKENS: Final = 8192  # thoughts count inside it; a JSON of a day's probabilities is far less
TIMEOUT_S: Final = 180.0
WORST_CHARS_PER_TOKEN: Final = 3.0  # a call's hold assumes denser text than the 4 chars a token we expect
TRIES: Final = 6
BACKOFF_S: Final = (2.0, 5.0, 15.0, 30.0, 60.0)
TRANSIENT_CODES: Final = frozenset({408, 429, 500, 502, 503, 504})


class SpendCapReached(RuntimeError):
    """The next call could pass the run's cap."""


@dataclass(frozen=True)
class Settings:
    spend: Path
    cap_usd: float
    model: str = MODEL_ID

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        e = dict(os.environ) if env is None else env
        spend, cap = e.get("COG_VERTEX_SPEND", "").strip(), e.get("COG_VERTEX_CAP_USD", "").strip()
        if not spend or not cap:
            raise RuntimeError("set COG_VERTEX_SPEND and COG_VERTEX_CAP_USD (the run's spend file and cap)")
        cap_usd = float(cap)
        if not cap_usd > 0:
            raise RuntimeError("COG_VERTEX_CAP_USD must be positive")
        return cls(Path(spend), cap_usd, e.get("COG_VERTEX_MODEL", "").strip() or MODEL_ID)


def cost_usd(tokens_in: float, tokens_out: float) -> float:
    return (tokens_in * PRICE_IN_PER_M + tokens_out * PRICE_OUT_PER_M) / 1e6


def worst_case_usd(prompt: str) -> float:
    return cost_usd(len(prompt) / WORST_CHARS_PER_TOKEN, MAX_OUTPUT_TOKENS)


@contextmanager
def _locked(path: Path) -> Iterator[dict[str, float]]:
    """The spend totals under an exclusive lock; what the block leaves in the dict is written back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(path.suffix + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            totals = json.loads(path.read_text()) if path.exists() else {}
            state = {"usd": float(totals.get("usd", 0.0)), "held": float(totals.get("held", 0.0)),
                     "calls": float(totals.get("calls", 0.0)), "tokens_in": float(totals.get("tokens_in", 0.0)),
                     "tokens_out": float(totals.get("tokens_out", 0.0))}
            yield state
            tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(state))
            tmp.replace(path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def hold(settings: Settings, worst: float) -> None:
    with _locked(settings.spend) as s:
        if s["usd"] + s["held"] + worst > settings.cap_usd:
            raise SpendCapReached(f"spent ${s['usd']:.2f} + held ${s['held']:.2f} + this call's ${worst:.3f} "
                                  f"> cap ${settings.cap_usd:.2f}")
        s["held"] += worst


def settle(settings: Settings, worst: float, tokens_in: int, tokens_out: int) -> float:
    actual = cost_usd(tokens_in, tokens_out)
    with _locked(settings.spend) as s:
        s["held"] = max(0.0, s["held"] - worst)
        s["usd"] += actual
        s["calls"] += 1
        s["tokens_in"] += tokens_in
        s["tokens_out"] += tokens_out
    return actual


def _status(error: Exception) -> int | None:
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    return code if isinstance(code, int) else None


@functools.cache
def _client() -> Any:
    genai = importlib.import_module("google.genai")
    types = importlib.import_module("google.genai.types")
    return genai.Client(vertexai=True, location=LOCATION,
                        http_options=types.HttpOptions(timeout=int(TIMEOUT_S * 1000)))


def _generate(model: str, prompt: str) -> tuple[str, int, int, str]:
    types = importlib.import_module("google.genai.types")
    config = types.GenerateContentConfig(
        temperature=0.0, max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level=THINKING))
    reply = _client().models.generate_content(model=model, contents=prompt, config=config)
    usage = reply.usage_metadata
    out = int(usage.candidates_token_count or 0) + int(usage.thoughts_token_count or 0)  # both billed as output
    finish = str(reply.candidates[0].finish_reason) if reply.candidates else "none"
    return reply.text or "", int(usage.prompt_token_count or 0), out, finish


def make_complete(settings: Settings, generate: Callable[[str, str], tuple[str, int, int, str]] = _generate,
                  sleep: Callable[[float], None] = time.sleep) -> Callable[[str], str]:
    """``complete(prompt) -> text`` with the run's cap held per call and transient errors retried."""
    calls_log = settings.spend.with_suffix(settings.spend.suffix + ".calls.jsonl")

    def complete(prompt: str) -> str:
        worst = worst_case_usd(prompt)
        hold(settings, worst)
        started, tokens_in, tokens_out = time.monotonic(), 0, 0
        try:
            for attempt in range(TRIES):
                try:
                    text, tokens_in, tokens_out, finish = generate(settings.model, prompt)
                    break
                except Exception as error:  # noqa: BLE001 - the SDK's errors share no base class
                    if _status(error) not in TRANSIENT_CODES or attempt == TRIES - 1:
                        raise
                    sleep(BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)])
        except BaseException:
            settle(settings, worst, 0, 0)  # a failed call is billed nothing we can see; free its hold
            raise
        settle(settings, worst, tokens_in, tokens_out)
        with open(calls_log, "a") as log:
            log.write(json.dumps({"in": tokens_in, "out": tokens_out, "finish": finish,
                                  "s": round(time.monotonic() - started, 2)}) + "\n")
        return text

    return complete


def complete(prompt: str) -> str:
    """The CLI's entry point (``--set client=anjo.sleep.mind.cog.vertex_client:complete``)."""
    return _default()(prompt)


@functools.cache
def _default() -> Callable[[str], str]:
    return make_complete(Settings.from_env())
