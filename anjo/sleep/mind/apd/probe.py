"""``probe-writer``: a few tiny writer calls on a core split's generator, to see the shape of its
replies before a life is written (MIND's data).

Each call is the writer's own (``life_v1.md``, at the writer's temperature, cap and reasoning
effort) on a first day with no beats. What is printed is the reply's shape, never its text: the
fields the message holds, the content's length and whether it carries think tags or a fence, the
reasoning field's length, whether the answer (``generators.answer``) is a JSON object and passes
the writer's own reading (``generate._read_day``), the finish reason, and the token counts as the
provider gave them. Live (``live``), the calls are held first on Fireworks' pool at their worst
case, under MIND's lock, and settled at what they cost; a call that fails costs its worst case.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from anjo.sleep import fireworks_api
from anjo.sleep.budget import Ledger
from anjo.sleep.mind.apd import gates, generate, retry
from anjo.sleep.mind.apd.generators import GENERATORS, Generator, answer
from anjo.sleep.mind.apd.judging import Rate
from anjo.sleep.mind.apd.types import LIFE_DAYS
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.mind.sweep import lock_path, mind_lock
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.holds import held, usd

MAX_PROBES: Final = 5
DEFAULT_PROBES: Final = 3
PROBED: Final = {name: found for name, found in GENERATORS.items() if found.model is not None}  # Fireworks' own
FIELD: Final = "independent film"
Call = Callable[..., ChatResult]  # ``fireworks_api.generate``'s signature


def messages(number: int) -> list[dict[str, Any]]:
    """Probe ``number``'s writer call: day 1 of a life, no earlier days, no beats."""
    payload = {"field": FIELD, "day": 1, "of": LIFE_DAYS, "date": generate.LIFE_START.isoformat(),
               "variation": number, "earlier": [], "plant": [], "ask": []}
    return [{"role": "system", "content": generate.LIFE_PROMPT.read_text(encoding="utf-8")},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _count(found: object) -> object:
    return found if isinstance(found, int) and not isinstance(found, bool) else None


def reply_shape(raw: Mapping[str, Any]) -> dict[str, Any]:
    """What a reply holds, none of its text."""
    choice = (raw.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    usage = raw.get("usage") or {}
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    shape: dict[str, Any] = {
        "model": raw.get("model"), "message_fields": sorted(message), "content_type": type(content).__name__,
        "content_chars": len(text), "think_tags": "<think>" in text, "fenced": "```" in text,
        "starts_with_brace": text.lstrip().startswith("{"),
        "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else None,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": _count(usage.get("prompt_tokens")), "completion_tokens": _count(usage.get("completion_tokens")),
        "total_tokens": _count(usage.get("total_tokens")),
        "reasoning_tokens": _count((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")),
    }
    return {**shape, **_answer_shape(answer(text))}


def _answer_shape(text: str) -> dict[str, Any]:
    try:
        body = gates.json_object(text)
    except gates.GateError:
        return {"answer_json": False, "answer_keys": None, "messages": None, "writer_reads": "not a JSON object"}
    messages_ = body.get("messages")
    try:
        generate._read_day(text, [])
        reads = "ok"
    except generate.GenerationFailed as failed:
        reads = f"{failed} (field {failed.field})"
    return {"answer_json": True, "answer_keys": sorted(body), "writer_reads": reads,
            "messages": len(messages_) if isinstance(messages_, list) else None}


def describe(number: int, shape: Mapping[str, Any]) -> str:
    return f"probe {number}: " + ", ".join(f"{name}={value}" for name, value in shape.items())


def run(generator: Generator, count: int, rate: Rate, out: Callable[[str], None], *,
        call: Call | None = None, costs: list[float] | None = None) -> float:
    """``count`` writer calls on ``generator``, each reply's shape printed; what they cost at
    ``rate``, each call counted at its worst case in ``costs`` until it returns. A call that fails
    is printed as its error, never retried."""
    if generator.model is None:
        raise ValueError("probe-writer probes a generator on Fireworks, not Spark Open")
    if not 1 <= count <= MAX_PROBES:
        raise ValueError(f"--n is 1 to {MAX_PROBES}")
    found = [] if costs is None else costs
    send: Call = call or fireworks_api.generate
    for number in range(1, count + 1):
        seen: list[Mapping[str, Any]] = []
        found.append(rate.worst)  # what it costs is unknown until it returns
        try:
            result = send(generator.model, messages(number), None, generate.TIMEOUT_S,
                          temperature=generate.WRITE_TEMPERATURE, max_tokens=retry.REPLY_MAX_TOKENS,
                          reasoning_effort=generator.reasoning_effort, on_reply=seen.append)
        except ModelApiError as error:
            out(f"probe {number}: failed: {error}")
            continue
        found[-1] = rate.usd(result.input_tokens, result.output_tokens)
        out(describe(number, {**reply_shape(seen[0] if seen else {}), "counted_out_tokens": result.output_tokens}))
    return sum(found)


def live(generator: Generator, count: int, rate: Rate, ledger: Ledger, out: Callable[[str], None], *,
         call: Call | None = None) -> float:
    """``run`` under MIND's lock (``MindBusy`` if another MIND command holds it), held on
    Fireworks' pool at ``count`` worst cases and settled at what the calls cost."""
    costs: list[float] = []
    key = f"apd:probe:writer:{generator.name}:{datetime.now(UTC):%Y%m%dT%H%M%S%f}"
    with mind_lock(lock_path(ledger, Path.cwd())), held(ledger, key, (FIREWORKS.pool,), usd(count * rate.worst),
                                                         lambda: sum(costs)):
        run(generator, count, rate, out, call=call, costs=costs)
    spent = sum(costs)
    out(f"Settled ${spent:.4f} on {FIREWORKS.pool}.")
    return spent


def plan_lines(generator: Generator, count: int, rate: Rate, pool: str) -> list[str]:
    worst = count * rate.worst
    return [f"Plan: probe-writer: {count} writer call(s) on {generator.name} ({generator.model}, reasoning effort "
            f"{generator.reasoning_effort}), day 1 with no beats.",
            f"Hold on {pool}: ${worst:.4f} ({count} x ${rate.worst:.4f}, each call at its cap), settled at what they "
            "cost.", "Prints each reply's shape, never its text.", "Nothing has run. Add --yes to start it."]
