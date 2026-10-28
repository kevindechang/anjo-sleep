"""``probe-writer``: a few tiny writer calls on a core split's generator, to see the shape of its
replies before a life is written (MIND's data).

Each call is the writer's own (``generate.writer_messages``, at the writer's temperature, cap and
reasoning effort) on a first day: an odd call with no beats, an even one with two (``BEATS``), so
a placement the writer gets wrong (counted from 1, say) shows. What is printed is the reply's
shape, never its text: the fields the message holds, the content's length and whether it carries
think tags or a fence, the reasoning field's length, whether an answer was taken from it or
refused by a rule (``generators.answer``), whether that answer is a JSON object and passes the
writer's own reading (``generate._read_day``), where its beats landed (``placement``: counts
only), the finish reason, and the token counts as the provider gave them. Live (``live``), the calls are held first on Fireworks' pool at their worst
case, under MIND's lock, and settled at what they cost; a call that fails costs its worst case.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from anjo.sleep import fireworks_api
from anjo.sleep.budget import Ledger
from anjo.sleep.mind.apd import gates, generate, retry
from anjo.sleep.mind.apd.generate import Beat, GenerationFailed
from anjo.sleep.mind.apd.generators import GENERATORS, Generator, Refused, answer
from anjo.sleep.mind.apd.judging import Rate
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.mind.sweep import lock_path, mind_lock
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.holds import held, usd

MAX_PROBES: Final = 5
DEFAULT_PROBES: Final = 3
PROBED: Final = {name: found for name, found in GENERATORS.items() if found.model is not None}  # Fireworks' own
FIELD: Final = "independent film"
Call = Callable[..., ChatResult]  # ``fireworks_api.generate``'s signature


BEATS: Final = (Beat("i1.o1", "the screening moved to Friday because the venue was double-booked", "plant"),
                Beat("i2.s", "whether to send the rough cut to the festival before the sound mix is done", "ask"))
MARKS: Final = {"i1.o1": "friday", "i2.s": "festival"}  # a word of each beat that its message holds


def beats_for(number: int) -> tuple[Beat, ...]:
    """Probe ``number``'s beats: two on an even call, none on an odd one."""
    return BEATS if number % 2 == 0 else ()


def messages(number: int, model: str) -> list[dict[str, Any]]:
    """Probe ``number``'s writer call to ``model``: day 1 of a life, no earlier days."""
    return generate.writer_messages(FIELD, 0, generate.LIFE_START, number, [], beats_for(number), model=model)


def _count(found: object) -> object:
    return found if isinstance(found, int) and not isinstance(found, bool) else None


def reply_shape(raw: Mapping[str, Any], beats: Sequence[Beat] = ()) -> dict[str, Any]:
    """What a reply to a day with ``beats`` holds, none of its text."""
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
    return {**shape, "beats": len(beats), **_answer_shape(text, choice.get("finish_reason"), beats)}


def _answer_shape(text: str, finish: object, beats: Sequence[Beat]) -> dict[str, Any]:
    unread: dict[str, Any] = {"answer_json": None, "answer_keys": None, "messages": None, **placement(None, beats)}
    try:
        found = answer(text, finish if isinstance(finish, str) else None)
    except Refused as refused:
        return {"answer": f"refused ({refused.rule})", **unread, "writer_reads": "refused"}
    try:
        body = gates.json_object(found)
    except gates.GateError:
        return {"answer": "ok", **unread, "answer_json": False, "writer_reads": "not a JSON object"}
    try:
        generate._read_day(found, beats, nothing_placed=True)  # as the writer reads a core split's generator
        reads = "ok"
    except GenerationFailed as failed:
        reads = f"{failed} (field {failed.field})"
    texts = body.get("messages")
    return {"answer": "ok", "answer_json": True, "answer_keys": sorted(body),
            "messages": len(texts) if isinstance(texts, list) else None, **placement(body, beats), "writer_reads": reads}


def placement(body: Mapping[str, Any] | None, beats: Sequence[Beat]) -> dict[str, Any]:
    """Where a day's beats landed, as counts: whether ``placed`` is valid (each beat once, at a
    message counted from 0), how many beats the message ``placed`` names holds, and how many the
    message before it does (a writer counting from 1). ``None`` with no beats or no answer."""
    if not beats or body is None:
        return {"placed_valid": None, "beats_at_placed": None, "beats_one_before_placed": None}
    texts, placed = body.get("messages"), body.get("placed")
    if not isinstance(texts, list) or not isinstance(placed, dict):
        return {"placed_valid": False, "beats_at_placed": 0, "beats_one_before_placed": 0}
    try:
        generate._check_placed(placed, beats, len(texts))
        valid = True
    except GenerationFailed:
        valid = False
    spots = [(placed.get(beat.id), MARKS[beat.id]) for beat in beats]
    return {"placed_valid": valid, "beats_at_placed": sum(_holds(texts, spot, mark) for spot, mark in spots),
            "beats_one_before_placed": sum(_holds(texts, spot, mark, back=1) for spot, mark in spots)}


def _holds(texts: Sequence[object], spot: object, mark: str, *, back: int = 0) -> bool:
    if not isinstance(spot, int) or isinstance(spot, bool) or not 0 <= spot - back < len(texts):
        return False
    text = texts[spot - back]
    return isinstance(text, str) and mark in text.lower()


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
            result = send(generator.model, messages(number, generator.model), None, generate.TIMEOUT_S,
                          temperature=generate.WRITE_TEMPERATURE, max_tokens=retry.REPLY_MAX_TOKENS,
                          reasoning_effort=generator.reasoning_effort, on_reply=seen.append)
        except ModelApiError as error:
            out(f"probe {number}: failed: {error}")
            continue
        found[-1] = rate.usd(result.input_tokens, result.output_tokens)
        shape = reply_shape(seen[0] if seen else {}, beats_for(number))
        out(describe(number, {**shape, "counted_out_tokens": result.output_tokens}))
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
            f"{generator.reasoning_effort}), each a first day: the odd ones with no beats, the even ones with two beats.",
            f"Hold on {pool}: ${worst:.4f} ({count} x ${rate.worst:.4f}, each call at its cap), settled at what they "
            "cost.", "Prints each reply's shape, never its text.", "Nothing has run. Add --yes to start it."]
