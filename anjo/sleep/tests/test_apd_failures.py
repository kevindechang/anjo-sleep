# mypy: disable-error-code="untyped-decorator"
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Why each generation attempt failed, a reasoning generator's reply shapes, and ``probe-writer``
(MIND's data): every failed attempt recorded with no text, the writer reading the answer and not
the reasoning, and the probe printing a reply's shape, never its words."""
from __future__ import annotations

import dataclasses
import json
import re
from functools import partial
from pathlib import Path
from typing import Any

import httpx
import pytest

from anjo.sleep import fireworks_api
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import calls, core_splits, fakes, gen_batch, generate, generators, pairs, probe
from anjo.sleep.mind.apd.backend import Halted
from anjo.sleep.mind.apd.live import RunsFailed
from anjo.sleep.mind.apd.ceiling import RunLedger
from anjo.sleep.mind.apd.judging import Rate
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.model_api import ChatResult
from anjo.sleep.stops import BudgetStop

DEEPSEEK = generators.DEEPSEEK
RATE = Rate(0.30, 1.20)
SPEC = next(spec for spec in core_splits.core_specs("core-train") if spec.prevalence > 0)
RECORD_KEYS = {"what", "name", "attempt", "step", "error", "reason", "day", "item", "field", "rule", "reply_chars",
               "answer_chars", "finish_reason", "rate_limits", "waits_s", "killed", "missing", "extra"}
BODY = '{"messages": ["a"], "placed": {}}'
DRAFT = '{"messages": ["a draft"], "placed": {}}'


class Recorded:
    """``fakes.generator_transport(shape)``, keeping each reply's body and the fake's whole answer
    to it (what the model wrote, even where its reply was cut)."""

    def __init__(self, shape: str) -> None:
        self.inner = fakes.generator_transport(shape)
        self.bodies: list[dict[str, Any]] = []
        self.answers: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        response = self.inner.handle_request(request)
        response.read()
        if response.status_code == 200:
            self.bodies.append(response.json())
            sent = json.loads(request.content)
            self.requests.append(sent)
            self.answers.append(json.loads(fakes.FakeLifeChat()(sent["model"], sent["messages"], None, 30.0).text))
        return response

    def texts(self) -> list[str]:
        """Every message the writer wrote and every item the extractor wrote, whole."""
        return [*(text for answer in self.answers for text in answer.get("messages", [])),
                *(item["text"] for answer in self.answers for item in answer.get("items", []))]


def _backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str) -> tuple[backends.Backend, Recorded]:
    """The fakes' backend with DeepSeek writing through the real ``gen_chat`` and Fireworks client,
    answered by the fake transport in ``shape``."""
    recorded = Recorded(shape)
    monkeypatch.setenv(fireworks_api.KEY_ENV, "fw-test")
    monkeypatch.setattr(fireworks_api, "generate", partial(fireworks_api.generate, transport=recorded.transport))
    backend = dataclasses.replace(backends.fake_backend(FIREWORKS), life_chat=backends.gen_chat(FIREWORKS),
                                  ledger=RunLedger(tmp_path / "budget.jsonl"), prices=cli.FAKE_PRICES,
                                  rates=cli.FAKE_RATES, generator=DEEPSEEK, generator_rate=RATE)
    return backend, recorded


def _logged(data: DataDir, made_id: str) -> list[calls.Call]:
    return [calls.Call.from_json(raw) for raw in data.call_logs()[made_id]]


def _no_text(texts: list[str], *places: str) -> None:
    assert texts, "the fake wrote nothing to look for"
    for text in texts:
        for place in places:
            assert text not in place


# ── The writer reads the answer ───────────────────────────────────────────────


def test_answer_takes_the_reasoning_and_the_one_fence_out() -> None:
    assert generators.answer(BODY) == BODY
    assert generators.answer(f"<think>\nfirst {{this}}\n</think>\n\n{BODY}", "stop") == BODY
    assert generators.answer(f"Here it is:\n```json\n{BODY}\n```\nDone.") == BODY
    assert generators.answer(f"<think>plan</think>\n```\n{BODY}\n```") == BODY
    # a draft fenced inside closed reasoning is not a fence of the answer's
    assert generators.answer(f"<think>a draft:\n```json\n{DRAFT}\n```\n</think>\n```json\n{BODY}\n```") == BODY
    # reasoning whose opening tag was the chat template's: the answer is what follows its close
    assert generators.answer(f"a draft:\n```json\n{DRAFT}\n```\n</think>\n\n{BODY}") == BODY
    fenced_inside = '{"messages": ["```json\\n{}\\n```"], "placed": {}}'
    assert generators.answer(fenced_inside) == fenced_inside  # an answer already is unchanged


@pytest.mark.parametrize(("text", "finish", "rule"), [
    (BODY, "length", "cut"),  # a reply cut at its cap is never a whole answer, even when it parses
    (f"<think>a draft:\n```json\n{DRAFT}\n```\nnow chec", "length", "cut"),
    (f"<think>a draft:\n```json\n{DRAFT}\n```\n", "stop", "open_think"),
    ("<think>never closed {", None, "open_think"),
    (f"<think>x</think><think>y {BODY}", "stop", "open_think"),
    (f"A draft:\n```json\n{DRAFT}\n```\nThe answer:\n```json\n{BODY}\n```", "stop", "fences"),
])
def test_answer_refuses_a_cut_reply_open_reasoning_and_two_fences(text: str, finish: str | None, rule: str) -> None:
    with pytest.raises(generators.Refused) as refused:
        generators.answer(text, finish)
    assert refused.value.rule == rule and DRAFT not in str(refused.value) and BODY not in str(refused.value)


@pytest.mark.parametrize(("shape", "reasoning"), [("plain", 0), ("think", 40), ("fenced", 0), ("reasoning_field", 40)])
def test_a_readable_reply_shape_writes_the_same_life_and_counts_its_reasoning(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str, reasoning: int) -> None:
    # Arrange: the same life written by plain replies, then by replies in this shape
    plain_backend, _ = _backend(tmp_path / "plain", monkeypatch, "plain")
    plain = gen_batch.generate_life(SPEC, plain_backend, plain_log := calls.CallLog())
    backend, _ = _backend(tmp_path / shape, monkeypatch, shape)
    log = calls.CallLog()

    # Act
    made = gen_batch.generate_life(SPEC, backend, log)

    # Assert: the answer was read, never the reasoning, and the reasoning was paid for as output
    assert made.life == plain.life and made.problems == plain.problems and made.attempts == 1
    written = [(c.kind, c.tokens_out) for c in log.calls if c.kind in ("write", "extract")]
    expected = [(c.kind, c.tokens_out + reasoning) for c in plain_log.calls if c.kind in ("write", "extract")]
    assert written == expected and not log.failures


@pytest.mark.parametrize(("shape", "rule", "finish"), [
    ("length", "cut", "length"), ("draft_in_cut_think", "cut", "length"), ("open_think", "open_think", "stop"),
    ("two_fences", "fences", "stop"),
])
def test_a_drafted_or_cut_answer_is_never_read_as_the_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         shape: str, rule: str, finish: str) -> None:
    # Arrange: every reply holds a whole day's JSON, but as a draft or cut at its cap
    backend, recorded = _backend(tmp_path, monkeypatch, shape)
    data, lines = DataDir(tmp_path / "pilot"), list[str]()

    # Act
    with pytest.raises(RunsFailed):
        gen_batch.generate_lives(data, [SPEC], backend, lines.append)

    # Assert: each attempt refused at the writer's first day, by its rule, and nothing was saved
    attempts = [raw for raw in data.failed() if raw["what"] == "generate-attempt"]
    assert len(attempts) == 3 and not data.lives("core-train")
    for raw in attempts:
        assert (raw["step"], raw["day"], raw["field"], raw["rule"], raw["finish_reason"]) == (
            "writer", 1, "reply", rule, finish)
        assert raw["reply_chars"] > 0 and raw["answer_chars"] is None
    assert all(f"[rule {rule}]" in line for line in lines if line.startswith("FAILED "))
    _no_text(recorded.texts(), (data.root / "failed.jsonl").read_text(encoding="utf-8"), "\n".join(lines))


# ── A day with nothing to place ───────────────────────────────────────────────


def test_a_generators_day_with_no_beats_may_leave_placed_out() -> None:
    beat = generate.Beat("i1.o1", "a fact", "plant")
    unplaced = '{"messages": ["a", "b"]}'
    # a core split's generator: a day with no beats reads a missing or empty ``placed`` as nothing placed
    assert generate._read_day(unplaced, [], nothing_placed=True) == (["a", "b"], {})
    for empty in ("null", "[]", "{}"):
        assert generate._read_day(f'{{"messages": ["a"], "placed": {empty}}}', [], nothing_placed=True) == (["a"], {})
    # a day with beats still needs them placed, as before
    for text in (unplaced, '{"messages": ["a"], "placed": []}', '{"messages": ["a"], "placed": null}'):
        with pytest.raises(generate.GenerationFailed, match="where its beats are") as failed:
            generate._read_day(text, [beat], nothing_placed=True)
        assert failed.value.field == "placed"
    with pytest.raises(generate.GenerationFailed, match="unplaced or unknown"):
        generate._read_day('{"messages": ["a"], "placed": {}}', [beat], nothing_placed=True)
    # Spark Open's reading (APD's own lives) is unchanged
    with pytest.raises(generate.GenerationFailed, match="where its beats are"):
        generate._read_day(unplaced, [])


@pytest.mark.parametrize(("shape", "generated"), [("unplaced_when_empty", True), ("never_placed", False)])
def test_a_generator_leaving_placed_out_of_a_beatless_day_still_writes_the_life(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str, generated: bool) -> None:
    backend, _ = _backend(tmp_path, monkeypatch, shape)
    data = DataDir(tmp_path / "pilot")

    if generated:
        gen_batch.generate_lives(data, [SPEC], backend, lambda line: None)
        assert len(data.lives("core-train")) == 1 and not data.failed()
        return
    with pytest.raises(RunsFailed):
        gen_batch.generate_lives(data, [SPEC], backend, lambda line: None)
    attempts = [raw for raw in data.failed() if raw["what"] == "generate-attempt"]
    assert len(attempts) == 3 and {(raw["step"], raw["field"]) for raw in attempts} == {("writer", "placed")}
    assert all(raw["reason"] == "a day doesn't say where its beats are" for raw in attempts)


def test_only_a_generators_writer_is_told_placed_is_always_there() -> None:
    def system(model: str) -> str:
        return str(generate.writer_messages("film", 0, generate.LIFE_START, 1, [], [], model=model)[0]["content"])

    prompt = generate.LIFE_PROMPT.read_text(encoding="utf-8")
    assert system(FIREWORKS.model) == prompt  # Spark Open's prompt, APD's lives' own, is unchanged
    assert DEEPSEEK.model is not None
    told = system(DEEPSEEK.model)
    assert told.startswith(prompt) and told[len(prompt):].strip() == generate.ALWAYS_PLACED
    assert len(generate.ALWAYS_PLACED.splitlines()) == 1 and "`placed`" in generate.ALWAYS_PLACED


# ── Each failed attempt, recorded ─────────────────────────────────────────────


def test_a_reply_cut_at_its_cap_fails_each_attempt_at_the_writer_and_says_so(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: every reply cut at its cap (finish length, the JSON unfinished)
    backend, recorded = _backend(tmp_path, monkeypatch, "length")
    data, lines = DataDir(tmp_path / "pilot"), list[str]()

    # Act
    with pytest.raises(RunsFailed):
        gen_batch.generate_lives(data, [SPEC], backend, lines.append)

    # Assert: three attempts, each failed at the writer's first day for its reply, cut at length
    attempts = [raw for raw in data.failed() if raw["what"] == "generate-attempt"]
    assert [raw["attempt"] for raw in attempts] == [0, 1, 2]
    for raw in attempts:
        assert set(raw) == RECORD_KEYS
        assert (raw["step"], raw["day"], raw["item"], raw["field"], raw["finish_reason"]) == (
            "writer", 1, None, "reply", "length")
        assert raw["error"] == "GenerationFailed" and raw["reply_chars"] > 0 and raw["rule"] == "cut"
        assert raw["reason"] == "a reply cut at its cap"
    assert [raw["what"] for raw in data.failed()] == ["generate-attempt"] * 3 + ["generate"]
    failed = [line for line in lines if line.startswith("FAILED ")]
    assert len(failed) == 3 and all("at the writer (day 1, field reply)" in line for line in failed)
    assert all("finish length" in line for line in failed)
    assert lines.index(failed[-1]) < next(i for i, line in enumerate(lines) if line.startswith("SKIPPED "))

    # The skipped life's calls are kept too, each reply's length and finish reason with them
    (made_id,) = data.call_logs()
    assert made_id.endswith(".failed-1")
    logged = _logged(data, made_id)
    assert [c.attempt for c in logged] == [0, 1, 2] and {c.finish for c in logged} == {"length"}

    # No model text anywhere it was recorded or printed
    folder = data.root / "calls"
    _no_text(recorded.texts(), (data.root / "failed.jsonl").read_text(encoding="utf-8"),
             *(path.read_text(encoding="utf-8") for path in folder.glob("*.jsonl")), "\n".join(lines))


def test_a_life_skipped_again_on_a_rerun_keeps_each_runs_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: a life whose every reply is cut, run twice
    backend, _ = _backend(tmp_path, monkeypatch, "length")
    data = DataDir(tmp_path / "pilot")

    # Act
    for _ in range(2):
        with pytest.raises(RunsFailed):
            gen_batch.generate_lives(data, [SPEC], backend, lambda line: None)

    # Assert: both runs' logs are kept, one per run, so ``calls`` sums what the ledger settled
    logs = data.call_logs()
    assert sorted(made_id.rsplit(".", 1)[1] for made_id in logs) == ["failed-1", "failed-2"]
    assert all([c.attempt for c in _logged(data, made_id)] == [0, 1, 2] for made_id in logs)


def test_a_rejected_request_skips_the_life_at_its_first_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: Fireworks refuses the request (an unknown parameter is a 400, never tried again)
    backend, _ = _backend(tmp_path, monkeypatch, "rejected")
    data, lines = DataDir(tmp_path / "pilot"), list[str]()

    # Act
    with pytest.raises(RunsFailed):
        gen_batch.generate_lives(data, [SPEC], backend, lines.append)

    # Assert: one attempt only, so a 400 can't be what failed two attempts around a third that ran
    (attempt,) = [raw for raw in data.failed() if raw["what"] == "generate-attempt"]
    assert (attempt["attempt"], attempt["step"], attempt["day"], attempt["error"]) == (0, "writer", 1, "ModelApiError")
    assert "HTTP 400" in attempt["reason"] and attempt["reply_chars"] is None
    (made_id,) = data.call_logs()
    assert [(c.kind, c.failed) for c in _logged(data, made_id)] == [("write", True)]


def test_a_failed_gate_is_recorded_by_gate_item_and_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the leak reader flags the first window it reads, once
    flagged: list[bool] = []

    class FlagsOnce(fakes.FakeGateReader):
        def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                     temperature: float | None = None) -> ChatResult:
            if calls.kind_of(messages) == "leak" and not flagged:
                flagged.append(True)
                return ChatResult(json.dumps({"flag": True}), (), model, 10, 2, "stop")
            return super().__call__(model, messages, tools, timeout_s, temperature=temperature)

    monkeypatch.setattr(fakes, "FakeGateReader", FlagsOnce)
    data, lines = tmp_path / "pilot", list[str]()

    # Act
    code = cli.main(["--fake", "--data", str(data), "--only", "1", "generate", "--split", "core-train"],
                    out=lines.append)

    # Assert: the first attempt failed the leak gate on item 1, from a day of the life; the second passed
    assert code == 0
    (raw,) = DataDir(data).failed()
    assert set(raw) == RECORD_KEYS and raw["what"] == "generate-attempt"
    assert (raw["attempt"], raw["step"], raw["error"], raw["item"]) == (0, "leak", "verdict", 1)
    assert 1 <= raw["day"] <= 10 and re.fullmatch(r"\w+ flagged from \d{4}-\d{2}-\d{2}", raw["reason"])
    (failed,) = [line for line in lines if line.startswith("FAILED ")]
    assert failed.startswith("FAILED attempt 1 of core-train life ") and f"(day {raw['day']}, item 1)" in failed

    # ``calls`` prints it with the calls
    lines.clear()
    assert cli.main(["--data", str(data), "calls"], out=lines.append) == 0
    assert "Failed attempts: 1" in lines and failed in lines


# ── probe-writer ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("shape", "seen"), [
    ("plain", ["think_tags=False", "starts_with_brace=True", "answer=ok", "writer_reads=ok", "finish_reason=stop"]),
    ("think", ["think_tags=True", "reasoning_tokens=40", "writer_reads=ok"]),
    ("fenced", ["fenced=True", "starts_with_brace=False", "answer_json=True", "writer_reads=ok"]),
    ("reasoning_field", ["reasoning_chars=26", "'reasoning_content'", "writer_reads=ok"]),
    ("length", ["finish_reason=length", "answer=refused (cut)", "writer_reads=refused"]),
    ("open_think", ["answer=refused (open_think)", "writer_reads=refused"]),
    ("two_fences", ["answer=refused (fences)", "writer_reads=refused"]),
])
def test_probe_prints_each_replys_shape_never_its_text(monkeypatch: pytest.MonkeyPatch, shape: str,
                                                       seen: list[str]) -> None:
    recorded, lines = Recorded(shape), list[str]()
    monkeypatch.setenv(fireworks_api.KEY_ENV, "fw-test")

    spent = probe.run(DEEPSEEK, 2, RATE, lines.append,
                      call=partial(fireworks_api.generate, transport=recorded.transport))

    assert [line.split(":")[0] for line in lines] == ["probe 1", "probe 2"]
    for part in seen:
        assert all(part in line for line in lines), (part, lines)
    assert 0 < spent < 2 * RATE.worst
    _no_text(recorded.texts(), "\n".join(lines))
    assert "Let me plan" not in "\n".join(lines) and "The person" not in "\n".join(lines)


def _one_based(model: str, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
               on_reply: Any = None, **kw: Any) -> ChatResult:
    """A writer that numbers its messages from 1 in ``placed``."""
    reply = fakes.FakeLifeChat()(model, messages, tools, timeout_s)
    body = json.loads(reply.text)
    text = json.dumps({**body, "placed": {beat: spot + 1 for beat, spot in body["placed"].items()}})
    on_reply({"model": model, "choices": [{"message": {"role": "assistant", "content": text},
                                           "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 800, "completion_tokens": 40, "total_tokens": 840}})
    return dataclasses.replace(reply, text=text)


def test_probe_places_two_beats_on_its_even_calls_and_says_where_they_landed(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded, lines = Recorded("plain"), list[str]()
    monkeypatch.setenv(fireworks_api.KEY_ENV, "fw-test")

    probe.run(DEEPSEEK, 2, RATE, lines.append, call=partial(fireworks_api.generate, transport=recorded.transport))
    probe.run(DEEPSEEK, 2, RATE, lines.append, call=_one_based)

    # The first call has no beats; the second two, each placed by its index from 0
    assert "beats=0" in lines[0] and "placed_valid=None" in lines[0]
    assert all(part in lines[1] for part in ("beats=2", "placed_valid=True", "beats_at_placed=2",
                                             "beats_one_before_placed=0", "writer_reads=ok"))
    # A writer that counts from 1 is seen: its beats sit one message before where it says
    assert all(part in lines[3] for part in ("beats=2", "placed_valid=False", "beats_at_placed=0",
                                             "beats_one_before_placed=2"))
    assert "writer_reads=a beat points outside the day's messages (field placed)" in lines[3]
    sent = [json.loads(body["messages"][-1]["content"]) for body in recorded.requests]
    assert [len(day["plant"]) + len(day["ask"]) for day in sent] == [0, 2]
    printed = "\n".join(lines)
    assert all(beat.say not in printed for beat in probe.BEATS)


def test_probe_counts_a_rejected_call_at_its_worst_case(monkeypatch: pytest.MonkeyPatch) -> None:
    lines: list[str] = []
    monkeypatch.setenv(fireworks_api.KEY_ENV, "fw-test")
    rejected = partial(fireworks_api.generate, transport=fakes.generator_transport("rejected"))
    assert probe.run(DEEPSEEK, 1, RATE, lines.append, call=rejected) == pytest.approx(RATE.worst)
    assert lines == [f"probe 1: failed: Fireworks chat with {DEEPSEEK.model} returned HTTP 400"]


def test_probe_writer_is_planned_first_and_refused_without_live(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_call(*args: Any, **kw: Any) -> ChatResult:
        raise AssertionError("the plan calls nothing")

    monkeypatch.setattr(fireworks_api, "generate", no_call)
    command = ["probe-writer", "--generator", "deepseek-v4p1-flash"]
    lines: list[str] = []

    assert cli.main(command, out=lines.append) == cli.REFUSED_CODE
    assert "needs --live" in lines[0]
    lines.clear()
    assert cli.main(["--live", *command], out=lines.append) == 0
    worst = 3 * Rate(*DEEPSEEK.list_prices or (0, 0)).worst
    assert lines[0].startswith("Plan: probe-writer: 3 writer call(s) on deepseek-v4p1-flash")
    assert "the even ones with two beats" in lines[0]
    assert f"Hold on {FIREWORKS.pool}: ${worst:.4f}" in lines[1] and lines[-1].startswith("Nothing has run")
    with pytest.raises(SystemExit):
        cli.main(["--live", "probe-writer", "--generator", "spark"], out=lines.append)  # not on Fireworks
    with pytest.raises(SystemExit):
        cli.main(["--live", *command, "--n", str(probe.MAX_PROBES + 1)], out=lines.append)


def test_probe_writer_runs_held_and_settles_what_it_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the shared ledger in a tmp_path data dir, Fireworks answering with think tags
    recorded, lines = Recorded("think"), list[str]()
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(fireworks_api.KEY_ENV, "fw-test")
    monkeypatch.setattr(fireworks_api, "generate", partial(fireworks_api.generate, transport=recorded.transport))

    # Act
    code = cli.main(["--live", "--yes", "probe-writer", "--generator", "deepseek-v4p1-flash", "--n", "2",
                     "--generator-prices", "0.30,1.20"], out=lines.append)

    # Assert: two calls, held on Fireworks' pool first and settled at what they cost, nothing left open
    assert code == 0 and len(recorded.bodies) == 2
    events = list(read_records(tmp_path / "sleep" / "budget.jsonl", PLAIN))
    (reserve,) = [e for e in events if e["op"] == "reserve"]
    (settle,) = [e for e in events if e["op"] == "settle"]
    assert str(reserve["job"]).startswith("apd:probe:writer:deepseek-v4p1-flash:") and reserve["pools"] == [
        FIREWORKS.pool]
    assert float(reserve["amount"]) == pytest.approx(2 * RATE.worst) and 0 < float(settle["amount"]) < 2 * RATE.worst
    assert RunLedger(tmp_path / "sleep" / "budget.jsonl").open_holds() == ()
    assert [line.split(":")[0] for line in lines if line.startswith("probe ")] == ["probe 1", "probe 2"]
    assert lines[-1] == f"Settled ${float(settle['amount']):.4f} on {FIREWORKS.pool}."
    _no_text(recorded.texts(), "\n".join(lines))


# ── A life stopped mid-attempt ────────────────────────────────────────────────


@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("stop", [BudgetStop("a pool can't hold it"), Halted("the command is ending")])
def test_a_life_stopped_mid_attempt_keeps_the_calls_it_made(tmp_path: Path, stop: Exception, pair: bool) -> None:
    # Arrange: the third call of the life (or the pair) stops the run (a hold without room, a kill)
    made: list[int] = []

    def stops_third(model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                    temperature: float | None = None) -> ChatResult:
        made.append(1)
        if len(made) == 3:
            raise stop
        return fakes.FakeLifeChat()(model, messages, tools, timeout_s, temperature=temperature)

    backend = dataclasses.replace(backends.fake_backend(FIREWORKS), life_chat=stops_third,
                                  ledger=RunLedger(tmp_path / "budget.jsonl"), prices=cli.FAKE_PRICES,
                                  rates=cli.FAKE_RATES)
    data = DataDir(tmp_path / "pilot")

    # Act
    with pytest.raises(type(stop)):
        if pair:
            pairs.generate_pairs(data, pairs.pair_specs()[:1], backend, lambda line: None)
        else:
            gen_batch.generate_lives(data, [SPEC], backend, lambda line: None)

    # Assert: the two calls it paid for are in its log, so ``calls`` still counts them
    (made_id,) = data.call_logs()
    assert made_id.endswith(".failed-1") and [c.kind for c in _logged(data, made_id)] == ["write", "extract"]
