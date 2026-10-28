# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Repairing a core split's failed attempt (MIND's data; ``--repair``, ``--keep-failed``): a leak or
a script audit that names its days has only those days written again, with the reason, and the gates
read again only where the text changed; a repair that keeps failing fails the attempt as today; a
failed attempt's life is kept apart under ``quarantine`` on request. Every model is a fake: no
network, no spend."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import calls, core_splits, fakes, gates, gen_batch, generate, generators, readers, repair
from anjo.sleep.mind.apd.calls import CallLog
from anjo.sleep.mind.apd.ceiling import RunLedger
from anjo.sleep.mind.apd.judging import GEMINI, MUSE, Rate
from anjo.sleep.mind.apd.store import DataDir, loads
from anjo.sleep.mind.apd.types import GateVerdict, Life, ProblemFile
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.deps import Prices

PRICES = Prices(1.25, 4.25, 1.5, 2.61)
RATES = {MUSE: Rate(1.25, 4.25), GEMINI: Rate(0.75, 3.75)}
MARK = "Funny how the budget and the date never line up."  # a conclusion, as a writer may add one
SPEC = core_splits.core_specs("core-train")[0]  # the live life that failed: 6 items, surfacing on days 8 to 10
LEAKY_DAY = 6  # 2026-09-12, before every item's surface
LifeChat = fakes.FakeLifeChat  # the fakes' own, kept before a test swaps the module's name
GateReader = fakes.FakeGateReader


@dataclasses.dataclass
class LeakyDay:
    """The fake writer and extractor, with ``MARK`` added to day ``day`` as first written; a
    rewrite of it drops it unless ``stubborn``. Only lives of the attempt seeds in ``seeds`` leak
    (every one when empty). Each write's day and each repair's payload are kept."""

    day: int = LEAKY_DAY
    stubborn: bool = False
    seeds: frozenset[int] = frozenset()
    inner: Any = dataclasses.field(default_factory=LifeChat)
    days: list[int] = dataclasses.field(default_factory=list)
    repairs: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        result: ChatResult = self.inner(model, messages, tools, timeout_s, temperature=temperature)
        if calls.kind_of(messages) != "write":
            return result
        body = json.loads(messages[-1]["content"])
        self.days.append(body["day"])
        if "reason" in body:
            self.repairs.append(body)
        leaks = body["day"] == self.day and (not self.seeds or body["variation"] in self.seeds)
        if not leaks or ("reason" in body and not self.stubborn):
            return result
        day = json.loads(result.text)
        return dataclasses.replace(result, text=json.dumps({**day, "messages": [*day["messages"], MARK]}))


@dataclasses.dataclass
class FlagsTheMark(GateReader):
    """The fake reader, flagging any leak window of the evidence view that holds ``MARK``; each
    leak request kept."""

    leaks: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        if calls.kind_of(messages) == "leak":
            body = json.loads(messages[-1]["content"])
            self.leaks.append(body)
            if body["view"] == "evidence" and any(MARK in day["text"] for day in body["days"]):
                return ChatResult(json.dumps({"flag": True}), (), model, 10, 2, "stop")
        return super().__call__(model, messages, tools, timeout_s, temperature=temperature)


@dataclasses.dataclass
class AuditsOnce:
    """The fake auditor, its first script audit finding ``issues``; the rest pass."""

    issues: tuple[str, ...]
    audits: int = 0
    inner: Any = dataclasses.field(default_factory=fakes.FakeGateAuditor)

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        if calls.kind_of(messages) == "audit":
            self.audits += 1
            if self.audits == 1:
                return ChatResult(json.dumps({"ok": False, "issues": list(self.issues)}), (), "muse-spark-1.3", 10, 2,
                                  "stop")
        result: ChatResult = self.inner(model, messages, tools, timeout_s, temperature=temperature)
        return result


def _backend(tmp_path: Path, **changes: Any) -> backends.Backend:
    """Written by DeepSeek and read by Gemini, as a core split is, held at real-like prices."""
    return dataclasses.replace(backends.fake_backend(FIREWORKS), ledger=RunLedger(tmp_path / "budget.jsonl"),
                               prices=PRICES, rates=RATES, generator=generators.DEEPSEEK, generator_rate=Rate(0.3, 1.2),
                               read_by=readers.GEMINI_READER, reader_rate=Rate(0.75, 3.75), **changes)


def _jobs(path: Path, op: str) -> dict[str, float]:
    return {str(e["job"]): float(e["amount"]) for e in read_records(path, PLAIN) if e["op"] == op}


# ── A leak ────────────────────────────────────────────────────────────────────


def test_a_leak_flagged_on_day_6_is_repaired_and_the_life_passes(tmp_path: Path) -> None:
    # Arrange: day 6's notes give an item away, as the live life's did ("evidence flagged from 2026-09-12")
    writer, reader, log = LeakyDay(), FlagsTheMark(), CallLog()
    backend = _backend(tmp_path, life_chat=writer, reader=reader, repair=True)

    # Act
    made = gen_batch.generate_life(SPEC, backend, log)

    # Assert: one attempt, one repair round that wrote day 6 again and nothing else
    assert made.attempts == 1 and made.repaired == 1 and gates.passed(made.verdicts)
    assert writer.days == [*range(1, 11), LEAKY_DAY]
    (asked,) = writer.repairs
    assert MARK in asked["before"] and "notes taken from" in asked["reason"]
    assert [day["day"] for day in asked["later"]] == list(range(LEAKY_DAY + 1, 11))
    assert all(MARK not in session.text for session in made.life.sessions)
    assert [s.text for s in made.life.sessions[LEAKY_DAY:]] == ["\n\n".join(d["messages"]) for d in asked["later"]]
    # the gates read again only what changed: the windows holding day 6, and the windows the first check
    # never reached (each item stopped at its first flag)
    fresh = sum(c.kind == "leak" and c.repair == 0 for c in log.calls)
    first, again = reader.leaks[:fresh], reader.leaks[fresh:]
    assert 0 < len(again) < fresh
    assert all("2026-09-12" in [day["date"] for day in body["days"]] or body not in first for body in again)
    assert len({json.dumps(body, sort_keys=True) for body in again}) == len(again)  # nothing read twice
    assert sum(c.kind == "audit" for c in log.calls) == 2  # the changed script is audited again
    # held and priced like an attempt, every hold settled
    reserved, settled = _jobs(tmp_path / "budget.jsonl", "reserve"), _jobs(tmp_path / "budget.jsonl", "settle")
    rounds = [job for job in reserved if f":{SPEC.seed}:0:r1:" in job]
    assert rounds and all(0 < settled[job] <= reserved[job] for job in rounds)
    assert sum(c.usd for c in log.calls if c.repair == 1) > 0
    shown = "\n".join(calls.breakdown({made.life.id: log.calls}))
    assert f"{made.life.id} attempt 1 as written" in shown and f"{made.life.id} attempt 1 round 1" in shown
    assert RunLedger(tmp_path / "budget.jsonl").open_holds() == ()


def test_a_repair_that_keeps_leaking_fails_the_attempt_as_today(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: every rewrite of day 6 leaks again; one attempt allowed
    monkeypatch.setattr(generate, "MAX_ATTEMPTS", 1)
    writer, log = LeakyDay(stubborn=True), CallLog()
    backend = _backend(tmp_path, life_chat=writer, reader=FlagsTheMark(), repair=True)

    # Act / Assert: two rounds, each writing day 6 again, then the attempt fails on the leak, as today
    with pytest.raises(generate.GenerationError, match="no core-train life passed the gates in 1 attempts"):
        gen_batch.generate_life(SPEC, backend, log)
    assert writer.days == [*range(1, 11), LEAKY_DAY, LEAKY_DAY] and len(writer.repairs) == repair.ROUNDS
    assert len(log.failures) == SPEC.prevalence  # every item's leak, each at day 6, recorded once as today
    assert {(f.step, f.day, f.reason) for f in log.failures} == {("leak", LEAKY_DAY, "evidence flagged from 2026-09-12")}
    assert {c.repair for c in log.calls} == {0, 1, 2}
    assert RunLedger(tmp_path / "budget.jsonl").open_holds() == ()


def test_without_repair_a_leak_fails_the_attempt_as_today(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate, "MAX_ATTEMPTS", 1)
    writer, log = LeakyDay(), CallLog()
    with pytest.raises(generate.GenerationError):
        gen_batch.generate_life(SPEC, _backend(tmp_path, life_chat=writer, reader=FlagsTheMark()), log)
    assert writer.days == list(range(1, 11)) and writer.repairs == []
    assert {f.step for f in log.failures} == {"leak"} and {c.repair for c in log.calls} == {0}


# ── The script audit ──────────────────────────────────────────────────────────


def test_an_audit_naming_a_day_has_that_day_written_again(tmp_path: Path) -> None:
    # Arrange: day 4 as first written calls something a problem, and the audit says so; its rewrite doesn't
    issue = "On 2026-09-10 the person calls the delay a problem."
    writer, auditor = LeakyDay(day=4), AuditsOnce((issue,))
    made = gen_batch.generate_life(SPEC, _backend(tmp_path, life_chat=writer, auditor=auditor, repair=True))
    assert made.attempts == 1 and made.repaired == 1 and auditor.audits == 2
    (asked,) = writer.repairs
    assert asked["day"] == 4 and issue in asked["reason"]


# ── What can be repaired ──────────────────────────────────────────────────────


def _life() -> tuple[Life, ProblemFile]:
    made = fakes.fake_life(SPEC)
    return made.life, made.problems


def test_fixes_name_each_day_with_its_reasons_or_none_when_a_rewrite_cannot_help() -> None:
    life, problems = _life()
    first, second = problems.items[0], problems.items[1]
    leak = GateVerdict("leak", first.id, False, "sessions flagged from 2026-09-12")
    audit = GateVerdict("script_audit", None, False, "1 issue(s) found")
    found = repair.fixes(life, problems, (leak, audit), ("On 2026-09-12 and 2026-09-08 it says so.",))
    assert found is not None and sorted(found) == [1, 5]  # day indexes, from 0
    assert first.topic in found[5][0] and "messages of" in found[5][0] and "2026-09-08" in found[1][0]
    # a gate a rewrite can't mend, an issue with no day, and too many days are left to a fresh attempt
    lost = GateVerdict("discoverability", second.id, False, "the reader found nothing")
    assert repair.fixes(life, problems, (leak, lost), ()) is None
    assert repair.fixes(life, problems, (audit,), ("The resolution does not follow.",)) is None
    many = " ".join(day.isoformat() for day in life.days[:repair.REPAIR_DAYS + 1])
    assert repair.fixes(life, problems, (audit,), (many,)) is None
    assert repair.fixes(life, problems, (), ()) is None  # nothing failed


def test_a_round_holds_its_writes_and_a_full_read_again() -> None:
    full = backends.life_calls(4)
    assert repair.round_calls(4, 2) == backends.GenCalls(4, full.reads, full.muse)


def test_the_memo_answers_a_request_it_sent_before_without_a_call() -> None:
    sent: list[str] = []

    def chat(model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
             temperature: float | None = None) -> ChatResult:
        sent.append(messages[-1]["content"])
        return ChatResult(f"reply {len(sent)}", (), model, 1, 1, "stop")

    memo: dict[str, ChatResult] = {}
    one, two = repair.Memo(chat, memo), repair.Memo(chat, memo)  # two holds of one attempt share it
    ask = [{"role": "system", "content": "# leak_v1"}, {"role": "user", "content": "{}"}]
    assert one("m", ask, None, 1.0, temperature=0.0).text == "reply 1"
    assert two("m", ask, None, 1.0, temperature=0.0).text == "reply 1" and len(sent) == 1
    assert two("other", ask, None, 1.0, temperature=0.0).text == "reply 2"


# ── The command line ──────────────────────────────────────────────────────────


def test_keep_failed_quarantines_a_failed_attempt_as_lives_are_stored(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the first attempt leaks on day 6, the second is clean
    monkeypatch.setattr(fakes, "FakeLifeChat", lambda: LeakyDay(seeds=frozenset({SPEC.seed})))
    monkeypatch.setattr(fakes, "FakeGateReader", FlagsTheMark)
    data, lines = tmp_path / "pilot", list[str]()

    # Act
    code = cli.main(["--fake", "--data", str(data), "--only", "1", "generate", "--split", "core-train",
                     "--keep-failed"], out=lines.append)

    # Assert: the failed attempt's life and sealed file are kept apart, with why; the passing life alone is a life
    (life,) = DataDir(data).lives("core-train")
    first = generate.life_id_for(SPEC.split, SPEC.field, SPEC.seed)
    assert code == 0 and life.id != first
    kept = data / "quarantine"
    failed = loads((kept / "lives" / f"{first}.r0.json").read_text(), Life)
    assert failed.id == first and MARK in failed.sessions[LEAKY_DAY - 1].text
    assert loads((kept / "sealed" / f"{first}.r0.json").read_text(), ProblemFile).life_id == first
    why = json.loads((kept / "why" / f"{first}.r0.json").read_text())
    assert (why["attempt"], why["round"]) == (0, 0)
    assert any(v["reason"] == "evidence flagged from 2026-09-12" for v in why["verdicts"])
    assert sorted(path.name for path in (data / "lives").iterdir()) == [f"{life.id}.json"]


def test_repair_and_keep_failed_are_for_the_core_splits_only(tmp_path: Path,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (["generate", "--repair"], ["generate", "--held-out", "--draw-seed", "x", "--keep-failed"]):
        with pytest.raises(SystemExit) as stopped:
            cli.main(["--fake", "--data", str(tmp_path / "pilot"), *argv], out=lambda line: None)
        assert stopped.value.code == 2 and "core splits" in capsys.readouterr().err
    assert not (tmp_path / "pilot").exists()


def test_the_plan_says_what_a_repair_round_may_hold(tmp_path: Path) -> None:
    lines: list[str] = []
    assert cli.main(["--live", "--prices", "1.25,4.25,1.5,2.61", "--judge-prices", "1.25,4.25,0.75,3.75",
                     "--data", str(tmp_path), "--only", "2", "generate", "--split", "core-train", "--repair"],
                    out=lines.append) == 0
    text = "\n".join(lines)
    assert f"up to {repair.ROUNDS} repair rounds" in text and "3 times" in text
