# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The gates' auditor given more room (MIND's data): Muse Spark spends its reasoning inside the
reply's cap, and on 2026-10-29 an audit came back cut at 8,192 tokens with nothing to read ("the
reply is not JSON; reply 0 chars, finish length"), which skipped its life. A reply cut at its cap
with no JSON object in it is asked once more with room for ``AUDITOR_ROOM_TOKENS``, as the morning
judge's is, at most ``AUDITOR_ROOMY_ASKS`` times a hold, and the hold covers that ask at its own
cap's worst case. Fakes only: no model, no network."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep import meta_api
from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import calls, core_splits, fakes, gen_batch, retry
from anjo.sleep.mind.apd.ceiling import RunLedger
from anjo.sleep.mind.apd.judging import MAX_IN_TOKENS, MAX_OUT_TOKENS, MUSE, Rate
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.model_api import ChatResult

MUSE_RATE = Rate(1.25, 4.25)  # Muse Spark's list prices, per million tokens in and out
AUDIT = [{"role": "system", "content": "# audit_v1"}, {"role": "user", "content": "{}"}]
VERDICT = json.dumps({"ok": True, "issues": []})
SPEC = next(spec for spec in core_splits.core_specs("core-train") if spec.prevalence > 0)


def _reply(text: str, finish: str, tokens_out: int) -> ChatResult:
    return ChatResult(text=text, tool_calls=(), model="muse-spark-1.3", input_tokens=3355, output_tokens=tokens_out,
                      finish_reason=finish)


class Answers:
    """A chat that answers its replies in turn, recording each call's timeout."""

    def __init__(self, *replies: ChatResult) -> None:
        self.replies, self.timeouts = list(replies), list[float]()

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        self.timeouts.append(timeout_s)
        return self.replies.pop(0)


CUT = _reply("", "length", MAX_OUT_TOKENS)  # 2026-10-29: the whole cap spent reasoning, nothing to read


def _auditor(inner: Answers, roomy: Answers, log: calls.CallLog | None = None) -> backends.CountedChat:
    """Two Muse calls held, and the one roomy ask (``GenCalls.muse`` and ``roomy``)."""
    return backends.CountedChat(inner, 2 + backends.AUDITOR_ROOMY_ASKS, MUSE_RATE, log=log, room=backends.Room(roomy))


def test_a_cut_reply_with_nothing_to_read_is_asked_once_more_with_room() -> None:
    # Arrange
    inner, roomy, log = Answers(CUT), Answers(_reply(VERDICT, "stop", 9_000)), calls.CallLog()
    auditor = _auditor(inner, roomy, log)

    # Act
    found = auditor("muse-spark", AUDIT, None, 1.0)

    # Assert: the roomy answer, waited for as long as its cap needs; both tries priced and logged
    assert found.text == VERDICT and len(inner.timeouts) == len(roomy.timeouts) == 1
    assert roomy.timeouts == [retry.reply_timeout_s(backends.AUDITOR_ROOM_TOKENS)]
    assert [(c.kind, c.finish, c.reply_chars) for c in log.calls] == [("audit", "length", 0), ("audit", "stop", len(VERDICT))]
    assert auditor.usd == pytest.approx(MUSE_RATE.usd(3355, MAX_OUT_TOKENS) + MUSE_RATE.usd(3355, 9_000))


@pytest.mark.parametrize("reply", [
    _reply(VERDICT, "length", MAX_OUT_TOKENS),  # cut, but its JSON is whole: read as it is
    _reply("", "stop", 10),  # not cut: the gates' own error, as ever
])
def test_only_a_cut_reply_with_no_json_object_is_asked_again(reply: ChatResult) -> None:
    inner, roomy = Answers(reply), Answers()
    assert _auditor(inner, roomy)("muse-spark", AUDIT, None, 1.0) is reply and roomy.timeouts == []


def test_a_hold_asks_with_room_at_most_its_roomy_asks() -> None:
    # Arrange: a second cut reply in the same hold
    inner, roomy = Answers(CUT, CUT), Answers(_reply(VERDICT, "stop", 9_000))
    auditor = _auditor(inner, roomy)

    # Act
    auditor("muse-spark", AUDIT, None, 1.0)
    second = auditor("muse-spark", AUDIT, None, 1.0)

    # Assert: the second comes back cut, for the gates to refuse (``gates.GateError``)
    assert backends.AUDITOR_ROOMY_ASKS == 1 and second is CUT and len(roomy.timeouts) == 1


def test_the_hold_covers_the_roomy_ask_at_its_own_caps_worst() -> None:
    # Arrange
    six = backends.life_calls(6)
    rates = {MUSE: MUSE_RATE}

    # Act
    held = backends.gen_holds(cli.FAKE_PRICES, rates, FIREWORKS.pool, six)["meta_usd"]
    before = backends.gen_holds(cli.FAKE_PRICES, rates, FIREWORKS.pool, dataclasses.replace(six, roomy=0))["meta_usd"]

    # Assert: seven Muse calls at 8,192 out, and one more at 16,384 out ($0.53 to $0.64 a six-item attempt)
    assert six.muse == 7 and six.roomy == backends.AUDITOR_ROOMY_ASKS == 1
    assert before == pytest.approx(7 * MUSE_RATE.usd(MAX_IN_TOKENS, MAX_OUT_TOKENS))
    assert held - before == pytest.approx(MUSE_RATE.usd(MAX_IN_TOKENS, backends.AUDITOR_ROOM_TOKENS))
    assert round(before, 2) == 0.53 and round(held, 2) == 0.64


def test_the_live_auditors_roomy_chat_asks_for_its_room(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[int | None] = []

    def chat(model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None,
             max_tokens: int | None = None) -> ChatResult:
        asked.append(max_tokens)
        return _reply(VERDICT, "stop", 10)

    monkeypatch.setattr(meta_api, "chat", chat)
    backends.gen_chat(FIREWORKS)("muse-spark", AUDIT, None, 1.0)
    backends.gen_chat(FIREWORKS, cap=backends.AUDITOR_ROOM_TOKENS)("muse-spark", AUDIT, None, 1.0)
    assert asked == [retry.REPLY_MAX_TOKENS, backends.AUDITOR_ROOM_TOKENS]


class CutsTheAudit(fakes.FakeGateAuditor):
    """The fakes' auditor, whose first script audit comes back cut with nothing to read."""

    def __init__(self) -> None:
        self.cut = False

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        if str(messages[0]["content"]).startswith("# audit_v1") and not self.cut:
            self.cut = True
            return CUT
        return super().__call__(model, messages, tools, timeout_s, temperature=temperature)


def test_a_life_whose_audit_came_back_cut_passes_on_its_first_attempt(tmp_path: Path) -> None:
    # Arrange
    log = calls.CallLog()
    backend = dataclasses.replace(backends.fake_backend(FIREWORKS), ledger=RunLedger(tmp_path / "budget.jsonl"),
                                  prices=cli.FAKE_PRICES, rates=cli.FAKE_RATES, auditor=CutsTheAudit(),
                                  auditor_room=fakes.FakeGateAuditor())

    # Act
    made = gen_batch.generate_life(SPEC, backend, log)

    # Assert
    assert made.attempts == 1
    assert [c.finish for c in log.calls if c.kind == "audit"] == ["length", "stop"]


def test_the_generate_plan_names_the_roomy_ask_in_the_auditors_hold(tmp_path: Path) -> None:
    lines: list[str] = []
    assert cli.main(["--live", "--prices", "1.25,4.25,1.5,2.61", "--judge-prices", "1.25,4.25,0.75,3.75", "--data",
                     str(tmp_path), "--only", "1", "generate"], out=lines.append) == 0
    (held,) = [line for line in "\n".join(lines).splitlines() if line.startswith("Hold on meta_usd")]
    assert "the gates' auditor (Muse Spark), and 1 ask with room for 16,384 tokens for a reply cut at its cap" in held
