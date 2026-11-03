# mypy: disallow-untyped-decorators=False
"""The wake's before and after on a judge's night (``app/judge_pair.py``): one personal question made
from what this judge said (her chooser's target, else their latest substantive message), asked of
base Glimmer and tonight's adapter at the end of the night; the collapse gate's everyday pair when
that can't be done; nothing when the gate didn't run. Fakes only: no GPU, no model."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from anjo.sleep.app import judge_pair
from anjo.sleep.app.judge_pair import GatePairs, PersonalPorts, from_message, from_target, save_pair
from anjo.sleep.gates.collapse import CollapsePair
from anjo.sleep.types import CheckpointRef, GateResult, NightResult

USER = "judge-a"
CKPT = CheckpointRef(id="p-aaaaaaaaaaaa-n1", person_id=USER, night=date(2026, 11, 3), uri="modal://adapters/x",
                     parent_id=None, rendering="r/low")
PASSED = GateResult("collapse", True, 0.0, 5.0)


def _result(status: str = "promoted", gate: GateResult | None = PASSED) -> NightResult:
    return NightResult(status=status, failed_at=None, checkpoint=CKPT if status == "promoted" else None,  # type: ignore[arg-type]
                       gates=MappingProxyType({} if gate is None else {"collapse": gate}), choices=(), cost_usd=0.0,
                       cost_parts=MappingProxyType({}), log_path=Path("log.jsonl"))


def _pair(i: int, *, base_dragged: bool = False, dragged: bool = False) -> CollapsePair:
    return CollapsePair(f"c{i:02d}", f"everyday question {i}?", f"base answer {i}", f"adapter answer {i}",
                        base_dragged, dragged)


def _gate(*pairs: CollapsePair) -> GatePairs:
    sink = GatePairs()
    sink(CKPT, list(pairs))
    return sink


class Serving:
    """Base Glimmer (adapter None) and the adapter by name, each one short reply; calls kept."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.asked: list[tuple[str | None, str]] = []

    def reply(self, adapter: CheckpointRef | None, question: str) -> str:
        self.asked.append((None if adapter is None else adapter.id, question))
        if self.fail:
            raise OSError("serving is down")
        return "base reply" if adapter is None else "adapter reply"


def _ports(question: str | None, serving: Serving) -> PersonalPorts:
    return PersonalPorts(question=lambda user_id: question, reply=serving.reply)


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    monkeypatch.setattr(judge_pair, "save_before_after", lambda user_id, **kw: out.append({"user": user_id, **kw}))
    return out


def test_a_target_becomes_a_question_the_judge_would_recognise() -> None:
    assert from_target("Hackathon demo") == "Thinking about hackathon demo: what should I focus on next?"
    assert from_target("choosing between two job offers.") == (
        "Thinking about choosing between two job offers: what should I focus on next?")
    assert from_target("NYC move") == "Thinking about NYC move: what should I focus on next?"
    assert from_target("  ") is None and from_target(None) is None


def test_a_message_is_kept_in_their_own_words() -> None:
    assert from_message("Should I take the offer with the better team?") == "Should I take the offer with the better team?"
    assert from_message("I'm building a sleep-learning AI and I keep second-guessing the demo") == (
        "I'm building a sleep-learning AI and I keep second-guessing the demo. What should I do next?")
    assert from_message("ok thanks") is None  # not substantive
    assert len(from_message("word " * 200) or "") <= judge_pair.MAX_QUESTION + len(judge_pair.ASK)


def test_the_personal_question_goes_to_base_glimmer_and_the_adapter(saved: list[dict[str, Any]]) -> None:
    serving = Serving()
    save_pair(USER, "2026-11-03", _result(), _gate(_pair(0)), _ports("Thinking about hackathon demo: what next?", serving))
    assert serving.asked == [(None, "Thinking about hackathon demo: what next?"),
                             ("p-aaaaaaaaaaaa-n1", "Thinking about hackathon demo: what next?")]
    assert saved == [{"user": USER, "question": "Thinking about hackathon demo: what next?", "before": "base reply",
                      "after": "adapter reply", "night": "2026-11-03", "before_model": "base Glimmer",
                      "after_model": "p-aaaaaaaaaaaa-n1", "promoted": True, "question_source": "personal"}]


def test_a_rejected_adapter_is_still_asked_by_name_and_says_so(saved: list[dict[str, Any]]) -> None:
    serving = Serving()
    held = GateResult("collapse", False, 20.0, 5.0, "dragged")
    save_pair(USER, "2026-11-03", _result("rejected", held), _gate(_pair(0)), _ports("q?", serving))
    assert serving.asked[1] == ("p-aaaaaaaaaaaa-n1", "q?")
    assert (saved[0]["promoted"], saved[0]["question_source"]) == (False, "personal")


@pytest.mark.parametrize("why", ["no_question", "serving_down"])
def test_without_a_personal_pair_the_gates_everyday_one_is_kept_and_labelled(saved: list[dict[str, Any]],
                                                                             why: str) -> None:
    serving = Serving(fail=why == "serving_down")
    gate = _gate(_pair(0), _pair(1, dragged=True), _pair(2, base_dragged=True))
    save_pair(USER, "2026-11-03", _result(), gate, _ports(None if why == "no_question" else "q?", serving))
    assert [(s["question"], s["before"], s["after"], s["question_source"]) for s in saved] == [
        ("everyday question 2?", "base answer 2", "adapter answer 2", "gate_everyday")]


@pytest.mark.parametrize("case", ["skipped", "never_ran"])
def test_no_gate_no_pair_and_no_call(saved: list[dict[str, Any]], case: str) -> None:
    serving = Serving()
    gate = GateResult("collapse", False, 0.0, 5.0, "off", skipped=True) if case == "skipped" else None
    save_pair(USER, "2026-11-03", _result("rejected", gate), GatePairs(), _ports("q?", serving))
    assert saved == [] and serving.asked == []


def test_a_gate_with_no_everyday_pair_still_gets_the_personal_one(saved: list[dict[str, Any]]) -> None:
    serving = Serving()
    save_pair(USER, "2026-11-03", _result(), _gate(), _ports("q?", serving))  # base replies weren't kept
    assert [s["question_source"] for s in saved] == ["personal"]


def test_the_question_comes_from_her_chooser_first_then_their_last_message(monkeypatch: pytest.MonkeyPatch) -> None:
    record: dict[str, Any] = {"choice": {"target_id": "t2", "considered": [
        {"target_id": "t1", "strength": 0.4}, {"target_id": "t2", "strength": 0.7}]}}
    labels = {"t1": "marathon plan", "t2": "hackathon demo"}
    messages = ["ok", "I keep putting off the decision between two job offers", "thanks!"]
    ask = judge_pair.personal_question(lambda: record, labels.get, lambda: messages)
    assert ask() == "Thinking about hackathon demo: what should I focus on next?"
    record["choice"] = {"target_id": None, "considered": [{"target_id": "t1", "strength": 0.4}]}
    assert ask() == "Thinking about marathon plan: what should I focus on next?"
    record.clear()
    assert ask() == "I keep putting off the decision between two job offers. What should I do next?"
    messages.clear()
    assert ask() is None


def test_replies_are_short_and_from_the_nights_own_client() -> None:
    sent: list[tuple[Any, int | None]] = []

    class Client:
        def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float,
                     seed: int | None = None, max_tokens: int | None = None) -> Any:
            sent.append((adapter, max_tokens))
            return type("R", (), {"text": "  a reply  "})()

    reply = judge_pair.client_reply(Client())
    assert reply(None, "q?") == "a reply" and reply(CKPT, "q?") == "a reply"
    assert sent == [(None, judge_pair.REPLY_TOKENS), (CKPT, judge_pair.REPLY_TOKENS)]
    assert judge_pair.REPLY_TOKENS <= 200
    with pytest.raises(ValueError):
        judge_pair.client_reply(type("C", (), {"complete": lambda *a, **k: type("R", (), {"text": " "})()})())(None, "q?")
