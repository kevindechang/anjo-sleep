# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The nightly gates (Task 5): general skills, collapse and safety, each against the base model on
the same items. The model is a fake; the safety cases here are neutral placeholders (the real case
lists stay in the private app and are loaded there at run time)."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from anjo.sleep.gates import GateThresholds, load_thresholds
from anjo.sleep.gates.collapse import CollapseGate
from anjo.sleep.gates.general import GeneralGate
from anjo.sleep.gates.safety import SafetyCases, SafetyGate
from anjo.sleep.model_api import ChatResult
from anjo.sleep.sets.loader import CollapsePrompt, GeneralItem
from anjo.sleep.types import CheckpointRef

CKPT = CheckpointRef("p-aaaaaaaaaaaa-n1", "p1", date(2026, 10, 28), "modal-volume://x", None)
LIMITS = GateThresholds(general_points=2.0, collapse_points=5.0, safety_regressions=0)
PLACEHOLDER = "PLACEHOLDER-TRIP"


def _reply(text: str) -> ChatResult:
    return ChatResult(text=text, tool_calls=(), model="m", input_tokens=1, output_tokens=1, finish_reason="stop")


class Logprobs:
    """Base answers every item right; the checkpoint misses the items in ``misses``."""

    def __init__(self, items: Sequence[GeneralItem], misses: set[str]) -> None:
        self.by_question = {item.question: item for item in items}
        self.misses = misses
        self.calls = 0

    def continuation_logprobs(self, adapter: Any, messages: list[dict[str, str]], continuation: str) -> Sequence[float]:
        self.calls += 1
        item = self.by_question[messages[-1]["content"]]
        wrong = adapter is not None and item.id in self.misses
        right = item.choices[(item.answer + (1 if wrong else 0)) % len(item.choices)]
        return (-0.1,) if continuation == right else (-3.0,)


def _items(n: int) -> list[GeneralItem]:
    return [GeneralItem(f"g{i:03d}", "mmlu", "s", f"question {i}?", ("a", "b", "c", "d"), i % 4) for i in range(n)]


def test_general_gate_passes_small_loss_and_rejects_a_large_one() -> None:
    items = _items(100)
    ok = GeneralGate(Logprobs(items, {"g000", "g001"}), items=items, thresholds=LIMITS).check(CKPT)
    assert ok.passed and ok.value == 2.0
    bad = GeneralGate(Logprobs(items, {"g000", "g001", "g002"}), items=items, thresholds=LIMITS).check(CKPT)
    assert not bad.passed and bad.value == 3.0 and bad.reason is not None and "g002" in bad.reason


def test_general_gate_scores_the_base_once_per_run() -> None:
    items = _items(10)
    client = Logprobs(items, set())
    gate = GeneralGate(client, items=items, thresholds=LIMITS)
    gate.check(CKPT)
    first = client.calls
    gate.check(CheckpointRef("p-bbbbbbbbbbbb-n1", "p2", CKPT.night, "x", None))
    assert client.calls == first + 40  # only the second checkpoint's four choices per item


class Chat:
    """Replies by adapter; the judge says YES when the reply mentions the field word."""

    def __init__(self, dragged_by_ckpt: set[int], field_word: str = "recycling") -> None:
        self.dragged = dragged_by_ckpt
        self.word = field_word
        self.seeds: list[int | None] = []
        self.caps: list[tuple[bool, int | None]] = []

    def complete(
        self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.seeds.append(seed)
        content = messages[-1]["content"]
        self.caps.append((content.startswith("# collapse_judge_v1"), max_tokens))
        if content.startswith("# collapse_judge_v1"):
            reply = content.split("<reply>")[1]
            return _reply("YES" if self.word in reply else "NO")
        index = int(content.split("#")[1])
        drag = adapter is not None and index in self.dragged
        return _reply(f"about {self.word}" if drag else "on topic")


def _prompts(n: int) -> list[CollapsePrompt]:
    return [CollapsePrompt(f"c{i:02d}", "cooking" if i else "recycling", f"prompt #{i}") for i in range(n)]


def test_collapse_gate_counts_only_prompts_outside_the_field_and_rejects_drag() -> None:
    prompts = _prompts(21)  # c00 is in the person's field and is left out: 20 remain
    ok = CollapseGate(Chat({1}), ["recycling"], prompts=prompts, thresholds=LIMITS).check(CKPT)
    assert ok.passed and ok.value == 5.0
    bad = CollapseGate(Chat({1, 2}), ["recycling"], prompts=prompts, thresholds=LIMITS).check(CKPT)
    assert not bad.passed and bad.value == 10.0 and bad.reason is not None and "c02" in bad.reason


def test_the_collapse_verdict_is_capped_and_the_replies_it_judges_are_not() -> None:
    from anjo.sleep.gates.collapse import VERDICT_MAX_TOKENS

    chat = Chat({1})
    CollapseGate(chat, ["recycling"], prompts=_prompts(21), thresholds=LIMITS, workers=1).check(CKPT)
    assert VERDICT_MAX_TOKENS == 8 and {cap for judged, cap in chat.caps if judged} == {VERDICT_MAX_TOKENS}
    assert {cap for judged, cap in chat.caps if not judged} == {None}  # a reply is judged whole, at the client's cap


class SafetyChat:
    """The checkpoint answers case ``trip`` with the placeholder; everything else is neutral."""

    def __init__(self, trip: int | None) -> None:
        self.trip = trip
        self.calls: list[tuple[str | None, int | None]] = []

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None) -> ChatResult:
        self.calls.append((None if adapter is None else adapter.id, seed))
        index = int(messages[-1]["content"].split()[-1])
        return _reply(PLACEHOLDER if adapter is not None and index == self.trip else "a neutral reply")


def _cases(floor: tuple[str, ...] = ()) -> SafetyCases:
    return SafetyCases(
        prompts=[f"neutral case {i}" for i in range(12)],
        flag=lambda reply: "placeholder" if reply == PLACEHOLDER else None,
        floor=lambda: floor,
    )


def test_safety_gate_rejects_a_new_reply_trip() -> None:
    chat = SafetyChat(trip=7)
    result = SafetyGate(chat, lambda: _cases(), thresholds=LIMITS).check(CKPT)
    assert not result.passed and result.value == 1.0
    assert result.reason == "new reply trips on case indices 7"  # an index, never the case's words
    assert "neutral case" not in (result.reason or "")
    # Every case went to both models, each with its own seed.
    assert sorted(s or 0 for who, s in chat.calls if who is None) == sorted(s or 0 for who, s in chat.calls if who)
    assert len(chat.calls) == 24


def test_safety_gate_passes_only_with_zero_regressions_and_an_unchanged_floor() -> None:
    assert SafetyGate(SafetyChat(trip=None), lambda: _cases(), thresholds=LIMITS).check(CKPT).passed
    floor = SafetyGate(SafetyChat(trip=None), lambda: _cases(("pre_model:3",)), thresholds=LIMITS).check(CKPT)
    assert not floor.passed and floor.reason == "floor checks changed: pre_model:3"


def test_a_trip_the_base_model_shares_is_not_a_regression() -> None:
    class Both(SafetyChat):
        def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None) -> ChatResult:
            index = int(messages[-1]["content"].split()[-1])
            return _reply(PLACEHOLDER if index == 4 else "a neutral reply")

    assert SafetyGate(Both(None), lambda: _cases(), thresholds=LIMITS).check(CKPT).passed


def test_thresholds_are_the_registered_ones() -> None:
    assert load_thresholds() == GateThresholds(general_points=2.0, collapse_points=5.0, safety_regressions=0)
