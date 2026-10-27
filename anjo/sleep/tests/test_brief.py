# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""The nightly brief (Task 2): a fold with two lines, served and training, that never mix."""
from __future__ import annotations

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.brief import BriefError, BriefState, compile_brief, fold_day
from anjo.sleep.model_api import ChatResult
from anjo.sleep.tests.sleep_fakes import SPARK, WONDER, brief_text, ev


class FakeChat:
    """Spark Open: echoes the evidence ids it was sent into the brief it writes."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, model, messages, tools, timeout_s, *, temperature=None):
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        text = messages[0]["content"]
        new = text.split("NEW:\n", 1)[1].split("\n\nBUDGET_TOKENS", 1)[0]
        prev = text.split("PREVIOUS:\n", 1)[1].split("\n\nNEW:", 1)[0]
        ids = [line.split("id=", 1)[1].split(" ", 1)[0] for line in new.splitlines() if "id=" in line]
        carried = [w for line in prev.splitlines() if line.startswith("ids:") for w in line[4:].split()]
        merged = dict.fromkeys([*carried, *ids])
        return ChatResult(text=brief_text(f"ids: {' '.join(merged)}"), tool_calls=(),
                          model=SPARK_OPEN, input_tokens=10, output_tokens=10, finish_reason="stop")


def _items():
    return [
        ev("own", "keep notes short", kind="preference", ancestry=frozenset()),
        ev("prac", "she liked the long take", kind="practice", ancestry=frozenset({SPARK})),
        ev("hw", "Memento came before Insomnia", kind="homework", ancestry=frozenset({WONDER})),
        ev("unk", "parsed preference", kind="preference", ancestry=None),
    ]


def _sent(chat: FakeChat, n: int = -1) -> str:
    return chat.calls[n]["messages"][0]["content"]


def test_training_brief_built_only_from_trainable_evidence():
    chat = FakeChat()
    compile_brief(None, _items(), include_hosted=False, chat=chat)
    sent = _sent(chat)
    assert "id=own" in sent and "id=prac" in sent
    assert "id=hw" not in sent and "id=unk" not in sent
    assert "INCLUDE_HOSTED: false" in sent


def test_training_brief_admits_hosted_only_when_hosted_may_train_and_still_not_unknown():
    chat = FakeChat()
    compile_brief(None, _items(), include_hosted=True, for_training=True, chat=chat)
    sent = _sent(chat)
    assert "id=hw" in sent and "id=unk" not in sent


def test_served_brief_holds_everything_homework_included():
    chat = FakeChat()
    compile_brief(None, _items(), include_hosted=True, chat=chat)
    sent = _sent(chat)
    assert all(f"id={i}" in sent for i in ("own", "prac", "hw", "unk"))


def test_brief_runs_on_spark_open_at_temperature_zero_with_the_registered_prompt():
    chat = FakeChat()
    compile_brief(None, _items()[:1], budget_tokens=1500, chat=chat)
    call = chat.calls[0]
    assert call["model"] == SPARK_OPEN and call["temperature"] == 0.0
    assert "BUDGET_TOKENS: 1500" in call["messages"][0]["content"]
    assert "PREVIOUS:\nnone" in call["messages"][0]["content"]


def test_served_and_training_briefs_fold_separately():
    chat = FakeChat()
    day1 = _items()
    day2 = [ev("own2", "no flattery", kind="preference", minutes=5)]

    state = fold_day(BriefState(None, None), day1, hosted_in_training=False, chat=chat)
    assert "hw" in (state.served or "") and "hw" not in (state.training or "")

    chat.calls.clear()
    nxt = fold_day(state, day2, hosted_in_training=False, chat=chat)
    served_call, training_call = (c["messages"][0]["content"] for c in chat.calls)
    assert _previous(served_call) == state.served
    assert _previous(training_call) == state.training and "hw" not in _previous(training_call)
    assert "own2" in (nxt.served or "") and "own2" in (nxt.training or "")
    assert "hw" in (nxt.served or "") and "hw" not in (nxt.training or "")


def _previous(sent: str) -> str:
    return sent.split("PREVIOUS:\n", 1)[1].split("\n\nNEW:", 1)[0]


def test_a_day_with_nothing_for_a_line_leaves_that_line_unchanged_without_a_call():
    chat = FakeChat()
    state = BriefState(brief_text("old"), brief_text("old"))
    out = fold_day(state, [ev("hw", kind="homework", ancestry=frozenset({WONDER}))],
                   hosted_in_training=False, chat=chat)
    assert out.training == state.training and len(chat.calls) == 1


def test_held_out_items_never_enter_a_brief():
    with pytest.raises(ValueError, match="held-out"):
        compile_brief(None, [ev("h", held_out=True)], chat=FakeChat())


def test_a_brief_missing_a_section_is_refused():
    def bad(model, messages, tools, timeout_s, *, temperature=None):
        return ChatResult("## Stated preferences\nnone", (), SPARK_OPEN, 1, 1, "stop")

    with pytest.raises(BriefError, match="section"):
        compile_brief(None, _items()[:1], chat=bad)


def test_a_brief_over_budget_is_refused():
    def long(model, messages, tools, timeout_s, *, temperature=None):
        return ChatResult(brief_text("word " * 4000), (), SPARK_OPEN, 1, 1, "stop")

    with pytest.raises(BriefError, match="budget"):
        compile_brief(None, _items()[:1], budget_tokens=100, chat=long)
