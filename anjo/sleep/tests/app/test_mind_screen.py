# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's domain screen on Spark Open (Task 17), behind the kernel's denylist: what she may
investigate. A research query never carries the person's names or a date, and a failed or
garbled screen refuses. Offline: the chat is a fake."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.ports import ScreenPort
from anjo.sleep.model_api import ChatResult, ModelApiError

NAMES = ("maraq", "Mara Quint")


@dataclass
class FakeChat:
    reply: str = '{"allowed": true}'
    error: bool = False
    calls: list[tuple[list[dict[str, Any]], int]] = field(default_factory=list)

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self.calls.append((messages, max_tokens))
        if self.error:
            raise ModelApiError("Baseten chat with spark timed out after 120s")
        return ChatResult(text=self.reply, tool_calls=(), model=SPARK_OPEN, input_tokens=300, output_tokens=9,
                          finish_reason="stop")


def _screen(chat: FakeChat) -> Any:
    from anjo.sleep.app.screen import SparkScreen

    return SparkScreen(names=NAMES, chat=chat)


def test_the_screen_fits_its_port() -> None:
    assert isinstance(_screen(FakeChat()), ScreenPort)


@pytest.mark.parametrize("query", [
    "Mara 85mm lens distance", "quint kitchen lighting", "lens rental on Thursday", "film festival October deadline",
    "shoot schedule 10/09", "permits 2026-10-09", "ask someone@example.org", "see example.org lenses", "zip 10012 rental",
])
def test_research_query_carries_no_personal_terms(query: str) -> None:
    chat = FakeChat()
    assert _screen(chat).screen(query, "research_query", 20_000) == it.Screened(allowed=False, tokens=0)
    assert chat.calls == []  # refused before any call


@pytest.mark.parametrize("query", [
    "85mm medium shot camera distance full frame", "f/1.4 depth of field at 3 m", "1/125 shutter handheld blur",
    "24-70mm zoom small kitchen", "march of the penguins camera rigs", "may a 50mm replace an 85mm",
])
def test_a_topical_research_query_goes_to_the_model(query: str) -> None:
    chat = FakeChat()
    assert _screen(chat).screen(query, "research_query", 20_000) == it.Screened(allowed=True, tokens=9)
    (messages, cap), = chat.calls
    from anjo.sleep.app.screen import SCREEN_MAX_TOKENS, SCREEN_PROMPT

    assert cap == SCREEN_MAX_TOKENS and messages[0] == {"role": "system", "content": SCREEN_PROMPT}
    assert messages[1] == {"role": "user", "content": f"Purpose: research_query\nText: {query}"}
    assert all(name.casefold() not in str(messages).casefold() for name in NAMES)


def test_questions_and_hypotheses_are_the_models_call() -> None:
    chat = FakeChat(reply='Thinking... {"allowed": false, "why": "it diagnoses the person"}')
    screened = _screen(chat).screen("Whether Mara Quint is avoiding the shoot", "question", 20_000)
    assert screened == it.Screened(allowed=False, tokens=9)
    assert _screen(FakeChat()).screen("The plan assumes room the kitchen lacks", "hypothesis", 20_000).allowed


@pytest.mark.parametrize("reply", ["", "yes", '{"allowed": "true"}', '{"verdict": true}', "[]"])
def test_a_reply_out_of_form_refuses(reply: str) -> None:
    assert _screen(FakeChat(reply=reply)).screen("scene 4 plan", "question", 20_000) == it.Screened(False, 9)


def test_a_failed_call_refuses_and_is_charged_its_cap(caplog: pytest.LogCaptureFixture) -> None:
    from anjo.sleep.app.screen import SCREEN_MAX_TOKENS

    screen = _screen(FakeChat(error=True))
    assert screen.screen("scene 4 plan", "question", 20_000) == it.Screened(False, SCREEN_MAX_TOKENS)
    assert screen.screen("scene 4 plan", "question", 40) == it.Screened(False, 40)
    assert "timed out" in caplog.text


def test_no_allowance_left_refuses_without_a_call() -> None:
    chat = FakeChat()
    assert _screen(chat).screen("scene 4 plan", "question", 0) == it.Screened(False, 0) and chat.calls == []


def test_screen_for_reads_the_persons_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.app import screen

    monkeypatch.setattr(screen, "person_names", lambda user_id: NAMES if user_id == "person-1" else ())
    built = screen.screen_for("person-1")
    assert built.names == NAMES and built.screen("Mara lens", "research_query", 100).allowed is False
