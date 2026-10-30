# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Spark Open behind MIND's model ports (Task 17): each call capped, a failed call charged its cap,
and the resolver shown the claim and the evidence only. Offline: every chat here is a fake."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import pytest

from anjo.sleep import baseten_api, spark_inquiry
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.ports import PredictionResolver, Thinker, WorldQuery
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.spark_inquiry import (
    RESOLVE_MAX_TOKENS,
    THINK_MAX_TOKENS,
    WORLD_MAX_TOKENS,
    SparkResolver,
    SparkThinker,
    SparkWorldQuery,
)
from anjo.sleep.types import Evidence

LOCKED = Evidence(
    id="e-d6-1", person_id="p", day=date(2026, 10, 7), kind="lesson", text="The courtyard is locked on Thursday.",
    source="message:e-d6-1", held_out=False, created_at=datetime(2026, 10, 7, 12, tzinfo=timezone.utc),
    model=None, ancestry=frozenset(),
)


@dataclass
class FakeChat:
    reply: str = "{}"
    output_tokens: int = 42
    error: bool = False
    calls: list[tuple[list[dict[str, Any]], int]] = field(default_factory=list)

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self.calls.append((messages, max_tokens))
        if self.error:
            raise ModelApiError("Baseten chat with spark returned HTTP 503")
        return ChatResult(text=self.reply, tool_calls=(), model=baseten_api.SPARK_OPEN, input_tokens=900,
                          output_tokens=self.output_tokens, finish_reason="stop")


def test_adapters_fit_the_ports() -> None:
    chat = FakeChat()
    assert isinstance(SparkThinker(chat), Thinker)
    assert isinstance(SparkWorldQuery(chat), WorldQuery)
    assert isinstance(SparkResolver(chat), PredictionResolver)


def test_thinker_sends_the_render_alone_and_caps_its_reply() -> None:
    chat = FakeChat(reply='{"op": "PARK", "question_id": "q2"}')
    reply = SparkThinker(chat).propose("RENDERED WORKSPACE", 20_000)
    assert reply == it.ModelReply(text='{"op": "PARK", "question_id": "q2"}', tokens=42, model=baseten_api.SPARK_OPEN)
    assert chat.calls == [([{"role": "user", "content": "RENDERED WORKSPACE"}], THINK_MAX_TOKENS)]
    SparkThinker(chat).propose("again", 300)
    assert chat.calls[-1][1] == 300


def test_a_failed_call_is_empty_text_charged_its_cap(caplog: pytest.LogCaptureFixture) -> None:
    chat = FakeChat(error=True)
    assert SparkThinker(chat).propose("r", 20_000) == it.ModelReply("", THINK_MAX_TOKENS, baseten_api.SPARK_OPEN)
    assert SparkWorldQuery(chat).ask("q", 250) == it.ModelReply("", 250, baseten_api.SPARK_OPEN)
    assert SparkResolver(chat).resolve("claim", [LOCKED]) == it.ResolverReply(
        "undetermined", RESOLVE_MAX_TOKENS, failed=True)
    assert "HTTP 503" in caplog.text


def test_nothing_left_means_no_call() -> None:
    chat = FakeChat()
    assert SparkThinker(chat).propose("r", 0) == it.ModelReply("", 0, baseten_api.SPARK_OPEN)
    assert SparkWorldQuery(chat).ask("q", 0).tokens == 0 and chat.calls == []


def test_world_query_asks_general_knowledge_under_a_fixed_preamble() -> None:
    chat = FakeChat(reply="About 3 m.")
    reply = SparkWorldQuery(chat).ask("How far back does an 85 mm medium shot need?", 20_000)
    assert reply.text == "About 3 m." and reply.tokens == 42
    (messages, cap), = chat.calls
    assert cap == WORLD_MAX_TOKENS and messages[0] == {"role": "system", "content": spark_inquiry.WORLD_PROMPT}
    assert messages[1] == {"role": "user", "content": "How far back does an 85 mm medium shot need?"}
    assert "general knowledge" in spark_inquiry.WORLD_PROMPT


def test_resolver_sees_only_the_claim_and_the_evidence() -> None:
    chat = FakeChat(reply='{"verdict": "false"}')
    reply = SparkResolver(chat).resolve("The courtyard is open for the scene 4 shoot", [LOCKED])
    assert reply == it.ResolverReply("false", 42)
    (messages, cap), = chat.calls
    assert cap == RESOLVE_MAX_TOKENS and messages[0] == {"role": "system", "content": spark_inquiry.RESOLVE_PROMPT}
    assert messages[1] == {"role": "user", "content": (
        "Claim: The courtyard is open for the scene 4 shoot\n\n"
        "Evidence:\n- (2026-10-07) The courtyard is locked on Thursday."
    )}
    for word in ("hypothes", "credence", "question"):
        assert word not in spark_inquiry.RESOLVE_PROMPT.casefold()


@pytest.mark.parametrize(("text", "verdict", "failed"), [
    ('{"verdict": "false"}', "false", False),
    ('```json\n{"verdict": "TRUE", "why": "the courtyard opened"}\n```', "true", False),
    ("Undetermined", "undetermined", False),
    (" false. ", "false", False),
    ("It is probably false", "undetermined", True),
    ('{"verdict": "maybe"}', "undetermined", True),
    ('{"answer": "true"}', "undetermined", True),
    ("", "undetermined", True),
])
def test_resolver_reads_its_verdict_leniently_and_counts_anything_else_failed(
    text: str, verdict: it.Verdict, failed: bool,
) -> None:
    """Only a real verdict marks evidence seen; a reply out of form is a failed call, retried later."""
    reply = SparkResolver(FakeChat(reply=text)).resolve("claim", [LOCKED])
    assert (reply.verdict, reply.failed, reply.tokens) == (verdict, failed, 42)


def test_the_default_chat_is_spark_open_on_baseten(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def post_chat(**kwargs: Any) -> ChatResult:
        seen.update(kwargs)
        return ChatResult(text="ok", tool_calls=(), model=baseten_api.SPARK_OPEN, input_tokens=1, output_tokens=1,
                          finish_reason="stop")

    monkeypatch.setattr(baseten_api, "post_chat", post_chat)
    chat: Callable[[list[dict[str, Any]], int], ChatResult] = spark_inquiry.spark_open
    chat([{"role": "user", "content": "x"}], 77)
    assert (seen["provider"], seen["url"], seen["key_env"], seen["model"]) == (
        "Baseten", baseten_api.URL, baseten_api.KEY_ENV, baseten_api.SPARK_OPEN)
    assert (seen["max_tokens"], seen["temperature"], seen["tools"]) == (77, 0.0, None)
