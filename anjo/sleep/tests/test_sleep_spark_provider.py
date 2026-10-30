"""Sleep's provider selection (``ANJO_SLEEP_SPARK``): one switch moves every Spark Open call, its
hold and the id it records to Fireworks. Default (unset or ``baseten``) is today's path. No network:
every client here is a fake."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pytest

from anjo.sleep.mind.apd.retry import reply_timeout_s

from anjo.sleep import baseten_api, fireworks_api, spark_inquiry, spark_provider
from anjo.sleep.budget import Bound, Ledger
from anjo.sleep.context.brief_store import folded_ancestry
from anjo.sleep.evidence.extract import Turn, extract
from anjo.sleep.model_api import ChatResult
from anjo.sleep.provenance import SPARK_OPEN_IDS
from anjo.sleep.types import Evidence

messages: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]


def _result(model: str) -> ChatResult:
    return ChatResult(text="{}", tool_calls=(), model=model, input_tokens=5, output_tokens=1, finish_reason="stop")


def test_default_provider() -> None:
    selected = spark_provider.from_env({})
    assert (selected.model, selected.chat, selected.pool, selected.price) == (
        baseten_api.SPARK_OPEN, baseten_api.chat, "baseten_sleep_usd", None)


def test_fireworks_provider() -> None:
    selected = spark_provider.from_env({"ANJO_SLEEP_SPARK": "fireworks"})
    assert (selected.model, selected.chat, selected.pool, selected.price) == (
        fireworks_api.SPARK_OPEN, fireworks_api.chat, "fireworks_sleep_usd", (1.25, 4.25))
    assert selected.model in SPARK_OPEN_IDS


def test_baseten_spelled_out_is_the_default() -> None:
    assert spark_provider.from_env({spark_provider.SETTING: "baseten"}) == spark_provider.from_env({})
    assert spark_provider.from_env({spark_provider.SETTING: " FIREWORKS "}) == spark_provider.from_env(
        {spark_provider.SETTING: "fireworks"})


def test_invalid_provider() -> None:
    with pytest.raises(ValueError, match="ANJO_SLEEP_SPARK"):
        spark_provider.from_env({"ANJO_SLEEP_SPARK": "unknown"})


def test_a_spark_chat_is_unchanged_without_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(spark_provider.SETTING, raising=False)
    posted: list[dict[str, Any]] = []

    def baseten(model: str, messages: Any, tools: Any, timeout_s: float, **kw: Any) -> ChatResult:
        posted.append({"model": model, "tools": tools, **kw})
        return _result(model)

    def never(*args: Any, **kw: Any) -> ChatResult:
        raise AssertionError("no Fireworks call without the switch")

    monkeypatch.setattr(baseten_api, "post_chat", baseten)
    monkeypatch.setattr(fireworks_api, "chat", never)
    result = baseten_api.chat(baseten_api.SPARK_OPEN, messages, None, 30.0, temperature=0.0, max_tokens=64, seed=7)
    assert result.model == baseten_api.SPARK_OPEN
    (posted_call,) = posted
    assert (posted_call["provider"], posted_call["url"], posted_call["key_env"], posted_call["model"]) == (
        "Baseten", baseten_api.URL, baseten_api.KEY_ENV, baseten_api.SPARK_OPEN)
    assert (posted_call["tools"], posted_call["temperature"], posted_call["max_tokens"], posted_call["extra"]) == (
        None, 0.0, 64, {"seed": 7})


def test_a_spark_chat_reaches_the_fireworks_client_when_switched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(spark_provider.SETTING, "fireworks")
    sent: list[dict[str, Any]] = []

    def fireworks(model: str, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None,
                  max_tokens: int | None = None, seed: int | None = None, transport: Any = None,
                  retries: Any = None) -> ChatResult:
        sent.append({"model": model, "messages": messages, "tools": tools, "timeout_s": timeout_s,
                     "temperature": temperature, "max_tokens": max_tokens, "seed": seed, "retries": retries})
        return _result(model)

    monkeypatch.setattr(fireworks_api, "chat", fireworks)
    result = baseten_api.chat(baseten_api.SPARK_OPEN, messages, None, 30.0, temperature=0.0, max_tokens=64, seed=7)
    assert sent == [{"model": fireworks_api.SPARK_OPEN, "messages": messages, "tools": None, "timeout_s": reply_timeout_s(64),
                     "temperature": 0.0, "max_tokens": 64, "seed": 7,
                     "retries": fireworks_api.SLEEP_RETRIES}]
    assert result.model == fireworks_api.SPARK_OPEN


def test_spark_inquiry_routes_through_the_same_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(spark_provider.SETTING, "fireworks")
    sent: list[dict[str, Any]] = []

    def fireworks(model: str, messages: Any, tools: Any, timeout_s: float, **kw: Any) -> ChatResult:
        sent.append({"model": model, "tools": tools, **kw})
        return _result(model)

    monkeypatch.setattr(fireworks_api, "chat", fireworks)
    reply = spark_inquiry.spark_open(messages, 10)
    (sent_call,) = sent
    assert (sent_call["model"], sent_call["tools"], sent_call["temperature"], sent_call["max_tokens"]) == (
        fireworks_api.SPARK_OPEN, None, 0.0, 10)
    assert reply.model == fireworks_api.SPARK_OPEN


def test_extraction_accepts_and_records_the_switched_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(spark_provider.SETTING, "fireworks")
    turn = Turn(source="message:0", role="user", text="I shoot on 16mm.", ancestry=frozenset())

    class Fake:
        def __call__(self, messages: list[dict[str, Any]]) -> ChatResult:
            return ChatResult(text='{"items": [{"turn": 0, "kind": "lesson", "text": "They shoot on 16mm."}]}',
                              tool_calls=(), model=fireworks_api.SPARK_OPEN, input_tokens=5, output_tokens=1,
                              finish_reason="stop")

    # The configured Baseten id is accepted too: both name the cleared model.
    accepted = extract("p", date(2026, 10, 28), [turn], imported=(), targets=(), known_ids=frozenset(),
                       now=datetime(2026, 10, 29, 2, tzinfo=timezone.utc), chat=Fake(),
                       model=baseten_api.SPARK_OPEN)
    assert len(accepted.items) == 1
    # With no id configured, the switched provider's id is both accepted and recorded.
    found = extract("p", date(2026, 10, 28), [turn], imported=(), targets=(), known_ids=frozenset(),
                    now=datetime(2026, 10, 29, 2, tzinfo=timezone.utc), chat=Fake())
    assert [item.model for item in found.items] == [fireworks_api.SPARK_OPEN]
    assert found.items[0].ancestry == frozenset({fireworks_api.SPARK_OPEN})


def test_a_brief_records_the_switched_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    item = Evidence(id="e-1", person_id="p", day=date(2026, 10, 28), kind="lesson", text="t", source="s",
                    held_out=False, created_at=datetime(2026, 10, 28, tzinfo=timezone.utc), model=None,
                    ancestry=frozenset())
    monkeypatch.delenv(spark_provider.SETTING, raising=False)
    assert folded_ancestry(frozenset(), [item]) == frozenset({baseten_api.SPARK_OPEN})
    monkeypatch.setenv(spark_provider.SETTING, "fireworks")
    assert folded_ancestry(frozenset(), [item]) == frozenset({fireworks_api.SPARK_OPEN})


def test_the_switch_moves_a_baseten_hold_to_the_fireworks_pool(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(spark_provider.SETTING, "fireworks")
    ledger = Ledger(tmp_path / "budget.jsonl")

    held = ledger.reserve("brief", ("baseten_sleep_usd",), Bound(rate_per_hour=4.0, timeout_s=3600.0))
    assert held is not None and held.pools == ("fireworks_sleep_usd",)
    assert ledger.used("fireworks_sleep_usd") == pytest.approx(4.0) and ledger.used("baseten_sleep_usd") == 0.0

    held.settle(1.5)
    assert ledger.used("fireworks_sleep_usd") == pytest.approx(1.5) and ledger.used("baseten_sleep_usd") == 0.0

    mind = ledger.reserve("mind", ("baseten_mind_usd",), Bound(rate_per_hour=2.0, timeout_s=3600.0))
    assert mind is not None and mind.pools == ("fireworks_sleep_usd",)
    mind.release()
    assert ledger.used("fireworks_sleep_usd") == pytest.approx(1.5) and ledger.used("baseten_mind_usd") == 0.0
