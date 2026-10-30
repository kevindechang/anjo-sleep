# mypy: ignore-errors
"""Morning judge selection and accounting, with no provider calls."""
import json
from dataclasses import replace

import pytest
import httpx

from anjo.sleep import baseten_api, fireworks_api, meta_api
from anjo.sleep.app import morning as app
from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.model_api import ChatResult, Retries
from anjo.sleep.probes.score import JudgeTokens, MuseJudge, judge_orders
from anjo.sleep.probes.morning import DEEPSEEK_JUDGE_USD_PER_CALL, _HeldDeepseekChat
from anjo.sleep.stops import BudgetStop
from anjo.sleep.tests.sleep_fakes import probe
from anjo.sleep.tests.test_morning import Rig


def test_selection_uses_same_prompt_and_settings(monkeypatch):
    calls = []

    def chat(model, messages, tools, timeout, **kwargs):
        calls.append((model, messages, tools, timeout, kwargs))
        return ChatResult('{"verdict":"tie"}', (), model, 3000, 500, "stop")

    monkeypatch.setattr(meta_api, "chat", chat)
    monkeypatch.setattr(fireworks_api, "chat", chat)
    monkeypatch.setattr(app, "judge_brief", lambda persona: "brief")
    monkeypatch.delenv(app.JUDGE_ENV, raising=False)
    muse = app.make_judge("film", JudgeTokens(), app._Resent())
    assert judge_orders(muse, probe(), "one", "two") == ("tie", False)
    monkeypatch.setenv(app.JUDGE_ENV, "deepseek")
    deepseek = app.make_judge("film", JudgeTokens(), app._Resent())
    assert judge_orders(deepseek, probe(), "one", "two") == ("tie", False)
    deepseek.again(probe(), "one", "two")
    assert [c[0] for c in calls] == ["muse-spark"] * 2 + [fireworks_api.DEEPSEEK_V4P1_FLASH] * 3
    assert calls[0][1:4] == calls[2][1:4] and calls[1][1:4] == calls[3][1:4]
    assert calls[0][4]["temperature"] == calls[2][4]["temperature"] == 0
    assert calls[4][4]["max_tokens"] == 16384
    assert all("retries" in c[4] for c in calls)


@pytest.mark.parametrize("model", [baseten_api.SPARK_OPEN, fireworks_api.SPARK_OPEN, "muse-spark-open"])
def test_spark_open_refused(model, monkeypatch):
    with pytest.raises(ValueError, match="Spark Open"):
        MuseJudge(lambda *a, **k: None, lambda _: "", model=model)
    monkeypatch.setenv(app.JUDGE_ENV, model)
    with pytest.raises(ValueError, match="muse or deepseek"):
        app.make_judge("film", JudgeTokens(), app._Resent())


def test_deepseek_accounting_and_record(tmp_path):
    ledger = Ledger(tmp_path / "budget.jsonl")
    rig = Rig(tmp_path, ledger=ledger)
    tokens = JudgeTokens()

    def chat(model, messages, tools, timeout, **kwargs):
        return ChatResult('{"verdict":"tie"}', (), model, 3000, 500, "stop")

    model = fireworks_api.DEEPSEEK_V4P1_FLASH
    rig.morning = replace(rig.morning, judge=MuseJudge(chat, lambda _: "brief", tokens, model),
                          judge_tokens=tokens, judge_model=model)
    done = rig.run()
    assert json.loads(done.path.read_text())["judge_model"] == model
    assert done.judge_usd == pytest.approx(done.judge_calls * (3000 * 0.30 + 500 * 1.20) / 1e6)
    events = read_records(ledger.path, PLAIN)
    assert next(e for e in events if e["op"] == "reserve")["pools"] == ["fireworks_sleep_usd", "judge_usd"]
    assert ledger.used("meta_usd") == 0
    assert ledger.used("judge_usd") == pytest.approx(done.judge_usd)


def test_deepseek_judge_pool_refusal_before_calls(tmp_path):
    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "judge_usd": 0})
    rig = Rig(tmp_path, ledger=ledger)
    judge = MuseJudge(lambda *a, **k: None, lambda _: "", model=fireworks_api.DEEPSEEK_V4P1_FLASH)
    rig.morning = replace(rig.morning, judge=judge)
    with pytest.raises(BudgetStop):
        rig.run()
    assert rig.judge.calls == 0 and rig.model.calls == []


def test_model_identity_cannot_disagree_with_judge(tmp_path):
    rig = Rig(tmp_path)
    judge = MuseJudge(lambda *a, **k: None, lambda _: "", model=fireworks_api.DEEPSEEK_V4P1_FLASH)
    assert replace(rig.morning, judge=judge).judge_model == fireworks_api.DEEPSEEK_V4P1_FLASH
    with pytest.raises(ValueError, match="does not match"):
        replace(rig.morning, judge=judge, judge_model="muse-spark-1.3")


def test_fireworks_chat_retries_transient_status(monkeypatch):
    requests = []
    resent = []

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"model": fireworks_api.DEEPSEEK_V4P1_FLASH,
                                       "choices": [{"message": {"content": '{"verdict":"tie"}'},
                                                    "finish_reason": "stop"}],
                                       "usage": {"prompt_tokens": 5, "completion_tokens": 3}})

    monkeypatch.setenv(fireworks_api.KEY_ENV, "test-key")
    result = fireworks_api.chat(fireworks_api.DEEPSEEK_V4P1_FLASH, [{"role": "user", "content": "x"}], None, 10,
                                temperature=0, transport=httpx.MockTransport(respond),
                                retries=Retries(sleep=lambda _: None, counted=lambda: resent.append(True)))
    assert result.input_tokens == 5 and len(requests) == 2 and resent == [True]
    assert requests[0] == requests[1] and "max_completion_tokens" not in requests[0]


@pytest.mark.parametrize("fail", [False, True])
def test_unmetered_or_failed_call_settles_full_bound(tmp_path, fail):
    ledger = Ledger(tmp_path / "budget.jsonl")

    def chat(*args, **kwargs):
        if fail:
            raise RuntimeError("provider failed")
        return ChatResult('{"verdict":"tie"}', (), "model", 0, 0, "stop")

    held_chat = _HeldDeepseekChat(chat, ledger, "morning:judge")
    if fail:
        with pytest.raises(RuntimeError):
            held_chat("model", [], None, 1)
    else:
        held_chat("model", [], None, 1)
    assert ledger.used("judge_usd") == pytest.approx(DEEPSEEK_JUDGE_USD_PER_CALL)
    assert ledger.used("fireworks_sleep_usd") == pytest.approx(DEEPSEEK_JUDGE_USD_PER_CALL)
    assert not ledger.open_holds()
