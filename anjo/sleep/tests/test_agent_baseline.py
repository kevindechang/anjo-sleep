# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The comparison's conventional agent (arm 2A), on fakes: the night's arm and the store's rules.

Off it is a no-op, so a night without the switch is the night it always was. On, it writes
``agent_note`` items that Spark Open's cleared ancestry alone would let train, so the kind is what
keeps every training input and every other reader away from them. Two nights carry nothing across
but the store and the brief, and the pass never spends past MIND's inquiry's token cap.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.agent_baseline.night import MAX_STEPS, POOL, SEARCH_K, AgentNight
from anjo.sleep.agent_baseline.types import AgentCall
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.budget import Ledger
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.evidence.store import AGENT_NOTE
from anjo.sleep.homework.research import WonderResearch
from anjo.sleep.mind.inquiry.types import TOKEN_CAP
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.meter import SpendMeter, WonderMeter
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.provenance import may_train
from anjo.sleep.tests.homework_fakes import FakeSearch, FakeWonder, research_script, study_web
from anjo.sleep.tests.night_fakes import NIGHT, World, own, step, steps, summary, window
from anjo.sleep.types import NightResult

UID = "p1"
NOTE_TEXT = "Optical sorters miss black film unless it is lit from below."
PRICES = Prices(spark_in_per_m=0.2, spark_out_per_m=0.6, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.95)


class Script:
    """Spark Open's replies in order (``done`` after them), and every prompt and cap it was given."""

    def __init__(self, *replies: Mapping[str, Any], tokens: int = 40) -> None:
        self.replies = [dict(reply) for reply in replies]
        self.tokens = tokens
        self.prompts: list[str] = []
        self.caps: list[int] = []

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self.caps.append(max_tokens)
        self.prompts.append("\n".join(str(message["content"]) for message in messages))
        reply = self.replies.pop(0) if self.replies else {"op": "done"}
        return ChatResult(text=json.dumps(reply), tool_calls=(), model=SPARK_OPEN, input_tokens=120,
                          output_tokens=min(self.tokens, max_tokens), finish_reason="stop")


class Burns:
    """Replies with a recall, writing exactly what its cap allows: only the cap can stop it."""

    def __init__(self, tokens: int) -> None:
        self.tokens = tokens
        self.caps: list[int] = []

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self.caps.append(max_tokens)
        return ChatResult(text=json.dumps({"op": "recall", "query": "sorting line"}), tool_calls=(),
                          model=SPARK_OPEN, input_tokens=10, output_tokens=min(self.tokens, max_tokens),
                          finish_reason="stop")


class Fails:
    """A provider that refuses every call: its usage is unknown, so the cap is charged whole."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self.calls += 1
        raise ModelApiError("the provider answered 500")


def _one(world: World, *, night: date = NIGHT, **deps: Any) -> NightResult:
    return run_night_batch_with(world.deps(**deps), {UID: night})[UID]


def _call(world: World, night: date = NIGHT) -> AgentCall:
    end = window(night)[1]
    return AgentCall(
        user_id=UID, night=night, as_of=end, at=end, store=world.evidence(UID), wonder=WonderMeter(12, 3.0),
        spend=SpendMeter(spark_in_per_m=PRICES.spark_in_per_m, spark_out_per_m=PRICES.spark_out_per_m),
        job=f"night:agent:{UID}:{night.isoformat()}", brief="She is rebuilding a film-plastics sorting line.",
    )


# ── off ──────────────────────────────────────────────────────────────────────


def test_off_is_a_no_op(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT)
    result = _one(world)
    assert "agent" not in [record["step"] for record in steps(result.log_path)]
    done = summary(result.log_path)
    assert "agent" not in done["cost_parts"] and "agent" not in done
    assert world.evidence(UID).agent_notes(NIGHT) == []


# ── on: the notes, and what refuses them ─────────────────────────────────────


def test_on_writes_agent_notes_that_training_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANJO_SLEEP_SPARK", raising=False)  # the default provider: Sleep's Baseten pool
    world = World(tmp_path)
    world.plan(UID, NIGHT)
    chat = Script({"op": "recall", "query": "sorting line"},
                  {"op": "note", "text": NOTE_TEXT, "cites": [own(UID, NIGHT, 0).id]},
                  {"op": "done"})
    ledger = Ledger(tmp_path / "budget.jsonl")
    result = _one(world, baseline=AgentNight(prices=PRICES, chat=chat, ledger=ledger))

    store = world.evidence(UID)
    [note] = store.agent_notes(NIGHT)
    assert note.kind == AGENT_NOTE and note.id.startswith("agent-") and note.text == NOTE_TEXT
    assert note.ancestry == frozenset({SPARK_OPEN}) and note.model == SPARK_OPEN
    assert may_train(note.ancestry, hosted_in_training=False)  # Spark Open's ancestry is cleared
    assert note not in store.trainable(NIGHT)  # so the kind is what keeps it out of training
    assert note not in store.stored(NIGHT) and note not in store.for_day(NIGHT)
    assert note.id not in {item.id for item in store.search(NOTE_TEXT, window(NIGHT)[1], 5)}
    assert note.id not in world.briefs(UID).folded_ids()

    done = summary(result.log_path)
    assert done["cost_parts"]["agent"] > 0 and done["agent"]["notes"] == [note.id]
    assert done["agent"]["stopped"] == "done" and step(result.log_path, "agent")["notes"] == 1
    assert step(result.log_path, "agent")["stopped"] == "done"
    order = [record["step"] for record in steps(result.log_path)]
    assert order.index("choose") < order.index("agent") < order.index("brief")  # after MIND's step
    assert ledger.used(POOL) == pytest.approx(done["cost_parts"]["agent"])
    assert f"night:agent:{UID}:{NIGHT}:spark:a1" in ledger.settlements()


def test_the_hold_lands_on_the_pool_the_provider_bills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The arm names Sleep's Spark pool, as ``night.mind`` does; the ledger moves the hold when
    ``ANJO_SLEEP_SPARK`` sends Spark Open to Fireworks, so it never holds a credit it won't spend."""
    world = World(tmp_path)

    def reserved(provider: str | None) -> tuple[str, ...]:
        if provider is None:
            monkeypatch.delenv("ANJO_SLEEP_SPARK", raising=False)
        else:
            monkeypatch.setenv("ANJO_SLEEP_SPARK", provider)
        ledger = Ledger(tmp_path / str(provider) / "budget.jsonl")
        AgentNight(prices=PRICES, chat=Script({"op": "done"}), ledger=ledger).night(_call(world))
        records = read_records(ledger.path, PLAIN)
        [hold] = [record for record in records if record["op"] == "reserve"]
        assert records[-1]["op"] == "settle"  # settled on the pool the hold names
        return tuple(hold["pools"])

    assert reserved(None) == ("baseten_sleep_usd",)  # unset: Spark Open bills Baseten
    assert reserved("fireworks") == ("fireworks_sleep_usd",)  # the switch: the ledger remaps it


def test_recall_drops_practice_scenes_without_shrinking_the_answer(tmp_path: Path) -> None:
    """Her rehearsal is not evidence: drop it after the store's cut, so a recall still returns
    ``SEARCH_K`` items rather than fewer whenever a practice scene ranked."""
    world = World(tmp_path)
    at = datetime.combine(NIGHT, time(22), tzinfo=timezone.utc)
    rehearsals = [replace(own(UID, NIGHT, i, f"sorting line rehearsal {i}"), id=f"prac-{i}", kind="practice",
                          created_at=at - timedelta(minutes=i)) for i in range(3)]
    world.evidence(UID).add([*(own(UID, NIGHT, i, f"sorting line item {i}") for i in range(6)), *rehearsals])
    chat = Script({"op": "recall", "query": "sorting line"}, {"op": "done"})
    AgentNight(prices=PRICES, chat=chat).night(_call(world))

    scratch = chat.prompts[1].split("## Tonight so far", 1)[1]
    shown = re.findall(r"- \[([^\]]+)\]", scratch)
    assert shown == [f"{UID}-{NIGHT.isoformat()}-e{i}" for i in (5, 4, 3, 2, 1)]
    assert len(shown) == SEARCH_K  # the practice scenes didn't shrink the answer


def test_research_stays_in_the_pass_and_never_reaches_the_store(tmp_path: Path) -> None:
    """Arm 2A's research must reach arm 2A alone: a source stored as homework would fold into the
    served brief and every arm's context tonight and later, so the pass keeps it in its scratchpad."""
    world = World(tmp_path)
    wonder = FakeWonder(research_script())
    chat = Script({"op": "research", "query": "optical sorting film plastics"},
                  {"op": "note", "text": NOTE_TEXT}, {"op": "done"})
    before = world.evidence(UID).stored(NIGHT)

    def research(call: AgentCall) -> WonderResearch:
        return WonderResearch(user_id=call.user_id, night=call.night, at=call.at, store=call.store,
                              meter=call.wonder, wonder=wonder, search=FakeSearch(),
                              fetcher=study_web().fetcher(), keep=False)

    outcome = AgentNight(prices=PRICES, chat=chat, research=research).night(_call(world))
    store = world.evidence(UID)
    assert outcome.stopped == "done" and outcome.receipt["research_calls"] == 1 and wonder.calls
    assert "wr-" in chat.prompts[1]  # the sources answered the pass, in its own scratchpad
    assert store.stored(NIGHT) == before  # nothing of the research folded in; the note is an agent_note
    assert [item.kind for item in store.agent_notes(NIGHT)] == [AGENT_NOTE]


# ── no state between nights ──────────────────────────────────────────────────


def test_nothing_is_carried_to_the_next_night(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT)
    second = NIGHT + timedelta(days=1)
    chat = Script({"op": "note", "text": NOTE_TEXT}, {"op": "done"}, {"op": "done"})
    _one(world, baseline=AgentNight(prices=PRICES, chat=chat))
    world.plan(UID, second)
    _one(world, night=second, baseline=AgentNight(prices=PRICES, chat=chat))

    [note] = world.evidence(UID).agent_notes(second)  # the first night's, and only it
    first_night_after, second_prompt = chat.prompts[1], chat.prompts[2]
    assert note.id in first_night_after  # tonight's own scratchpad shows what it wrote
    assert note.id not in second_prompt and note.text not in second_prompt
    assert own(UID, NIGHT, 0).text in second_prompt  # what it reads is the person's evidence


# ── the cap ──────────────────────────────────────────────────────────────────


def test_the_night_stays_inside_minds_token_cap(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT)
    chat = Burns(5_000)
    outcome = AgentNight(prices=PRICES, chat=chat).night(_call(world))
    assert outcome.stopped == "token_cap"
    assert chat.caps == [TOKEN_CAP - 5_000 * step_number for step_number in range(4)]
    assert outcome.tokens_used == TOKEN_CAP == 4 * 5_000
    assert outcome.steps == outcome.calls == len(chat.caps) == 4 <= MAX_STEPS


def test_a_failed_call_is_charged_its_cap(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT)
    chat = Fails()
    outcome = AgentNight(prices=PRICES, chat=chat).night(_call(world))
    assert outcome.stopped == "malformed" and outcome.tokens_used == TOKEN_CAP  # charged whole
    assert outcome.calls == 1 and outcome.notes == () and world.evidence(UID).agent_notes(NIGHT) == []
