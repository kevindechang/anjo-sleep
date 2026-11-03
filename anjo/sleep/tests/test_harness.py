# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""Harness learning: the strategy log, the one-edit loop, and the reported-only 2H arm.

Fakes only: no model, GPU or network. The switch-off tests compare what the night and the morning
write with the harness code never reached; the switch-on tests drive the loop through a scripted
Spark Open and a scripted critic.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.budget import Ledger
from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.harness.loop import (
    HARNESS_MIN_MARGIN,
    HARNESS_MIN_WINS,
    HarnessCall,
    run_harness,
)
from anjo.sleep.harness.store import (
    STRATEGY_MAX,
    STRATEGY_WORDS,
    JsonlStrategyStore,
    StrategyEdit,
    StrategyError,
    strategy_block,
)
from anjo.sleep.night import story as story_module
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.night.story import StoryError, story_from
from anjo.sleep.practice.transcripts import PracticeTranscript
from anjo.sleep.probes.arms import ArmDeps, reply
from anjo.sleep.probes.morning import COMMITTED_ARMS, HARNESS_PAIR, PAIRS, Morning, run_morning
from anjo.sleep.probes.score import JudgeError
from anjo.sleep.reports import results
from anjo.sleep.stops import BudgetStop, Cancelled
from anjo.sleep.tests.night_fakes import NIGHT, SparkChat, Trainer, World, step, steps
from anjo.sleep.tests.sleep_fakes import (
    FakeContexts,
    FakeLogprobs,
    FakeModel,
    FakeRegistry,
    StoreSpy,
    ckpt,
    probe,
    taste,
)

# A default full morning: the six registered arms and the pairs between them (2A and 4M need ANJO_SLEEP_ARMS_EXTRA).
REGISTERED_PAIRS = {name for name, (a, b) in PAIRS.items() if a in COMMITTED_ARMS and b in COMMITTED_ARMS}

AT = datetime.combine(NIGHT, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=22)
LINE = "Ask for the outline before writing a draft."


def result(text: str) -> ChatResult:
    return ChatResult(text=text, tool_calls=(), model=SPARK_OPEN, input_tokens=20, output_tokens=5,
                      finish_reason="stop")


def _store(tmp_path: Path, lines: tuple[str, ...] = (), *, night: date = NIGHT - timedelta(days=1)) -> JsonlStrategyStore:
    """A store whose existing lines were promoted on an earlier night, so tonight may edit once."""
    store = JsonlStrategyStore(tmp_path / "strategies.jsonl", "p1", codec=PLAIN, guard=NO_GUARD)
    for line in lines:
        store.promote(night, StrategyEdit("add", line=line), wins=5, ties=1, losses=2,
                      ancestry=frozenset({SPARK_OPEN}), at=AT)
    return store


def transcripts(count: int = 8, *, lost: bool = True) -> tuple[PracticeTranscript, ...]:
    """The night's practice: short scenes, each one a reply that lost to the teacher unless told not.
    A teacher-target scene carries her losing reply as the critic's rejected candidate, as practice
    records it; the teacher's winning reply is never hers."""
    return tuple(
        PracticeTranscript(
            id=f"pr-{n}", person_id="p1", night=NIGHT, scene=n, message=f"Message {n}: a new brief.",
            reply=f"Teacher reply {n}." if lost else f"Reply {n}.",
            context=f"C for {n}.", f_ids=("e1",),
            ancestry=frozenset({SPARK_OPEN}), created_at=AT + timedelta(seconds=n),
            target="teacher" if lost else "glimmer",
            rejected=f"Reply {n}." if lost else None,
            rejected_from="glimmer" if lost else None,
        )
        for n in range(count)
    )


class Glimmer:
    """Her serving Glimmer: every probe is one reply, and its calls are kept."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float,
                 seed: int | None = None) -> ChatResult:
        self.calls.append({"adapter": adapter, "messages": messages, "temperature": temperature, "seed": seed})
        return result(f"Probe reply {len(self.calls)}")


class Chat:
    """Spark Open and the critic, scripted by role: the proposal, the vet, then a verdict a scene."""

    def __init__(self, proposal: str | None = None, vet: str = '{"allow": true}',
                 outcomes: tuple[str, ...] = ()) -> None:
        self.proposal, self.vet, self.outcomes = proposal, vet, outcomes
        self.prompts: list[str] = []
        self.compares = 0
        self.vet_calls = 0
        self.fail: Exception | None = None

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *,
                 temperature: float | None = None, max_tokens: int | None = None,
                 seed: int | None = None) -> ChatResult:
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if prompt.startswith("# harness_propose_v1"):
            if self.fail is not None:
                raise self.fail
            assert self.proposal is not None
            return result(self.proposal)
        if prompt.startswith("# harness_vet_v1"):
            self.vet_calls += 1
            return result(self.vet)
        if prompt.startswith("# harness_compare_v1"):
            outcome = self.outcomes[self.compares // 2]
            order = self.compares % 2
            self.compares += 1
            if outcome == "edited":
                return result('{"winner": "B"}' if order == 0 else '{"winner": "A"}')
            if outcome == "current":
                return result('{"winner": "A"}' if order == 0 else '{"winner": "B"}')
            return result('{"winner": "tie"}')
        raise AssertionError("an unexpected Spark Open call")


def harness(tmp_path: Path, chat: Chat, *, lines: tuple[str, ...] = (), lost: bool = True,
            count: int = 8) -> tuple[dict[str, Any], JsonlStrategyStore, Glimmer]:
    store = _store(tmp_path, lines)
    glimmer = Glimmer()
    call = HarnessCall(
        user_id="p1", night=NIGHT, seed=7, at=AT, store=store, transcripts=transcripts(count, lost=lost),
        glimmer=glimmer, chat=chat, adapter=None,
    )
    return run_harness(call), store, glimmer


# ── the store ────────────────────────────────────────────────────────────────


def test_a_promotion_appends_a_version_with_its_evidence_and_folds_the_list(tmp_path: Path) -> None:
    store = _store(tmp_path)
    version = store.promote(NIGHT, StrategyEdit("add", line=LINE), wins=5, ties=1, losses=2,
                            ancestry=frozenset({SPARK_OPEN}), at=AT)
    assert version.version == 1 and version.night == NIGHT and version.wins == 5
    assert store.lines() == (LINE,)
    assert store.versions() == (version,)
    record = json.loads(store.path.read_text(encoding="utf-8"))
    assert record["edit"] == {"action": "add", "line": LINE}
    assert record["ancestry"] == [SPARK_OPEN] and store.export()["strategies"] == [record]


def test_a_strategy_line_is_at_most_forty_words(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(StrategyError, match=str(STRATEGY_WORDS)):
        store.promote(NIGHT, StrategyEdit("add", line=" ".join(["word"] * (STRATEGY_WORDS + 1))),
                      wins=5, ties=0, losses=0, ancestry=None, at=AT)
    assert store.lines() == ()


def test_the_list_and_its_versions_fold_only_through_the_night_asked(tmp_path: Path) -> None:
    store = _store(tmp_path, (LINE,), night=NIGHT)
    later = NIGHT + timedelta(days=1)
    store.promote(later, StrategyEdit("add", line="A later way."), wins=5, ties=0, losses=0,
                  ancestry=None, at=AT)
    assert store.lines(until=NIGHT) == (LINE,)
    assert store.lines(until=later) == (LINE, "A later way.")
    assert store.lines() == (LINE, "A later way.")  # a caller that names no night reads the present
    assert [version.night for version in store.versions(until=NIGHT)] == [NIGHT]


def test_an_add_on_a_full_list_is_refused_and_a_replace_prunes_it(tmp_path: Path) -> None:
    lines = tuple(f"Line {n}." for n in range(STRATEGY_MAX))
    store = _store(tmp_path, lines)
    with pytest.raises(StrategyError, match="full"):
        store.promote(NIGHT, StrategyEdit("add", line="Another."), wins=5, ties=0, losses=0,
                      ancestry=None, at=AT)
    assert store.lines() == lines
    store.promote(NIGHT, StrategyEdit("replace", target=lines[0], line="A different first line."),
                  wins=5, ties=0, losses=2, ancestry=None, at=AT)
    assert len(store.lines()) == STRATEGY_MAX and store.lines()[0] == "A different first line."


# ── the loop ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("outcomes,promoted,why", [
    ((("edited",) * 5 + ("tie",) * 3), True, "5 wins, margin 5"),
    ((("edited",) * 5 + ("current",) * 2 + ("tie",)), True, "5 wins, margin 3"),
    ((("edited",) * 4 + ("tie",) * 4), False, "4 wins"),
    ((("edited",) * 5 + ("current",) * 3), False, "margin 2"),
])
def test_the_promotion_bar_edges(tmp_path: Path, outcomes: tuple[str, ...], promoted: bool, why: str) -> None:
    chat = Chat('{"action": "add", "line": "Ask for the outline before writing a draft."}', outcomes=outcomes)
    done, store, _ = harness(tmp_path, chat)
    assert (done["wins"], done["ties"], done["losses"]) == (
        outcomes.count("edited"), outcomes.count("tie"), outcomes.count("current"))
    assert done["promoted"] is promoted, why
    assert (store.lines() == (LINE,)) is promoted
    if promoted:
        assert done["learned"] == LINE and done["list_size"] == 1
    else:
        assert "learned" not in done and done["list_size"] == 0
    assert HARNESS_MIN_WINS == 5 and HARNESS_MIN_MARGIN == 3


def test_a_reduced_probe_set_never_promotes_even_at_five_wins(tmp_path: Path) -> None:
    chat = Chat('{"action": "add", "line": "' + LINE + '"}',
                outcomes=("edited",) * 5 + ("tie",))
    done, store, _ = harness(tmp_path, chat, count=6)
    assert done["compared"] == 6 and done["wins"] == 5 and done["promoted"] is False
    assert store.lines() == () and "learned" not in done


def test_a_line_naming_safety_can_be_removed_or_replaced(tmp_path: Path) -> None:
    safe = "Keep the safe word."
    remove = Chat('{"action": "remove", "target": "' + safe + '"}', outcomes=("edited",) * 8)
    done, store, _ = harness(tmp_path / "remove", remove, lines=(safe,))
    assert done["vetted"] is True and done["promoted"] is True and store.lines() == ()
    assert done["learned"] == f"removed: {safe}"

    replace_chat = Chat('{"action": "replace", "target": "' + safe + '", "line": "' + LINE + '"}',
                        outcomes=("edited",) * 8)
    done, store, _ = harness(tmp_path / "replace", replace_chat, lines=(safe,))
    assert done["promoted"] is True and store.lines() == (LINE,)


def test_the_loop_never_proposes_without_a_lost_or_flagged_reply(tmp_path: Path) -> None:
    chat = Chat('{"action": "add", "line": "Ask for the outline."}')
    done, store, glimmer = harness(tmp_path, chat, lost=False)
    assert done["proposed"] is None and done["calls"] == 0
    assert chat.prompts == [] and glimmer.calls == [] and store.lines() == ()


@pytest.mark.parametrize("proposal", [
    "not json",
    '{"action": "add", "line": "One line.", "extra": 1}',
    '{"action": "remove", "target": "a line that is not there"}',
    '{"action": "add", "line": "' + " ".join(["word"] * 41) + '"}',
    '{"action": "change", "line": "One line."}',
])
def test_a_proposal_that_is_not_one_valid_edit_is_no_edit(tmp_path: Path, proposal: str) -> None:
    chat = Chat(proposal)
    done, store, glimmer = harness(tmp_path, chat)
    assert done["proposed"] is None and done["promoted"] is False
    assert done["calls"] == 1 and chat.vet_calls == 0 and glimmer.calls == [] and store.lines() == ()


def test_a_replace_with_the_same_line_is_no_edit(tmp_path: Path) -> None:
    chat = Chat('{"action": "replace", "target": "Line 0.", "line": "Line 0."}')
    done, store, glimmer = harness(tmp_path, chat, lines=("Line 0.",))
    assert done["proposed"] is None and "itself" in done["why"]
    assert done["calls"] == 1 and chat.vet_calls == 0 and glimmer.calls == []
    assert store.lines() == ("Line 0.",)


def test_the_keyword_guard_refuses_safety_before_the_critic(tmp_path: Path) -> None:
    chat = Chat('{"action": "add", "line": "Always mention the crisis line."}')
    done, store, glimmer = harness(tmp_path, chat)
    assert done["proposed"] is not None and done["vetted"] is False
    assert "crisis" in done["vetoed"] and done["calls"] == 1
    assert chat.vet_calls == 0 and glimmer.calls == [] and store.lines() == ()


def test_the_vet_refuses_a_reason_and_no_promotion_follows(tmp_path: Path) -> None:
    chat = Chat('{"action": "add", "line": "Ask for the outline."}',
                vet='{"allow": false, "reason": "that is a prompt hack"}', outcomes=("edited",) * 8)
    done, store, glimmer = harness(tmp_path, chat)
    assert done["vetted"] is False and done["vetoed"] == "that is a prompt hack"
    assert chat.vet_calls == 1 and glimmer.calls == [] and done["promoted"] is False
    assert store.lines() == ()


def test_a_full_list_gives_no_add_and_a_replace_prunes(tmp_path: Path) -> None:
    lines = tuple(f"Line {n}." for n in range(STRATEGY_MAX))
    chat = Chat('{"action": "add", "line": "Another."}')
    done, store, _ = harness(tmp_path / "add", chat, lines=lines)
    assert done["proposed"] is None and "full" in done["why"] and store.lines() == lines

    chat = Chat('{"action": "replace", "target": "Line 0.", "line": "' + LINE + '"}',
                outcomes=("edited",) * 8)
    done, store, _ = harness(tmp_path / "replace", chat, lines=lines)
    assert done["promoted"] is True and len(store.lines()) == STRATEGY_MAX
    assert store.lines()[0] == LINE and "Line 0." not in store.lines()
    assert done["list_size"] == STRATEGY_MAX and done["learned"] == LINE


def test_a_failing_spark_call_is_an_error_not_an_edit(tmp_path: Path) -> None:
    chat = Chat('{"action": "add", "line": "Ask for the outline."}', outcomes=("edited",) * 8)
    chat.fail = ModelApiError("down")
    with pytest.raises(ModelApiError):
        harness(tmp_path, chat)
    assert len(chat.prompts) == 1 and chat.vet_calls == 0 and chat.compares == 0


def test_one_edit_a_night_holds_on_a_rerun_and_a_backfill(tmp_path: Path) -> None:
    done, store, _ = harness(tmp_path, Chat('{"action": "add", "line": "' + LINE + '"}',
                                            outcomes=("edited",) * 8))
    assert done["promoted"] is True and store.lines() == (LINE,)

    again = Chat('{"action": "add", "line": "Another way."}', outcomes=("edited",) * 8)
    same = run_harness(HarnessCall(
        user_id="p1", night=NIGHT, seed=7, at=AT, store=store, transcripts=transcripts(),
        glimmer=Glimmer(), chat=again, adapter=None))
    assert same["promoted"] is False and same["why"] == "the list was already edited tonight"
    assert again.prompts == [] and store.lines() == (LINE,)

    older = Chat('{"action": "add", "line": "Another way."}', outcomes=("edited",) * 8)
    backfill = run_harness(HarnessCall(
        user_id="p1", night=NIGHT - timedelta(days=1), seed=7, at=AT, store=store,
        transcripts=transcripts(), glimmer=Glimmer(), chat=older, adapter=None))
    assert backfill["promoted"] is False and "later night" in backfill["why"]
    assert backfill["list_size"] == 0 and older.prompts == [] and store.lines() == (LINE,)


def test_the_proposer_sees_her_rejected_reply_never_the_teachers_winning_one(tmp_path: Path) -> None:
    losing = transcripts(1)[0]  # the teacher won; practice kept her reply as the critic's rejected one
    teacher_won = replace(losing, scene=1, reply="TEACHER WIN", rejected="A teacher reply.",
                          rejected_from="teacher")
    chat = Chat("not json")
    done = run_harness(HarnessCall(
        user_id="p1", night=NIGHT, seed=7, at=AT, store=_store(tmp_path), transcripts=(losing, teacher_won),
        glimmer=Glimmer(), chat=chat, adapter=None))
    assert done["proposed"] is None
    assert "Her reply (the critic rejected it): Reply 0." in chat.prompts[0]
    assert "TEACHER WIN" not in chat.prompts[0]

    only_teacher = Chat("not json")
    skipped = run_harness(HarnessCall(
        user_id="p1", night=NIGHT, seed=7, at=AT, store=_store(tmp_path / "skip"), transcripts=(teacher_won,),
        glimmer=Glimmer(), chat=only_teacher, adapter=None))
    assert skipped["why"] == "no reply lost or was flagged tonight" and only_teacher.prompts == []


def test_2h_is_2c_plus_the_marked_block_and_empty_is_2c_exactly() -> None:
    model = FakeModel()
    d = ArmDeps(model=model, adapters=FakeRegistry({"p1": ckpt("p1")}), contexts=FakeContexts(),
                evidence=lambda uid: StoreSpy([]), logprobs=FakeLogprobs(lambda adapter, messages: "A"),
                donors={}, until=NIGHT, strategies=lambda uid, until: (LINE,))
    reply("2C", "p1", probe(), deps=d)
    reply("2H", "p1", probe(), deps=d)
    two_c, two_h = model.calls[0]["messages"], model.calls[1]["messages"]
    assert two_h[0]["content"] == two_c[0]["content"] + "\n\nHow you work best with them:\n- " + LINE
    assert two_h[1]["content"] == two_c[1]["content"] and model.calls[1]["adapter"] is None

    empty = replace(d, strategies=lambda uid, until: (), context_cache={})
    model.calls.clear()
    reply("2C", "p1", probe(), deps=empty)
    reply("2H", "p1", probe(), deps=empty)
    assert model.calls[0]["messages"] == model.calls[1]["messages"]


def test_2h_reads_the_list_through_the_night_it_is_scored_on(tmp_path: Path) -> None:
    store = _store(tmp_path, (LINE,), night=NIGHT)
    later = NIGHT + timedelta(days=1)
    store.promote(later, StrategyEdit("add", line="A later way."), wins=5, ties=0, losses=0,
                  ancestry=None, at=AT)
    model = FakeModel()
    deps = ArmDeps(model=model, adapters=FakeRegistry({"p1": ckpt("p1")}), contexts=FakeContexts(),
                   evidence=lambda uid: StoreSpy([]), logprobs=FakeLogprobs(lambda adapter, messages: "A"),
                   donors={}, until=NIGHT, strategies=lambda uid, until: store.lines(until=until))
    reply("2C", "p1", probe(), deps=deps)
    reply("2H", "p1", probe(), deps=deps)
    assert model.calls[1]["messages"][0]["content"] == model.calls[0]["messages"][0]["content"] + (
        "\n\nHow you work best with them:\n- " + LINE)

    newer = replace(deps, until=later, context_cache={})
    model.calls.clear()
    reply("2C", "p1", probe(), deps=newer)
    reply("2H", "p1", probe(), deps=newer)
    assert "A later way." in model.calls[1]["messages"][0]["content"]


def test_the_strategy_block_is_empty_text_for_an_empty_list() -> None:
    assert strategy_block("C", ()) == "C"
    assert strategy_block(None, (LINE,)) == "How you work best with them:\n- " + LINE


# ── the morning ──────────────────────────────────────────────────────────────


PRICES = Prices(spark_in_per_m=0.2, spark_out_per_m=0.6, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.95)


class Judge:
    def __init__(self) -> None:
        self.calls = 0

    def verdict(self, p: Any, first: str, second: str) -> str:
        self.calls += 1
        return "A" if len(first) > len(second) else "B" if len(second) > len(first) else "tie"


class RoomyFailingJudge(Judge):
    """The judge fails on the first ``failures`` verdicts of one scene's personal arms, then its
    retry can answer; the next failure has no retry, so the retry pool's size decides the outcome."""

    def __init__(self, failures: int = 8) -> None:
        super().__init__()
        self.failures = failures

    def verdict(self, p: Any, first: str, second: str) -> str:
        self.calls += 1
        if self.failures and p.id == "q3" and {first, second} == {"reply from p1-n1", "reply from base"}:
            self.failures -= 1
            raise JudgeError("unreadable")
        return "A"

    def again(self, p: Any, first: str, second: str) -> str:
        return "tie"


class Failing2H(FakeModel):
    """Glimmer answers every registered arm; a prompt carrying the strategy block (2H) fails."""

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float,
                 seed: int | None = None) -> ChatResult:
        if any("How you work best with them:" in message["content"] for message in messages):
            raise ModelApiError("2H is down")
        return super().complete(adapter, messages, temperature=temperature, seed=seed)


class Distinct2H(FakeModel):
    """Glimmer answers 2H with a text only 2H says, so a judge can fail only its comparisons."""

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float,
                 seed: int | None = None) -> ChatResult:
        if any("How you work best with them:" in message["content"] for message in messages):
            self.calls.append({"adapter": adapter, "messages": messages, "temperature": temperature, "seed": seed})
            return ChatResult("2H reply", (), "base", 10, 3, "stop")
        return super().complete(adapter, messages, temperature=temperature, seed=seed)


class TwoHOnlyJudge(Judge):
    """An unreadable answer only where one reply is 2H's, so registered comparisons still read."""

    def verdict(self, p: Any, first: str, second: str) -> str:
        self.calls += 1
        if "2H reply" in (first, second):
            raise JudgeError("unreadable")
        return "A" if len(first) > len(second) else "B" if len(second) > len(first) else "tie"


class TwoHJudgeError(Judge):
    """The judge fails at its provider only where one reply is 2H's."""

    def verdict(self, p: Any, first: str, second: str) -> str:
        self.calls += 1
        if "2H reply" in (first, second):
            raise ModelApiError("the judge is down")
        return "A" if len(first) > len(second) else "B" if len(second) > len(first) else "tie"


class TwoHBudgetStop(Judge):
    """A 2H judge call its per-call hold can't reserve, as DeepSeek's is when its pool is spent."""

    def verdict(self, p: Any, first: str, second: str) -> str:
        self.calls += 1
        if "2H reply" in (first, second):
            raise BudgetStop("fireworks_sleep_usd/judge_usd can't hold the worst case of a 2H call")
        return "A" if len(first) > len(second) else "B" if len(second) > len(first) else "tie"


class TwoHAndContext(FakeModel):
    """2H and 2C (and every context arm) each say a text of their own, so a judge can tell 2H's two
    comparisons apart: against arm 1, and against 2C."""

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float,
                 seed: int | None = None) -> ChatResult:
        if any("How you work best with them:" in message["content"] for message in messages):
            self.calls.append({"adapter": adapter, "messages": messages, "temperature": temperature, "seed": seed})
            return ChatResult("2H reply", (), "base", 10, 3, "stop")
        if any("CTX" in message["content"] for message in messages):
            self.calls.append({"adapter": adapter, "messages": messages, "temperature": temperature, "seed": seed})
            return ChatResult("reply with context", (), "base", 10, 3, "stop")
        return super().complete(adapter, messages, temperature=temperature, seed=seed)


class OneUnderTwoOver(Judge):
    """Unreadable on one 2H probe against 1 (under a quarter of four), and on two against 2C (over
    it): the first count is written, then the second drops the arm."""

    def verdict(self, p: Any, first: str, second: str) -> str:
        self.calls += 1
        if "2H reply" in (first, second):
            other = second if first == "2H reply" else first
            if (other == "reply from base" and p.id == "q1") or (
                    other == "reply with context" and p.id in {"q2", "q3"}):
                raise JudgeError("unreadable")
        return "A" if len(first) > len(second) else "B" if len(second) > len(first) else "tie"


class TwoHCancelled(Judge):
    """A cancelled night reaches the judge on a 2H call: it must stop the morning, never be swallowed."""

    def verdict(self, p: Any, first: str, second: str) -> str:
        self.calls += 1
        if "2H reply" in (first, second):
            raise Cancelled("the night was cancelled")
        return "A" if len(first) > len(second) else "B" if len(second) > len(first) else "tie"


class RegisteredBudgetStop(Judge):
    """Every registered comparison hits a pool with no room: the morning stops as it did with the switch off."""

    def verdict(self, p: Any, first: str, second: str) -> str:
        self.calls += 1
        raise BudgetStop("meta_usd/judge_usd can't hold the worst case of the morning")


class MorningRig:
    def __init__(self, tmp_path: Path, *, model: Any = None, judge: Any = None,
                 strategies: Any = None, ledger: Ledger | None = None) -> None:
        self.model = model or FakeModel()
        self.judge = judge or Judge()
        self.store = StoreSpy([])
        self.registry = FakeRegistry({"p1": ckpt("p1", 1), "donor": ckpt("donor", 1)})
        self.probes = [probe("q1", "new_work"), probe("q2", "vocab"), probe("q3", "tone_hold"), taste("t1")]
        deps = ArmDeps(model=self.model, adapters=self.registry, contexts=FakeContexts(),
                       evidence=lambda uid: self.store, logprobs=FakeLogprobs(lambda adapter, messages: "A"),
                       donors={"p1": "donor"}, until=NIGHT,
                       strategies=strategies or (lambda uid, until: ()))
        self.morning = Morning(arms=deps, sealed=lambda uid: self.probes, judge=self.judge, prices=PRICES,
                               ledger=ledger, root=tmp_path / "results", timer=lambda: 0.0)

    def run(self, night: date = NIGHT) -> Any:
        return run_morning("p1", night, self.registry.adapters["p1"], morning=self.morning,
                           label="development", promoted=True)

    @property
    def scores(self) -> Path:
        return self.morning.root / "p1" / "scores"  # type: ignore[operator]


def test_the_switch_is_on_unless_it_is_explicitly_zero() -> None:
    from anjo.sleep.harness import HARNESS_ENV, harness_on

    assert harness_on({}) is True and harness_on({HARNESS_ENV: " "}) is True
    assert harness_on({HARNESS_ENV: "1"}) is True
    assert harness_on({HARNESS_ENV: "0"}) is False
    assert harness_on({HARNESS_ENV: "yes"}) is False  # no new refusal: any other word stays off


def test_morning_off_is_the_six_arms_and_the_registered_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "0")
    rig, again = MorningRig(tmp_path / "a"), MorningRig(tmp_path / "b")
    first = rig.run()
    second = again.run()
    record = json.loads(first.path.read_text(encoding="utf-8"))
    assert set(record["arms"]) == set(COMMITTED_ARMS) and set(record["pairs"]) == REGISTERED_PAIRS
    assert "2H" not in record["arms"] and HARNESS_PAIR not in record["pairs"]
    assert first.path.read_bytes() == second.path.read_bytes()  # the same run, the same bytes


def test_morning_on_adds_2h_and_its_pair_but_not_a_registered_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "0")
    off = MorningRig(tmp_path / "off")
    off_record = json.loads(off.run().path.read_text(encoding="utf-8"))
    off_calls = list(off.model.calls)

    monkeypatch.delenv("ANJO_SLEEP_HARNESS", raising=False)  # unset is on
    on = MorningRig(tmp_path / "on")
    on_done = on.run()
    on_record = json.loads(on_done.path.read_text(encoding="utf-8"))
    assert set(on_record["arms"]) == set(COMMITTED_ARMS) | {"2H"}
    assert set(on_record["pairs"]) == REGISTERED_PAIRS | {HARNESS_PAIR}
    assert on_record["pairs"][HARNESS_PAIR]["overall"]["n"] == len(on.probes)
    for arm in COMMITTED_ARMS:
        if arm == "1":
            continue
        assert on_record["arms"][arm] == off_record["arms"][arm]
    for pair in REGISTERED_PAIRS:
        assert on_record["pairs"][pair] == off_record["pairs"][pair]
    # The six registered arms' questions are the same before the harness arm is asked.
    assert on.model.calls[:len(off_calls)] == off_calls


def test_the_morning_hands_2h_the_night_it_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    seen: list[date] = []
    rig = MorningRig(tmp_path, strategies=lambda uid, until: seen.append(until) or ())
    night = NIGHT - timedelta(days=4)
    rig.run(night=night)
    assert seen and set(seen) == {night}  # the backfilled night, never the wall's or a later one


def test_a_corrupt_strategy_log_leaves_2h_as_2c_and_the_morning_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    path = tmp_path / "strategies.jsonl"
    path.write_text("this is not a strategy\n", encoding="utf-8")
    broken = JsonlStrategyStore(path, "p1", codec=PLAIN, guard=NO_GUARD)
    rig = MorningRig(tmp_path, strategies=lambda uid, until: broken.lines(until=until))
    record = json.loads(rig.run().path.read_text(encoding="utf-8"))
    assert record["unavailable"] == {} and HARNESS_PAIR in record["pairs"]
    per_arm = sum(1 for p in rig.probes if p.kind != "taste_ab")  # taste_ab asks no model
    two_c = [call["messages"] for call in rig.model.calls[per_arm:2 * per_arm]]
    two_h = [call["messages"] for call in rig.model.calls[-per_arm:]]
    assert two_h == two_c  # a list that can't be read is no list: 2H is 2C exactly


def test_a_failing_2h_reply_leaves_every_registered_byte_and_the_morning_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "0")
    off = MorningRig(tmp_path / "off")
    off_record = json.loads(off.run().path.read_text(encoding="utf-8"))

    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    on = MorningRig(tmp_path / "on", model=Failing2H(), strategies=lambda uid, until: (LINE,))
    on_record = json.loads(on.run().path.read_text(encoding="utf-8"))
    assert on_record["unavailable"] == {"2H": "harness_unavailable"}
    assert HARNESS_PAIR not in on_record["pairs"] and "2H" not in on_record["arms"]
    assert on.model.calls == off.model.calls  # the registered arms asked exactly the switch-off run
    for arm in COMMITTED_ARMS:
        if arm != "1":
            assert on_record["arms"][arm] == off_record["arms"][arm]
    for pair in REGISTERED_PAIRS:
        assert on_record["pairs"][pair] == off_record["pairs"][pair]


def test_a_failing_2h_prepare_leaves_it_unavailable_not_the_morning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")

    def broken(uid: str, until: date | None) -> tuple[str, ...]:
        raise RuntimeError("the list's reader broke")

    rig = MorningRig(tmp_path, strategies=broken)
    done = rig.run()
    record = json.loads(done.path.read_text(encoding="utf-8"))
    assert record["unavailable"] == {"2H": "harness_unavailable"}
    assert HARNESS_PAIR not in record["pairs"] and set(record["arms"]) == set(COMMITTED_ARMS)
    assert all(record["arms"][arm]["overall"]["n"] == len(rig.probes) for arm in COMMITTED_ARMS if arm != "1")


def test_a_judge_that_cannot_read_2h_drops_it_not_the_morning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "0")
    off = MorningRig(tmp_path / "off")
    off_record = json.loads(off.run().path.read_text(encoding="utf-8"))

    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    on = MorningRig(tmp_path / "on", model=Distinct2H(), judge=TwoHOnlyJudge(),
                    strategies=lambda uid, until: (LINE,))
    done = on.run()
    record = json.loads(done.path.read_text(encoding="utf-8"))
    assert record["unavailable"] == {"2H": "harness_unavailable"}
    assert "2H" not in record["arms"] and HARNESS_PAIR not in record["pairs"]
    assert all(row["pair"] != "2H_vs_1" for row in record["outcomes"])
    for arm in COMMITTED_ARMS:
        if arm != "1":
            assert record["arms"][arm] == off_record["arms"][arm]
    for pair in REGISTERED_PAIRS:
        assert record["pairs"][pair] == off_record["pairs"][pair]


def _same_registered_scores(record: dict[str, Any], off_record: dict[str, Any]) -> None:
    """The registered arms and pairs, byte for byte as the switch-off run wrote them."""
    assert json.dumps(record["arms"], sort_keys=True) == json.dumps(off_record["arms"], sort_keys=True)
    assert json.dumps(record["pairs"], sort_keys=True) == json.dumps(off_record["pairs"], sort_keys=True)


def test_a_2h_judge_error_drops_2h_and_writes_the_registered_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "0")
    off = MorningRig(tmp_path / "off")
    off_record = json.loads(off.run().path.read_text(encoding="utf-8"))

    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    ledger = Ledger(tmp_path / "budget.jsonl")
    on = MorningRig(tmp_path / "on", model=Distinct2H(), judge=TwoHJudgeError(), ledger=ledger,
                    strategies=lambda uid, until: (LINE,))
    done = on.run()
    record = json.loads(done.path.read_text(encoding="utf-8"))
    assert record["unavailable"] == {"2H": "harness_unavailable"}
    assert "2H" not in record["arms"] and HARNESS_PAIR not in record["pairs"]
    assert all(row["pair"] not in {"2H_vs_1", HARNESS_PAIR} for row in record["outcomes"])
    assert "no_verdict" not in record
    _same_registered_scores(record, off_record)
    assert done.judge_usd > 0 and not ledger.open_holds()  # the 2H calls that did run settled


def test_a_2h_judge_budget_stop_drops_2h_and_writes_the_registered_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "0")
    off = MorningRig(tmp_path / "off")
    off_record = json.loads(off.run().path.read_text(encoding="utf-8"))

    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    on = MorningRig(tmp_path / "on", model=Distinct2H(), judge=TwoHBudgetStop(),
                    strategies=lambda uid, until: (LINE,))
    record = json.loads(on.run().path.read_text(encoding="utf-8"))
    assert record["unavailable"] == {"2H": "harness_unavailable"}
    assert "2H" not in record["arms"] and HARNESS_PAIR not in record["pairs"]
    assert all(row["pair"] not in {"2H_vs_1", HARNESS_PAIR} for row in record["outcomes"])
    _same_registered_scores(record, off_record)


def test_a_dropped_2h_leaves_no_2h_no_verdict_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "0")
    off = MorningRig(tmp_path / "off", model=TwoHAndContext(), judge=OneUnderTwoOver())
    off.probes = [probe("q1"), probe("q2"), probe("q3"), probe("q4")]  # four judged: one is not over a quarter
    off_done = off.run()
    off_record = json.loads(off_done.path.read_text(encoding="utf-8"))

    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    on = MorningRig(tmp_path / "on", model=TwoHAndContext(), judge=OneUnderTwoOver(),
                    strategies=lambda uid, until: (LINE,))
    on.probes = [probe("q1"), probe("q2"), probe("q3"), probe("q4")]
    done = on.run()
    record = json.loads(done.path.read_text(encoding="utf-8"))
    # 2H has one unread on 2H_vs_1, under the cap, then two on 2H_vs_2C, over it: the arm is dropped.
    assert record["unavailable"] == {"2H": "harness_unavailable"}
    assert "2H" not in record["arms"] and HARNESS_PAIR not in record["pairs"]
    assert all(row["pair"] not in {"2H_vs_1", HARNESS_PAIR} for row in record["outcomes"])
    # Neither 2H count is written, and only the registered sum is reported.
    assert "no_verdict" not in record and done.no_verdict == off_done.no_verdict == 0
    # The file is the switch-off run's but for the code, byte for byte.
    assert (json.dumps({**record, "unavailable": {}}, sort_keys=True, indent=1)
            == off_done.path.read_text(encoding="utf-8"))
    _same_registered_scores(record, off_record)


def test_a_registered_judge_budget_stop_still_stops_the_morning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    rig = MorningRig(tmp_path, judge=RegisteredBudgetStop(), strategies=lambda uid, until: (LINE,))
    with pytest.raises(BudgetStop):
        rig.run()
    assert not rig.scores.exists()  # no score file, the switch-off run's stop


def test_a_cancelled_2h_judge_call_still_stops_the_morning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    rig = MorningRig(tmp_path, model=Distinct2H(), judge=TwoHCancelled(),
                     strategies=lambda uid, until: (LINE,))
    with pytest.raises(Cancelled):
        rig.run()
    assert not rig.scores.exists()  # a cancelled night is never read as a harness failure


def test_a_registered_pairs_retries_are_the_same_with_the_switch_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "0")
    off = MorningRig(tmp_path / "off", judge=RoomyFailingJudge())
    off.probes.append(probe("q4", "new_work"))  # four judged probes: one no-verdict is not over a quarter
    off_record = json.loads(off.run().path.read_text(encoding="utf-8"))

    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    on = MorningRig(tmp_path / "on", judge=RoomyFailingJudge())
    on.probes.append(probe("q4", "new_work"))
    on_record = json.loads(on.run().path.read_text(encoding="utf-8"))
    # Eight failures exhaust a seven-retry registered pool: the last pair's scene has no verdict. A
    # shared pool sized over 2H would give it an eighth retry and change the registered score.
    assert off_record["no_verdict"]["4_vs_2F"] == on_record["no_verdict"]["4_vs_2F"] == 1
    for arm in COMMITTED_ARMS:
        if arm != "1":
            assert on_record["arms"][arm] == off_record["arms"][arm]
    for pair in REGISTERED_PAIRS:
        assert on_record["pairs"][pair] == off_record["pairs"][pair]


def test_the_results_table_carries_the_harness_pair_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    rig = MorningRig(tmp_path)
    rig.run()
    night = rig.morning.root / "p1" / "nights" / NIGHT.isoformat()  # type: ignore[operator]
    night.mkdir(parents=True)
    (night / "summary.json").write_text(json.dumps({"status": "promoted", "night": NIGHT.isoformat(),
                                                    "label": "development", "gates": {}, "cost_usd": 0.0}))
    text = results.table(["p1"], root=rig.morning.root)
    assert f"pair {HARNESS_PAIR}" in text
    assert all(f"arm {arm}" in text for arm in COMMITTED_ARMS) and "arm 2H" not in text


# ── the night ────────────────────────────────────────────────────────────────


class CapturingTrainer(Trainer):
    """The night's trainer, keeping the dataset's rows so a test can look for a strategy in them."""

    def __init__(self, world: World) -> None:
        super().__init__(world)
        self.rows = ""

    def train(self, user_id: str, night: date, prev: Any, data: Any, name: str) -> Any:
        self.rows = data.to_jsonl()
        return super().train(user_id, night, prev, data, name)


class HarnessSpark(SparkChat):
    """Practice's Spark Open, and the harness's proposal, vet and comparison."""

    def __init__(self, *, proposal: str | None, fail: bool = False, wins: int = 8,
                 fail_at: str | None = None) -> None:
        super().__init__()
        self.proposal, self.fail, self.wins, self.fail_at = proposal, fail, wins, fail_at
        self.comparisons = 0
        self.practice_prompts: list[str] = []

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *,
                 temperature: float | None = None, max_tokens: int | None = None,
                 seed: int | None = None) -> ChatResult:
        content = messages[-1]["content"]
        if content.startswith("# harness_propose_v1"):
            if self.fail or self.fail_at == "propose":
                raise ModelApiError("down")
            assert self.proposal is not None
            return result(self.proposal)
        if content.startswith("# harness_vet_v1"):
            if self.fail_at == "vet":
                raise ModelApiError("down")
            return result('{"allow": true}')
        if content.startswith("# harness_compare_v1"):
            outcome = "edited" if self.comparisons // 2 < self.wins else "current"
            order = self.comparisons % 2
            self.comparisons += 1
            return result('{"winner": "B"}' if outcome == "edited" and order == 0
                          else '{"winner": "A"}' if outcome == "edited" else
                          '{"winner": "A"}' if order == 0 else '{"winner": "B"}')
        self.practice_prompts.append(content)
        return super().__call__(model, messages, tools, timeout_s, temperature=temperature,
                                max_tokens=max_tokens, seed=seed)


def _night(tmp_path: Path, spark: HarnessSpark, *, strategies: JsonlStrategyStore | None = None,
           capture: bool = False, lost: bool = True) -> tuple[World, Any, str]:
    world = World(tmp_path)
    world.spark = spark  # type: ignore[assignment]
    world.plan("p1", NIGHT)
    if lost:  # tonight's replies lost or were flagged: the harness has something to learn from
        world.transcripts("p1").add(list(transcripts()))
    trainer = CapturingTrainer(world) if capture else None
    changes: dict[str, Any] = {"strategies": lambda uid: strategies or _store(tmp_path / "empty"),
                               "glimmer": world.glimmer}
    if trainer is not None:
        changes["trainer"] = trainer
    result = run_night_batch_with(world.deps(**changes), {"p1": NIGHT})["p1"]
    return world, result, "" if trainer is None else trainer.rows


def _tree(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_switch_off_writes_no_harness_step_and_the_same_night(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "0")
    world_a, one, _ = _night(tmp_path / "a", HarnessSpark(proposal=None))
    world_b, two, _ = _night(tmp_path / "b", HarnessSpark(proposal=None))
    assert "harness" not in [record["step"] for record in steps(one.log_path)]
    assert not (world_a.root / "p1" / "strategies.jsonl").exists()
    assert one.status == two.status == "promoted"
    def normalized(world: World, root: Path) -> dict[str, bytes]:
        # The log path is the only absolute path in the night's files.
        return {name: body.replace(str(world.logs("p1")).encode(), b"<logs>")
                for name, body in _tree(root).items()}

    assert normalized(world_a, world_a.logs("p1")) == normalized(world_b, world_b.logs("p1"))


def test_a_promoted_strategy_is_in_the_list_and_the_story(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    store = _store(tmp_path / "strategies")
    world, result, _ = _night(tmp_path / "night", HarnessSpark(
        proposal='{"action": "add", "line": "Ask for the outline before writing a draft."}'),
        strategies=store)
    receipt = step(result.log_path, "harness")
    assert receipt["promoted"] is True and receipt["wins"] == 8 and receipt["learned"] == LINE
    assert receipt["list_size"] == 1 and receipt["calls"] == 18 and receipt["usd"] > 0
    assert store.lines() == (LINE,)
    story = story_from(json.loads((result.log_path.parent / "story.json").read_text(encoding="utf-8")))
    assert story.learn.learned == (LINE,)
    assert "learned" in (result.log_path.parent / "story.json").read_text(encoding="utf-8")


def test_a_failing_harness_call_still_trains_and_the_list_stays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    store = _store(tmp_path / "strategies", (LINE,))
    world, result, _ = _night(tmp_path / "night", HarnessSpark(proposal=None, fail=True), strategies=store)
    receipt = step(result.log_path, "harness")
    assert receipt["error"].startswith("ModelApiError") and receipt["promoted"] is False
    assert receipt["calls"] == 1  # the proposal call it made, not zero
    assert store.lines() == (LINE,)
    assert result.status == "promoted" and step(result.log_path, "train")["status"] == "ok"
    story = story_from(json.loads((result.log_path.parent / "story.json").read_text(encoding="utf-8")))
    assert story.learn.learned == ()


def test_a_failed_vet_counts_the_proposal_and_vet_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    store = _store(tmp_path / "strategies", (LINE,))
    world, result, _ = _night(tmp_path / "night", HarnessSpark(
        proposal='{"action": "add", "line": "Ask for the outline."}', fail_at="vet"), strategies=store)
    receipt = step(result.log_path, "harness")
    assert receipt["error"].startswith("ModelApiError") and receipt["calls"] == 2
    assert receipt["promoted"] is False and store.lines() == (LINE,)
    assert result.status == "promoted"


@pytest.mark.parametrize("error", [
    AttributeError("broken"), ImportError("broken"), LookupError("broken"), AssertionError("broken"),
])
def test_a_harness_error_the_night_can_survive_never_fails_the_night(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception,
) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")

    def broken(call: HarnessCall) -> dict[str, Any]:
        raise error

    monkeypatch.setattr("anjo.sleep.night.run.run_harness", broken)
    world, result, _ = _night(tmp_path, HarnessSpark(proposal=None))
    assert result.status == "promoted"  # the switch-off night's status
    receipt = step(result.log_path, "harness")
    assert receipt["promoted"] is False and type(error).__name__ in receipt["error"]


def test_a_cancelled_night_stops_at_the_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")

    def cancelled(call: HarnessCall) -> dict[str, Any]:
        raise Cancelled("the night was cancelled")

    monkeypatch.setattr("anjo.sleep.night.run.run_harness", cancelled)
    world, result, _ = _night(tmp_path, HarnessSpark(proposal=None))
    assert result.status == "failed" and result.failed_at == "harness"
    assert step(result.log_path, "harness")["error"].startswith("Cancelled")


def test_the_strategy_never_reaches_practice_rows_seeds_or_dreams(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
    secret = "A strategy line that must stay out of practice and rows."
    store = _store(tmp_path / "strategies", (secret,))
    world, result, rows = _night(tmp_path / "night", HarnessSpark(
        proposal='{"action": "add", "line": "Every answer ends with a next step."}'),
        strategies=store, capture=True)
    assert result.status == "promoted"
    assert secret not in rows
    assert secret not in (world.root / "p1" / "practice.jsonl").read_text(encoding="utf-8")
    assert secret not in (world.root / "p1" / "evidence.jsonl").read_text(encoding="utf-8")
    assert secret not in (world.root / "p1" / "briefs.jsonl").read_text(encoding="utf-8")
    assert all(secret not in prompt for prompt in world.spark.practice_prompts)


def test_only_the_harnesss_learned_field_is_optional_in_a_story() -> None:
    @dataclass(frozen=True)
    class Defaulted:
        required: int
        extra: int = 0

    with pytest.raises(StoryError, match="extra"):  # a defaulted field is required like every other
        story_module._record(Defaulted, {"required": 1}, "story")


def test_the_harness_imports_no_probes_or_reports() -> None:
    root = Path(__file__).parents[1]
    for path in sorted((root / "harness").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        modules += [node.module for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module is not None]
        assert not [m for m in modules if m.startswith(("anjo.sleep.probes", "anjo.sleep.reports"))], path.name


def test_practice_training_and_dreams_never_import_the_strategy_list() -> None:
    root = Path(__file__).parents[3]
    for name in ("anjo/sleep/practice/run.py", "anjo/sleep/practice/full.py", "anjo/sleep/train/dataset.py",
                 "anjo/sleep/train/terms.py", "anjo/sleep/night/dreams.py", "anjo/sleep/night/core_dreams.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        modules = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        modules += [node.module for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module is not None]
        assert not [m for m in modules if m.startswith("anjo.sleep.harness")], name
