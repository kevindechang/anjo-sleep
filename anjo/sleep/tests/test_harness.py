# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""Harness learning: the strategy log, the one-edit loop, and the reported-only 2H arm.

Fakes only: no model, GPU or network. The switch-off tests compare what the night and the morning
write with the harness code never reached; the switch-on tests drive the loop through a scripted
Spark Open and a scripted critic.
"""
from __future__ import annotations

import ast
import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
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
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.night.story import story_from
from anjo.sleep.practice.transcripts import PracticeTranscript
from anjo.sleep.probes.arms import ArmDeps, reply
from anjo.sleep.probes.morning import COMMITTED_ARMS, HARNESS_PAIR, PAIRS, Morning, run_morning
from anjo.sleep.reports import results
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


def _store(tmp_path: Path, lines: tuple[str, ...] = ()) -> JsonlStrategyStore:
    store = JsonlStrategyStore(tmp_path / "strategies.jsonl", "p1", codec=PLAIN, guard=NO_GUARD)
    for line in lines:
        store.promote(NIGHT, StrategyEdit("add", line=line), wins=5, ties=1, losses=2,
                      ancestry=frozenset({SPARK_OPEN}), at=AT)
    return store


def transcripts(count: int = 8, *, lost: bool = True) -> tuple[PracticeTranscript, ...]:
    """The night's practice: short scenes, each one a reply that lost to the teacher unless told not."""
    return tuple(
        PracticeTranscript(
            id=f"pr-{n}", person_id="p1", night=NIGHT, scene=n, message=f"Message {n}: a new brief.",
            reply=f"Reply {n}.", context=f"C for {n}.", f_ids=("e1",),
            ancestry=frozenset({SPARK_OPEN}), created_at=AT + timedelta(seconds=n),
            target="teacher" if lost else "glimmer",
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


def test_2h_is_2c_plus_the_marked_block_and_empty_is_2c_exactly() -> None:
    model = FakeModel()
    d = ArmDeps(model=model, adapters=FakeRegistry({"p1": ckpt("p1")}), contexts=FakeContexts(),
                evidence=lambda uid: StoreSpy([]), logprobs=FakeLogprobs(lambda adapter, messages: "A"),
                donors={}, until=NIGHT, strategies=lambda uid: (LINE,))
    reply("2C", "p1", probe(), deps=d)
    reply("2H", "p1", probe(), deps=d)
    two_c, two_h = model.calls[0]["messages"], model.calls[1]["messages"]
    assert two_h[0]["content"] == two_c[0]["content"] + "\n\nHow you work best with them:\n- " + LINE
    assert two_h[1]["content"] == two_c[1]["content"] and model.calls[1]["adapter"] is None

    empty = replace(d, strategies=lambda uid: (), context_cache={})
    model.calls.clear()
    reply("2C", "p1", probe(), deps=empty)
    reply("2H", "p1", probe(), deps=empty)
    assert model.calls[0]["messages"] == model.calls[1]["messages"]


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


class MorningRig:
    def __init__(self, tmp_path: Path) -> None:
        self.model, self.judge, self.store = FakeModel(), Judge(), StoreSpy([])
        self.registry = FakeRegistry({"p1": ckpt("p1", 1), "donor": ckpt("donor", 1)})
        self.probes = [probe("q1", "new_work"), probe("q2", "vocab"), probe("q3", "tone_hold"), taste("t1")]
        deps = ArmDeps(model=self.model, adapters=self.registry, contexts=FakeContexts(),
                       evidence=lambda uid: self.store, logprobs=FakeLogprobs(lambda adapter, messages: "A"),
                       donors={"p1": "donor"}, until=NIGHT, strategies=lambda uid: ())
        self.morning = Morning(arms=deps, sealed=lambda uid: self.probes, judge=self.judge, prices=PRICES,
                               root=tmp_path / "results", timer=lambda: 0.0)

    def run(self) -> Any:
        return run_morning("p1", NIGHT, self.registry.adapters["p1"], morning=self.morning,
                           label="development", promoted=True)

    @property
    def scores(self) -> Path:
        return self.morning.root / "p1" / "scores"  # type: ignore[operator]


def test_morning_off_is_the_six_arms_and_the_registered_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANJO_SLEEP_HARNESS", raising=False)
    rig, again = MorningRig(tmp_path / "a"), MorningRig(tmp_path / "b")
    first = rig.run()
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "0")
    second = again.run()
    record = json.loads(first.path.read_text(encoding="utf-8"))
    assert set(record["arms"]) == set(COMMITTED_ARMS) and set(record["pairs"]) == REGISTERED_PAIRS
    assert "2H" not in record["arms"] and HARNESS_PAIR not in record["pairs"]
    assert first.path.read_bytes() == second.path.read_bytes()  # the same run, the same bytes


def test_morning_on_adds_2h_and_its_pair_but_not_a_registered_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    off = MorningRig(tmp_path / "off")
    off_record = json.loads(off.run().path.read_text(encoding="utf-8"))
    off_calls = list(off.model.calls)

    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "1")
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

    def __init__(self, *, proposal: str | None, fail: bool = False, wins: int = 8) -> None:
        super().__init__()
        self.proposal, self.fail, self.wins = proposal, fail, wins
        self.comparisons = 0
        self.practice_prompts: list[str] = []

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *,
                 temperature: float | None = None, max_tokens: int | None = None,
                 seed: int | None = None) -> ChatResult:
        content = messages[-1]["content"]
        if content.startswith("# harness_propose_v1"):
            if self.fail:
                raise ModelApiError("down")
            assert self.proposal is not None
            return result(self.proposal)
        if content.startswith("# harness_vet_v1"):
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
    monkeypatch.delenv("ANJO_SLEEP_HARNESS", raising=False)
    world_a, one, _ = _night(tmp_path / "a", HarnessSpark(proposal=None))
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "0")
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
    assert store.lines() == (LINE,)
    assert result.status == "promoted" and step(result.log_path, "train")["status"] == "ok"
    story = story_from(json.loads((result.log_path.parent / "story.json").read_text(encoding="utf-8")))
    assert story.learn.learned == ()


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
