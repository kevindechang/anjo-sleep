# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Replay (Task 17b): start states from development lives, branches masked to their night, one
shared label per start state, the Brier score on the frozen set, and the two datasets. The models
are ``replay.fakes``; the kernel and the loop are the real ones."""
from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.mind.apd import types as apd
from anjo.sleep.mind.apd.arms import load_brief, run_arm
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.replay import dataset, fakes, resolve, sample, score
from anjo.sleep.mind.replay.types import (
    REPLAY_K,
    CallEnd,
    TOKEN_PENALTY,
    Branch,
    Label,
    ReplayRefused,
    Scored,
    StartState,
)
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.provenance import GLIMMER, HOSTED
from anjo.sleep.train.dataset import Dataset
from anjo.sleep.types import Evidence

BRIEF = load_brief()


def _dev(seed: int = 0, life: apd.Life | None = None) -> tuple[apd.Life, apd.ArmRun]:
    life = life or fakes.fake_life(seed)
    return life, run_arm("M", life, fakes.fake_models())


def _starts(seed: int = 0) -> tuple[apd.Life, tuple[StartState, ...]]:
    life, run = _dev(seed)
    return life, sample.start_states(life, run)


def _branches(start: StartState, life: apd.Life, models: sample.ReplayModels | None = None) -> tuple[Branch, ...]:
    return sample.run_branches(start, life, models or fakes.fake_replay_models(), brief=BRIEF)


def _later(life: apd.Life, start: StartState) -> list[Evidence]:
    return [item for item in life.evidence if item.created_at > start.as_of]


CountingChat = fakes.FakeLabelChat  # it keeps every prompt it was sent


# ── Start states ──────────────────────────────────────────────────────────────


def test_start_states_are_each_nights_live_questions_with_two_hypotheses() -> None:
    life, starts = _starts()
    assert [start.night for start in starts] == list(life.days[1:])  # night 0 raised no question
    for start in starts:
        assert start.question_id == "q1" and start.as_of.date() == start.night
        assert start.frozen_ids == ("q1.h2a", "q1.h2b", "q1.else")
        assert start.hypotheses[-1][1] == it.CATCH_ALL
        nights = [e.step.entry.night for e in start.events if isinstance(e.step.entry, it.RunStart)]
        assert max(nights) == start.night  # the log ends with that night
    assert len({start.id for start in starts}) == len(starts)


def _as_person(life: apd.Life, person: str) -> apd.Life:
    """The same life under another id: what a real person's would look like."""
    return dataclasses.replace(
        life, id=person, evidence=tuple(dataclasses.replace(item, person_id=person) for item in life.evidence),
        targets={day: tuple(dataclasses.replace(t, person_id=person) for t in held) for day, held in life.targets.items()},
    )


def test_replay_uses_only_synthetic_lives() -> None:
    life, run = _dev()
    real = _as_person(life, "u-4f2a")
    typed = dataclasses.replace(life, sessions=(apd.Session(life.days[0], "typed by a person", "keyboard"),))
    for refused in (real, fakes.fake_life(0, split="held_out"), typed, dataclasses.replace(life, evidence=(
            dataclasses.replace(life.evidence[0], model="keyboard"), *life.evidence[1:]))):
        with pytest.raises(ReplayRefused):
            sample.start_states(refused, dataclasses.replace(run, life_id=refused.id))
    with pytest.raises(ReplayRefused):  # a run of another arm, or of another life
        sample.start_states(life, dataclasses.replace(run, arm="M-kernel"))
    with pytest.raises(ReplayRefused):
        sample.start_states(life, dataclasses.replace(run, life_id="development-fake-film-9"))
    start = sample.start_states(life, run)[0]
    with pytest.raises(ReplayRefused):
        _branches(start, real)
    labels = resolve.LabelBook().with_label(Label(start.id, "q1.h2a", SPARK_OPEN))
    with pytest.raises(ReplayRefused):  # a start state from a life the dataset wasn't given
        dataset.build([fakes.fake_life(1)], [start], {start.id: _branches(start, life)}, labels, brief=BRIEF)
    with pytest.raises(ReplayRefused):
        dataset.build([life, real], [start], {start.id: _branches(start, life)}, labels, brief=BRIEF)
    with pytest.raises(ReplayRefused):
        resolve.label_all([start], {life.id: typed}, resolve.SparkLabeler(chat=fakes.FakeLabelChat()))


# ── Branches ──────────────────────────────────────────────────────────────────


def test_replay_tools_time_masked() -> None:
    life, starts = _starts()
    start = starts[0]  # the end of night 1: the window (day 2) and the lock (day 3) come later
    later = _later(life, start)
    assert later and all("window" in item.text or "locked" in item.text for item in later)
    branches = _branches(start, life)
    prompts = [turn.prompt for branch in branches for turn in branch.turns]
    assert any('evidence_search "courtyard window scene 4"' in prompt for prompt in prompts)  # it looked
    for prompt in prompts:
        assert not any(item.id in prompt or item.text in prompt for item in later)
        assert "## Today's new evidence" not in prompt  # no notice pass: nothing arrives in a branch
    seen: list[str] = []

    class Spy:
        def __init__(self) -> None:
            self.inner = fakes.FakeGlimmer()
            self.calls = self.inner.calls

        def propose(self, rendered: str, max_tokens: int) -> it.ModelReply:
            seen.append(rendered)
            return self.inner.propose(rendered, max_tokens)

    models = dataclasses.replace(fakes.fake_replay_models(), thinker_for=lambda seed, night: Spy())
    _branches(start, life, models)
    assert seen and all(not any(item.text in text for item in later) for text in seen)


class _Predicts:
    """The M thinker, but its first search becomes a prediction on the kitchen."""

    def __init__(self) -> None:
        self.inner, self.done = fakes.FakeGlimmer(), False

    def propose(self, rendered: str, max_tokens: int) -> it.ModelReply:
        reply = self.inner.propose(rendered, max_tokens)
        if self.done or '"INVESTIGATE"' not in reply.text:
            return reply
        self.done = True
        return dataclasses.replace(reply, text=json.dumps({
            "op": "PREDICT", "question_id": "q1", "claim": "She shoots scene 4 in the kitchen",
            "p_true_given": {"q1.h2a": 0.9, "q1.h2b": 0.2, "q1.else": 0.5}, "horizon_days": 3}))


class _FailingResolver:
    """The M run's resolver, failing every call: nothing is marked seen, so a branch asks again."""

    def resolve(self, claim: str, evidence: Sequence[Evidence]) -> it.ResolverReply:
        return it.ResolverReply(verdict="undetermined", tokens=0, failed=True)


def test_the_resolver_in_a_branch_is_masked_to_its_night_and_named_by_its_reply() -> None:
    life = fakes.fake_life(0)
    run = run_arm("M", life, dataclasses.replace(fakes.fake_models(), thinker=_Predicts(), resolver=_FailingResolver()))
    start = sample.start_states(life, run)[1]  # the prediction was made the night before, and is unseen
    chat = fakes.FakeResolverChat()
    branches = _branches(start, life, dataclasses.replace(fakes.fake_replay_models(), resolver_chat=chat))
    sent = [str(messages[-1]["content"]) for messages in chat.sent]
    assert sent and any(item.text in text for text in sent for item in life.evidence if item.day == start.night)
    assert not any(item.text in text for text in sent for item in _later(life, start))
    assert all(branch.ancestry is not None and SPARK_OPEN in branch.ancestry for branch in branches)
    for named in (sorted(HOSTED)[0], ""):  # the model the reply names, not one declared beside it
        other = fakes.FakeResolverChat(model=named)
        tainted = _branches(start, life, dataclasses.replace(fakes.fake_replay_models(), resolver_chat=other))
        assert all(named in (branch.ancestry or ()) for branch in tainted)
        assert not dataset.cleared(start, tainted)


def test_a_world_answer_or_verdict_the_m_run_never_attributed_never_trains() -> None:
    life = fakes.fake_life(0)
    world_run = run_arm("M", life, dataclasses.replace(
        fakes.fake_models(), thinker=fakes.FakeGlimmer(seed=3, temperature=0.8)))
    verdict_run = run_arm("M", life, dataclasses.replace(fakes.fake_models(), thinker=_Predicts()))
    for run, entry in ((world_run, it.Answered), (verdict_run, it.Resolved)):
        starts = sample.start_states(life, run)
        holding = [s for s in starts if any(isinstance(e.step.entry, entry) for e in s.events)]
        assert holding and all(start.ancestry is None for start in holding)
        assert all(start.ancestry is not None for start in starts if start not in holding)
        book = resolve.label_all(starts, {life.id: life}, resolve.SparkLabeler(chat=fakes.FakeLabelChat()))
        branches = {start.id: _branches(start, life) for start in starts}
        built = dataset.build([life], starts, branches, book, brief=BRIEF)
        assert not (_ids(built.thinker1) | _ids(built.shuffled)) & {start.id for start in holding}


def test_branches_differ_by_seed_and_repeat_exactly() -> None:
    life, starts = _starts()
    first, again = _branches(starts[0], life), _branches(starts[0], life)
    assert len(first) == REPLAY_K
    assert first == again
    assert len({branch.seed for branch in first}) == REPLAY_K
    assert len({tuple(branch.credences.values()) for branch in first} | {(b.tokens,) for b in first}) > 1


def test_every_branch_scored_on_the_frozen_hypothesis_set() -> None:
    life, starts = _starts()
    for start in starts:
        for branch in _branches(start, life):
            assert tuple(branch.credences) == start.frozen_ids
            assert sum(branch.credences.values()) == pytest.approx(1.0)
    start = starts[0]
    q = start.question_id
    final = [
        it.Hypothesis(f"{q}.h2a", q, "The kitchen leaves no room", 0.3),
        it.Hypothesis(f"{q}.h7a", q, "A third way she added tonight", 0.2),
        it.Hypothesis(f"{q}.h2b", q, "The window", 0.4),
        it.Hypothesis(it.catch_all_id(q), q, it.CATCH_ALL, 0.1),
    ]
    vector = sample.frozen_credences(final, start)
    assert vector == pytest.approx({f"{q}.h2a": 0.3, f"{q}.h2b": 0.4, f"{q}.else": 0.3})  # added → catch-all
    branch = _branches(start, life)[0]
    label = Label(start.id, f"{q}.h2a", SPARK_OPEN)
    scored = score.score_branch(branch, start, label)
    p = branch.credences
    expected = -sum((p[h] - (1.0 if h == label.holds else 0.0)) ** 2 for h in start.frozen_ids)
    assert scored.brier == pytest.approx(expected)
    assert scored.score == pytest.approx(expected - TOKEN_PENALTY * branch.tokens / 1000)
    off_set = dataclasses.replace(branch, credences={f"{q}.h2a": 0.5, f"{q}.h7a": 0.5})
    with pytest.raises(ValueError, match="frozen"):
        score.score_branch(off_set, start, label)
    with pytest.raises(ValueError, match="frozen"):
        score.score_branch(branch, start, Label(start.id, f"{q}.h7a", SPARK_OPEN))


# ── Labels ────────────────────────────────────────────────────────────────────


def test_truth_written_once_per_start_state_and_shared() -> None:
    life, starts = _starts()
    chat = CountingChat()
    book = resolve.label_all(starts, {life.id: life}, resolve.SparkLabeler(chat=chat))
    asked = [start for start in starts if _later(life, start)]
    assert len(chat.sent) == len(asked)  # one call per start state with later evidence, none without
    assert set(book.labels) == {start.id for start in starts}
    with pytest.raises(resolve.LabelRewrite):
        book.with_label(Label(starts[0].id, None, SPARK_OPEN))
    determined = [start for start in starts if book.labels[start.id].holds is not None]
    for start in determined:
        label = book.labels[start.id]
        scored = score.score_start(_branches(start, life), start, label)
        assert len(scored) == REPLAY_K  # every branch against the one label
        for each in scored:
            p = each.branch.credences
            assert each.brier == pytest.approx(-sum((p[h] - (h == label.holds)) ** 2 for h in start.frozen_ids))
    assert book.labels[starts[0].id].holds == "q1.h2a"  # the lock settles it for the kitchen
    assert book.labels[starts[-1].id].holds is None  # nothing after the last night


def test_label_prompt_never_contains_branch_text() -> None:
    life, starts = _starts()
    start = starts[0]
    branches = _branches(start, life)
    chat = CountingChat()
    resolve.SparkLabeler(chat=chat).label(start, life)
    (messages,) = chat.sent
    text = "\n".join(str(m["content"]) for m in messages)
    assert messages[0]["content"] == resolve.RESOLVE_PROMPT
    replies = {turn.reply.text for branch in branches for turn in branch.turns}
    assert replies and not any(reply in text for reply in replies)
    assert not any(line in text for branch in branches for turn in branch.turns
                   for line in turn.prompt.splitlines() if line.startswith("- #"))
    earlier = [item for item in life.evidence if item.created_at <= start.as_of]
    assert not any(item.text in text for item in earlier)  # only what came after the start state
    assert all(item.text in text for item in _later(life, start))
    assert all(statement in text for _, statement in start.hypotheses)


def test_a_failed_or_unreadable_label_is_undetermined() -> None:
    life, starts = _starts()
    for reply in ("I think B", '{"holds": "Q"}', '{"holds": 2}', ""):
        def answer(messages: list[dict[str, Any]], max_tokens: int, text: str = reply) -> ChatResult:
            return dataclasses.replace(fakes.FakeLabelChat()(messages, max_tokens), text=text)

        label = resolve.SparkLabeler(chat=answer).label(starts[0], life)
        assert label.holds is None and label.failed
    assert resolve.parse_label('{"holds": "undetermined"}', starts[0]) == (None, False)
    assert resolve.parse_label('{"holds": "C"}', starts[0]) == ("q1.else", False)
    assert resolve.parse_label('{"holds": "D"}', starts[0]) == (None, True)


# ── The two datasets ──────────────────────────────────────────────────────────


def _labelled(seed: int = 0) -> tuple[apd.Life, tuple[StartState, ...], dict[str, tuple[Branch, ...]], resolve.LabelBook]:
    life, starts = _starts(seed)
    book = resolve.label_all(starts, {life.id: life}, resolve.SparkLabeler(chat=fakes.FakeLabelChat()))
    branches = {start.id: _branches(start, life) for start in starts}
    return life, starts, branches, book


def _ids(data: Dataset) -> set[str]:
    return {example.id.split(":b", 1)[0] for example in data.examples}


def test_undetermined_start_state_dropped_for_all_branches() -> None:
    life, starts, branches, book = _labelled()
    undetermined = [start.id for start in starts if book.labels[start.id].holds is None]
    assert undetermined
    built = dataset.build([life], starts, branches, book, brief=BRIEF)
    for data in (built.thinker1, built.shuffled):
        assert not _ids(data) & set(undetermined)
    assert built.dropped["undetermined"] == len(undetermined)
    sampled = sample.sample_all([(life, starts)], fakes.fake_replay_models(), book, brief=BRIEF)
    assert set(sampled) == {start.id for start in starts if start.id not in undetermined}  # never branched


def test_replay_dataset_refuses_uncleared_provenance() -> None:
    life, starts, branches, book = _labelled()
    kept = [start for start in starts if book.labels[start.id].holds is not None]
    hosted = sorted(HOSTED)[0]
    clean = dataset.build([life], kept, branches, book, brief=BRIEF)
    assert clean.dropped["provenance"] == 0 and len(clean.thinker1.examples) > 0
    first = kept[0]

    def tainted_turn(branch: Branch) -> Branch:
        turn = branch.turns[0]
        return dataclasses.replace(branch, turns=(dataclasses.replace(turn, reply=dataclasses.replace(
            turn.reply, model=hosted)), *branch.turns[1:]))

    for taint in (
        {first.id: tuple(tainted_turn(b) if b.index == 0 else b for b in branches[first.id])},
        {first.id: tuple(dataclasses.replace(b, ancestry=(b.ancestry or frozenset()) | {hosted})
                         for b in branches[first.id])},
        {first.id: tuple(dataclasses.replace(b, ancestry=None) for b in branches[first.id])},
    ):
        built = dataset.build([life], kept, {**branches, **taint}, book, brief=BRIEF)
        assert built.dropped["provenance"] == 1
        for data in (built.thinker1, built.shuffled):
            assert first.id not in _ids(data)
    tainted = [dataclasses.replace(first, ancestry=frozenset({SPARK_OPEN, hosted})), *kept[1:]]
    assert dataset.build([life], tainted, branches, book, brief=BRIEF).dropped["provenance"] == 1
    world = dataclasses.replace(fakes.fake_replay_models(), world=fakes.FakeWorld(model=hosted))
    assert any(hosted in (b.ancestry or ()) for b in _branches(first, life, world))  # a world answer's model


def test_a_label_trains_only_when_spark_opens_reply_names_it() -> None:
    life, starts, branches, book = _labelled()
    kept = [start for start in starts if book.labels[start.id].usable]
    first = kept[0]
    for named in (sorted(HOSTED)[0], ""):
        def answer(messages: list[dict[str, Any]], max_tokens: int, model: str = named) -> ChatResult:
            return dataclasses.replace(fakes.FakeLabelChat()(messages, max_tokens), model=model)

        label = resolve.SparkLabeler(chat=answer).label(first, life)
        assert label.holds is not None and label.model == named and not label.usable
        relabelled = resolve.LabelBook({**book.labels, first.id: label})
        built = dataset.build([life], kept, branches, relabelled, brief=BRIEF)
        assert built.dropped["provenance"] == 1
        assert first.id not in _ids(built.thinker1) | _ids(built.shuffled)
        assert first.id not in sample.sample_all([(life, kept)], fakes.fake_replay_models(), relabelled, brief=BRIEF)

    def broken(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        raise ModelApiError("Baseten chat: HTTP 500")

    failed = resolve.SparkLabeler(chat=broken).label(first, life)
    assert (failed.holds, failed.model, failed.failed, failed.tokens) == (None, "", True, resolve.LABEL_MAX_TOKENS)


def test_dataset_rows_are_glimmers_own_turns_behind_the_brief() -> None:
    life, starts, branches, book = _labelled()
    built = dataset.build([life], starts, branches, book, brief=BRIEF)
    turns = {(b.start_id, b.index, t.seq): t for bs in branches.values() for b in bs for t in b.turns}
    for example in (*built.thinker1.examples, *built.shuffled.examples):
        start_id, rest = example.id.split(":b", 1)
        index, seq = rest.split(":t", 1)
        turn = turns[(start_id, int(index), int(seq))]
        assert example.prompt == (("user", f"{BRIEF}\n\n{turn.prompt}"),)
        assert example.completion == turn.reply.text and turn.reply.model == GLIMMER
        assert not example.teacher


def test_both_datasets_take_one_whole_trajectory_from_the_same_start_states() -> None:
    lives: list[apd.Life] = []
    starts_all: list[StartState] = []
    branches_all: dict[str, tuple[Branch, ...]] = {}
    labels = resolve.LabelBook()
    for seed in range(3):
        life, starts, branches, book = _labelled(seed)
        lives.append(life)
        starts_all.extend(starts)
        branches_all.update(branches)
        for label in book.labels.values():
            labels = labels.with_label(label)
    built = dataset.build(lives, starts_all, branches_all, labels, brief=BRIEF)
    determined = [s for s in starts_all if labels.labels[s.id].holds is not None]
    assert built.starts == len(determined) and _ids(built.thinker1) == _ids(built.shuffled) == {s.id for s in determined}
    assert len(built.thinker1.examples) != len(built.shuffled.examples)  # never trimmed to match
    assert not any(built.thinker1.dropped.values()) and not any(built.shuffled.dropped.values())  # no row cut
    for data in (built.thinker1, built.shuffled):
        for start in determined:
            rows = [e for e in data.examples if e.id.startswith(f"{start.id}:b")]
            (index,) = {int(e.id.rsplit(":b", 1)[1].split(":t", 1)[0]) for e in rows}  # one branch each
            winner = branches_all[start.id][index]
            assert [e.id for e in rows] == [f"{start.id}:b{index}:t{t.seq}" for t in winner.targets]  # whole
    again = dataset.build(lives, starts_all, branches_all, labels, brief=BRIEF)
    assert again == built  # seeded: the same inputs make the same datasets
    longest = max(e.tokens for e in built.thinker1.examples)
    tight = dataset.build(lives, starts_all, branches_all, labels, brief=BRIEF, max_seq_tokens=longest - 1)
    assert tight.dropped["too_long"] >= 1 and _ids(tight.thinker1) == _ids(tight.shuffled)
    assert tight.starts + tight.dropped["too_long"] == built.starts  # a long row drops its start state from both


# ── Selection ─────────────────────────────────────────────────────────────────


def _scored(scores: Sequence[float], tokens: Sequence[int], complete: Sequence[bool] | None = None) -> tuple[Scored, ...]:
    done = complete or [True] * len(scores)
    return tuple(
        Scored(Branch("s", n, n, {"q1.h2a": 0.5, "q1.else": 0.5}, tokens[n], (), done[n], frozenset({GLIMMER})),
               brier=value, score=value)
        for n, value in enumerate(scores)
    )


def _best_index(scored: Sequence[Scored]) -> int | None:
    found = dataset.best(scored)
    return None if found is None else found.branch.index


def test_the_best_complete_trajectory_wins_and_ties_go_to_fewer_tokens() -> None:
    assert _best_index(_scored([-0.2, -0.1, -0.5, -0.9], [10, 10, 10, 10])) == 1
    assert _best_index(_scored([-0.1, -0.1, -0.5, -0.9], [300, 200, 10, 10])) == 1
    assert _best_index(_scored([-0.1, -0.2, -0.5, -0.9], [1, 1, 1, 1], [False, True, True, True])) == 1
    assert _best_index(_scored([-0.1], [1], [False])) is None


def test_shuffle_happens_before_selection() -> None:
    scored = _scored([-0.1, -0.4, -0.7, -1.0], [100, 100, 100, 100])
    picks = set()
    for seed in range(60):
        permuted = dataset.permute(scored, random.Random(seed))
        assert sorted(s.score for s in permuted) == sorted(s.score for s in scored)
        assert [s.branch for s in permuted] == [s.branch for s in scored]  # the scores move, the branches stay
        winner = dataset.pick_shuffled(scored, random.Random(seed))
        assert winner == dataset.best(permuted)  # the winner is picked on the permuted scores
        picks.add(winner.branch.index if winner else None)
    assert picks == {0, 1, 2, 3}  # a random branch, not the best one shuffled after the pick


def test_sampling_skips_nothing_it_was_given_and_counts_tokens() -> None:
    life, starts = _starts()
    calls: list[int] = []

    def factory(seed: int, night: date) -> sample.RecordingThinker:
        calls.append(seed)
        assert night == sample.branch_night(starts[0])  # dated on the branch's own night
        return fakes.FakeGlimmer(seed, 0.8)

    models = dataclasses.replace(fakes.fake_replay_models(), thinker_for=factory)
    branches = sample.run_branches(starts[0], life, models, brief=BRIEF, k=3)
    assert len(branches) == 3 == len(calls) and calls == [b.seed for b in branches]
    for branch in branches:
        assert branch.tokens >= sum(turn.reply.tokens for turn in branch.turns)
        assert branch.complete and branch.turns
        assert branch.ancestry is not None and branch.ancestry <= {SPARK_OPEN, GLIMMER}


def test_the_night_label_of_a_branch_counts_k5_afresh() -> None:
    life, starts = _starts()
    start = starts[0]
    stalled = [e for e in start.events if e.step.focus == start.question_id]
    assert stalled  # the M run worked the question that night, then stalled on it
    branches = _branches(start, life)
    assert all(len(branch.turns) >= 3 for branch in branches)  # each branch still gets its own night
    assert start.night + timedelta(days=1) == sample.branch_night(start)


def test_only_applied_turns_that_ended_at_a_stop_token_train() -> None:
    life, starts = _starts()

    class Cut:
        """The branch thinker, but every second call is cut at its token cap and the third fails."""

        def __init__(self, seed: int) -> None:
            self.inner, self.calls = fakes.FakeGlimmer(seed, 0.8), list[CallEnd]()

        def propose(self, rendered: str, max_tokens: int) -> it.ModelReply:
            reply = self.inner.propose(rendered, max_tokens)
            n = len(self.calls) + 1
            self.calls.append(CallEnd(None, failed=True) if n == 3 else CallEnd("length" if n % 2 == 0 else "stop"))
            return dataclasses.replace(reply, text="") if n == 3 else reply

    clean = _branches(starts[0], life)
    assert all(branch.complete and not branch.failed for branch in clean)
    for branch in clean:
        rejected = [t.seq for t in branch.turns if t.seq not in branch.applied]
        assert set(branch.applied) <= {t.seq for t in branch.turns}
        assert {t.seq for t in branch.targets} == set(branch.applied) and branch.day == sample.branch_night(starts[0])
        assert not set(rejected) & {t.seq for t in branch.targets}
    cut = _branches(starts[0], life, dataclasses.replace(fakes.fake_replay_models(), thinker_for=lambda s, n: Cut(s)))
    for branch in cut:
        assert all(branch.ends[t.seq].finish == "stop" and t.seq in branch.applied for t in branch.targets)
        assert any(end.finish == "length" for end in branch.ends.values())
        assert not any(branch.ends[t.seq].finish == "length" for t in branch.targets)
        assert branch.failed and not branch.complete  # a failed call makes a branch incomplete
    built = dataset.build([life], starts[:1], {starts[0].id: cut}, resolve.LabelBook().with_label(
        Label(starts[0].id, "q1.h2a", SPARK_OPEN)), brief=BRIEF)
    assert built.dropped["incomplete"] == 1 and not built.thinker1.examples
