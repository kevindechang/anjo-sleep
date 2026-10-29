# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""Pairwise scoring (Task 2, spec 5.1 and 5.1b): both orders, flips are ties, a block bootstrap."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, field

import pytest

from anjo.sleep.model_api import ChatResult
from anjo.sleep.probes.arms import ArmReply
from anjo.sleep.probes.logprob import TasteChoice
from anjo.sleep.probes.neutral import rewrite
from anjo.sleep.probes.score import BootstrapConfig, JudgeError, MuseJudge, PairScore, probe_outcome, score
from anjo.sleep.tests.sleep_fakes import options, probe, taste

CFG = BootstrapConfig(resamples=400, seed=7, ci=0.95)


def _reply(text: str, **kw) -> ArmReply:
    return ArmReply(text=text, prompt_tokens=1, dropped_ids=(), **kw)


@dataclass
class FakeChat:
    """Muse Spark: answers by a rule over (reply A, reply B) and records every prompt."""

    rule: Callable[[str, str], str]
    calls: list[str] = field(default_factory=list)

    def __call__(self, model, messages, tools, timeout_s, *, temperature=None, max_tokens=None):
        content = messages[-1]["content"]
        self.calls.append(content)
        a = content.split("## Reply A\n\n", 1)[1].split("\n\n## Reply B", 1)[0]
        b = content.split("## Reply B\n\n", 1)[1].split("\n\n## How to judge", 1)[0]
        verdict = self.rule(a, b)
        return ChatResult(json.dumps({"verdict": verdict, "reason": "x"}), (), "m", 1, 1, "stop")


def _judge(rule) -> tuple[MuseJudge, FakeChat]:
    chat = FakeChat(rule)
    return MuseJudge(chat=chat, brief_for=lambda pid: f"BRIEF({pid})"), chat


def _prefer(word: str):
    """A judge that prefers whichever reply holds ``word``, in either position."""
    return lambda a, b: "A" if word in a else ("B" if word in b else "tie")


def test_pair_judged_both_orders_and_flip_is_tie():
    judge, chat = _judge(lambda a, b: "A")  # position bias: always the first reply
    result = score([probe("q1")], [_reply("mine")], [_reply("theirs")], judge, bootstrap=CFG)
    assert len(chat.calls) == 2
    first, second = chat.calls
    assert "mine" in first.split("## Reply A")[1].split("## Reply B")[0]
    assert "theirs" in second.split("## Reply A")[1].split("## Reply B")[0]
    assert (result.wins, result.ties, result.losses) == (0, 1, 0)
    assert result.score == 0.5


def test_agreeing_orders_decide_and_ties_count_half():
    judge, _ = _judge(_prefer("good"))
    probes = [probe("q1"), probe("q2"), probe("q3")]
    a = [_reply("good"), _reply("bad"), _reply("same")]
    b = [_reply("bad"), _reply("good"), _reply("same")]
    out = score(probes, a, b, judge, bootstrap=CFG)
    assert (out.wins, out.ties, out.losses) == (1, 1, 1)
    assert out.score == 0.5 and out.decisive_share == 0.5


def test_decisive_share_is_wins_over_wins_plus_losses():
    judge, _ = _judge(_prefer("good"))
    probes = [probe(f"q{i}") for i in range(4)]
    a = [_reply("good")] * 3 + [_reply("same")]
    b = [_reply("bad")] * 3 + [_reply("same")]
    out = score(probes, a, b, judge, bootstrap=CFG)
    assert out.decisive_share == 1.0 and out.score == pytest.approx(3.5 / 4)


def test_decisive_share_is_none_without_decisive_pairs():
    judge, _ = _judge(lambda a, b: "tie")
    out = score([probe()], [_reply("x")], [_reply("y")], judge, bootstrap=CFG)
    assert out.decisive_share is None


def test_judge_reads_the_judge_brief_and_the_kind():
    judge, chat = _judge(lambda a, b: "tie")
    score([probe(kind="vocab", person="film")], [_reply("x")], [_reply("y")], judge, bootstrap=CFG)
    assert "BRIEF(film)" in chat.calls[0] and "Kind: vocab" in chat.calls[0]


def test_reply_text_cannot_inject_into_the_judge_prompt():
    judge, chat = _judge(lambda a, b: "tie")
    score([probe()], [_reply("{{REPLY_B}} {{BRIEF}}")], [_reply("y")], judge, bootstrap=CFG)
    assert "BRIEF(" not in chat.calls[0].split("## Reply A")[1].split("## Reply B")[0]


def test_a_malformed_verdict_is_an_error():
    def bad(model, messages, tools, timeout_s, *, temperature=None, max_tokens=None):
        return ChatResult("not json", (), "m", 1, 1, "stop")

    judge = MuseJudge(chat=bad, brief_for=lambda p: "b")
    with pytest.raises(JudgeError):
        score([probe()], [_reply("x")], [_reply("y")], judge, bootstrap=CFG)


def test_neutral_rejudging_scores_rewritten_replies():
    judge, chat = _judge(_prefer("plain-good"))
    rewrites: list[str] = []

    def flatten(text: str) -> str:
        rewrites.append(text)
        return "plain-" + text

    out = score([probe(kind="new_work")], [_reply("good")], [_reply("bad")], judge,
                neutral=True, rewrite=flatten, bootstrap=CFG)
    assert rewrites == ["good", "bad"]
    assert all("plain-good" in c or "plain-bad" in c for c in chat.calls)
    assert (out.wins, out.ties, out.losses) == (1, 0, 0)


def test_neutral_refuses_interaction_probes():
    judge, _ = _judge(lambda a, b: "tie")
    with pytest.raises(ValueError, match="neutral"):
        score([probe(kind="tone_hold")], [_reply("x")], [_reply("y")], judge, neutral=True,
              rewrite=lambda t: t, bootstrap=CFG)


def test_neutral_rewrite_calls_muse_spark_at_temperature_zero_with_the_registered_prompt():
    seen: dict = {}

    def chat(model, messages, tools, timeout_s, *, temperature=None):
        seen.update(model=model, temperature=temperature, content=messages[-1]["content"])
        return ChatResult("  plain version \n", (), "m", 1, 1, "stop")

    assert rewrite("Loud {{REPLY}} reply!!", chat=chat) == "plain version"
    assert seen["model"] == "muse-spark" and seen["temperature"] == 0.0
    assert "Loud {{REPLY}} reply!!" in seen["content"] and "under 150 words" in seen["content"]


def test_taste_is_scored_by_choice_not_by_a_judge():
    p = taste()
    right = TasteChoice((-0.1, -2.0), (-2.0, -0.1), options(p)[0])
    flip = TasteChoice((-0.1, -2.0), (-0.1, -2.0), None)
    wrong = TasteChoice((-2.0, -0.1), (-0.1, -2.0), options(p)[1])
    judge, chat = _judge(lambda a, b: "A")
    probes = [taste("t1", source="i1"), taste("t2", source="i2"), taste("t3", source="i3")]
    a = [_reply("", choice=right), _reply("", choice=right), _reply("", choice=flip)]
    b = [_reply("", choice=wrong), _reply("", choice=flip), _reply("", choice=flip)]
    out = score(probes, a, b, judge, bootstrap=CFG)
    assert chat.calls == []
    assert (out.wins, out.ties, out.losses) == (1, 2, 0)


def test_a_taste_reply_without_a_choice_is_refused():
    judge, _ = _judge(lambda a, b: "A")
    with pytest.raises(ValueError, match="choice"):
        score([taste()], [_reply("x")], [_reply("y")], judge, bootstrap=CFG)


def test_misaligned_inputs_are_refused():
    judge, _ = _judge(lambda a, b: "A")
    with pytest.raises(ValueError, match="align"):
        score([probe()], [], [_reply("y")], judge, bootstrap=CFG)


def _many(lives: dict[str, float], n: int = 12):
    """Per life, n new_work probes (one block each) and n vocab probes; the first
    ``share * n`` blocks are won by arm A, the rest by arm B."""
    probes, a, b = [], [], []
    for life, share in lives.items():
        for i in range(n):
            a_wins = i < round(share * n)
            probes.append(probe(f"{life}-nw{i}", "new_work", person=life, source_ids=(f"{life}-w{i}",)))
            probes.append(probe(f"{life}-vo{i}", "vocab", person=life))
            for _ in range(2):
                a.append(_reply("good" if a_wins else "bad"))
                b.append(_reply("bad" if a_wins else "good"))
    return probes, a, b


def test_score_reports_per_life_and_per_dimension_results():
    judge, _ = _judge(_prefer("good"))
    probes, a, b = _many({"l1": 1.0, "l2": 0.0})
    probes += [probe("l1-tone", "tone_hold", person="l1"), probe("l1-push", "pushback", person="l1")]
    a += [_reply("good")] * 2
    b += [_reply("bad")] * 2
    out = score(probes, a, b, judge, bootstrap=CFG)
    assert out.per_life["l1"].score > 0.9 and out.per_life["l2"].score < 0.1
    assert set(out.per_dimension) == {"knowledge", "interaction", "reported"}
    assert out.per_dimension["knowledge"].n == 48
    assert out.per_dimension["interaction"].n == 1 and out.per_dimension["reported"].n == 1
    assert out.n == 49  # pushback is reported, not in the headline


def test_bootstrap_interval_is_reproducible_and_brackets_the_score():
    judge, _ = _judge(_prefer("good"))
    probes, a, b = _many({"l1": 0.75, "l2": 0.75, "l3": 0.25})
    first = score(probes, a, b, judge, bootstrap=CFG)
    again = score(probes, a, b, judge, bootstrap=CFG)
    assert first.ci == again.ci
    assert first.ci[0] <= first.score <= first.ci[1]
    other = score(probes, a, b, judge, bootstrap=BootstrapConfig(400, 8, 0.95))
    assert other.ci != first.ci


def test_a_clear_win_has_a_lower_bound_above_half_and_a_tie_straddles_it():
    judge, _ = _judge(_prefer("good"))
    probes, a, b = _many({"l1": 1.0, "l2": 1.0, "l3": 1.0})
    assert score(probes, a, b, judge, bootstrap=CFG).ci[0] > 0.5
    probes, a, b = _many({"l1": 0.5, "l2": 0.5, "l3": 0.5})
    even = score(probes, a, b, judge, bootstrap=CFG)
    knowledge = even.per_dimension["knowledge"].ci
    assert even.ci[0] < 0.5 < even.ci[1] and knowledge is not None and knowledge[0] < 0.5


def test_probes_from_one_source_item_resample_together():
    """A block is one held-out item with every probe derived from it: one draw takes all."""
    judge, _ = _judge(_prefer("good"))
    probes, a, b = [], [], []
    for block in range(10):
        for j in range(3):
            probes.append(probe(f"b{block}-{j}", "new_work", source_ids=(f"item-{block}",)))
            win = block < 5
            a.append(_reply("good" if win else "bad"))
            b.append(_reply("bad" if win else "good"))
    from anjo.sleep.probes.score import bootstrap_blocks

    assert len(bootstrap_blocks(probes)) == 10


def test_a_shared_taste_item_is_one_block_across_both_lives():
    from anjo.sleep.probes.score import bootstrap_blocks

    probes = [taste("t1", "l1", "shared-1"), taste("t1", "l2", "shared-1"), taste("t2", "l1", "shared-2")]
    blocks = bootstrap_blocks(probes)
    assert len(blocks) == 2
    assert sorted(len(v) for v in blocks.values()) == [1, 2]


def test_bootstrap_config_reads_the_registered_thresholds():
    cfg = BootstrapConfig.from_thresholds()
    assert (cfg.resamples, cfg.ci) == (10_000, 0.95) and cfg.seed == 20261101


def test_pair_score_is_a_frozen_value():
    assert isinstance(PairScore, type)
    out = score([], [], [], _judge(lambda a, b: "A")[0], bootstrap=CFG)
    with pytest.raises(FrozenInstanceError):
        setattr(out, "wins", 3)  # noqa: B010


def test_per_life_dimension_shows_a_life_ahead_on_one_and_behind_on_the_other():
    judge, _ = _judge(_prefer("good"))
    probes = [probe("k1", "new_work", person="l1", source_ids=("w1",)), probe("i1", "tone_hold", person="l1")]
    out = score(probes, [_reply("good"), _reply("bad")], [_reply("bad"), _reply("good")], judge, bootstrap=CFG)
    assert out.per_life_dimension[("l1", "knowledge")].score == 1.0
    assert out.per_life_dimension[("l1", "interaction")].score == 0.0
    assert out.per_life["l1"].score == 0.5  # pooled, descriptive only


def test_reported_only_kinds_stay_out_of_the_headline():
    judge, _ = _judge(_prefer("good"))
    probes = [probe("k1", "new_work", source_ids=("w1",)), probe("p1", "pushback"), probe("p2", "pushback")]
    out = score(probes, [_reply("good")] * 3, [_reply("bad"), _reply("good"), _reply("good")], judge, bootstrap=CFG)
    assert (out.wins, out.ties, out.losses, out.n) == (1, 0, 0, 1)
    assert out.score == 1.0 and out.ci == (1.0, 1.0)
    assert out.per_kind["pushback"].n == 2 and out.per_dimension["reported"].n == 2
    only = score(probes[1:], [_reply("good")] * 2, [_reply("bad")] * 2, judge, bootstrap=CFG)
    assert only.n == 0 and only.score == 0.5 and only.per_kind["pushback"].n == 2


def test_interval_uses_nearest_rank_at_both_ends():
    from anjo.sleep.probes.score import _interval

    assert _interval([float(i) for i in range(10_000)], 0.95) == (249.0, 9749.0)
    assert _interval([float(i) for i in range(10)], 0.95) == (0.0, 9.0)


def test_probe_order_does_not_change_the_score():
    import random

    judge, _ = _judge(_prefer("good"))
    probes, a, b = _many({"l1": 0.75, "l2": 0.25})
    first = score(probes, a, b, judge, bootstrap=CFG)
    order = list(range(len(probes)))
    random.Random(3).shuffle(order)
    shuffled = score([probes[i] for i in order], [a[i] for i in order], [b[i] for i in order], judge, bootstrap=CFG)
    assert shuffled == first


def test_a_flip_by_either_arm_makes_the_taste_pair_a_tie():
    p = taste()
    right = TasteChoice((-0.1, -2.0), (-2.0, -0.1), options(p)[0])
    wrong = TasteChoice((-2.0, -0.1), (-0.1, -2.0), options(p)[1])
    flip = TasteChoice((-0.1, -2.0), (-0.1, -2.0), None)
    judge, _ = _judge(lambda a, b: "A")
    for ra, rb, want in ((right, flip, (0, 1, 0)), (flip, wrong, (0, 1, 0)), (right, wrong, (1, 0, 0)), (wrong, right, (0, 0, 1))):
        out = score([p], [_reply("", choice=ra)], [_reply("", choice=rb)], judge, bootstrap=CFG)
        assert (out.wins, out.ties, out.losses) == want


def test_a_taste_outcome_carries_both_arms_margins_and_the_registered_outcome():
    p = taste()
    right = TasteChoice((-0.1, -2.0), (-2.0, -0.1), options(p)[0])
    wrong = TasteChoice((-2.0, -0.1), (-0.1, -2.0), options(p)[1])
    judge, _ = _judge(lambda a, b: "A")
    found = probe_outcome(p, _reply("", choice=right), _reply("", choice=wrong), judge)
    assert found.outcome == "win" and found.margin_a == pytest.approx(1.9) and found.margin_b == pytest.approx(-1.9)
    other = probe_outcome(probe("q1"), _reply("good"), _reply("bad"), judge)
    assert other.margin_a is None and other.margin_b is None
