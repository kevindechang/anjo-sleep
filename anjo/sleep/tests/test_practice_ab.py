# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The practice A/B (``tools.practice_ab``): plan-first; on the same scenes and seeds; judged by a
family neither arm uses (Gemini), blind, in both orders, a flip a tie and a dropped scene its arm's
loss; and the default changes only when arm B is non-inferior at the bar fixed before any run. Critic
agreement with Spark Open is reported, never a bar."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.night.deps import NightOptions, Prices
from anjo.sleep.practice.full import FullPractice
from anjo.sleep.practice.roles import TODAY, Roles
from anjo.sleep.tests.test_full_practice import ADAPTER, Glimmer, Spark, _evidence, _scenes, scored
from anjo.sleep.tests.test_practice_roles import DEEPSEEK, DeepSeek, _critic
from anjo.sleep.tools import practice_ab

UID = "synthetic-p1"
NIGHTS = [date(2026, 10, 18) + timedelta(days=i) for i in range(10)]
CASES = [arg for night in NIGHTS for arg in ("--case", f"{UID}:film_student:{night.isoformat()}")]
ARGS = [*CASES, "--arm-b", "critic=deepseek"]
PRICES = Prices(spark_in_per_m=1.0, spark_out_per_m=2.0, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.95)
LABELS = ("arm a", "arm b", "arm_a", "arm_b", "deepseek", "spark", "critic", "gemini", "person=", "teacher=", "roles")


def _refuse(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("the plan sends nothing")


def test_practice_ab_plan_prints_the_bar_its_scenes_and_holds_and_sends_nothing(
        capsys: pytest.CaptureFixture[str]) -> None:
    assert practice_ab.SCENES_NEEDED == 155  # 95% one-sided, margin 0.10, power 80% at the widest spread
    assert practice_ab.main(ARGS, wiring=_refuse, critic_port=_refuse, judge=_refuse, ledger=_refuse) == 0
    out = capsys.readouterr().out
    for word in ("160 scenes (the bar needs 155)", "baseten_sleep_usd", "fireworks_sleep_usd", "vertex_usd",
                 "modal_usd", "both orders", "non-inferior", "never a bar", "expected"):
        assert word in out
    with pytest.raises(SystemExit):  # too few scenes for the bar: refused before anything is built
        practice_ab.main([*CASES[:4], "--arm-b", "critic=deepseek", "--out", "x", "--go"], wiring=_refuse)
    with pytest.raises(SystemExit):
        practice_ab.main([*CASES, "--arm-b", "person=spark"], wiring=_refuse)  # arm B is today's
    assert practice_ab.JUDGE_FAMILY not in practice_ab.families(TODAY) | practice_ab.families(Roles(critic="deepseek"))


class Judge:
    """Gemini as the blind judge: prefers the example whose reply to learn is the teacher's."""

    def __init__(self, always: str | None = None) -> None:
        self.always = always
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        self.prompts.append(prompt)
        first = prompt.split("## Example 1", 1)[1].split("## Example 2", 1)[0]
        verdict = self.always or ("1" if "Reply to learn: TEACHER" in first else "2")
        return json.dumps({"verdict": verdict}), 1_000, 50


def _wiring(spark: Spark, deepseek: DeepSeek) -> Any:
    def build(trainer: Any, options: NightOptions) -> Any:
        assert trainer is None and options.practice_roles is not None
        roles: Roles = options.practice_roles
        critic = _critic(deepseek) if roles.critic == "deepseek" else None

        def practice(chat: Any, night: Any) -> FullPractice:
            return FullPractice(chat=chat, client=Glimmer(), adapter=lambda u: ADAPTER, seed=7,
                                snapshot=lambda u, m: (None, frozenset({SPARK_OPEN})), critic=critic,
                                third_party=lambda u: True, lazy_teacher=roles.teacher == "lazy")

        store = SimpleNamespace(trainable=lambda night: _evidence())
        return SimpleNamespace(practice=practice, evidence=lambda uid: store, spark_chat=spark, prices=PRICES,
                               wake=None, ledger=None)

    return build


def _go(tmp_path: Path, judge: Judge, capsys: pytest.CaptureFixture[str], *, cut: Any = lambda n: False,
        ) -> tuple[Spark, DeepSeek, list[Any]]:
    spark = Spark(_scenes(), scored(5, (6, 7, 8), lambda text: False))  # arm A: Glimmer's third, nothing flagged
    deepseek = DeepSeek(scored(9, (5, 6, 7), lambda text: text.startswith("GLIMMER")), cut=cut)  # every Glimmer one misses
    code = practice_ab.main([*ARGS, "--out", str(tmp_path), "--go"], wiring=_wiring(spark, deepseek),
                            critic_port=lambda night, ledger: _critic(deepseek), judge=judge, ledger=None)
    assert code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    return spark, deepseek, lines


def test_practice_ab_is_blind_and_order_swapped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    judge = Judge()
    spark, deepseek, lines = _go(tmp_path, judge, capsys)
    arms, (retest, new), judged, decision = lines[:20], lines[20:22], lines[22], lines[23]
    # Same scenes: arm A's are handed to arm B, so the person is asked once a case.
    assert sum(c["messages"][-1]["content"].startswith("# scenes_v1") for c in spark.calls) == 10
    assert {a["roles"]["critic"] for a in arms[0::2]} == {"spark"} and {a["roles"]["critic"] for a in arms[1::2]} == {DEEPSEEK}
    # Critic agreement on the very same prompts, reported against Spark Open, never a bar.
    asked, judged_by_b = spark.critic_prompts(), deepseek.prompts  # a case: A's 16, then their retest / B's 16, then A's
    assert len(asked) == len(judged_by_b) == 10 * 32
    for case in range(10):
        arm_a = asked[32 * case:32 * case + 16]
        assert asked[32 * case + 16:32 * case + 32] == arm_a == judged_by_b[32 * case + 16:32 * case + 32]
    assert retest["critic"] == "spark-retest" and retest["agree_best_with_spark"] == 1.0
    assert new["critic"] == "deepseek" and new["flag_rate"] == 0.75 and new["agree_best_with_spark"] == 0.0
    # Blind: no arm, role or model named; both orders, so 320 judge calls for 160 scenes.
    assert len(judge.prompts) == 320
    assert not any(label in prompt.lower() for prompt in judge.prompts for label in LABELS)
    for first, second in zip(judge.prompts[::2], judge.prompts[1::2], strict=True):
        one, two = (p.split("## Example 2")[0].split("## Example 1")[1] for p in (first, second))
        assert one != two  # the second order shows the other arm's example first
    assert judged["judge"]["win"] == 160 and judged["score_b"] == 1.0 and judged["non_inferior"] is True
    assert decision == {"switch_default": True}  # the downstream judgement decides, not agreement with Spark
    saved = (tmp_path / "arm_b.jsonl").read_text().splitlines()
    assert len(saved) == 160 and (tmp_path / "arm_a.jsonl").exists()


def test_practice_ab_counts_a_dropped_scene_as_its_arms_loss(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Arm B's critic cuts every 4th verdict, so B drops 4 of each case's 16 scenes; the judge always picks
    # what it sees first, so every scene both kept flips to a tie.
    _, _, lines = _go(tmp_path, Judge(always="1"), capsys, cut=lambda n: n % 4 == 0)
    judged = lines[22]
    assert judged["judge"]["dropped_b"] == 40 and judged["judge"]["flips"] == 120 and judged["judge"]["tie"] == 120
    assert judged["score_b"] == pytest.approx((120 * 0.5) / 160)  # the 40 dropped scenes are B's losses
    assert judged["lower_bound"] < judged["bar"] == 0.4 and judged["non_inferior"] is False
    assert lines[23] == {"switch_default": False}
