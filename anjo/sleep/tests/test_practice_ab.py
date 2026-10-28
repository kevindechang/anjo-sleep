# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The practice A/B (``tools.practice_ab``): plan-first; on the same scenes and seeds; the critics
compared on the very same prompts (Spark Open's own retest as the floor); the teacher material judged
blind, in both orders, a flip a tie; and the default changes only when every pre-set bar holds."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.deps import NightOptions, Prices
from anjo.sleep.practice.full import FullPractice
from anjo.sleep.practice.roles import TODAY, Roles
from anjo.sleep.tests.test_full_practice import ADAPTER, Glimmer, Spark, _evidence, _scenes, scored
from anjo.sleep.tests.test_practice_roles import GEMINI, Gemini, _critic
from anjo.sleep.tools import practice_ab

UID = "synthetic-p1"
ARGS = ["--user", UID, "--persona", "film_student", "--night", "2026-10-28", "--arm-b", "critic=gemini"]
PRICES = Prices(spark_in_per_m=1.0, spark_out_per_m=2.0, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.95)
LABELS = ("arm a", "arm b", "arm_a", "arm_b", "gemini", "spark", "critic", "person=", "teacher=", "roles")


def _refuse(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("the plan sends nothing")


def test_practice_ab_plan_prints_holds_and_sends_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    assert practice_ab.main(ARGS, wiring=_refuse, critic_port=_refuse, judge_chat=_refuse, ledger=_refuse) == 0
    out = capsys.readouterr().out
    for word in ("baseten_sleep_usd", "vertex_usd", "meta_usd", "modal_usd", "both orders", "flag ratio"):
        assert word in out
    with pytest.raises(SystemExit):
        practice_ab.main([*ARGS[:-1], "person=spark"], wiring=_refuse)  # arm B is today's: nothing to compare


class Judge:
    """Muse Spark as the blind judge: prefers the example whose reply to learn is the teacher's."""

    def __init__(self, always: str | None = None) -> None:
        self.always = always
        self.prompts: list[str] = []

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, **kw: Any) -> ChatResult:
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        first = prompt.split("## Example 1", 1)[1].split("## Example 2", 1)[0]
        verdict = self.always or ("1" if "Reply to learn: TEACHER" in first else "2")
        return ChatResult(text=json.dumps({"verdict": verdict}), tool_calls=(), model="m", input_tokens=1,
                          output_tokens=1, finish_reason="stop")


def _wiring(spark: Spark, gemini: Gemini) -> Any:
    def build(trainer: Any, options: NightOptions) -> Any:
        assert trainer is None and options.practice_roles is not None
        roles: Roles = options.practice_roles
        critic = _critic(gemini) if roles.critic == "gemini" else None

        def practice(chat: Any, night: Any) -> FullPractice:
            return FullPractice(chat=chat, client=Glimmer(), adapter=lambda u: ADAPTER, seed=7,
                                snapshot=lambda u, m: (None, frozenset({SPARK_OPEN})), critic=critic,
                                third_party=lambda u: True, lazy_teacher=roles.teacher == "lazy")

        store = SimpleNamespace(trainable=lambda night: _evidence())
        return SimpleNamespace(practice=practice, evidence=lambda uid: store, spark_chat=spark, prices=PRICES,
                               wake=None, ledger=None)

    return build


def _go(tmp_path: Path, judge: Judge, capsys: pytest.CaptureFixture[str]) -> tuple[Spark, Gemini, list[Any]]:
    spark = Spark(_scenes(), scored(5, (6, 7, 8), lambda text: False))  # arm A: Glimmer's third, nothing flagged
    gemini = Gemini(scored(9, (5, 6, 7), lambda text: text.startswith("GLIMMER")))  # every Glimmer one misses
    code = practice_ab.main([*ARGS, "--out", str(tmp_path), "--go"], wiring=_wiring(spark, gemini),
                            critic_port=lambda night, ledger: _critic(gemini), judge_chat=judge, ledger=None)
    assert code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    return spark, gemini, lines


def test_practice_ab_is_blind_and_order_swapped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    judge = Judge()
    spark, gemini, lines = _go(tmp_path, judge, capsys)
    arm_a, arm_b, *critics, judged, decision = lines
    # Same scenes: arm A's are handed to arm B, so the person is asked once.
    assert sum(c["messages"][-1]["content"].startswith("# scenes_v1") for c in spark.calls) == 1
    assert arm_a["roles"]["critic"] == "spark" and arm_b["roles"]["critic"] == GEMINI
    assert arm_a["scenes"] == arm_b["scenes"] == 16 and arm_b["teacher_targets"] == 16 and arm_a["teacher_targets"] == 0
    # The critics on the very same prompts: arm A's 16 again to Spark Open and to Gemini.
    asked_a = spark.critic_prompts()[:16]
    assert spark.critic_prompts()[16:] == asked_a and gemini.prompts[16:] == asked_a
    retest, new = critics[1], critics[2]
    assert retest["critic"] == "spark-retest" and retest["agree_best"] == 1.0 and critics[0]["flag_rate"] == 0.0
    assert new["critic"] == GEMINI and new["flag_rate"] == 0.75 and new["agree_best"] == 0.0
    # Blind: no arm, role or model named; both orders, so 32 judge calls for 16 scenes.
    assert len(judge.prompts) == 32
    assert not any(label in prompt.lower() for prompt in judge.prompts for label in LABELS)
    for first, second in zip(judge.prompts[::2], judge.prompts[1::2], strict=True):
        one, two = (p.split("## Example 2")[0].split("## Example 1")[1] for p in (first, second))
        assert one != two  # the second order shows the other arm's example first
    assert judged["judge"]["win"] == 16 and judged["score_b"] == 1.0
    # A judge that always prefers what it sees first flips every time: all ties.
    _, _, flipped = _go(tmp_path / "again", Judge(always="1"), capsys)
    assert flipped[-2]["judge"]["flips"] == 16 and flipped[-2]["score_b"] == 0.5
    # The bars were set before the result: a critic flagging far more than Spark Open keeps today's default,
    # however the judge liked arm B.
    assert decision["bars"]["judge"] is True and decision["bars"]["flag_ratio"] is False
    assert decision["switch_default"] is False
    saved = [json.loads(line) for line in (tmp_path / "arm_b.jsonl").read_text().splitlines()]
    assert len(saved) == 16 and (tmp_path / "arm_a.jsonl").exists()
    assert TODAY == Roles()
