# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Full practice caps each Spark Open role's output: the person (the scenes), the critic and the
teacher, each at what it writes plus room for the model's reasoning. A reply cut at its cap is
never used as if whole: a cut teacher is no candidate, a cut critic is no verdict, and cut scenes
stop practice. Fakes only."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.model_api import ChatResult
from anjo.sleep.practice.full import (
    REASONING_ROOM,
    SCENES_ASKED,
    SPARK_CAPS,
    VISIBLE_TOKENS,
    FullPractice,
)
from anjo.sleep.practice.run import PracticeError
from anjo.sleep.serving.client import MAX_TOKENS as GLIMMER_REPLY_CAP
from anjo.sleep.tests.sleep_fakes import ev
from anjo.sleep.types import CheckpointRef

NIGHT = date(2026, 10, 28)
AT = datetime(2026, 10, 28, 23, 30, tzinfo=timezone.utc)
ADAPTER = CheckpointRef("p-abc-n2", "p1", date(2026, 10, 27), "modal-volume://anjo-sleep-adapters/p-abc-n2", None)
LONG_SENTENCE = " ".join(["word"] * 24) + "."


def _result(text: str, finish: str = "stop", out: int = 5) -> ChatResult:
    return ChatResult(text=text, tool_calls=(), model="m", input_tokens=100, output_tokens=out, finish_reason=finish)


def role_of(messages: list[dict[str, str]]) -> str:
    content = messages[-1]["content"]
    return "person" if content.startswith("# scenes_v1") else "critic" if content.startswith("# critic_v1") else "teacher"


class Spark:
    """Spark Open playing the person, the teacher and the critic; ``cut`` names the roles whose
    replies end at their cap."""

    def __init__(self, cut: frozenset[str] = frozenset()) -> None:
        self.cut = cut
        self.caps: list[tuple[str, int | None]] = []

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None,
                 max_tokens: int | None = None) -> ChatResult:
        assert model == SPARK_OPEN
        role = role_of(messages)
        self.caps.append((role, max_tokens))
        finish = "length" if role in self.cut else "stop"
        if role == "person":
            text = json.dumps({"scenes": [f"New brief {i}: where would you start?" for i in range(SCENES_ASKED)]})
            return _result(text if finish == "stop" else text[:40], finish, out=900)
        if role == "critic":
            return _result('{"scores": {"A": 3, "B": 3, "C": 3, "D": 9}, "misses_constraint": []}', finish, out=700)
        return _result("TEACHER: start with the infeed.", finish, out=1200)


class Glimmer:
    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None) -> ChatResult:
        return _result(f"GLIMMER on {messages[-1]['content'][:20]}")


def _run(spark: Spark) -> Any:
    practice = FullPractice(chat=spark, client=Glimmer(), adapter=lambda uid: ADAPTER, seed=7,
                            snapshot=lambda uid, message: (None, None))
    return practice.run("p1", NIGHT, [ev("own", "I run a recycling startup. Never use jargon.")], None, at=AT)


def test_spark_caps_are_applied_per_role() -> None:
    spark = Spark()
    run = _run(spark)
    for role in ("person", "critic", "teacher"):
        assert {cap for asked, cap in spark.caps if asked == role} == {SPARK_CAPS[role]}, role
    assert [role for role, _ in spark.caps].count("person") == 1
    # The night log's counts by role: calls, the provider's tokens, and replies cut at the cap.
    assert run.spark["teacher"] == {"calls": 16, "in": 1600, "out": 19200, "cut": 0}
    assert run.spark["critic"]["calls"] == 16 and run.spark["person"]["out"] == 900

    # Each cap is the role's visible reply and the same room for reasoning; the teacher's reply is
    # held to the bound its Glimmer rivals have (a longer one is cut and never trained).
    assert all(SPARK_CAPS[role] == VISIBLE_TOKENS[role] + REASONING_ROOM for role in SPARK_CAPS)
    assert VISIBLE_TOKENS["teacher"] == GLIMMER_REPLY_CAP
    longest_scenes = json.dumps({"scenes": [" ".join([LONG_SENTENCE] * 3)] * SCENES_ASKED})  # 20 three-sentence messages
    widest_verdict = json.dumps({"scores": dict.fromkeys("ABCD", 10), "misses_constraint": list("ABCD")})
    assert estimate_tokens(longest_scenes) <= VISIBLE_TOKENS["person"]
    assert estimate_tokens(widest_verdict) <= VISIBLE_TOKENS["critic"]


def test_a_cut_teacher_is_no_candidate_and_a_cut_critic_no_verdict() -> None:
    run = _run(Spark(cut=frozenset({"teacher"})))
    assert run.teacher_targets == 0 and run.spark["teacher"]["cut"] == 16
    assert all(t.target == "glimmer" and t.rejected_from != "teacher" for t in run.transcripts)

    judged = _run(Spark(cut=frozenset({"critic"})))
    assert judged.critic_unreadable == 16 and judged.pairs == 0 and judged.spark["critic"]["cut"] == 16
    assert judged.teacher_targets == 0  # the cut verdict named the teacher best: never read


def test_cut_scenes_stop_practice_and_say_so() -> None:
    with pytest.raises(PracticeError, match="cap"):
        _run(Spark(cut=frozenset({"person"})))
