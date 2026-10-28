# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Full practice caps each Spark Open role's output: the person (the scenes), the critic and the
teacher, each at what it writes plus twice the most reasoning seen. A cap never changes training
data silently: a cut teacher is no candidate, a cut critic drops its scene, a cut list of scenes
drops the scene it cut, and every cut is counted in the night log. Fakes only."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.brief import REASONING_ROOM
from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.model_api import ChatResult
from anjo.sleep.practice.full import (
    SCENES_ASKED,
    SPARK_CAPS,
    VISIBLE_TOKENS,
    FullPractice,
)
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

    def __init__(self, cut: frozenset[str] = frozenset(), *, whole_scenes: int = 10, teacher: str = "TEACHER: start with the infeed.") -> None:
        self.cut, self.whole_scenes, self.teacher = cut, whole_scenes, teacher
        self.caps: list[tuple[str, int | None]] = []

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None,
                 max_tokens: int | None = None) -> ChatResult:
        assert model == SPARK_OPEN
        role = role_of(messages)
        self.caps.append((role, max_tokens))
        finish = "length" if role in self.cut else "stop"
        if role == "person":
            scenes = [f"New brief {i}: where would you start?" for i in range(SCENES_ASKED)]
            text = json.dumps({"scenes": scenes})
            if finish != "stop":  # cut part way through the scene after ``whole_scenes`` whole ones
                text = text[:text.index(scenes[self.whole_scenes]) + 9]
            return _result(text, finish, out=900)
        if role == "critic":
            return _result('{"scores": {"A": 3, "B": 3, "C": 3, "D": 9}, "misses_constraint": []}', finish, out=700)
        return _result(self.teacher, finish, out=1200)


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

    # Each cap is the role's visible reply and the same room for reasoning: twice the most seen (the
    # longest live call wrote about 8.5k tokens against a visible brief of 2,000 at most), so a cap
    # bounds a runaway call and never bites a call like those measured.
    assert all(SPARK_CAPS[role] == VISIBLE_TOKENS[role] + REASONING_ROOM for role in SPARK_CAPS)
    assert REASONING_ROOM >= 2 * (8_500 - 2_000)
    assert VISIBLE_TOKENS["teacher"] == GLIMMER_REPLY_CAP
    longest_scenes = json.dumps({"scenes": [" ".join([LONG_SENTENCE] * 3)] * SCENES_ASKED})  # 20 three-sentence messages
    widest_verdict = json.dumps({"scores": dict.fromkeys("ABCD", 10), "misses_constraint": list("ABCD")})
    assert estimate_tokens(longest_scenes) <= VISIBLE_TOKENS["person"]
    assert estimate_tokens(widest_verdict) <= VISIBLE_TOKENS["critic"]


def test_a_cut_teacher_is_no_candidate() -> None:
    run = _run(Spark(cut=frozenset({"teacher"})))
    assert run.teacher_targets == 0 and run.spark["teacher"]["cut"] == 16 and len(run.transcripts) == 16
    assert all(t.target == "glimmer" and t.rejected_from != "teacher" for t in run.transcripts)


def test_a_teacher_reply_past_glimmers_bound_is_no_candidate() -> None:
    # It stopped on its own, but longer than any Glimmer reply can be: never a target or a side.
    long_reply = "TEACHER: " + " ".join(["word"] * (4 * GLIMMER_REPLY_CAP))
    assert estimate_tokens(long_reply) > GLIMMER_REPLY_CAP
    run = _run(Spark(teacher=long_reply))
    assert run.teacher_targets == 0 and run.spark["teacher"]["cut"] == 16
    assert all(t.reply != long_reply and t.rejected != long_reply for t in run.transcripts)


def test_cut_critic_drops_the_pair_and_is_logged(tmp_path: Path) -> None:
    judged = _run(Spark(cut=frozenset({"critic"})))  # the cut verdict named the teacher best: never read
    assert judged.transcripts == () and judged.pairs == 0 and judged.teacher_targets == 0
    assert judged.capped_scenes == 16 and judged.spark["critic"]["cut"] == 16
    assert judged.critic_unreadable == 0  # a cut is not an unreadable reply: no first-candidate target

    from anjo.sleep.tests.night_fakes import step
    from anjo.sleep.tests.test_full_practice import Spark as NightSpark
    from anjo.sleep.tests.test_full_practice import _full_night, _scenes

    class CutCritic(NightSpark):
        def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *,
                     temperature: float | None = None, max_tokens: int | None = None) -> ChatResult:
            if messages[-1]["content"].startswith("# critic_v1"):
                return _result('{"scores": {"A": 3', "length", out=max_tokens or 0)
            return super().__call__(model, messages, tools, timeout_s, temperature=temperature, max_tokens=max_tokens)

    result, _, _ = _full_night(tmp_path, CutCritic(_scenes()))
    done = step(result.log_path, "practice")
    assert done["scenes"] == 0 and done["pairs"] == 0 and done["capped_scenes"] == 16
    assert done["spark"]["critic"]["cut"] == 16 and done["spark"]["critic"]["calls"] == 16


def test_cut_person_drops_the_scene_not_the_night() -> None:
    run = _run(Spark(cut=frozenset({"person"})))  # ten whole scenes, then one cut part way
    assert len(run.transcripts) == 10 and run.capped_scenes == 1 and run.spark["person"]["cut"] == 1
    assert {t.message for t in run.transcripts} == {f"New brief {i}: where would you start?" for i in range(10)}
    nothing = _run(Spark(cut=frozenset({"person"}), whole_scenes=0))  # cut inside the first scene
    assert nothing.transcripts == () and nothing.capped_scenes == 1
