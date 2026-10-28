# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Who plays practice's roles (``ANJO_SLEEP_PRACTICE_ROLES``): Spark Open plays all three unless set.
Glimmer may play the person; a third-party critic (Gemini 3.8 Flash on Vertex) may score a synthetic
person's candidates, never a real person's; the teacher (always Spark Open) may be asked only where it
could become the target (``teacher=lazy``). The critic's words never reach a transcript, the teacher
or anything that trains, so its model never enters a training row's ancestry."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.model_api import ChatResult
from anjo.sleep.practice import roles
from anjo.sleep.practice.critics import ThirdPartyCritic
from anjo.sleep.practice.full import SCENES_ASKED, VISIBLE_TOKENS, FullPractice
from anjo.sleep.provenance import CLEARED, GLIMMER, may_train
from anjo.sleep.tests.test_full_practice import (
    ADAPTER,
    AT,
    NIGHT,
    Glimmer,
    Spark,
    _dataset,
    _evidence,
    _scenes,
    call_no,
    candidates_of,
    scored,
)

GEMINI = "gemini-3.8-flash"
CRITIC_WORDS = "GEMINI-SAYS-candidate-B-is-sloppy"  # free text a critic might add: it must never travel


class Gemini:
    """A third-party critic: the critic prompt in, JSON scores out (with free text it was never asked for)."""

    def __init__(self, critic: Any, *, raw: str | None = None) -> None:
        self.critic, self.raw = critic, raw
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        self.prompts.append(prompt)
        if self.raw is not None:
            return self.raw, 900, 40
        return json.dumps({**self.critic(candidates_of(prompt)), "why": CRITIC_WORDS}), 900, 40


class PersonGlimmer(Glimmer):
    """Glimmer's client, also playing the person (the base model, at the person's cap)."""

    def __init__(self, scenes: list[str]) -> None:
        super().__init__()
        self.scenes = scenes
        self.person: list[dict[str, Any]] = []

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None,
                 max_tokens: int | None = None) -> ChatResult:
        if messages[-1]["content"].startswith("# scenes_v1"):
            self.person.append({"adapter": adapter, "seed": seed, "max_tokens": max_tokens, "temperature": temperature})
            return ChatResult(text=json.dumps({"scenes": self.scenes}), tool_calls=(), model="m", input_tokens=10,
                              output_tokens=5, finish_reason="stop")
        return super().complete(adapter, messages, temperature=temperature, seed=seed)


def _critic(gemini: Gemini) -> ThirdPartyCritic:
    return ThirdPartyCritic(model=GEMINI, ask=gemini, usd_per_m_in=0.75, usd_per_m_out=3.75)


def _run(spark: Spark, glimmer: Glimmer | None = None, uid: str = "p1", **kw: Any) -> Any:
    practice = FullPractice(chat=spark, client=glimmer or Glimmer(), adapter=lambda u: ADAPTER, seed=7,
                            snapshot=lambda u, m: (f"C-SNAPSHOT for {m[:12]}", frozenset({SPARK_OPEN})), **kw)
    return practice, practice.run(uid, NIGHT, _evidence(), None, at=AT)


def test_practice_roles_default_is_today() -> None:
    assert roles.practice_roles({}) == roles.TODAY == roles.Roles("spark", "spark", "always")
    assert roles.practice_roles({roles.ROLES_ENV: " "}) == roles.TODAY
    assert roles.practice_roles({roles.ROLES_ENV: "glimmer"}) == roles.Roles("glimmer", "spark", "always")
    assert roles.practice_roles({roles.ROLES_ENV: "person=glimmer, critic=gemini,teacher=lazy"}) == roles.Roles(
        "glimmer", "gemini", "lazy")
    for wrong in ("teacher=glimmer", "critic=glimmer", "person=gemini", "critic", "person=glimmer,person=spark",
                  "judge=gemini"):
        with pytest.raises(roles.RolesError, match=roles.ROLES_ENV):
            roles.practice_roles({roles.ROLES_ENV: wrong})

    # Unset, practice is today's: Spark Open writes the scenes, answers as the teacher and judges.
    spark = Spark(_scenes())
    practice, run = _run(spark)
    assert practice.max_spark_calls == 1 + 2 * 16
    assert len(spark.critic_prompts()) == 16 and len(spark.teacher_calls()) == 16
    assert spark.calls[0]["messages"][-1]["content"].startswith("# scenes_v1")
    assert run.critic_usd == 0.0 and run.roles == {"person": "spark", "teacher": "spark", "critic": "spark"}
    assert run.spark["person"]["calls"] == 1 and run.spark["critic"]["calls"] == 16


def test_third_party_critic_never_enters_training_ancestry(tmp_path: Path) -> None:
    """The teacher writes its candidate from F and the message alone, before any critic runs (and, lazy,
    from those alone after one did); the critic's reply is read for scores and letters only. So a
    critic that may not train is in no transcript, no evidence item, no training row and no ancestry."""
    misses = scored(9, (5, 6, 7), lambda text: text.startswith("GLIMMER") and call_no(text) // 3 % 2 == 0)
    for teacher in ("always", "lazy"):
        gemini, spark = Gemini(misses), Spark(_scenes())
        practice, run = _run(spark, critic=_critic(gemini), third_party=lambda uid: True, lazy_teacher=teacher == "lazy")
        assert len(gemini.prompts) >= 16 and spark.critic_prompts() == []  # the scores came from the third party
        assert run.roles["critic"] == GEMINI and run.critic_usd == pytest.approx(
            len(gemini.prompts) * (900 * 0.75 + 40 * 3.75) / 1e6)
        assert any(t.target == "teacher" for t in run.transcripts)  # its scores still choose
        for transcript in run.transcripts:
            assert transcript.ancestry is not None and transcript.ancestry <= CLEARED
            assert may_train(transcript.ancestry, hosted_in_training=False)
            assert GEMINI not in transcript.ancestry
            fields = (transcript.message, transcript.reply, transcript.rejected or "", transcript.context or "",
                      transcript.evidence().text)
            assert not any(CRITIC_WORDS in field or '"scores"' in field for field in fields)
        for call in spark.teacher_calls():  # the teacher reads F and the message, never a verdict
            assert all(CRITIC_WORDS not in m["content"] and "scores" not in m["content"] for m in call["messages"])
            assert [m["role"] for m in call["messages"]] == ["system", "user"]
        data, _ = _dataset(tmp_path / teacher, list(run.transcripts), dropout=0.0)
        assert data.examples and "ancestry" not in data.dropped
        assert all(CRITIC_WORDS not in json.dumps(e.prompt) + e.completion for e in data.examples)
        assert all(CRITIC_WORDS not in json.dumps(p.prompt) + p.chosen + p.rejected for p in data.pairs)


def test_a_real_persons_candidates_never_reach_a_third_party_critic() -> None:
    gemini, spark = Gemini(scored(5, (6, 7, 8), lambda text: False)), Spark(_scenes())
    _, run = _run(spark, critic=_critic(gemini), third_party=lambda uid: uid.startswith("synthetic-"))
    assert gemini.prompts == [] and len(spark.critic_prompts()) == 16  # Vertex sees synthetic people only
    assert run.roles["critic"] == "spark" and run.critic_usd == 0.0


def test_the_lazy_teacher_is_asked_only_where_it_could_become_the_target() -> None:
    all_miss_odd = scored(9, (5, 6, 7), lambda text: text.startswith("GLIMMER") and call_no(text) // 3 % 2 == 1)
    always_spark, lazy_spark = Spark(_scenes(), all_miss_odd), Spark(_scenes(), all_miss_odd)
    _, always = _run(always_spark)
    lazy_practice, lazy = _run(lazy_spark, lazy_teacher=True)
    assert len(always_spark.teacher_calls()) == 16 and len(lazy_spark.teacher_calls()) == 8  # odd scenes only
    assert [(t.target, t.reply) for t in lazy.transcripts] == [(t.target, t.reply) for t in always.transcripts]
    assert lazy.teacher_targets == always.teacher_targets == 8
    assert len(lazy_spark.critic_prompts()) == 16 + 8  # a second verdict where the teacher came in
    assert lazy_practice.max_spark_calls == 1 + 2 * 16 + 16  # its worst: every scene asks the teacher and twice the critic
    # A pair with the teacher as the worst is lost where the teacher is never asked: the A/B counts it.
    assert lazy.pairs <= always.pairs


def test_glimmer_plays_the_person_on_the_base_model_at_the_persons_cap() -> None:
    spark, glimmer = Spark(_scenes()), PersonGlimmer(_scenes())
    practice, run = _run(spark, glimmer, glimmer_person=True)
    assert not any(c["messages"][-1]["content"].startswith("# scenes_v1") for c in spark.calls)
    [person] = glimmer.person
    assert person["adapter"] is None and person["max_tokens"] == VISIBLE_TOKENS["person"]
    assert person["seed"] is not None
    assert practice.max_spark_calls == 2 * 16
    assert len(run.transcripts) == 16 and run.roles["person"] == "glimmer" and run.spark["person"]["calls"] == 0
    assert all(t.ancestry == frozenset({SPARK_OPEN, GLIMMER}) for t in run.transcripts)
    _, again = _run(Spark(_scenes()), PersonGlimmer(_scenes()), glimmer_person=True)
    assert [t.message for t in again.transcripts] == [t.message for t in run.transcripts]
    assert SCENES_ASKED == 20


def test_the_night_prices_a_third_party_critic_into_practice(tmp_path: Path) -> None:
    from anjo.sleep.tests.night_fakes import step
    from anjo.sleep.tests.test_full_practice import _full_night

    gemini = Gemini(scored(4, (5, 6, 8), lambda text: False))
    result, _, _ = _full_night(tmp_path, Spark(_scenes(), scored(4, (5, 6, 8), lambda text: False)),
                               practice_kw={"critic": _critic(gemini), "third_party": lambda uid: True})
    done = step(result.log_path, "practice")
    assert done["roles"]["critic"] == GEMINI and done["critic_usd"] == pytest.approx(16 * (900 * 0.75 + 40 * 3.75) / 1e6)
    assert result.cost_parts["practice"] >= done["critic_usd"]
