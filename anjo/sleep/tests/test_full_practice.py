# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Full practice (Task 7): 16 to 20 cue-free scenes a night with the same F; four candidates per
turn (three from her personal Glimmer, one Spark Open teacher, all on one shared prefix); a Spark
Open critic marks the best and the worst, Glimmer wins ties, and the teacher can be best only when
all three Glimmer candidates miss a constraint stated in F. The best reply trains on pre-practice C
(or nothing under dropout), and the (best, worst) pairs are kept, decontaminated on both sides."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.evidence.store import JsonlEvidenceStore
from anjo.sleep.model_api import ChatResult
from anjo.sleep.practice.full import CANDIDATE_TEMPERATURE, SCENES_ASKED, FullPractice
from anjo.sleep.practice.transcripts import JsonlTranscriptStore, PracticeTranscript
from anjo.sleep.probes.decontam import OverlapDecontaminator
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.tests.sleep_fakes import WONDER, ev, probe
from anjo.sleep.train.dataset import build_night_dataset
from anjo.sleep.types import CheckpointRef, Probe

NIGHT = date(2026, 10, 28)
AT = datetime(2026, 10, 28, 23, 30, tzinfo=timezone.utc)
ADAPTER = CheckpointRef("p-abc-n2", "p1", date(2026, 10, 27), "modal-volume://anjo-sleep-adapters/p-abc-n2", None)
_LABEL = re.compile(r"^\[([A-Z])\]$")

Critic = Callable[[dict[str, str]], dict[str, Any]]


def _result(text: str, finish: str = "stop") -> ChatResult:
    return ChatResult(text=text, tool_calls=(), model="m", input_tokens=10, output_tokens=5, finish_reason=finish)


def _scenes(n: int = SCENES_ASKED, cued: int = 0) -> list[str]:
    clean = [f"I have a new sorting brief, number {i}. Where would you start?" for i in range(n - cued)]
    return clean + [f"As we discussed, redo plan {i}." for i in range(cued)]


def candidates_of(prompt: str) -> dict[str, str]:
    """The critic prompt's candidates by label."""
    found: dict[str, list[str]] = {}
    label: str | None = None
    inside = prompt.split("<candidates>", 1)[1].split("</candidates>", 1)[0]
    for line in inside.splitlines():
        match = _LABEL.match(line.strip())
        if match:
            label = match.group(1)
            found[label] = []
        elif label is not None:
            found[label].append(line)
    return {key: "\n".join(lines).strip() for key, lines in found.items()}


def call_no(text: str) -> int:
    """A Glimmer candidate's call number: three calls a scene, so ``n % 3`` is its place in the scene
    and ``n // 3`` the scene."""
    found = re.search(r"#(\d+)", text)
    assert found is not None
    return int(found.group(1))


def scored(teacher: int, glimmer: Sequence[int], misses: Callable[[str], bool]) -> Critic:
    """A critic that scores the teacher and each Glimmer candidate (by its place) as given."""

    def critic(cands: dict[str, str]) -> dict[str, Any]:
        scores = {}
        for label, text in cands.items():
            scores[label] = teacher if text.startswith("TEACHER") else glimmer[call_no(text) % 3]
        return {"scores": scores, "misses_constraint": [label for label, text in cands.items() if misses(text)]}

    return critic


class Spark:
    """Spark Open: writes the scenes, plays the teacher (with F as its system prompt) and the critic."""

    def __init__(self, scenes: list[str], critic: Critic | None = None, *, raw_critic: str | None = None) -> None:
        self.scenes = scenes
        self.critic = critic or scored(5, (6, 7, 8), lambda text: False)
        self.raw_critic = raw_critic
        self.calls: list[dict[str, Any]] = []

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None) -> ChatResult:
        assert model == SPARK_OPEN and tools is None
        self.calls.append({"messages": messages, "temperature": temperature})
        content = messages[-1]["content"]
        if content.startswith("# scenes_v1"):
            return _result(json.dumps({"scenes": self.scenes}))
        if content.startswith("# critic_v1"):
            return _result(self.raw_critic if self.raw_critic is not None else json.dumps(self.critic(candidates_of(content))))
        assert messages[0]["role"] == "system"
        return _result(f"TEACHER reply to: {content[:30]}")

    def teacher_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["messages"][0]["role"] == "system"]

    def critic_prompts(self) -> list[str]:
        return [c["messages"][-1]["content"] for c in self.calls if c["messages"][-1]["content"].startswith("# critic_v1")]


class Glimmer:
    def __init__(self, finish: Callable[[int], str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.finish = finish or (lambda n: "stop")

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None) -> ChatResult:
        self.calls.append({"adapter": adapter, "messages": messages, "seed": seed, "temperature": temperature})
        n = len(self.calls) - 1
        return _result(f"GLIMMER #{n} on: {messages[-1]['content'][:24]}", self.finish(n))


def _evidence() -> list[Any]:
    return [
        ev("own", "I run a recycling startup. Never use jargon with me.", minutes=1),
        ev("hw", "Homework found optical sorters miss black plastic.", kind="homework",
           ancestry=frozenset({WONDER}), minutes=3),
    ]


def _practice(spark: Spark, glimmer: Glimmer, **kw: Any) -> FullPractice:
    return FullPractice(chat=spark, client=glimmer, adapter=lambda uid: ADAPTER, seed=7,
                        snapshot=lambda uid, message: (f"C-SNAPSHOT for {message[:12]}", frozenset({SPARK_OPEN})), **kw)


def _run(spark: Spark, glimmer: Glimmer | None = None, **kw: Any) -> Any:
    return _practice(spark, glimmer or Glimmer(), **kw).run("p1", NIGHT, _evidence(), None, at=AT)


def test_full_practice_runs_16_to_20_scenes() -> None:
    spark = Spark(_scenes())
    run = _run(spark)
    assert len(run.transcripts) == 16  # the default, from the 20 asked for
    assert f"Write {SCENES_ASKED} messages" in spark.calls[0]["messages"][-1]["content"]
    assert len(_run(Spark(_scenes()), scenes=20).transcripts) == 20
    cued = _run(Spark(_scenes(cued=3)), scenes=20)
    assert len(cued.transcripts) == 17 and cued.dropped_cued == 3
    for wrong in (8, 15, 21):
        with pytest.raises(ValueError):
            _practice(Spark(_scenes()), Glimmer(), scenes=wrong)
    assert _practice(Spark(_scenes()), Glimmer()).max_spark_calls == 1 + 2 * 16


def test_practice_candidates_use_full_context_and_train_on_c(tmp_path: Path) -> None:
    spark, glimmer = Spark(_scenes()), Glimmer()
    run = _run(spark, glimmer)
    teacher = spark.teacher_calls()
    assert len(glimmer.calls) == 3 * 16 and len(teacher) == 16
    for scene in range(16):
        prefixes = [c["messages"] for c in glimmer.calls[3 * scene: 3 * scene + 3]] + [teacher[scene]["messages"]]
        assert all(p == prefixes[0] for p in prefixes)  # one shared prefix: F, then the message
        system = prefixes[0][0]["content"]
        assert prefixes[0][0]["role"] == "system" and "Never use jargon" in system and "optical sorters" not in system
    assert all(c["adapter"] == ADAPTER for c in glimmer.calls)  # her personal Glimmer
    assert all(c["temperature"] == CANDIDATE_TEMPERATURE for c in glimmer.calls)
    seeds = [c["seed"] for c in glimmer.calls]
    assert len(set(seeds)) == len(seeds)
    again = Glimmer()
    _run(Spark(_scenes()), again)
    assert [c["seed"] for c in again.calls] == seeds
    for transcript in run.transcripts:
        assert transcript.context is not None and transcript.context.startswith("C-SNAPSHOT")
        assert "Never use jargon" not in transcript.context  # the input is C, never F
    data, _ = _dataset(tmp_path, list(run.transcripts), dropout=0.0)
    for example in data.examples:
        assert example.prompt[0][0] == "system" and example.prompt[0][1].startswith("C-SNAPSHOT")
    dropped, _ = _dataset(tmp_path / "d", list(run.transcripts), dropout=0.9)
    assert any(len(e.prompt) == 1 and e.prompt[0][0] == "user" for e in dropped.examples)  # nothing, under dropout


def test_teacher_picked_only_when_all_glimmer_candidates_miss_a_constraint() -> None:
    def glimmer_number(text: str) -> int:
        return call_no(text) % 3 if text.startswith("GLIMMER") else -1

    two_miss = _run(Spark(_scenes(), scored(9, (5, 6, 7), lambda t: glimmer_number(t) in (0, 1))))
    assert all(t.target == "glimmer" and t.reply.startswith("GLIMMER") for t in two_miss.transcripts)
    assert all(call_no(t.reply) % 3 == 2 for t in two_miss.transcripts)  # the best Glimmer, 7
    all_miss = _run(Spark(_scenes(), scored(9, (5, 6, 7), lambda t: t.startswith("GLIMMER"))))
    assert all(t.target == "teacher" and t.reply.startswith("TEACHER") for t in all_miss.transcripts)
    assert all_miss.teacher_targets == 16
    teacher_misses_too = _run(Spark(_scenes(), scored(9, (5, 6, 7), lambda t: True)))
    assert all(t.target == "glimmer" for t in teacher_misses_too.transcripts)
    tie = _run(Spark(_scenes(), scored(7, (5, 6, 7), lambda t: t.startswith("GLIMMER"))))
    assert all(t.target == "glimmer" for t in tie.transcripts)  # Glimmer wins ties
    spark = Spark(_scenes(), scored(9, (5, 6, 7), lambda t: t.startswith("GLIMMER")))
    switched_off = _run(spark, teacher=False)
    assert all(t.target == "glimmer" for t in switched_off.transcripts) and spark.teacher_calls() == []


def test_the_critic_sees_candidates_blind_and_in_a_seeded_order() -> None:
    spark = Spark(_scenes())
    _run(spark)
    prompts = spark.critic_prompts()
    assert len(prompts) == 16
    assert all("teacher" not in p.lower() and "spark" not in p.lower() and "glimmer" not in p.lower()
               for p in [p.replace("GLIMMER #", "").replace("TEACHER reply", "") for p in prompts])
    positions = {list(candidates_of(p).values()).index(next(v for v in candidates_of(p).values() if v.startswith("TEACHER")))
                 for p in prompts}
    assert len(positions) > 1  # the teacher isn't always in one place
    again = Spark(_scenes())
    _run(again)
    assert again.critic_prompts() == prompts


def test_best_and_worst_are_kept_as_a_pair() -> None:
    run = _run(Spark(_scenes(), scored(4, (5, 6, 8), lambda t: False)))
    for transcript in run.transcripts:
        assert transcript.reply.startswith("GLIMMER") and call_no(transcript.reply) % 3 == 2
        assert transcript.rejected is not None and transcript.rejected.startswith("TEACHER")
        assert transcript.rejected_from == "teacher"
    assert run.pairs == 16
    flat = _run(Spark(_scenes(), scored(6, (6, 6, 6), lambda t: False)))
    assert flat.pairs == 0 and all(t.rejected is None for t in flat.transcripts)  # no preference, no pair


def test_an_unreadable_critic_keeps_the_first_glimmer_reply_and_no_pair() -> None:
    run = _run(Spark(_scenes(), raw_critic="I like them all"))
    assert run.critic_unreadable == 16 and run.pairs == 0
    assert all(t.target == "glimmer" and t.rejected is None for t in run.transcripts)


def test_a_cut_candidate_never_becomes_a_target() -> None:
    every_cut = _run(Spark(_scenes()), Glimmer(finish=lambda n: "length"))
    assert every_cut.transcripts == () and every_cut.dropped_cut == 16  # the teacher alone never trains
    one_cut = _run(Spark(_scenes(), scored(9, (5, 6, 7), lambda t: t.startswith("GLIMMER"))),
                   Glimmer(finish=lambda n: "length" if n % 3 == 0 else "stop"))
    assert all(t.target == "glimmer" for t in one_cut.transcripts)  # two Glimmers left: no teacher


def test_homework_never_reaches_full_practice() -> None:
    with pytest.raises(ValueError):
        _practice(Spark(_scenes()), Glimmer()).run("p1", NIGHT, _evidence(), object(), at=AT)


def test_transcripts_keep_cleared_ancestry() -> None:
    run = _run(Spark(_scenes()))
    assert all(t.ancestry == frozenset({SPARK_OPEN, GLIMMER}) for t in run.transcripts)


# ── the dataset: provenance, decontamination and pairs ──────────────────────


def _dataset(tmp_path: Path, transcripts: list[PracticeTranscript], *, sealed: list[Probe] | None = None,
             decontam: Any = None, dropout: float = 0.6) -> Any:
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = JsonlEvidenceStore(tmp_path / "evidence.jsonl", "p1", codec=PLAIN, guard=NO_GUARD)
    log = JsonlTranscriptStore(tmp_path / "practice.jsonl", "p1", codec=PLAIN, guard=NO_GUARD)
    store.add([t.evidence() for t in transcripts])
    log.add(transcripts)
    return build_night_dataset(store, None, NIGHT, sealed or [], decontam or OverlapDecontaminator(), 1_000_000,
                               transcripts=log, seed=7, dropout=dropout)


def test_teacher_candidate_marked_in_dataset(tmp_path: Path) -> None:
    even_scenes_miss = scored(9, (5, 6, 7), lambda t: t.startswith("GLIMMER") and (call_no(t) // 3) % 2 == 0)
    run = _run(Spark(_scenes(), even_scenes_miss))
    teacher = [t for t in run.transcripts if t.target == "teacher"]
    assert teacher and len(teacher) < len(run.transcripts)
    stored = JsonlTranscriptStore(tmp_path / "log.jsonl", "p1", codec=PLAIN, guard=NO_GUARD)
    stored.add(list(run.transcripts))
    assert stored.get(teacher[0].id) == teacher[0]  # provenance survives the log
    data, _ = _dataset(tmp_path / "data", list(run.transcripts))
    rows = [json.loads(line) for line in data.to_jsonl().splitlines()]
    by_id = {row["id"]: row for row in rows}
    for transcript in run.transcripts:
        assert by_id[transcript.id]["provenance"] == transcript.target
    assert {e.id for e in data.examples if e.teacher} == {t.id for t in teacher}


class Spy:
    def __init__(self) -> None:
        self.inner = OverlapDecontaminator()
        self.texts: list[str] = []

    def check(self, texts: Sequence[str], sealed: Sequence[Probe]) -> tuple[bool, ...]:
        self.texts.extend(texts)
        return self.inner.check(texts, sealed)


def test_practice_output_goes_through_decontam(tmp_path: Path) -> None:
    run = _run(Spark(_scenes(), scored(4, (5, 6, 8), lambda t: False)))
    spy = Spy()
    data, _ = _dataset(tmp_path, list(run.transcripts), decontam=spy, sealed=[probe()])
    for transcript in run.transcripts:  # every text that could train, both sides of each pair
        assert any(transcript.reply in text for text in spy.texts)
        assert transcript.rejected is not None and any(transcript.rejected in text for text in spy.texts)
    assert len(data.examples) == 16 and len(data.pairs) == 16


def _pair(i: int, reply: str, rejected: str) -> PracticeTranscript:
    return PracticeTranscript(
        id=f"pr-{i:02d}", person_id="p1", night=NIGHT, scene=i, message=f"A new festival brief, number {i}. Where to start?",
        reply=reply, context="## Their brief\nShe edits her own films.", f_ids=("e1",),
        ancestry=frozenset({SPARK_OPEN, GLIMMER}), created_at=AT.replace(minute=i), target="glimmer",
        rejected=rejected, rejected_from="glimmer",
    )


def test_dpo_pairs_decontaminated_on_both_sides(tmp_path: Path) -> None:
    sealed = probe(prompt="How should I cut the ferry scene so the last line lands on her face?")
    leak = "Cut the ferry scene so the last line lands on her face, then hold."
    transcripts = [
        _pair(1, "Start with the strongest image and cut toward it.", "Use jargon-heavy pacing notes."),
        _pair(2, leak, "Start anywhere."),  # the chosen side carries the probe
        _pair(3, "Begin with sound, then picture.", leak),  # the rejected side carries it
    ]
    data, probe_dropped = _dataset(tmp_path, transcripts, sealed=[sealed], dropout=0.0)
    assert probe_dropped == 1 and {e.id for e in data.examples} == {"pr-01", "pr-03"}
    assert [p.id for p in data.pairs] == ["pr-01"]
    assert data.pairs_dropped["probe"] == 2
    [pair] = data.pairs
    assert pair.chosen == transcripts[0].reply and pair.rejected == transcripts[0].rejected
    assert pair.prompt == next(e.prompt for e in data.examples if e.id == "pr-01")
    rows = [json.loads(line) for line in data.to_jsonl(pairs=True).splitlines()]
    pair_rows = [row for row in rows if "chosen" in row]
    assert len(pair_rows) == 1 and pair_rows[0]["rejected"] == [{"role": "assistant", "content": transcripts[0].rejected}]
    assert not any("chosen" in json.loads(line) for line in data.to_jsonl().splitlines())


def test_only_tonights_pairs_are_kept(tmp_path: Path) -> None:
    old = PracticeTranscript(
        id="pr-old", person_id="p1", night=date(2026, 10, 20), scene=0, message="An old brief. Where to start?",
        reply="Start small.", context=None, f_ids=(), ancestry=frozenset({SPARK_OPEN, GLIMMER}),
        created_at=datetime(2026, 10, 20, 23, 0, tzinfo=timezone.utc), rejected="Start big.", rejected_from="glimmer",
    )
    store = JsonlEvidenceStore(tmp_path / "evidence.jsonl", "p1", codec=PLAIN, guard=NO_GUARD)
    log = JsonlTranscriptStore(tmp_path / "practice.jsonl", "p1", codec=PLAIN, guard=NO_GUARD)
    tonight = _pair(1, "Start with the strongest image.", "Start anywhere.")
    store.add([old.evidence(), tonight.evidence()])
    log.add([old, tonight])
    data, _ = build_night_dataset(store, date(2026, 10, 25), NIGHT, [], OverlapDecontaminator(), 1_000_000,
                                  transcripts=log, seed=7, dropout=0.0)
    assert {e.id for e in data.examples} == {"pr-old", "pr-01"} and [p.id for p in data.pairs] == ["pr-01"]


# ── in the night ──────────────────────────────────────────────────────────────


def test_the_night_holds_full_practices_worst_case_and_logs_its_counts(tmp_path: Path) -> None:
    from anjo.sleep.budget import Ledger
    from anjo.sleep.context.brief_store import TrainingBriefs
    from anjo.sleep.context.builder import RecallSettings
    from anjo.sleep.context.recall_bm25 import Bm25Recall
    from anjo.sleep.context.training import TrainingContext
    from anjo.sleep.evidence.jsonl import read_records
    from anjo.sleep.night.holds import spark_worst_usd
    from anjo.sleep.night.run import run_night_batch_with
    from anjo.sleep.tests.night_fakes import NIGHT as NIGHT_DATE
    from anjo.sleep.tests.night_fakes import World, step

    world = World(tmp_path)
    world.plan("p1", NIGHT_DATE)
    world.spark.scenes = SCENES_ASKED
    critic = Spark(_scenes(), scored(4, (5, 6, 8), lambda t: False))

    def spark(model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None) -> ChatResult:
        content = messages[-1]["content"]
        if content.startswith(("# brief_v1", "# scenes_v1")):
            return world.spark(model, messages, tools, timeout_s, temperature=temperature)
        return critic(model, messages, tools, timeout_s, temperature=temperature)

    def practice(chat: Any, night: date) -> FullPractice:
        recall = Bm25Recall(world.evidence, until=night)
        context = TrainingContext(TrainingBriefs(world.briefs), recall, RecallSettings(k=6, budget_tokens=1200))
        return FullPractice(chat=chat, client=Glimmer(), snapshot=context.snapshot, adapter=world.registry.last_promoted,
                            seed=7)

    ledger = Ledger(tmp_path / "budget.jsonl")
    deps = world.deps(spark_chat=spark, practice=practice, ledger=ledger)
    result = run_night_batch_with(deps, {"p1": NIGHT_DATE})["p1"]
    assert result.status == "promoted"
    done = step(result.log_path, "practice")
    assert done["scenes"] == 16 and done["pairs"] == 16 and done["teacher_targets"] == 0
    assert done["critic_unreadable"] == 0
    [held] = [e for e in read_records(ledger.path, PLAIN) if e["op"] == "reserve" and e["job"].startswith("night:practice:")]
    assert held["amount"] == pytest.approx(spark_worst_usd(deps.prices, 1 + 2 * 16))
