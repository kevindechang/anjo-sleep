# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Day dreams and replay seeding (``ANJO_SLEEP_DAY_DREAMS``): the switch and its refusals, earlier
practice rows under tonight's refusals, her Glimmer's messages parsed from a cut answer, the
decontaminator and probe rules, the receipt's counts, and the rows' round trip into the job. No
network, no model: the fakes of the objective's own tests."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.night.day_dreams import (
    DAY_DREAMS,
    DREAM_FAILURES_MAX,
    DayDream,
    DreamDay,
    GlimmerDayDreams,
    NoDayDreams,
    messages_from,
)
from anjo.sleep.night.dreams import Terms
from anjo.sleep.night.meter import SpendMeter
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.tests.night_fakes import World, own
from anjo.sleep.tests.test_sleep_objective import (
    AT,
    GEMINI,
    NIGHT,
    Transcripts,
    _NoLeak,
    _Tokens,
    _data,
    _today,
    _transcript,
)
from anjo.sleep.train import colab_train, objective, sleep_job, sleep_loss, terms
from anjo.sleep.train.dataset import SeedRow
from anjo.sleep.train.lora import LoraCfg, training_config
from anjo.sleep.types import Probe

SCRIPT = Path(__file__).parents[3]


class _Answer:
    def __init__(self, text: str, finish: str = "stop") -> None:
        from anjo.sleep.model_api import ChatResult

        self.result = ChatResult(text=text, tool_calls=(), model=GLIMMER, input_tokens=10, output_tokens=5,
                                 finish_reason=finish)


class _Client:
    def __init__(self, answers: list[Any]) -> None:
        self.answers = answers
        self.calls: list[tuple[Any, list[dict[str, str]], int | None, float]] = []

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None,
                 max_tokens: int | None = None) -> Any:
        self.calls.append((adapter, messages, seed, temperature))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer.result


def _lines(n: int) -> str:
    return "\n".join(json.dumps({"message": f"Message {i}: what should I do about scene {i}?"}) for i in range(n))


def _day(text: str = "Notes: they edit at night.", ancestry: frozenset[str] | None = frozenset({GLIMMER})) -> DreamDay:
    return DreamDay(text=text, ancestry=ancestry)


def _dream(i: int, *, ancestry: frozenset[str] | None = frozenset({GLIMMER})) -> DayDream:
    return DayDream(id=f"d{i}", message=f"Message {i}", reply=f"Reply {i}", ancestry=ancestry)


def _replay(i: int, *, created_at: Any = AT, target: str = "glimmer") -> Any:
    return replace(_transcript(i), created_at=created_at, target=target)


def test_the_switch_off_sends_todays_payload_and_builds_todays_rows(tmp_path: Path) -> None:
    seed = {objective.OBJECTIVE_ENV: "sft_dpo_seed", objective.CAPACITY_ENV: "0"}
    off = objective.night_cfg(seed)
    assert off == objective.night_cfg({**seed, objective.DAY_DREAMS_ENV: "0"}) and not off.day_dreams
    assert objective.night_terms(seed) == Terms(seed=True) == objective.night_terms({**seed, objective.DAY_DREAMS_ENV: "0"})
    sent = training_config(off, 12, warm=True, revision="a" * 40, night=NIGHT)
    assert sent["terms"] == {"fold_rank": None, "dreams": False} and "day_dreams" not in sent["terms"]
    assert sleep_job.seed_plan(sleep_loss, sent, eot=11, pad=0, long_term_rank=0, begun=0.0).day_dreams is False

    made = [_transcript(0), _replay(1)]
    data = _data(made)
    data = replace(data, examples=(data.examples[0], replace(data.examples[1], new=False)))
    rows, dropped = terms.seed_rows(data, Transcripts(made), [_today(frozenset())], [],
                                    hosted_in_training=False)  # replay is off: earlier practice is unseen
    assert [row.id for row in rows] == ["t0"] and dropped["missing"] == 0
    written = json.loads(replace(data, seeds=rows).to_jsonl(terms=True).splitlines()[-1])
    assert "group" not in written  # a today's row serializes exactly as before


def test_the_switch_is_seeded_only_and_rides_with_capacity() -> None:
    seed = {objective.OBJECTIVE_ENV: "sft_dpo_seed", objective.CAPACITY_ENV: "0"}
    on = objective.night_cfg({**seed, objective.DAY_DREAMS_ENV: "1"})
    assert on.day_dreams and on.objective == "sft_dpo_seed"
    assert objective.night_terms({**seed, objective.DAY_DREAMS_ENV: "1"}) == Terms(seed=True, day_dreams=True)
    capacity = objective.night_cfg({**seed, objective.DAY_DREAMS_ENV: "1", objective.CAPACITY_ENV: "1"})
    assert (capacity.day_dreams, capacity.r, capacity.passes) == (True, 32, 3)
    for bad in ({objective.OBJECTIVE_ENV: "sft_dpo", objective.DAY_DREAMS_ENV: "1"},
                {**seed, objective.DAY_DREAMS_ENV: "yes"}):
        with pytest.raises(objective.ObjectiveError):
            objective.night_cfg(bad)
    with pytest.raises(ValueError):
        LoraCfg(objective="sft_dpo", day_dreams=True)


def _earlier(made: list[Any]) -> Any:
    """The dataset of ``made`` as earlier practice, plus one example with no transcript (``gone``)."""
    data = _data(made)
    gone = replace(data.examples[0], id="gone")  # nothing dates it: it is considered after the dated rows
    return replace(data, examples=tuple(replace(example, new=False) for example in (*data.examples, gone)))


def test_a_replay_refusal_past_the_cap_is_not_counted() -> None:
    dated = [_replay(i, created_at=AT - timedelta(minutes=i)) for i in range(terms.SEED_REPLAY_MAX)]
    rows, dropped = terms.seed_rows(_earlier(dated), Transcripts(dated), [], [], hosted_in_training=False, replay=True)
    assert len(rows) == terms.SEED_REPLAY_MAX and dropped["missing"] == 0  # the cap closed before it was read

    short = dated[:-1]
    rows, dropped = terms.seed_rows(_earlier(short), Transcripts(short), [], [], hosted_in_training=False, replay=True)
    assert len(rows) == terms.SEED_REPLAY_MAX - 1 and dropped["missing"] == 1  # the cap still had room


def test_replay_rows_are_newest_first_and_capped() -> None:
    made = [_transcript(0), _replay(1, created_at=AT - timedelta(days=1)),
            _replay(2, created_at=AT - timedelta(days=2))]
    data = _data(made)
    data = replace(data, examples=(data.examples[0], *(replace(e, new=False) for e in data.examples[1:])))
    rows, dropped = terms.seed_rows(data, Transcripts(made), [_today(frozenset())], [],
                                    hosted_in_training=False, replay=True)
    assert [row.id for row in rows] == ["t0", "t1", "t2"]
    assert all(row.group is None for row in rows)  # earlier practice stays an ordinary seeding row
    assert all(value == 0 for value in dropped.values())

    many = [_replay(i, created_at=AT - timedelta(minutes=i)) for i in range(30)]
    data = _data(many)
    data = replace(data, examples=tuple(replace(example, new=False) for example in data.examples))
    rows, _ = terms.seed_rows(data, Transcripts(many), [_today(frozenset())], [],
                              hosted_in_training=False, replay=True)
    assert len(rows) == terms.SEED_REPLAY_MAX == 24 and [row.id for row in rows] == [f"t{i}" for i in range(24)]


def test_a_replay_row_is_refused_for_every_reason_tonights_rows_are() -> None:
    made = [_replay(0)]
    data = replace(_data(made), examples=(replace(_data(made).examples[0], new=False),))

    teacher = [_replay(1, target="teacher")]
    rows, dropped = terms.seed_rows(replace(_data(teacher), examples=(replace(_data(teacher).examples[0], new=False),)),
                                    Transcripts(teacher), [], [], hosted_in_training=False, replay=True)
    assert rows == () and dropped["teacher_target"] == 1

    gone = replace(data.examples[0], id="gone")
    rows, dropped = terms.seed_rows(replace(data, examples=(gone,)), Transcripts(made), [], [],
                                    hosted_in_training=False, replay=True)
    assert rows == () and dropped["missing"] == 1

    bare = replace(_replay(2), context=None)
    rows, dropped = terms.seed_rows(replace(_data([bare]), examples=(replace(_data([bare]).examples[0], new=False),)),
                                    Transcripts([bare]), [], [], hosted_in_training=False, replay=True)
    assert rows == () and dropped["no_context"] == 1

    rows, dropped = terms.seed_rows(data, Transcripts(made), [_today(frozenset({GEMINI}))], [],
                                    hosted_in_training=False, replay=True)
    assert rows == () and dropped["uncleared"] == 1  # the day's item came from Gemini

    sealed = Probe("q1", "p1", "new_work",
                   "the late tomatoes need a sooner planting date and a warmer bed this year", None, None, ())
    held = replace(_replay(3), context=f"Notes: {sealed.prompt}.")
    rows, dropped = terms.seed_rows(replace(_data([held]), examples=(replace(_data([held]).examples[0], new=False),)),
                                    Transcripts([held]), [], [sealed], hosted_in_training=False, replay=True)
    assert rows == () and dropped["probe"] == 1

    rows, dropped = terms.seed_rows(data, Transcripts(made), [], [], hosted_in_training=False, replay=True,
                                    max_seq_tokens=1)
    assert rows == () and dropped["truncated"] == 1


def test_the_day_dreams_call_returns_the_messages_and_her_replies() -> None:
    from anjo.sleep.night.day_dreams import DREAM_MIX

    assert DAY_DREAMS == 16 and DREAM_MIX == (8, 4, 4) and sum(DREAM_MIX) == DAY_DREAMS
    client = _Client([_Answer(_lines(DAY_DREAMS)), *(_Answer(f"Her reply {i}") for i in range(DAY_DREAMS))])
    source = GlimmerDayDreams(client, seed=5)
    made = source.dream("p1", NIGHT, None, _day())
    assert source.available and len(made) == DAY_DREAMS and len(client.calls) == DAY_DREAMS + 1
    assert all(call[1][0] == {"role": "system", "content": _day().text} for call in client.calls)
    assert all(call[3] == 1.0 for call in client.calls)  # her own sampling, as MIND's dreams
    assert made[0].message == "Message 0: what should I do about scene 0?" and made[0].reply == "Her reply 0"
    assert made[0].ancestry == frozenset({GLIMMER}) and NoDayDreams().dream("p1", NIGHT, None, _day()) == ()
    anchored = _Client([_Answer('{"message": "One more?"}'), _Answer("Yes.")])
    made = GlimmerDayDreams(anchored, seed=5).dream("p1", NIGHT, None,
                                                    _day(ancestry=frozenset({"person"})))
    assert made[0].ancestry == frozenset({"person", GLIMMER})  # her text joins the day's ancestry


def test_a_cut_unreadable_or_failed_answer_yields_fewer_dreams_not_a_failed_night() -> None:
    cut = _Answer(_lines(3) + "\n{\"message\": \"a line the cap cut", finish="length")
    client = _Client([cut, _Answer("one"), _Answer("two"), _Answer("three")])
    assert len(GlimmerDayDreams(client, seed=1).dream("p1", NIGHT, None, _day())) == 3

    assert GlimmerDayDreams(_Client([_Answer("not json at all")]), seed=1).dream("p1", NIGHT, None, _day()) == ()
    assert messages_from('```json\n{"message": "a"}\n{"message": "b"}\n```') == ["a", "b"]
    from anjo.sleep.model_api import ModelApiError

    failed = _Client([ModelApiError("the serving call failed")])
    assert GlimmerDayDreams(failed, seed=1).dream("p1", NIGHT, None, _day()) == ()
    assert GlimmerDayDreams(_Client([]), seed=1).dream("p1", NIGHT, None, _day("")) == ()  # no day, no call


def test_a_failed_reply_costs_one_dream_and_three_in_a_row_end_the_asking() -> None:
    from anjo.sleep.model_api import ModelApiError

    client = _Client([_Answer(_lines(4)), _Answer("one"), ModelApiError("the reply call failed"),
                      _Answer("three"), _Answer("four")])
    made = GlimmerDayDreams(client, seed=1).dream("p1", NIGHT, None, _day())
    assert [dream.reply for dream in made] == ["one", "three", "four"]  # one failure skipped one dream, not the rest

    client = _Client([_Answer(_lines(5)), *[ModelApiError("down") for _ in range(DREAM_FAILURES_MAX)],
                      _Answer("never")])
    made = GlimmerDayDreams(client, seed=1).dream("p1", NIGHT, None, _day())
    assert made == () and len(client.calls) == 1 + DREAM_FAILURES_MAX  # three in a row stop the asking


def test_day_dream_rows_obey_the_decontaminator_the_probe_rule_and_the_length() -> None:
    day = _day("Notes: they edit at night. What they told you today:\n- They switched the short to 16mm today.")
    rows, dropped = terms.day_dream_rows([_dream(0), _dream(1)], day, [], _NoLeak(), hosted_in_training=False)
    assert len(rows) == 2 and rows[0].context == day.text and rows[0].group == "dream"
    assert all(value == 0 for value in dropped.values())

    rows, dropped = terms.day_dream_rows([_dream(0, ancestry=frozenset({GEMINI}))], day, [], _NoLeak(),
                                         hosted_in_training=False)
    assert rows == () and dropped["uncleared"] == 1

    class _Leak:
        def check(self, texts: Any, sealed: Any) -> tuple[bool, ...]:
            return (True,) * len(texts)

    rows, dropped = terms.day_dream_rows([_dream(0)], day, [], _Leak(), hosted_in_training=False)
    assert rows == () and dropped["probe"] == 1

    sealed = Probe("q1", "p1", "new_work", "the late tomatoes need a sooner planting date and a warmer bed", None, None, ())
    rows, dropped = terms.day_dream_rows([_dream(0)], _day(f"Notes: {sealed.prompt}"), [sealed], _NoLeak(),
                                         hosted_in_training=False)
    assert rows == () and dropped["probe"] == 1

    rows, dropped = terms.day_dream_rows([_dream(0)], day, [], _NoLeak(), hosted_in_training=False, max_seq_tokens=1)
    assert rows == () and dropped["truncated"] == 1
    rows, dropped = terms.day_dream_rows([_dream(0)], _day(""), [], _NoLeak(), hosted_in_training=False)
    assert rows == () and dropped["no_context"] == 1


class _DaySource:
    available = True

    def __init__(self, made: tuple[DayDream, ...]) -> None:
        self.made = made
        self.days: list[DreamDay] = []

    def dream(self, user_id: str, night: Any, adapter: Any, day: DreamDay) -> tuple[DayDream, ...]:
        self.days.append(day)
        return self.made


def test_the_night_counts_replay_rows_and_day_dreams_in_its_receipt(tmp_path: Path) -> None:
    new, old = _transcript(0), replace(_replay(1), created_at=AT - timedelta(days=1))
    world = World(tmp_path)
    world.evidence("p1").add([own("p1", NIGHT, 0)])
    world.transcripts("p1").add([new, old])
    data = _data([new, old])
    data = replace(data, examples=(data.examples[0], replace(data.examples[1], new=False)))
    source = _DaySource((_dream(0), _dream(1, ancestry=frozenset({GEMINI}))))
    deps = world.deps(terms=Terms(seed=True, day_dreams=True), day_dreams=source)
    spend = SpendMeter(spark_in_per_m=0.2, spark_out_per_m=0.6)

    data, receipt = terms.attach_night_terms(deps, "p1", NIGHT, world.evidence("p1"), [], data, spend)

    assert receipt["seeds"] == 2 and receipt["seed_replay"] == 1  # the new row and the replay one
    assert receipt["day_dreams"]["generated"] == 2 and receipt["day_dreams"]["kept"] == 1
    assert receipt["day_dreams"]["dropped"]["uncleared"] == 1 and receipt["day_dreams"]["seconds"] == 60.0
    assert source.days[0].text.startswith("What you know about them, from your notes:")  # notes + today
    assert "sorting line" in source.days[0].text and receipt["anchors"] == terms.ANCHORS
    assert [(row.id, row.group) for row in data.seeds] == [("t0", None), ("t1", None), ("d0", "dream")]
    assert spend.parts()["practice"] > 0  # her serving GPU's minute, as MIND's dreams


def test_a_failed_day_dream_source_leaves_the_night_with_zero_dreams(tmp_path: Path) -> None:
    made = [_transcript(0)]
    world = World(tmp_path)
    world.evidence("p1").add([own("p1", NIGHT, 0)])
    world.transcripts("p1").add(made)
    deps = world.deps(terms=Terms(seed=True, day_dreams=True), day_dreams=_DaySource(()))
    data, receipt = terms.attach_night_terms(deps, "p1", NIGHT, world.evidence("p1"), [], _data(made),
                                             SpendMeter(spark_in_per_m=0.2, spark_out_per_m=0.6))
    assert receipt["day_dreams"]["generated"] == receipt["day_dreams"]["kept"] == 0
    assert [row.id for row in data.seeds] == ["t0"]  # the night still has its seeding row


def test_an_uncleared_day_asks_nothing_of_her_glimmer(tmp_path: Path) -> None:
    made = [_transcript(0)]
    world = World(tmp_path)
    world.evidence("p1").add([replace(own("p1", NIGHT, 0), ancestry=frozenset({GEMINI}))])
    world.transcripts("p1").add(made)
    source = _DaySource((_dream(0),))
    deps = world.deps(terms=Terms(seed=True, day_dreams=True), day_dreams=source)
    spend = SpendMeter(spark_in_per_m=0.2, spark_out_per_m=0.6)

    _, receipt = terms.attach_night_terms(deps, "p1", NIGHT, world.evidence("p1"), [], _data(made), spend)

    assert source.days == []  # no day reached her Glimmer: nothing was asked
    assert receipt["day_dreams"]["skipped"] == "uncleared" and receipt["day_dreams"]["generated"] == 0
    assert receipt["day_dreams"]["kept"] == 0 and spend.parts()["practice"] == 0.0  # and nothing was spent


def test_a_day_holding_a_sealed_probe_asks_nothing_of_her_glimmer(tmp_path: Path) -> None:
    sealed = Probe("q1", "p1", "new_work",
                   "the late tomatoes need a sooner planting date and a warmer bed this year", None, None, ())
    made = [replace(_transcript(0), context=f"Notes: {sealed.prompt}.")]
    world = World(tmp_path)
    world.evidence("p1").add([own("p1", NIGHT, 0)])
    world.transcripts("p1").add(made)
    source = _DaySource((_dream(0),))
    deps = world.deps(terms=Terms(seed=True, day_dreams=True), day_dreams=source)
    spend = SpendMeter(spark_in_per_m=0.2, spark_out_per_m=0.6)

    _, receipt = terms.attach_night_terms(deps, "p1", NIGHT, world.evidence("p1"), [sealed], _data(made), spend)

    assert source.days == [] and receipt["day_dreams"]["skipped"] == "probe"
    assert receipt["day_dreams"]["generated"] == 0 and spend.parts()["practice"] == 0.0


def test_the_day_is_the_newest_cleared_transcript_with_a_context() -> None:
    newest = replace(_transcript(0), context="Notes: a Gemini summary.", ancestry=frozenset({GEMINI}))
    empty = replace(_transcript(1), created_at=AT - timedelta(minutes=1), context="")
    kept = replace(_transcript(2), created_at=AT - timedelta(minutes=2), context="Notes: they edit at night.")
    made = [newest, empty, kept]

    notes, ancestry = terms.day_notes(_data(made), Transcripts(made), hosted_in_training=False)

    assert notes == kept.context and ancestry == kept.ancestry  # the uncleared and the empty one are not the day
    day = terms.day_dream_context(_data(made), Transcripts(made), [_today(frozenset())], hosted_in_training=False)
    assert "they edit at night" in day.text and "Gemini summary" not in day.text


def test_a_dream_group_round_trips_through_the_job_payload() -> None:
    sent = training_config(LoraCfg(objective="sft_dpo_seed", day_dreams=True), 12, warm=True, revision="a" * 40,
                           night=NIGHT)
    assert sent["terms"] == {"fold_rank": None, "dreams": False, "day_dreams": True}
    loaded = colab_train.load_config(json.dumps(sent))
    assert loaded["terms"]["day_dreams"] is True
    assert sleep_job.seed_plan(sleep_loss, loaded, eot=11, pad=0, long_term_rank=0, begun=0.0).day_dreams is True
    with pytest.raises(ValueError):
        colab_train.load_config(json.dumps({**sent, "terms": {**sent["terms"], "day_dreams": "yes"}}))

    row = SeedRow(id="d0", context="What they told you today:\n- The short is 16mm.", message="What now?",
                  reply="Shoot on the 16mm.", tokens=20, group="dream")
    written = [json.loads(line) for line in replace(_data([]), seeds=(row,)).to_jsonl(terms=True).splitlines()]
    assert written[0]["group"] == "dream"
    by_kind = sleep_job.term_rows(colab_train, written)
    cfg = {"max_seq_length": 8192, "terms": {"fold_rank": None, "dreams": False, "day_dreams": True}}
    template = {"reasoning_strength": "low", "current_date": NIGHT.isoformat()}
    batch, _ = sleep_job.seed_batch(colab_train, sleep_loss, _Tokens(), [], [], by_kind, cfg, template)
    assert batch.seeds[0].group == "dream"
    plain = replace(row, group=None)
    written = [json.loads(line) for line in replace(_data([]), seeds=(plain,)).to_jsonl(terms=True).splitlines()]
    assert "group" not in written[0]
    batch, _ = sleep_job.seed_batch(colab_train, sleep_loss, _Tokens(), [], [],
                                    sleep_job.term_rows(colab_train, written), cfg, template)
    assert batch.seeds[0].group is None


def test_dream_split_and_the_gain_and_band_rule() -> None:
    from anjo.sleep.night import day_dreams

    assert day_dreams.DAY_DREAM_GROUP == sleep_loss.DREAM_KIND == "dream"  # the job's trial marker
    seeds = [sleep_loss.SeedSeq((1, 2), (7, 8, 1, 2), (3, 11), group="dream" if i < 14 else None)
             for i in range(16)]
    held, groups = sleep_loss.dream_split(seeds, 3)
    assert len(held) == sleep_loss.DREAM_HELD == 4 and all(seeds[i].group == "dream" for i in held)
    assert [len(group) for group in groups] == [4, 4, 2] and len({i for group in groups for i in group} | set(held)) == 14
    assert sleep_loss.dream_split(seeds, 3) == (held, groups)  # fixed by the plan seed
    assert sleep_loss.dream_split(seeds, 4) != (held, groups)
    assert sleep_loss.dream_kept(1.0, 0.98, 0.5, 0.5)  # 2% below the untouched tau
    assert not sleep_loss.dream_kept(1.0, 0.995, 0.5, 0.5)  # not the required 1%
    assert not sleep_loss.dream_kept(1.0, 0.98, 0.5, 0.494)  # anchors beyond the band
    assert sleep_loss.dream_kept(1.0, 0.98, 0.5, 0.495)  # exactly on it


def test_day_dreams_and_the_trial_code_never_import_the_probes_or_reports() -> None:
    import ast

    for name in ("anjo/sleep/night/day_dreams.py", "anjo/sleep/train/sleep_loss.py"):
        tree = ast.parse((SCRIPT / name).read_text(encoding="utf-8"))
        modules = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        modules += [node.module for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module is not None]
        assert not [m for m in modules if m.startswith(("anjo.sleep.probes", "anjo.sleep.reports"))], name
