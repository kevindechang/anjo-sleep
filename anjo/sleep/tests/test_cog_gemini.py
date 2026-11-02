"""The Gemini baseline (gemini.py), with fake clients only: no call leaves this process."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from anjo.sleep.mind.cog.gemini import (
    GeminiInquirer,
    GeminiPredictor,
    apply_platt,
    extract_json,
    fit_platt,
    parse_probabilities,
)
from anjo.sleep.mind.cog.run import LifeRecord, QueryRecord, run_life
from anjo.sleep.mind.cog.types import Event, LifeMeta, Query, Question
from anjo.sleep.tests.test_cog_run import FakeLife

QID = re.compile(r"^- (\S+) \(about Day", re.MULTILINE)


class FakeGemini:
    """Answers every query id in the prompt with ``p``, inside a fenced block; keeps the prompts."""

    def __init__(self, p: float = 0.7, drop: int = 0) -> None:
        self.p, self.drop, self.prompts = p, drop, []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        ids = QID.findall(prompt)[self.drop:]
        return "Sure, here you go:\n```json\n" + json.dumps({q: self.p for q in ids}) + "\n```"


class FakeChooser:
    def __init__(self, reply: str) -> None:
        self.reply, self.prompts = reply, []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def _first_prompt(life: FakeLife, *, stats: bool = False, day: int = 5) -> str:
    fake = FakeGemini()
    pred = GeminiPredictor(fake, cache_dir=None, model_label="fake", stats=stats)
    pred.begin(life.meta)
    for d in range(day + 1):
        pred.observe(d, life.events(d))
    pred.predict(life.queries(day))
    return fake.prompts[-1]


# -- reading replies --
def test_json_is_found_fenced_bare_or_in_prose() -> None:
    assert extract_json('```json\n{"a": 0.2}\n```') == {"a": 0.2}
    assert extract_json('{"a": 0.2}') == {"a": 0.2}
    assert extract_json('My guess: {"a": 0.2, "b": {"c": 1}} -- hope it helps {x}') == {"a": 0.2, "b": {"c": 1}}
    assert extract_json("no json here") is None and extract_json("[1, 2]") is None


def test_probabilities_are_clamped_and_missing_ids_counted() -> None:
    probs, missing, failed = parse_probabilities(
        '{"a": 0.0, "b": 1, "c": "70%", "d": 85, "e": "x", "f": true, "g": -0.3}',
        ["a", "b", "c", "d", "e", "f", "g", "h"])
    assert probs["a"] == 0.01 and probs["b"] == 0.99
    assert probs["c"] == pytest.approx(0.7) and probs["d"] == pytest.approx(0.85)
    assert probs["e"] == probs["f"] == probs["g"] == probs["h"] == 0.5
    assert missing == 4 and not failed
    probs, missing, failed = parse_probabilities("I cannot say.", ["a", "b"])
    assert probs == {"a": 0.5, "b": 0.5} and missing == 2 and failed


# -- the prompt --
def test_the_prompt_is_dated_text_never_tokens() -> None:
    life = FakeLife(3, n_days=10)
    prompt = _first_prompt(life)
    first = life.events(0)[0]
    assert f"Day 0 ({['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][life.meta.start_weekday]}): {first.text}" in prompt
    assert all(e.text in prompt for d in range(6) for e in life.events(d))
    assert "Day 6 (" not in prompt.split("Today is")[0]  # nothing after the asked day
    assert "day=" not in prompt and "('did'" not in prompt
    for q in life.queries(5):
        assert f"- {q.qid} (about Day {q.target_day}" in prompt and q.text in prompt
    assert "Today is Day 5" in prompt and "JSON object" in prompt and "counted for you" not in prompt


def test_one_call_per_day_and_records_carry_usage() -> None:
    fake = FakeGemini(p=0.7, drop=1)
    life = FakeLife(4, n_days=8)
    pred = GeminiPredictor(fake, cache_dir=None, model_label="fake")
    rec = run_life(life, pred, query_days={2, 5})
    assert len(fake.prompts) == 2 and {q.asked_day for q in rec.queries} == {2, 5}
    assert sorted({q.p for q in rec.queries}) == [0.5, 0.7]  # the first id of each day was dropped
    assert rec.usage["calls"] == 2 and rec.usage["missing_ids"] == 2 and rec.usage["queries"] == 12
    assert rec.usage["prompt_chars"] == sum(len(p) for p in fake.prompts)
    assert rec.usage["response_chars"] > 0 and rec.run == "gemini"


def test_the_cache_makes_reruns_free(tmp_path: Path) -> None:
    life = FakeLife(5, n_days=6)
    first = FakeGemini()
    rec1 = run_life(life, GeminiPredictor(first, cache_dir=tmp_path, model_label="gemini-x"))
    files = list((tmp_path / "gemini-x").glob("*.json"))
    assert len(files) == len(first.prompts) == 5
    again = FakeGemini(p=0.1)  # a cache hit never reaches it
    rec2 = run_life(life, GeminiPredictor(again, cache_dir=tmp_path, model_label="gemini-x"))
    assert again.prompts == [] and rec2.queries == rec1.queries
    assert rec2.usage["calls"] == 0 and rec2.usage["cache_hits"] == 5
    other = FakeGemini(p=0.1)
    run_life(life, GeminiPredictor(other, cache_dir=tmp_path, model_label="another-model"))
    assert len(other.prompts) == 5


def test_client_errors_propagate() -> None:
    def broken(prompt: str) -> str:
        raise RuntimeError("quota")

    with pytest.raises(RuntimeError, match="quota"):
        run_life(FakeLife(1, n_days=3), GeminiPredictor(broken, cache_dir=None, model_label="x"))


# -- the stats mode --
def test_stats_mode_adds_a_table_counted_up_to_the_asked_day() -> None:
    meta = LifeMeta("p", 30, 0, ("run", "gym"), (), ("weather",))
    fake = FakeGemini()
    pred = GeminiPredictor(fake, cache_dir=None, model_label="fake", stats=True)
    pred.begin(meta)
    script = {
        0: [Event(0, "did", "run", "Ran 5k.", ("did", "run")),
            Event(0, "plan", "gym", "Gym tomorrow.", ("plan", "gym"), value=1.0),
            Event(0, "message", "work", "Late email.", ("message", "late"))],
        1: [Event(1, "skipped", "gym", "No gym.", ("skipped", "gym")),
            Event(1, "mood", "self", "Low.", ("mood", "low"), value=2.0)],
        7: [Event(7, "did", "run", "Ran again.", ("did", "run"))],
        9: [Event(9, "did", "run", "A third run.", ("did", "run"))],
    }
    for d in range(9):
        pred.observe(d, script.get(d, []))
    pred.predict([Query("q1", 8, 9, "run", "Will they run on Tuesday?", ("query", "run"), "count")])
    prompt = fake.prompts[-1]
    assert "counted for you" in prompt and pred.name == "gemini_stats"
    run_row = next(line for line in prompt.splitlines() if line.startswith("run ") and "/" in line)
    assert run_row.split()[1:3] == ["2/0", "0/0"]  # Mondays: done twice; Tuesdays: nothing yet
    gym_row = next(line for line in prompt.splitlines() if line.startswith("gym ") and "/" in line)
    assert gym_row.split()[2] == "0/1"
    assert "Plans kept: 0 of 1 that came due." in prompt
    assert "Late-night messages: 0 in the last 3 days, 0 in the last 7." in prompt  # day 0 is 8 days back
    assert "A third run." not in prompt
    plain = _first_prompt(FakeLife(3, n_days=10))
    stats = _first_prompt(FakeLife(3, n_days=10), stats=True)
    assert stats != plain and stats.startswith(plain.split("\n\nToday is")[0])


# -- Platt recalibration --
def _records(ps: np.ndarray, ys: np.ndarray) -> list[LifeRecord]:
    qs = tuple(QueryRecord(qid=f"q{i}", subject="run", family="count", asked_day=0, target_day=1,
                           p=float(p), u=None, outcome=bool(y), oracle_p=0.5, state_p=0.5)
               for i, (p, y) in enumerate(zip(ps, ys)))
    return [LifeRecord(run="gemini", predictor="gemini", inquirer=None, seed=0, person_id="p", n_days=2,
                       start_weekday=0, topics=(), shift_days=(), shift_subjects=None, queries=qs, asks=())]


def test_platt_scaling_tempers_an_overconfident_model() -> None:
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.2, 0.8, size=4000)
    ys = rng.random(4000) < true_p
    stated = 1 / (1 + np.exp(-3 * np.log(true_p / (1 - true_p))))  # logits tripled
    a, b = fit_platt(_records(stated, ys))
    assert a == pytest.approx(1 / 3, abs=0.05) and abs(b) < 0.1
    assert apply_platt(0.9, (a, b)) < 0.9 and apply_platt(0.5, (1.0, 0.0)) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        fit_platt([])


def test_the_recalibrated_predictor_reuses_the_cache(tmp_path: Path) -> None:
    life = FakeLife(6, n_days=5)
    run_life(life, GeminiPredictor(FakeGemini(0.9), cache_dir=tmp_path, model_label="m"))
    later = FakeGemini(0.9)
    rec = run_life(life, GeminiPredictor(later, cache_dir=tmp_path, model_label="m", platt=(0.5, 0.0)))
    assert later.prompts == [] and rec.run == "gemini_platt"
    expected = 1 / (1 + math.exp(-0.5 * math.log(0.9 / 0.1)))
    assert all(q.p == pytest.approx(expected) for q in rec.queries)


# -- the inquirer --
def _questions() -> list[Question]:
    return [Question("w", 3, "work", "How is work?", ("ask", "work"), 0.05,
                     options=(("answer", "work", "calm"), ("answer", "work", "busy"))),
            Question("t", 3, "weather", "Nice weather?", ("ask", "weather"), 0.2)]


def test_gemini_inquirer_asks_what_the_reply_names() -> None:
    life = FakeLife(2, n_days=6)
    chooser = FakeChooser('```json\n{"ask": "w"}\n```')
    inq = GeminiInquirer(chooser, cache_dir=None, model_label="fake")
    inq.begin(life.meta)
    for d in range(4):
        inq.observe(d, life.events(d))
    upcoming = list(life.queries(3))
    assert inq.choose(3, _questions(), upcoming) == "w"
    prompt = chooser.prompts[-1]
    assert "- w (cost 0.05): How is work? Possible answers: answer work calm / answer work busy" in prompt
    assert "- t (cost 0.20): Nice weather?" in prompt
    assert all(q.qid in prompt for q in upcoming) and life.events(3)[0].text in prompt
    assert inq.choose(3, [], upcoming) is None and len(chooser.prompts) == 1


def test_gemini_inquirer_stays_silent_on_null_and_on_nonsense() -> None:
    for reply, key in (('{"ask": null}', "silent"), ('{"ask": "zzz"}', "invalid"), ("no idea", "invalid")):
        inq = GeminiInquirer(FakeChooser(reply), cache_dir=None, model_label="fake")
        inq.begin(LifeMeta("p", 5, 0, (), (), ()))
        assert inq.choose(0, _questions(), []) is None
        assert inq.usage[key] == 1


def test_gemini_inquirer_in_the_run_loop_counts_its_usage() -> None:
    life = FakeLife(8, n_days=5)
    reply = FakeChooser('{"ask": null}')
    rec = run_life(life, GeminiPredictor(FakeGemini(), cache_dir=None, model_label="m"),
                   GeminiInquirer(reply, cache_dir=None, model_label="m"))
    assert rec.asks == () and rec.inquirer == "gemini" and rec.usage["inquirer.calls"] == 5
    assert "Day 4" in reply.prompts[-1]
