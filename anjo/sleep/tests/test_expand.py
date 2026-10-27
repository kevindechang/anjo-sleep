# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Life expansion (Task 12a): a one-line life becomes a ``PersonFile`` on Spark Open, through the
``ModelClient`` port. The output is validated strictly and refused, never repaired."""
from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.lives.draw import Life, Pair, load_pairs
from anjo.sleep.model_api import ChatResult
from anjo.sleep.personas.model import (
    MIN_HELD_OUT_PREFERENCES,
    MIN_NEW_WORK,
    MIN_PLANS,
    PLANNED_NIGHTS,
    NEW_WORK_FIRST_DAY,
    to_visible_dict,
    validate_person_file,
)
from anjo.sleep.sim.expand import PROMPT_PATH, ExpandError, expand_life

PAIRS = load_pairs()
PAIR = PAIRS[0]
FIRST, SECOND = PAIR.lives
LAST_DAY = PLANNED_NIGHTS + 7


def person_json(life: Life, pair: Pair) -> dict[str, Any]:
    """A valid file for ``life``: its registered line and plans, the pair's shared taste items with
    this life's labels, and the rest written by the 'model'."""
    visible_prefs = [{"id": f"{life.id}-v{i}", "text": f"{life.field} habit {i} of {life.id}", "domain": "craft"}
                     for i in range(60)]
    held_prefs = [{"id": f"{life.id}-h{i}", "text": f"hidden {life.field} taste {i} of {life.id}", "domain": "craft"}
                  for i in range(15)]
    taste = [{"id": t.id, "pref_id": held_prefs[i]["id"], "prompt": t.prompt, "options": list(t.options),
              "pick": t.pick[life.id]} for i, t in enumerate(pair.taste_ab)]
    taste += [{"id": f"{life.id}-t{i}", "pref_id": held_prefs[i]["id"], "prompt": f"Which suits {life.id} {i}?",
               "options": [f"left {i}", f"right {i}"], "pick": i % 2} for i in range(len(taste), 15)]
    title = f"Project of {life.id}"
    work = [{"id": f"{life.id}-w{i}", "day": NEW_WORK_FIRST_DAY + i % (LAST_DAY - 1),
             "parent_project": title, "text": f"New job {i} arrives for {life.id}"} for i in range(26)]
    return {
        "visible": {
            "person_id": life.id, "name": f"Name of {life.id}", "age": 40, "field": life.field,
            "field_domains": ["craft", "shops"], "line": life.line,
            "project": {"title": title, "logline": "A logline.", "constraints": ["c1", "c2", "c3"]},
            "preferences": visible_prefs,
            "tone_requests": [{"id": f"tone{i}", "text": f"Tone {i}"} for i in range(3)],
            "stated_views": [{"id": "view0", "text": "A view."}],
            "plans": [{"day": p.day, "text": p.text} for p in life.plans],
        },
        "held_out": {"preferences": held_prefs, "taste_ab": taste, "new_work": work},
    }


class Client:
    """A ``ModelClient`` that answers with ``make(life_id)`` for whichever life the prompt names."""

    def __init__(self, make: Callable[[Life, Pair], dict[str, Any]] = person_json, raw: str | None = None) -> None:
        self.make, self.raw, self.calls = make, raw, []  # type: ignore[var-annotated]

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None) -> ChatResult:
        self.calls.append({"adapter": adapter, "messages": messages, "temperature": temperature, "seed": seed})
        prompt = messages[-1]["content"]
        life = next(lf for pair in PAIRS for lf in pair.lives if f"person_id: {lf.id}\n" in prompt)
        pair = next(p for p in PAIRS if life in p.lives)
        text = self.raw if self.raw is not None else json.dumps(self.make(life, pair))
        return ChatResult(text=text, tool_calls=(), model=SPARK_OPEN, input_tokens=10, output_tokens=10, finish_reason="stop")


def test_expand_life_splits_visible_and_held_out() -> None:
    client = Client()
    pf = expand_life(FIRST, 7, client=client)
    validate_person_file(pf)
    assert pf.visible.person_id == FIRST.id and pf.visible.line == FIRST.line
    assert len(pf.held_out.new_work) >= MIN_NEW_WORK
    assert all(NEW_WORK_FIRST_DAY <= w.day <= LAST_DAY and w.parent_project == pf.visible.project.title
               for w in pf.held_out.new_work)
    assert len(pf.held_out.preferences) >= MIN_HELD_OUT_PREFERENCES
    assert len(pf.held_out.taste_ab) >= len(pf.held_out.preferences)
    assert len(pf.visible.plans) >= MIN_PLANS
    visible = json.dumps(to_visible_dict(pf.visible)).lower()
    for pref in pf.held_out.preferences:  # none of the held-out part is visible
        assert pref.text.lower() not in visible
    for item in pf.held_out.taste_ab:
        assert item.prompt.lower() not in visible and all(o.lower() not in visible for o in item.options)
    for work in pf.held_out.new_work:
        assert work.text.lower() not in visible
    call = client.calls[0]
    assert call["adapter"] is None and call["temperature"] == 0.0 and call["seed"] == 7


def test_the_prompt_is_the_registered_file_with_the_life_filled_in() -> None:
    client = Client()
    expand_life(FIRST, 1, client=client)
    prompt = client.calls[0]["messages"][-1]["content"]
    assert prompt.startswith("# expand_v1") and FIRST.line in prompt and "{{" not in prompt
    assert PROMPT_PATH.read_text(encoding="utf-8").startswith("# expand_v1")


def test_pair_lives_share_taste_items_with_opposite_labels() -> None:
    client = Client()
    one, two = (expand_life(life, 3, client=client) for life in (FIRST, SECOND))
    shared = {t.id for t in PAIR.taste_ab}
    a = {t.id: t for t in one.held_out.taste_ab if t.id in shared}
    b = {t.id: t for t in two.held_out.taste_ab if t.id in shared}
    assert set(a) == set(b) == shared and shared
    for item_id in shared:
        assert a[item_id].options == b[item_id].options and a[item_id].prompt == b[item_id].prompt
        assert a[item_id].pick != b[item_id].pick  # opposite labels
    only_own = {t.id for t in one.held_out.taste_ab} - shared
    assert only_own.isdisjoint({t.id for t in two.held_out.taste_ab})


def _broken(change: Callable[[dict[str, Any]], None]) -> Callable[[Life, Pair], dict[str, Any]]:
    def make(life: Life, pair: Pair) -> dict[str, Any]:
        data = copy.deepcopy(person_json(life, pair))
        change(data)
        return data
    return make


def _drop_new_work(d: dict[str, Any]) -> None:
    del d["held_out"]["new_work"][MIN_NEW_WORK - 1:]


def _late_arrival(d: dict[str, Any]) -> None:
    d["held_out"]["new_work"][0]["day"] = LAST_DAY + 1


def _early_arrival(d: dict[str, Any]) -> None:
    d["held_out"]["new_work"][0]["day"] = 1


def _same_day(d: dict[str, Any]) -> None:
    for w in d["held_out"]["new_work"]:
        w["day"] = 5


def _flipped_shared(d: dict[str, Any]) -> None:
    t = d["held_out"]["taste_ab"][0]
    t["pick"] = 1 - t["pick"]


def _missing_shared(d: dict[str, Any]) -> None:
    d["held_out"]["taste_ab"][0]["prompt"] = "A different question entirely"


def _leaked_preference(d: dict[str, Any]) -> None:
    d["visible"]["preferences"][0]["text"] = d["held_out"]["preferences"][0]["text"]


def _leaked_work(d: dict[str, Any]) -> None:
    d["visible"]["stated_views"][0]["text"] = d["held_out"]["new_work"][0]["text"]


def _leaked_taste(d: dict[str, Any]) -> None:
    d["visible"]["stated_views"][0]["text"] = "I always say: " + d["held_out"]["taste_ab"][5]["prompt"]


def _few_plans(d: dict[str, Any]) -> None:
    del d["visible"]["plans"][2:]


def _other_line(d: dict[str, Any]) -> None:
    d["visible"]["line"] = "Somebody else entirely."


def _wrong_id(d: dict[str, Any]) -> None:
    d["visible"]["person_id"] = "someone-else"


def _lost_plan(d: dict[str, Any]) -> None:
    d["visible"]["plans"][0]["text"] = "A plan the registered life never had."


def _few_held_prefs(d: dict[str, Any]) -> None:
    del d["held_out"]["preferences"][14:]


def _wrong_project(d: dict[str, Any]) -> None:
    d["held_out"]["new_work"][0]["parent_project"] = "Another project"


@pytest.mark.parametrize("change", [
    _drop_new_work, _late_arrival, _early_arrival, _same_day, _flipped_shared, _missing_shared,
    _leaked_preference, _leaked_work, _leaked_taste, _few_plans, _other_line, _wrong_id, _lost_plan,
    _few_held_prefs, _wrong_project,
])
def test_a_file_that_breaks_a_rule_is_refused_not_repaired(change: Callable[[dict[str, Any]], None]) -> None:
    with pytest.raises(ExpandError):
        expand_life(FIRST, 1, client=Client(_broken(change)))


@pytest.mark.parametrize("raw", ["", "not json", "[1, 2]", '{"visible": 3, "held_out": 4}', '{"visible": {}}',
                                 "```json\n{}\n```"])
def test_output_that_is_not_a_person_file_is_refused(raw: str) -> None:
    with pytest.raises(ExpandError):
        expand_life(FIRST, 1, client=Client(raw=raw))


def test_a_life_outside_every_pair_is_refused() -> None:
    stray = Life("X1", "PX", "ceramics", "A potter.", "taste", FIRST.plans)
    with pytest.raises(ExpandError, match="pair"):
        expand_life(stray, 1, client=Client())
