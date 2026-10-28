# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False
"""The hero persona (a film student who also founds an AI startup): safe text, and the default person."""
import json
import re
from pathlib import Path

import pytest

from anjo.sleep.app import demo_live
from anjo.sleep.app.demo_live import AnjoFloor
from anjo.sleep.demo import words
from anjo.sleep.night import daily
from anjo.sleep.personas import model as pm
from anjo.sleep.sim import accelerated

SLEEP = Path(pm.__file__).parents[1]
HERO, HERO_DONOR = "film_founder", "film_founder_donor"


def _strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _texts(name):
    return _strings(json.loads((SLEEP / "personas" / f"{name}.json").read_text()))


@pytest.mark.parametrize("name", [HERO, HERO_DONOR])
def test_hero_persona_has_no_crisis_content(name):
    from anjo.core.crisis import NO_CRISIS, crisis_tier, is_crisis, is_soft_crisis

    floor = AnjoFloor()
    texts = _texts(name) + [s.prompt for s in words.SCENES]
    assert len(texts) > 300
    for text in texts:
        assert not is_crisis(text) and not is_soft_crisis(text), text
        assert crisis_tier(text) == NO_CRISIS, text
        assert floor.fixed_reply(text) is None, text
    blob = " ".join(texts).lower()
    assert not re.findall(r"suicid|self-harm|kill myself|end it all|hopeless|worthless|overdose", blob)


def test_hero_is_an_adult_with_the_asked_for_shape():
    pf = pm.load_person_file(SLEEP / "personas" / f"{HERO}.json")
    v = pf.visible
    assert 18 <= v.age <= 25 and v.project.title == "Cutlist"
    blob = " ".join(_texts(HERO)).lower()
    for trait in ("cofounder", "runway", "deadline", "alone", "family", "feature"):
        assert trait in blob
    assert v.name not in {"Kevin"} and "kevin" not in blob


def test_hero_is_the_default_person():
    assert demo_live.DEMO_PERSONA == HERO
    assert accelerated.FAKE_PERSONA == HERO
    assert daily.DEFAULT_PERSONA == HERO
    assert (SLEEP / "personas" / f"{HERO}.json").is_file()
    # the page's starter scenes are the hero's: every one is in the first person, about his work
    assert [s.id for s in words.SCENES] == ["midnight-deck", "rough-cut", "nobody-to-tell"]
    assert "Teo" in words.COPY["headline"] and "Mara" not in json.dumps(dict(words.COPY))
    example = json.loads((SLEEP / "examples" / "synthetic" / "persona.json").read_text())
    assert example["name"] == "Teo Halvorsen"


def test_the_default_persona_runs_the_daily_command_without_naming_one(monkeypatch):
    monkeypatch.delenv(daily.USERS_ENV, raising=False)
    with pytest.raises(SystemExit):  # no account for the hero: it stops, naming him
        daily.main(["--phase", "day"])
