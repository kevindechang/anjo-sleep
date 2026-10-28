# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The night view (``demo/night_view.py``): her latest night as one loop, dream → rehearse → learn →
check → wake, built from fake stories with no network. A nap stops after the dream; a stage that
didn't run says why; a failure names its step, never its error."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.demo import words
from anjo.sleep.demo.heldout import HeldOutText
from anjo.sleep.demo.night_view import LINKS, story_view
from anjo.sleep.demo.page import WEB_DIR
from anjo.sleep.demo.panel import MindSources, mind_view, night_record
from anjo.sleep.night import story as st
from anjo.sleep.tests.test_night_story import FULL, PROMOTED, TOLD

NIGHT = date(2026, 10, 30)


def _by(view: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {stage["stage"]: stage for stage in view["stages"]}


def _numbers(stage: dict[str, Any]) -> dict[str, Any]:
    return {n["name"]: n.get("value", ("not_run", n.get("not_run"))) for n in stage["numbers"]}


def test_the_loop_runs_left_to_right_with_one_key_number_and_its_evidence() -> None:
    view = story_view([(3, FULL), (2, TOLD["napped, MIND on"])])
    assert (view["state"], view["night"], view["date"], view["status"], view["napped"]) == (
        "ok", 3, FULL.night.isoformat(), "promoted", False)
    assert view["sleep"] == {"kind": "deep", "pressure": 0.62, "threshold": 0.5, "reason": "pressure"}
    assert [s["stage"] for s in view["stages"]] == ["dream", "rehearse", "learn", "check", "wake"]
    stages = _by(view)
    assert {name: (s["state"], s["key"]) for name, s in stages.items()} == {
        "dream": ("ran", 2), "rehearse": ("ran", 16), "learn": ("ran", 8), "check": ("ran", 1), "wake": ("ran", 28)}
    assert {name: s["link"] for name, s in stages.items()} == LINKS
    assert LINKS == {"dream": "mind-questions", "rehearse": "dailies", "learn": None, "check": "mind-gates",
                     "wake": "mind-chart"}
    assert stages["dream"]["cites"] == ["e-0a1b", "e-2c3d"]  # the events the dream rests on, ids only
    assert _numbers(stages["dream"]) == {"questions": 2, "ignitions": 1, "threads": 1, "boundaries": 1}
    assert _numbers(stages["rehearse"]) == {"flagged": 6, "corrected": 2, "seeded": 1}
    assert _numbers(stages["learn"]) == {"surprise": 1.1, "surprise_rows": 8, "channel_loss": 0.42, "stamp": "proj-3",
                                         "seeding_kl": 0.031, "fold": 8, "core_adaptation": 0.2}
    check = stages["check"]
    assert (check["of"], check["promoted"]) == (1, True)  # a switched-off gate is never counted as passed
    assert check["gates"] == [{"gate": "general", "kind": "pass", "value": 0.0, "threshold": 2.0},
                              {"gate": "safety", "kind": "skip", "value": 0.0, "threshold": 0.0}]
    assert _numbers(stages["wake"]) == {"no_verdict": 1, "self_model_predicted": 0.3, "self_model_actual": 0.25}


def test_what_didnt_run_is_greyed_with_its_reason() -> None:
    stages = _by(story_view([(1, TOLD["promoted"])]))
    assert stages["wake"]["state"] == "not_run" and stages["wake"]["reason"] == st.SCORED_IN_MORNING
    assert stages["wake"]["key"] is None
    numbers = _numbers(stages["learn"])
    assert numbers["channel_loss"] == ("not_run", st.CHANNEL_NOT_YET) and numbers["fold"] == ("not_run", st.FOLD_NOT_YET)
    quiet = _by(story_view([(1, TOLD["nothing_new"])]))
    assert quiet["dream"]["state"] == "not_run" and quiet["dream"]["reason"] == st.MIND_OFF
    assert all((quiet[name]["state"], quiet[name]["reason"]) == ("not_run", st.NOTHING_NEW)
               for name in ("rehearse", "learn", "check"))


def test_a_nap_stops_after_the_dream_learning_tomorrow() -> None:
    for name in ("napped, MIND on", "napped, MIND off"):
        view = story_view([(2, TOLD[name])])
        assert view["napped"] is True and view["sleep"]["kind"] == "nap"
        after = view["stages"][1:]
        assert [(s["state"], s["reason"]) for s in after] == [("not_run", st.LEARNING_TOMORROW)] * 4


def test_a_failure_names_its_step_never_its_error() -> None:
    broke = replace(TOLD["failed at gates"], check=st.Failed("gates", "ServingError: /v1/completions on host-7"))
    view = story_view([(4, broke)])
    stages = _by(view)
    assert (stages["check"]["state"], stages["check"]["step"]) == ("failed", "gates")
    assert stages["wake"] == {**stages["wake"], "state": "not_run", "reason": "the night stopped at gates"}
    morning = replace(FULL, wake=st.Wake(st.Failed("morning", "TooManyNoVerdicts: 3 of 3"), st.NotRun("x")))
    assert (_by(story_view([(1, morning)]))["wake"]["state"], _by(story_view([(1, morning)]))["wake"]["step"]) == (
        "failed", "morning")
    shown = json.dumps([view, story_view([(1, morning)])])
    assert "ServingError" not in shown and "host-7" not in shown and "TooManyNoVerdicts" not in shown


def test_no_night_none_and_an_untold_story_unavailable() -> None:
    assert story_view(None) == {"state": "unavailable"}
    assert story_view([]) == {"state": "none"}
    assert story_view([(5, None)]) == {"state": "unavailable", "night": 5}


# ── the mind panel carries it ────────────────────────────────────────────────


def _summary(status: str) -> dict[str, Any]:
    return {"night": NIGHT.isoformat(), "status": status, "gates": {}, "choices": []}


def test_the_panel_tells_each_night_from_its_story_file_or_else_its_log() -> None:
    from_log = night_record(_summary("promoted"), PROMOTED)
    assert from_log.story == st.tell(NIGHT, "promoted", PROMOTED)
    from_file = night_record(_summary("promoted"), PROMOTED, story=replace(FULL, night=NIGHT).record())
    assert from_file.story == replace(FULL, night=NIGHT)
    unreadable = night_record(_summary("promoted"), PROMOTED, story={"version": 9})
    assert unreadable.story is None and unreadable.status == "promoted"  # only the story is unavailable
    view = mind_view(MindSources(nights=(from_log,)), HeldOutText([]))
    assert view["story"]["state"] == "ok" and view["story"]["night"] == 1
    assert mind_view(MindSources(), HeldOutText([]))["story"] == {"state": "unavailable"}


# ── the page's script ────────────────────────────────────────────────────────


def test_night_js_reads_only_words_in_the_copy_and_draws_with_no_html() -> None:
    script = (WEB_DIR / "night.js").read_text()
    read = set(re.findall(r"copy\.([a-zA-Z0-9_]+)", script))
    assert read and read <= set(words.COPY), sorted(read - set(words.COPY))
    for stage in st.STAGES:
        assert {f"story_stage_{stage}", f"story_line_{stage}"} <= set(words.COPY), stage
    for name in ("questions", "ignitions", "threads", "boundaries", "flagged", "corrected", "seeded", "surprise",
                 "surprise_rows", "channel_loss", "stamp", "seeding_kl", "fold", "core_adaptation", "no_verdict",
                 "self_model_predicted", "self_model_actual"):
        assert f"story_number_{name}" in words.COPY, name
    assert "innerHTML" not in script and "eval(" not in script and "localStorage" not in script
    assert "http://" not in script and "https://" not in script


NODE = shutil.which("node")
SHARED = frozenset({"tag", "num", "label"})  # the page's small utilities, styled once in demo.css


def _styled_elsewhere() -> set[str]:
    """Every class another stylesheet styles: a stage drawn with one inherits its look (``.skip`` is
    the skip link, which once pulled a switched-off gate out of its card)."""
    return {name for sheet in WEB_DIR.glob("*.css") if sheet.name != "mind.css"
            for name in re.findall(r"\.([a-zA-Z][\w-]*)", sheet.read_text())}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_night_js_draws_a_nap_and_a_full_night_in_a_fake_page(tmp_path: Path) -> None:
    payloads = {"full": story_view([(3, FULL)]), "nap": story_view([(2, TOLD["napped, MIND on"])])}
    harness = tmp_path / "harness.js"
    harness.write_text((Path(__file__).parent / "night_js_harness.js").read_text())
    for name, payload in payloads.items():
        handed = {"questions": {"state": "none", "items": []}, "story": payload}  # the /mind payload, as mind.js hands it over
        run = subprocess.run([str(NODE), str(harness), str(WEB_DIR / "night.js"), json.dumps(dict(words.COPY)),
                              json.dumps(handed)], capture_output=True, text=True, timeout=60, check=False)
        assert run.returncode == 0, run.stderr
        drawn = json.loads(run.stdout)
        assert [s["stage"] for s in drawn["stages"]] == list(st.STAGES), name
        assert set(drawn["classes"]) & _styled_elsewhere() <= SHARED, name
        if name == "nap":
            assert [s["state"] for s in drawn["stages"]] == ["not_run"] * 5  # MIND on: the dream deferred too
            assert words.COPY["story_nap"] in drawn["text"]
        else:
            assert [s["state"] for s in drawn["stages"]] == ["ran"] * 5
            assert "#mind-questions" in drawn["links"] and "#mind-gates" in drawn["links"]
