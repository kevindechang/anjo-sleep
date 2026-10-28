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
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.demo import words
from anjo.sleep.demo.heldout import HeldOutText
from anjo.sleep.demo.night_view import LINKS, story_view
from anjo.sleep.demo.page import WEB_DIR
from anjo.sleep.demo.panel import MindSources, mind_view, night_record
from anjo.sleep.night import story as st
from anjo.sleep.tests.test_night_story import CORE_DREAM, FULL, PROMOTED, TOLD, _napped, _with

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
    assert _numbers(stages["dream"]) == {"questions": 2, "ignitions": 1, "threads": 1, "boundaries": 1,
                                         "dreams": 9, "dream_candidates": 12, "dream_keep_rate": 0.75}
    assert _numbers(stages["rehearse"]) == {"flagged": 6, "corrected": 2, "seeded": 4, "seeded_threads": 1}
    assert _numbers(stages["learn"]) == {
        "surprise": True, "surprise_rows": 8, "channel_loss": 0.42, "stamp": "9f86d081884c", "seed_rows": 10,
        "seeding_kl_before": 0.05, "seeding_kl_after": 0.031, "imitation_rows": 3, "imitation_reward": 0.6, "fold": 8,
        "protect": "anchors", "protect_minutes": 0.4, "core_adaptation": 0.2}
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
    assert numbers["channel_loss"] == ("not_run", st.CHANNEL_OFF) and numbers["fold"] == ("not_run", st.TERMS_OFF)
    assert {numbers[name] for name in ("seeding", "imitation", "protect")} == {("not_run", st.TERMS_OFF)}
    assert numbers["surprise"] == ("not_run", st.NO_SURPRISE)
    assert _numbers(stages["dream"])["dreams"] == ("not_run", st.TERMS_OFF)
    assert _numbers(stages["rehearse"])["seeded"] == ("not_run", st.CORE_OFF)
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


def test_a_reused_practice_says_whose_it_was_never_that_it_ran() -> None:
    earlier = NIGHT - timedelta(days=1)
    rehearse = _by(story_view([(2, replace(FULL, rehearse=st.Reused(earlier, 12)))]))["rehearse"]
    assert rehearse == {**rehearse, "state": "reused", "key": 12, "reused_from": earlier.isoformat(), "numbers": []}


def test_a_nap_the_core_dreamt_in_shows_its_dream_then_stops() -> None:
    view = story_view([(2, st.tell(NIGHT, "napped", _with(_napped(mind=True), "notice", CORE_DREAM)))])
    dream = view["stages"][0]
    assert (dream["state"], dream["key"], dream["cites"]) == ("ran", 1, ["e-0a1b", "e-2c3d"])  # its one ignition
    assert [(s["state"], s["reason"]) for s in view["stages"][1:]] == [("not_run", st.LEARNING_TOMORROW)] * 4


def _claims(story: st.Story) -> bool:
    claimed: bool = story_view([(1, story)])["rehearsed_the_dream"]
    return claimed


def test_the_lede_claims_the_whole_loop_only_when_it_ran() -> None:
    assert _claims(FULL) is True  # the dream picked threads, practice was seeded on them, and she woke better
    dreamt, rehearsed = FULL.dream, FULL.rehearse
    assert isinstance(dreamt, st.Dream) and isinstance(rehearsed, st.Rehearse)
    for story in (TOLD["promoted"], TOLD["napped, MIND on"], TOLD["failed at gates"],
                  replace(FULL, dream=replace(dreamt, threads=st.NotRun(st.NO_CURRICULUM))),
                  replace(FULL, rehearse=replace(rehearsed, seeded=st.NotRun(st.FELL_BACK))),
                  replace(FULL, rehearse=st.Reused(NIGHT, 16)), replace(FULL, check=st.Check((), promoted=False))):
        assert _claims(story) is False
    assert "rehearsed_the_dream" not in story_view([]) and "rehearsed_the_dream" not in story_view(None)


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
    older = night_record(_summary("promoted"), PROMOTED, story={**FULL.record(), "version": 1})
    assert older.story == st.tell(NIGHT, "promoted", PROMOTED)  # an older schema's story: told again from the log
    broken = [r if r["step"] != "dataset" else {"step": "dataset", "status": "ok"} for r in PROMOTED]
    untellable = night_record(_summary("promoted"), broken, story={"version": 9})  # nor its log: no dataset count
    assert untellable.story is None and untellable.status == "promoted"  # only the story is unavailable
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
    for name in ("questions", "ignitions", "threads", "boundaries", "dreams", "dream_candidates", "dream_keep_rate",
                 "flagged", "corrected", "seeded", "seeded_threads", "surprise", "surprise_rows", "channel_loss", "stamp",
                 "seeding", "seed_rows", "seeding_kl_before", "seeding_kl_after", "imitation", "imitation_rows",
                 "imitation_reward", "fold", "protect", "protect_minutes", "core_adaptation", "no_verdict",
                 "self_model", "self_model_predicted", "self_model_actual"):
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
    payloads = {"full": story_view([(3, FULL)]), "nap": story_view([(2, TOLD["napped, MIND on"])]),
                "reused": story_view([(2, replace(FULL, rehearse=st.Reused(NIGHT - timedelta(days=1), 12)))]),
                "none": story_view([])}
    harness = tmp_path / "harness.js"
    harness.write_text((Path(__file__).parent / "night_js_harness.js").read_text())
    for name, payload in payloads.items():
        handed = {"questions": {"state": "none", "items": []}, "story": payload}  # the /mind payload, as mind.js hands it over
        run = subprocess.run([str(NODE), str(harness), str(WEB_DIR / "night.js"), json.dumps(dict(words.COPY)),
                              json.dumps(handed)], capture_output=True, text=True, timeout=60, check=False)
        assert run.returncode == 0, run.stderr
        drawn = json.loads(run.stdout)
        assert set(drawn["classes"]) & _styled_elsewhere() <= SHARED, name
        lede = {"full": "story_lede", "nap": "story_lede_plain", "reused": "story_lede_plain"}.get(name)
        assert drawn["lede"] == ({"text": words.COPY[lede], "hidden": False} if lede else {"text": "", "hidden": True}), name
        if name == "none":
            assert drawn["stages"] == [] and words.COPY["story_none"] in drawn["text"]
            continue
        assert [s["stage"] for s in drawn["stages"]] == list(st.STAGES), name
        if name == "nap":
            assert [s["state"] for s in drawn["stages"]] == ["not_run"] * 5  # MIND on: the dream deferred too
            assert words.COPY["story_nap"] in drawn["text"]
        elif name == "reused":
            assert [s["state"] for s in drawn["stages"]] == ["ran", "reused", "ran", "ran", "ran"]
            assert "12" in drawn["text"] and (NIGHT - timedelta(days=1)).isoformat() in drawn["text"]
        else:
            assert [s["state"] for s in drawn["stages"]] == ["ran"] * 5
            assert "#mind-questions" in drawn["links"] and "#mind-gates" in drawn["links"]


def test_a_kept_brief_says_so_and_never_sends_its_error() -> None:
    kept = story_view([(3, replace(FULL, brief=st.Kept("2026-10-28", "BriefError: the brief ran to its cap")))])
    assert kept["brief_kept"] == "2026-10-28" and "BriefError" not in json.dumps(kept)
    assert story_view([(3, replace(FULL, brief=st.Kept(st.EMPTY, "BriefError: x")))])["brief_kept"] == st.EMPTY
    assert story_view([(3, FULL)])["brief_kept"] is None


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_night_js_says_when_the_brief_was_kept(tmp_path: Path) -> None:
    harness = tmp_path / "harness.js"
    harness.write_text((Path(__file__).parent / "night_js_harness.js").read_text())
    for kept, line in (("2026-10-28", words.COPY["story_brief_kept"].replace("{night}", "2026-10-28")),
                       (st.EMPTY, words.COPY["story_brief_kept_empty"]), (None, None)):
        story = FULL if kept is None else replace(FULL, brief=st.Kept(kept, "BriefError: x"))
        handed = {"questions": {"state": "none", "items": []}, "story": story_view([(3, story)])}
        run = subprocess.run([str(NODE), str(harness), str(WEB_DIR / "night.js"), json.dumps(dict(words.COPY)),
                              json.dumps(handed)], capture_output=True, text=True, timeout=60, check=False)
        assert run.returncode == 0, run.stderr
        drawn = json.loads(run.stdout)
        assert [s["state"] for s in drawn["stages"]] == ["ran"] * 5  # the loop ran on the brief it kept
        if line is None:
            assert "loop-brief" not in drawn["classes"]
        else:
            assert line in drawn["text"] and "loop-brief" in drawn["classes"] and "BriefError" not in drawn["text"]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_night_js_says_why_a_number_is_missing_once_in_plain_words(tmp_path: Path) -> None:
    harness = tmp_path / "harness.js"
    harness.write_text((Path(__file__).parent / "night_js_harness.js").read_text())
    handed = {"questions": {"state": "none", "items": []}, "story": story_view([(1, TOLD["promoted"])])}
    run = subprocess.run([str(NODE), str(harness), str(WEB_DIR / "night.js"), json.dumps(dict(words.COPY)),
                          json.dumps(handed)], capture_output=True, text=True, timeout=60, check=False)
    assert run.returncode == 0, run.stderr
    text = json.loads(run.stdout)["text"]
    assert "not yet" not in text  # a record that's off or missing says why, never that it's coming
    assert text.count(st.TERMS_OFF) == 2  # once in the dream, once for the four terms it groups in learn
    assert text.count(st.CORE_OFF) == 2 and text.count(st.CHANNEL_OFF) == 1  # the dream's three, then rehearse
    assert st.ADAPTATION_NOT_YET in text and st.SELF_MODEL_NOT_YET in text  # no record says either: plain lines


def test_the_page_claims_nothing_before_the_story_says_what_ran() -> None:
    page = (WEB_DIR / "index.html").read_text()
    assert re.search(r'<p id="night-loop-lede" class="night-lede" hidden></p>', page)  # night.js fills it from the story
    assert "{{story_lede}}" not in page
