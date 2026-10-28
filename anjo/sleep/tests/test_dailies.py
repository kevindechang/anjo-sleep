# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Dailies: the live page's morning card, its engine. One night's card from what the app read (what
he took from the night, one line he'd now say differently, what the night cost), and the before and
after line pre-generated per checkpoint pair. The routes and the app's reads are tested in
``tests/app/test_demo_dailies.py``; the night phase that runs the pre-generation in ``test_daily.py``."""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import Ledger
from anjo.sleep.demo import daily_lines, dailies, words
from anjo.sleep.demo.daily_lines import DailyLines, DailyLinesError
from anjo.sleep.demo.dailies import DailiesError, DailySources, NightFiles, dailies_view, one_line
from anjo.sleep.demo.panel import PanelError
from anjo.sleep.demo.heldout import HeldOutText
from anjo.sleep.demo.starters import Side
from anjo.sleep.night.holds import usd
from anjo.sleep.tests.inquiry_fakes import PERSON, Ws, day, ev, notice_op
from anjo.sleep.types import CheckpointRef, Evidence, Target

NOW = datetime(2026, 10, 4, 3, 0, tzinfo=timezone.utc)
SCENE = words.DAILY_SCENE
HELD = "He would rather the beta launch wait until the festival rough cut is locked and screened twice."
PARTS = {"extract": 0.01, "practice": 0.05, "training": 0.12, "gates": 0.02, "brief": 0.004, "mind": 0.003}
QUIET = {"extract": 0.01, "practice": 0.0, "training": 0.0, "gates": 0.0, "brief": 0.002, "mind": 0.0}


def _ckpt(n: int) -> CheckpointRef:
    return CheckpointRef(f"p-film-n{n}", PERSON, day(n), f"uri://n{n}", None, "gabc/low")


def _summary(n: int, status: str, *, parts: dict[str, float] | None = PARTS, action: str | None = "revisit_thread",
             failed_at: str | None = None) -> dict[str, Any]:
    considered = [
        {"want_id": "w1", "kind": "understand", "target_id": "t-bench", "action": "revisit_thread", "strength": 0.7,
         "weight": 1.0, "status": "won" if action else "held_back", "reason": None if action else "below_threshold"},
        {"want_id": "w2", "kind": "follow_up", "target_id": "t-pricing", "action": "prepare_commitment",
         "strength": 0.4, "weight": 1.0, "status": "held_back", "reason": "lost_to_target"},
    ]
    record: dict[str, Any] = {
        "night": day(n).isoformat(), "status": status, "failed_at": failed_at, "label": "development",
        "checkpoint": None, "gates": {}, "trained": status in ("promoted", "rejected"),
        "choices": [{"opportunity": "away", "action": action, "target_id": "t-bench" if action else None,
                     "question_id": None, "considered": considered}],
    }
    if parts is not None:
        record.update(cost_usd=sum(parts.values()), cost_parts=dict(parts))
    return record


def _steps(*, practice_s: float = 300.0, gpu_minutes: float = 4.5, gates_s: float = 120.0,
           morning_s: float | None = 180.0, dailies_s: float | None = 30.0) -> tuple[dict[str, Any], ...]:
    steps: list[dict[str, Any]] = [
        {"step": "choose", "status": "ok", "receipt": {"outcome": "ran"}},
        {"step": "practice", "status": "ok", "seconds": practice_s},
        {"step": "train", "status": "ok", "receipt": {"gpu_minutes": gpu_minutes}},
        {"step": "gates", "status": "ok", "seconds": gates_s},
        {"step": "note", "status": "ok", "outcome": "no_reason"},
    ]
    if morning_s is not None:
        steps.append({"step": "morning", "status": "ok", "seconds": morning_s})
    if dailies_s is not None:
        steps.append({"step": "dailies", "status": "ok", "seconds": dailies_s})
    return tuple(steps)


def _quiet(n: int) -> NightFiles:
    return NightFiles(_summary(n, "nothing_new", parts=QUIET, action=None), ())


def _target(tid: str, label: str, kind: str = "thread") -> Target:
    return Target(tid, PERSON, kind, label, None, "open", 1, 0, 0.5 if kind == "thread" else None, day(1))  # type: ignore[arg-type]


TARGETS = (_target("t-bench", "Whether the park bench earns its six minutes"),
           _target("t-pricing", "Pricing the festival tier", "commitment"),
           _target("t-later", "Something he raises next week"))


def _evidence() -> tuple[Evidence, ...]:
    return (
        ev("e1", "the bench runs long", n=1, target="t-bench"),
        ev("e2", "the bench again", n=2, target="t-bench"),  # a target he came back to: moved
        ev("e3", "a price for festivals", n=2, target="t-pricing"),  # first mentioned tonight: opened
        ev("e4", "practice on pricing", n=2, kind="practice", target="t-later"),  # hers, not his: moves nothing
        ev("e5", "next week's thing", n=3, target="t-later"),  # a later night's
    )


def _workspace(subject: str = "Six minutes of bench against a festival cut that has to move") -> Any:
    items = _evidence()
    w = Ws()
    w.start(2)
    w.arrive(*items[:3])
    w.ok(notice_op(["e1", "e2"], subject=subject, targets=("t-bench",)))  # raised from tonight's evidence
    w.ok(notice_op(["e1"], kind="gap", subject="An older question from night one", targets=("t-bench",)))
    return w.ws


def _answer(calls: list[tuple[str | None, str]], text: Callable[[str], str] | None = None) -> Any:
    def answer(checkpoint: CheckpointRef | None, scene: words.Scene) -> Side:
        key = None if checkpoint is None else checkpoint.id
        calls.append((key, scene.id))
        return Side(text(key or "base") if text else f"From {key or 'base'}. The rest stays.", 400, 0.001)

    return answer


def _lines(*pairs: tuple[CheckpointRef | None, CheckpointRef], text: Callable[[str], str] | None = None) -> DailyLines:
    kept: DailyLines | None = None
    for before, after in pairs:
        kept = daily_lines.generate(SCENE, before, after, _answer([], text), keep=kept, person="film_founder", now=NOW)
    return kept or DailyLines(person="film_founder", generated_at=NOW.isoformat(), pairs=())


def _sources(nights: list[NightFiles], /, **changes: Any) -> DailySources:
    base: dict[str, Any] = {
        "person": PERSON, "nights": nights, "evidence": _evidence(), "targets": TARGETS, "workspace": _workspace(),
        "history": (_ckpt(1), _ckpt(2)), "lines": _lines((None, _ckpt(1)), (_ckpt(1), _ckpt(2))),
        "settlements": {}, "day_index": {},
    }
    return DailySources(**{**base, **changes})


def _view(sources: DailySources, held: HeldOutText, **kw: Any) -> dict[str, Any]:
    """The card with the reply check passing everything unless a test says otherwise."""
    return dailies_view(sources, held, **{"passes": lambda _text: True, **kw})


TWO = [NightFiles(_summary(1, "promoted"), _steps()), NightFiles(_summary(2, "promoted"), _steps())]
NOBODY = HeldOutText(())


# ── the card ──────────────────────────────────────────────────────────────────


def test_dailies_card_sections_from_a_night() -> None:
    card = _view(_sources(TWO), NOBODY)
    assert (card["state"], card["night"], card["date"], card["status"]) == ("ok", 2, day(2).isoformat(), "promoted")
    assert card["morning"] == day(3).isoformat()  # the night's clock: the morning after it, never the server's
    assert [(n["night"], n["date"]) for n in card["nights"]] == [(2, day(2).isoformat()), (1, day(1).isoformat())]

    learned = card["learned"]
    assert learned["state"] == "ok"
    assert [(t["label"]["text"], t["change"], t["kind"]) for t in learned["targets"]] == [
        ("Pricing the festival tier", "opened", "commitment"),
        ("Whether the park bench earns its six minutes", "moved", "thread")]  # practice and later days move nothing
    chose = learned["chose"]
    assert (chose["action"], chose["target"]["text"], chose["outcome"]) == (
        "revisit_thread", "Whether the park bench earns its six minutes", "ran")
    assert [(b["kind"], b["target"]["text"], b["reason"]) for b in learned["beat"]] == [
        ("follow_up", "Pricing the festival tier", "lost_to_target")]
    assert [(q["kind"], q["subject"]["text"]) for q in learned["questions"]] == [
        ("contradiction", "Six minutes of bench against a festival cut that has to move")]  # night one's isn't new

    line = card["line"]
    assert (line["state"], line["label"], line["scene"]["id"]) == ("ok", "pre-generated", SCENE.id)
    assert (line["before_night"], line["after_night"]) == (1, 2)
    assert (line["before"]["text"], line["after"]["text"]) == ("From p-film-n1.", "From p-film-n2.")

    cost = card["cost"]
    assert (cost["state"], cost["source"]) == ("ok", "summary")
    assert [part["part"] for part in cost["parts"]] == list(dailies.CARD_PARTS)

    first = _view(_sources(TWO), NOBODY, night=1)  # a card by night
    assert (first["night"], first["line"]["before_night"], first["line"]["before"]["text"]) == (
        1, 0, "From base.")  # night one's before is the base model
    assert [q["subject"]["text"] for q in first["learned"]["questions"]] == ["An older question from night one"]
    with pytest.raises(DailiesError):
        _view(_sources(TWO), NOBODY, night=3)


def test_a_malformed_night_fails_where_it_is_read() -> None:
    with pytest.raises(PanelError):  # the app reads it inside its guarded read: the card is unavailable
        NightFiles({"status": "promoted"})
    with pytest.raises(PanelError):
        NightFiles({**_summary(1, "promoted"), "choices": "not a list"})
    assert NightFiles(_summary(1, "promoted"), _steps()).record.status == "promoted"


def test_dailies_before_his_first_night_and_unreadable_sources() -> None:
    assert _view(_sources([]), NOBODY)["state"] == "none"
    closed = _view(_sources(TWO, nights=None), NOBODY)
    assert closed["state"] == "unavailable" and closed["night"] is None
    assert all(closed[name]["state"] == "unavailable" for name in ("learned", "line", "cost"))
    partial = _view(_sources(TWO, workspace=None, lines=None, settlements=None), NOBODY)
    assert (partial["learned"]["state"], partial["line"]["state"]) == ("unavailable", "unavailable")
    assert partial["cost"]["state"] == "ok"  # the summary still carries the night's own parts
    quiet = _view(_sources([_quiet(1)], evidence=(), workspace=_workspace(), lines=_lines()), NOBODY)
    assert quiet["learned"]["state"] == "none"  # nothing reached him, nothing chosen, nothing beaten


def test_dailies_text_passes_the_held_out_filter() -> None:
    held = HeldOutText([HELD])
    targets = (*TARGETS[:1], _target("t-pricing", f"About this: {HELD}", "commitment"), TARGETS[2])
    lines = _lines((None, _ckpt(1)), (_ckpt(1), _ckpt(2)), text=lambda key: f"{HELD} From {key}.")
    card = _view(_sources(TWO, targets=targets, workspace=_workspace(f"Maybe: {HELD}"), lines=lines), held)
    shown = json.dumps(card)
    assert HELD not in shown and " ".join(HELD.split()[2:10]) not in shown
    learned = card["learned"]
    assert learned["targets"][0]["label"] == {"text": None, "withheld": True}
    assert learned["beat"][0]["target"] == {"text": None, "withheld": True}
    assert learned["questions"][0]["subject"] == {"text": None, "withheld": True}
    assert card["line"]["after"] == {"text": None, "withheld": True}

    with pytest.raises(TypeError):  # no default: a card is never built without the reply check
        dailies_view(_sources(TWO), NOBODY)  # type: ignore[call-arg]

    # A reply the reply check trips is held back too, both sides of the line.
    checked = _view(_sources(TWO), NOBODY, passes=lambda text: "p-film-n2" not in text)
    assert (checked["line"]["before"], checked["line"]["after"]) == (
        {"text": None, "withheld": True}, {"text": None, "withheld": True})


def test_dailies_no_new_checkpoint_says_so() -> None:
    lines = _lines((None, _ckpt(1)), (_ckpt(1), _ckpt(2)))  # even with pairs on file
    for status in ("rejected", "nothing_new", "failed"):
        nights = [NightFiles(_summary(1, "promoted"), _steps()), NightFiles(_summary(2, status), _steps())]
        card = _view(_sources(nights, history=(_ckpt(1),), lines=lines), NOBODY)
        line = card["line"]
        assert (line["state"], line["reason"]) == ("none", "no_new_checkpoint"), status
        assert line["before"] is None and line["after"] is None and "From" not in json.dumps(line)
    # A promoted night whose pair was never generated says that, and shows nothing either.
    missing = _view(_sources(TWO, lines=_lines((None, _ckpt(1)))), NOBODY)["line"]
    assert (missing["state"], missing["reason"], missing["after"]) == ("none", "not_generated", None)


def test_dailies_line_pregenerated_per_checkpoint_pair(tmp_path: Path) -> None:
    calls: list[tuple[str | None, str]] = []
    first = daily_lines.generate(SCENE, None, _ckpt(1), _answer(calls), keep=None, person="film_founder", now=NOW)
    assert calls == [(None, SCENE.id), ("p-film-n1", SCENE.id)]  # one scene: the night's before, then its after
    second = daily_lines.generate(SCENE, _ckpt(1), _ckpt(2), _answer(calls), keep=first, person="film_founder", now=NOW)
    assert calls[2:] == [("p-film-n1", SCENE.id), ("p-film-n2", SCENE.id)]  # a pair is generated together
    again = daily_lines.generate(SCENE, _ckpt(1), _ckpt(2), _answer(calls), keep=second, person="film_founder", now=NOW)
    assert len(calls) == 4 and again.pairs == second.pairs  # a pair it holds asks nothing
    assert [(p.night, p.before, p.after) for p in again.pairs] == [
        (day(1).isoformat(), None, "p-film-n1"), (day(2).isoformat(), "p-film-n1", "p-film-n2")]
    assert again.find("p-film-n1", "p-film-n2", SCENE) is not None
    assert again.find("p-film-n2", "p-film-n3", SCENE) is None  # a pair answers only its own two checkpoints

    path = tmp_path / "daily_lines.json"
    path.write_text(daily_lines.to_json(again))
    loaded = daily_lines.load(path)
    assert loaded == again and loaded is not None and loaded.label == "pre-generated"
    assert daily_lines.load(tmp_path / "absent.json") is None
    with pytest.raises(DailyLinesError):
        daily_lines.from_json(json.dumps({**json.loads(daily_lines.to_json(again)), "label": "live"}))

    # The card reads the pair its night names: night 3's before is night 2's checkpoint, never a stale pair.
    three = [*TWO, NightFiles(_summary(3, "promoted"), _steps())]
    history = (_ckpt(1), _ckpt(2), _ckpt(3))
    assert _view(_sources(three, history=history, lines=again), NOBODY)["line"]["reason"] == "not_generated"
    # A scene whose words changed since retires its pairs, by the prompt's hash.
    reworded = words.Scene(SCENE.id, SCENE.title, SCENE.prompt + " Keep it short.")
    assert again.find("p-film-n1", "p-film-n2", reworded) is None


def test_one_line_is_the_first_sentence_they_say_differently() -> None:
    before = "I'm so sorry about tonight. The export bug is on us. We'll fix it by Friday."
    after = "You lost three hours tonight, and that's on us. The export bug is on us. We'll fix it by Friday."
    assert one_line(before, after) == ("I'm so sorry about tonight.", "You lost three hours tonight, and that's on us.")
    assert one_line("Same words. Then this.", "Same words. Then that!") == ("Then this.", "Then that!")
    assert one_line("**Same.** Again.", "- Same.\nAgain.") is None  # markup aside, he'd say it the same way
    assert one_line("Same.", "Same. And one more thing.") == (None, "And one more thing.")  # said only after
    long = " ".join(["word"] * 80) + "."
    cut = one_line("Short.", long)
    assert cut is not None and cut[1] is not None and len(cut[1].split()) == dailies.LINE_WORDS and cut[1].endswith("…")


# ── what the night cost ───────────────────────────────────────────────────────


def _settle(ledger: Ledger, job: str, amount: float, pool: Any = "baseten_sleep_usd") -> None:
    held = ledger.reserve(job, (pool,), usd(1.0))
    assert held is not None
    held.settle(amount)


def _ledger(tmp_path: Path) -> Ledger:
    ledger = Ledger(tmp_path / "budget.jsonl")
    night, other = day(2).isoformat(), day(1).isoformat()
    _settle(ledger, f"sim:{PERSON}:d2:s7:a1", 0.006)
    _settle(ledger, f"morning:{night}:{PERSON}:arms:a1", 0.15, "modal_usd")
    _settle(ledger, f"morning:{night}:{PERSON}:judge:a1", 0.04, "meta_usd")
    _settle(ledger, f"dailies:{night}:{PERSON}:a1", 0.02, "modal_usd")
    _settle(ledger, f"night:practice:{PERSON}:{night}:a1", 0.06)
    _settle(ledger, f"modal_train:{night}:{PERSON}:a1", 0.13, "modal_usd")
    _settle(ledger, f"night:extract:{PERSON}:{night}:a1", 0.011)
    _settle(ledger, f"night:notice:{PERSON}:{night}:spark:a1", 0.002, "baseten_mind_usd")
    _settle(ledger, f"night:note:{PERSON}:{night}:a1", 0.004, "modal_usd")  # the note's GPU: not in the summary
    _settle(ledger, f"night:homework:{PERSON}:{night}:t-bench:c1:a1", 0.003, "meta_usd")  # Wonder: nor this
    _settle(ledger, f"colab_train:{night}:{PERSON}:a1", 0.7, "colab_cu")  # compute units, never dollars
    _settle(ledger, f"night:practice:{PERSON}:{other}:a1", 0.9)  # another night's
    _settle(ledger, "sim:someone-else:d2:s7:a1", 0.9)  # another person's
    assert ledger.reserve(f"morning:{night}:{PERSON}:extra:a1", ("modal_usd",), usd(5.0)) is not None  # a hold
    assert ledger.reserve(f"modal_train:{night}:{PERSON}:a2", ("modal_usd",), usd(5.0)) is not None  # never settled
    return ledger


def _parts(card: dict[str, Any]) -> dict[str, tuple[Any, Any, Any]]:
    return {p["part"]: (p["usd"], p["gpu_minutes"], p["source"]) for p in card["cost"]["parts"]}


def test_dailies_cost_from_summary_else_ledger(tmp_path: Path) -> None:
    settled = _ledger(tmp_path).settlements()
    assert f"morning:{day(2).isoformat()}:{PERSON}:extra:a1" not in settled  # a hold is never a settlement
    assert f"colab_train:{day(2).isoformat()}:{PERSON}:a1" not in settled  # dollars only
    nights = [_quiet(1), NightFiles(_summary(2, "promoted"), _steps())]
    card = _view(_sources(nights, settlements=settled, day_index={day(2): 2}), NOBODY)
    parts = _parts(card)
    assert card["cost"]["source"] == "summary"
    assert parts["day"] == (pytest.approx(0.006), None, "ledger")  # the simulator's calls; nothing timed the day
    assert parts["practice"] == (pytest.approx(0.05), pytest.approx(5.0), "summary")
    assert parts["training"] == (pytest.approx(0.12), pytest.approx(4.5), "summary")
    assert parts["morning"] == (pytest.approx(0.21), pytest.approx(3.5), "ledger")  # the run and its dailies line
    # Extraction, gates, the brief and MIND from the summary; the note's GPU and Wonder from the ledger.
    assert parts["rest"] == (pytest.approx(0.037 + 0.007), pytest.approx(2.0), "summary")
    assert card["cost"]["usd"] == pytest.approx(0.006 + 0.05 + 0.12 + 0.21 + 0.044)
    assert card["cost"]["gpu_minutes"] == pytest.approx(15.0)  # measured durations only, never a hold's bound

    # A summary without its cost parts: the night's own ledger keys, and the card says so.
    bare = [_quiet(1), NightFiles(_summary(2, "promoted", parts=None), _steps())]
    fallback = _view(_sources(bare, settlements=settled, day_index={day(2): 2}), NOBODY)
    parts = _parts(fallback)
    assert fallback["cost"]["source"] == "ledger"
    assert parts["practice"] == (pytest.approx(0.06), pytest.approx(5.0), "ledger")
    assert parts["training"] == (pytest.approx(0.13), pytest.approx(4.5), "ledger")
    assert parts["rest"] == (pytest.approx(0.013 + 0.007), pytest.approx(2.0), "ledger")  # the same things
    colab = {**settled, f"colab_train:{day(2).isoformat()}:{PERSON}:a1": 0.7}  # given one anyway: not dollars
    assert _parts(_view(_sources(bare, settlements=colab, day_index={day(2): 2}), NOBODY))["training"][0] == (
        pytest.approx(0.13))
    # No ledger and no day marker: those parts are unrecorded, never guessed.
    blind = _parts(_view(_sources(bare, settlements=None), NOBODY))
    assert blind["day"] == (None, None, None) and blind["practice"] == (None, pytest.approx(5.0), None)


def test_dailies_projects_a_month_against_the_ceiling() -> None:
    nights = [_quiet(1), NightFiles(_summary(2, "promoted"), _steps())]
    projection = _view(_sources(nights), NOBODY)["cost"]["projection"]
    trained, quiet = 0.01 + 0.05 + 0.12 + 0.02 + 0.004, 0.012  # the ceiling counts all but MIND
    assert projection["label"] == "development" and projection["ceiling_usd"] == dailies.CEILING_USD
    assert (projection["nights"], projection["trained_share"]) == (2, 0.5)
    assert projection["month_usd"] == pytest.approx(30 * (0.5 * trained + 0.5 * quiet))
    assert projection["worst_month_usd"] == pytest.approx(30 * trained) and projection["under_ceiling"] is True
    first = _view(_sources(nights), NOBODY, night=1)["cost"]["projection"]  # only the nights up to it
    assert (first["nights"], first["trained_share"], first["worst_month_usd"]) == (1, 0.0, None)
    assert first["month_usd"] == pytest.approx(30 * quiet)
    failed = NightFiles(_summary(3, "failed", failed_at="train"), ())  # a night that started training counts
    later = _view(_sources([*nights, failed]), NOBODY)["cost"]["projection"]
    assert later["trained_share"] == pytest.approx(2 / 3)
