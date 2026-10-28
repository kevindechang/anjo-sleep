# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""Dailies on the sleep host: the morning card's routes behind the 18+ visit, read from the real
stores of the account that plays the film student, and the night phase's step that pre-generates
its before and after line. Only the model and the serving backend are fakes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.demo import daily_lines, words
from anjo.sleep.tests.app.test_demo_mind import _ckpt, _day, _service, _write_night
from anjo.sleep.tests.app.test_demo_page import (  # noqa: F401  (account is a fixture)
    ACCOUNT,
    FakeServing,
    ScriptedModel,
    _arms,
    _client,
    _demo_dir,
    _visit,
    account,
)
from anjo.sleep.types import NightResult

pytestmark = pytest.mark.usefixtures("account")

API = "/api/sleep/demo"
PARTS = {"extract": 0.01, "practice": 0.05, "training": 0.12, "gates": 0.02, "brief": 0.004, "mind": 0.0}


def _night(n: int, status: str = "promoted") -> None:
    """A night as the night run leaves it: its summary carries its cost parts, its log the timed steps."""
    from anjo.sleep.app.ownership import night_logs

    _write_night(n, status)
    folder = night_logs(ACCOUNT) / _day(n).isoformat()
    summary = json.loads((folder / "summary.json").read_text())
    (folder / "summary.json").write_text(json.dumps({**summary, "cost_parts": PARTS, "cost_usd": sum(PARTS.values())}))
    with open(folder / "log.jsonl", "a", encoding="utf-8") as log:
        for step in ({"step": "practice", "status": "ok", "seconds": 240.0},
                     {"step": "morning", "status": "ok", "seconds": 120.0}):
            log.write(json.dumps(step) + "\n")


def _result(n: int, status: str = "promoted") -> NightResult:
    return NightResult(status, None, _ckpt(n) if status == "promoted" else None, {}, (), 0.0, {},  # type: ignore[arg-type]
                       Path("log.jsonl"))


def _two_nights(model: ScriptedModel) -> None:
    from anjo.sleep.app.demo_pairs import after_night
    from anjo.sleep.app.storage import checkpoint_registry

    for n in (1, 2):
        _night(n)
        checkpoint_registry().promote(ACCOUNT, _ckpt(n))
        after_night(ACCOUNT, "film_founder", _day(n), _result(n), arms=_arms(model, FakeServing()))


def _day_marker(n: int, index: int) -> None:
    from anjo.sleep.app.storage import SLEEP_DIR, data_root

    folder = data_root() / SLEEP_DIR / "daily"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"film_founder-{_day(n).isoformat()}.json").write_text(json.dumps(
        {"persona": "film_founder", "date": _day(n).isoformat(), "day_index": index, "status": "ok", "failure": None,
         "failed_turns": 0}))


def _settle(job: str, amount: float) -> None:
    from anjo.sleep.app.storage import SLEEP_DIR, data_root
    from anjo.sleep.budget import LEDGER_NAME, Ledger
    from anjo.sleep.night.holds import usd

    held = Ledger(data_root() / SLEEP_DIR / LEDGER_NAME).reserve(job, ("baseten_sleep_usd",), usd(1.0))
    assert held is not None
    held.settle(amount)


def test_dailies_route_needs_a_visit() -> None:
    _two_nights(ScriptedModel(reply=lambda n: f"Reply {n} opens here. Then the same rest."))
    _day_marker(2, 9)
    _settle(f"sim:{ACCOUNT}:d9:s20261028:a1", 0.007)
    client = _client(_service())
    for path in ("/dailies", "/dailies/1"):
        assert client.get(f"{API}{path}").status_code == 401  # behind the 18+ visit, like every model-written word
    headers = _visit(client)
    card = client.get(f"{API}/dailies", headers=headers).json()
    assert (card["state"], card["night"], card["date"], card["status"]) == ("ok", 2, _day(2).isoformat(), "promoted")
    line = card["line"]
    assert (line["state"], line["before_night"], line["after_night"]) == ("ok", 1, 2)
    assert (line["before"]["text"], line["after"]["text"]) == ("Reply 2 opens here.", "Reply 3 opens here.")
    parts = {p["part"]: p for p in card["cost"]["parts"]}
    assert (parts["day"]["usd"], parts["day"]["source"]) == (pytest.approx(0.007), "ledger")  # by the day's marker
    assert (parts["practice"]["usd"], parts["practice"]["gpu_minutes"]) == (pytest.approx(0.05), pytest.approx(4.0))
    assert (parts["morning"]["gpu_minutes"], parts["morning"]["source"]) == (pytest.approx(2.0), "ledger")  # its line's hold
    assert ACCOUNT not in json.dumps(card)  # no account id, and no ledger key, reaches the page

    first = client.get(f"{API}/dailies/1", headers=headers).json()
    assert (first["night"], first["line"]["before_night"], first["line"]["before"]["text"]) == (
        1, 0, "Reply 0 opens here.")
    missing = client.get(f"{API}/dailies/9", headers=headers)
    assert missing.status_code == 404 and missing.json()["detail"] == words.COPY["unknown_night"]
    assert client.get(f"{API}/dailies/0", headers=headers).status_code == 422


def test_dailies_show_nothing_for_an_account_that_isnt_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    _two_nights(ScriptedModel())
    monkeypatch.setenv("ANJO_SLEEP_DEMO_USER_ID", "someone-real")
    client = _client(_service())
    card = client.get(f"{API}/dailies", headers=_visit(client)).json()
    assert card["state"] == "unavailable" and card["night"] is None
    assert all(card[name]["state"] == "unavailable" for name in ("learned", "line", "cost"))


def test_after_night_pregenerates_the_line_and_holds_its_gpu_time() -> None:
    from anjo.sleep.app.demo_dailies import DAILY_LINES_FILE
    from anjo.sleep.app.demo_pairs import LINE_MAX_S, after_night
    from anjo.sleep.serving.client import LOAD_WAIT_S, REQUEST_TIMEOUT_S
    from anjo.sleep.app.storage import SLEEP_DIR, data_root
    from anjo.sleep.budget import LEDGER_NAME, Ledger

    model = ScriptedModel(reply=lambda n: f"Reply {n}.")
    _two_nights(model)
    assert len(model.calls) == 4  # each night: its before, then its after, one scene
    assert all(words.DAILY_SCENE.prompt in json.dumps(call["messages"]) for call in model.calls)
    lines = daily_lines.load(_demo_dir() / DAILY_LINES_FILE)
    assert lines is not None and lines.label == "pre-generated"
    assert [(p.before, p.after) for p in lines.pairs] == [(None, "p-film-n1"), ("p-film-n1", "p-film-n2")]
    ledger = Ledger(data_root() / SLEEP_DIR / LEDGER_NAME)
    assert set(ledger.settlements()) == {f"dailies:{_day(n).isoformat()}:{ACCOUNT}:a1" for n in (1, 2)}
    assert ledger.open_holds() == ()
    reserved = [json.loads(line) for line in ledger.path.read_text().splitlines() if '"reserve"' in line]
    # The hold's bound covers its whole timed block: two answers, each waiting for a new adapter's
    # folder and then for its reply, with room to spare, so a slow line never records an overage.
    assert {event["timeout_s"] for event in reserved} == {LINE_MAX_S}
    assert LINE_MAX_S > 2 * (REQUEST_TIMEOUT_S + LOAD_WAIT_S)

    # Run again for the same night, or after a night that promoted nothing: nothing is asked.
    after_night(ACCOUNT, "film_founder", _day(2), _result(2), arms=_arms(model, FakeServing()))
    _night(3, "rejected")
    after_night(ACCOUNT, "film_founder", _day(3), _result(3, "rejected"), arms=_arms(model, FakeServing()))
    assert len(model.calls) == 4
    with pytest.raises(ValueError):  # only the demo's own persona, on a synthetic account
        after_night(ACCOUNT, "someone_else", _day(2), _result(2), arms=_arms(model, FakeServing()))


def test_a_line_the_crisis_check_trips_is_withheld() -> None:
    from anjo.evals.crisis_reply_cases import REPLY_TRIPS

    trip = REPLY_TRIPS[0][1]
    _two_nights(ScriptedModel(reply=lambda n: trip if n == 1 else f"Reply {n} opens here."))
    client = _client(_service())
    line = client.get(f"{API}/dailies/1", headers=_visit(client)).json()["line"]
    assert line["state"] == "ok"
    assert (line["before"], line["after"]) == ({"text": None, "withheld": True}, {"text": None, "withheld": True})
    assert trip not in json.dumps(line)


def test_a_malformed_night_makes_the_card_unavailable_not_an_error() -> None:
    from anjo.sleep.app.ownership import night_logs

    _two_nights(ScriptedModel())
    summary = night_logs(ACCOUNT) / _day(2).isoformat() / "summary.json"
    summary.write_text(json.dumps({**json.loads(summary.read_text()), "choices": "not a list"}))
    client = _client(_service())
    response = client.get(f"{API}/dailies", headers=_visit(client))
    assert response.status_code == 200
    card = response.json()
    assert card["state"] == "unavailable" and all(card[name]["state"] == "unavailable" for name in ("learned", "line", "cost"))


def test_every_word_the_sheet_shows_is_in_the_copy() -> None:
    import re

    from anjo.sleep.demo.dailies import CARD_PARTS
    from anjo.sleep.demo.page import WEB_DIR

    script = (WEB_DIR / "dailies.js").read_text()
    named = set(re.findall(r"copy\.(\w+)", script))
    built = {f"cost_part_{p}" for p in CARD_PARTS} | {f"cost_source_{s}" for s in ("summary", "ledger")} | {
        f"dailies_{c}" for c in ("opened", "moved")} | {f"dailies_line_{r}" for r in ("no_new_checkpoint", "not_generated")}
    assert named and not (named | built) - set(words.COPY)
    assert {"dailies_line_missing", "dailies_before_unknown"} <= named  # an honest label, never "nothing"


def test_a_night_starting_mid_line_cant_sweep_its_live_hold() -> None:
    from anjo.sleep.app.demo_pairs import line_for
    from anjo.sleep.app.storage import SLEEP_DIR, checkpoint_registry, data_root
    from anjo.sleep.budget import LEDGER_NAME, Ledger
    from anjo.sleep.demo.service import Answer
    from anjo.sleep.night.sweep import NightBusy, swept_night

    ledger = Ledger(data_root() / SLEEP_DIR / LEDGER_NAME)
    tried: list[str] = []

    class Arms:
        def awake(self) -> bool:
            return True

        def answer_at(self, question: str, checkpoint: Any) -> Answer:
            with pytest.raises(NightBusy), swept_night(ledger, lambda: (0, 0)):
                pass  # the line holds the night lock: no night start settles its hold under it
            tried.append("swept")
            return Answer("A reply.", 400, 600, 1.0)

    _night(1)
    checkpoint_registry().promote(ACCOUNT, _ckpt(1))
    assert line_for(ACCOUNT, _day(1), Arms()) is not None
    assert tried == ["swept", "swept"] and ledger.open_holds() == ()
