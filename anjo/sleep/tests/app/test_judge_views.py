# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types)
"""What a judge's own pages read: her bet about her own practice as she made it and as it resolved,
the live pass's wants on the MIND panel, and the opener built from the want she picked when no note
waits. Fakes and in-memory logs only."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from anjo.sleep.app import judge_views, live_mind
from anjo.sleep.mind import self_predict
from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry.log import MemoryWorkspaceLog
from anjo.sleep.tests.inquiry_fakes import night
from anjo.sleep.tests.test_mind_loop_focus import NIGHT_START, _commit, _record, _seeded


def _bet() -> tuple[MemoryWorkspaceLog, Any]:
    w, targets = _seeded(second=False)
    log = MemoryWorkspaceLog(tuple(w.events))
    return log, _commit(log, w, targets)


def test_the_bet_reads_as_she_made_it_before_it_resolves() -> None:
    log, committed = _bet()
    bet = judge_views.self_prediction(log.events())
    assert bet is not None and bet["outcome"] is None
    assert bet["claim"] == committed.predictions[0].claim
    p = self_predict.PRIORS[0] * self_predict.P_IMPROVES + self_predict.PRIORS[1] * self_predict.P_NO_CHANGE \
        + self_predict.CATCH_ALL * self_predict.P_OTHER
    assert bet["p"] == pytest.approx(p, abs=1e-3)


def test_a_resolved_bet_keeps_the_probability_she_gave_it() -> None:
    log, committed = _bet()
    made = judge_views.self_prediction(log.events())
    self_predict.resolve(log, kernel.fold(log.events()), night=NIGHT_START, at=night(2), record=_record(3),
                         predictions=committed.predictions, history=())
    settled = judge_views.self_prediction(log.events())
    assert settled is not None and made is not None
    assert settled["outcome"] == "true" and settled["p"] == made["p"]


def test_no_bet_no_self_prediction() -> None:
    w, _ = _seeded(second=False)
    assert judge_views.self_prediction(tuple(w.events)) is None
    assert judge_views.self_prediction(()) is None


def test_the_live_pass_leads_the_panel_and_the_opener(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.demo.panel import MindSources
    from anjo.sleep.tests.night_fakes import thread

    target = thread("judge-1", "the kitchen shoot")
    tid = target.id
    monkeypatch.setattr(judge_views, "_sources", lambda _uid: MindSources(nights=(), targets=(target,), events=()))
    live_mind.save_record("judge-1", {
        "at": datetime(2026, 11, 3, 15, tzinfo=timezone.utc).isoformat(), "state": "ok", "thinker": "spark-open",
        "notice": {"new": 2, "questions_opened": 1},
        "choice": {"action": "inquire", "target_id": tid, "question_id": "q-1", "kind": "resolve",
                   "considered": [
                       {"want_id": "w-2", "kind": "revisit", "target_id": tid, "action": "revisit_thread",
                        "strength": 0.2, "status": "lost", "reason": "lost"},
                       {"want_id": "w-1", "kind": "resolve", "target_id": tid, "action": "inquire",
                        "strength": 0.6, "status": "won", "reason": None}]},
    })
    view = judge_views.mind_payload("judge-1")
    assert view["wants"]["live"] is True
    assert [w["status"] for w in view["wants"]["items"]] == ["won", "lost"]
    assert view["wants"]["items"][0]["target"] == {"text": "the kitchen shoot", "withheld": False}
    assert view["live"]["questions_opened"] == 1 and view["live"]["choice"]["action"] == "inquire"
    monkeypatch.setattr("anjo.sleep.app.demo_mind.return_payload", lambda *_a: {"state": "none"})
    opener, kind = judge_views._opener("judge-1", judge_views._sources("judge-1"))
    assert kind == "want" and opener is not None and "the kitchen shoot" in opener
