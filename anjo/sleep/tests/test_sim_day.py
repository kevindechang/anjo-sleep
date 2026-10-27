# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""A simulated day before the night (Task 5): every line carries known ancestry when the replies
it read are known, a turn Glimmer can't answer ends the day short instead of failing the night,
and the night log records the counts."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.sim.day import TurnFailed, run_day
from anjo.sleep.tests.night_fakes import NIGHT, World, step, steps

PERSONA = load_person_file(Path(__file__).parents[1] / "personas" / "founder.json").visible
T0 = datetime(2026, 10, 28, 9, 0, tzinfo=timezone.utc)


class Model:
    def complete(self, system: str, messages: list[dict[str, str]], *, temperature: float, seed: int) -> str:
        return f"Line {len(messages)} about the pilot."


class Transport:
    def __init__(self, fail_at: int | None = None) -> None:
        self.sent = 0
        self.fail_at = fail_at

    def send(self, session_id: str, text: str, *, cleared: bool) -> str:
        if self.fail_at is not None and self.sent == self.fail_at:
            raise TurnFailed("Glimmer is down: the turn stored nothing")
        self.sent += 1
        return f"Reply {self.sent}."


class Served:
    """The recorded ancestry of each reply Glimmer served (``NewestReply``'s shape)."""

    def of_reply(self, session_id: str) -> frozenset[str] | None:
        return frozenset({GLIMMER})


def _clock() -> object:
    ticks = iter(T0 + timedelta(minutes=i) for i in range(10))
    return lambda: next(ticks)


def test_a_simulated_day_through_the_transport_has_known_ancestry() -> None:
    day = run_day(PERSONA, 1, 7, model=Model(), transport=Transport(), replies=Served(), clock=_clock())  # type: ignore[arg-type]
    assert day.failed == 0 and day.answered == len(day.turns) // 2 > 0
    users = [t for t in day.turns if t.role == "user"]
    assert users[0].ancestry == frozenset({SPARK_OPEN})
    assert all(t.ancestry == frozenset({SPARK_OPEN, GLIMMER}) for t in users[1:])
    assert all(t.ancestry == frozenset({GLIMMER}) for t in day.turns if t.role == "assistant")


def test_a_failed_turn_ends_the_day_short_not_in_error() -> None:
    day = run_day(PERSONA, 1, 7, model=Model(), transport=Transport(fail_at=2), replies=Served(), clock=_clock())  # type: ignore[arg-type]
    assert day.short and (day.answered, day.failed) == (2, 1) and day.window[0] < day.window[1]


def test_the_night_logs_a_short_day_and_runs_on_its_window(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    day = run_day(PERSONA, 1, 7, model=Model(), transport=Transport(fail_at=0), replies=Served(), clock=_clock())  # type: ignore[arg-type]
    result = run_night_batch_with(world.deps(), {"p1": NIGHT}, days={"p1": day})["p1"]
    assert result.status == "promoted"
    record = step(result.log_path, "simulate")
    assert record == {"step": "simulate", "status": "ok", "uses": "existing", "system": "chat route", "noop": False,
                      "answered": 0, "failed_turns": 1, "short_day": True}
    assert steps(result.log_path)[0]["step"] == "simulate"
    assert step(result.log_path, "close_day")["window"] == [day.window[0].isoformat(), day.window[1].isoformat()]


def test_the_dev_api_transport_reads_the_stream_and_fails_a_turn_it_cannot_answer() -> None:
    from anjo.sleep.app.sim_transport import DevApiTransport

    calls: list[str] = []
    down = {"now": False}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.headers["Authorization"] == "Bearer t"
        if request.url.path == "/api/sleep/dev/sessions":
            return httpx.Response(200, json={"session_id": "s-1", "cleared": True})
        if down["now"]:
            return httpx.Response(200, text='event: error\ndata: {"error": "unavailable"}\n\n')
        return httpx.Response(200, text='event: token\ndata: {"text": "Start "}\n\nevent: token\ndata: {"text": "small."}\n\n'
                                        'event: done\ndata: {}\n\n')

    transport = DevApiTransport("http://127.0.0.1:8000", "t", transport=httpx.MockTransport(handler))
    assert transport.send("sim-founder-d1", "Where do I start?", cleared=True) == "Start small."
    down["now"] = True
    with pytest.raises(TurnFailed):
        transport.send("sim-founder-d1", "And then?", cleared=True)
    assert calls == ["/api/sleep/dev/sessions", "/api/chat/s-1/message", "/api/chat/s-1/message"]
