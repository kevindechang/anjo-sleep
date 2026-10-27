# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The dev API transport (``anjo.sleep.app.sim_transport``), which needs private Anjo, so it lives
with the adapter tests the public repo doesn't collect."""
from __future__ import annotations

import httpx
import pytest

from anjo.sleep.app.sim_transport import DevApiTransport
from anjo.sleep.sim.day import TurnFailed


def test_the_dev_api_transport_reads_the_stream_and_fails_a_turn_it_cannot_answer() -> None:
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
