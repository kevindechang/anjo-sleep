# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The research query's personal-terms screen (``anjo.sleep.app.screen``), which needs private Anjo,
so it lives with the adapter tests the public repo doesn't collect."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from anjo.sleep.app.screen import SparkScreen
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.model_api import ChatResult
from anjo.sleep.tests.mind_night_fakes import LifeThinker, Researches, mind_night
from anjo.sleep.tests.night_fakes import step, summary
from anjo.sleep.tests.test_night_mind import UID, _night, _world


def test_research_query_carries_no_personal_terms(tmp_path: Path) -> None:
    world = _world(tmp_path)

    def allowing(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        return ChatResult('{"allowed": true}', (), SPARK_OPEN, 40, 5, "stop")

    research = Researches()
    thinker = LifeThinker(targets=(f"{UID}-t1",), tool="research",
                          queries=["how Maya Lin rehearses", "demo rehearsal schedules"])
    mind = mind_night(world, thinker, research=research,
                      screen=lambda spark: SparkScreen(names=("Maya Lin",), chat=spark(allowing)))
    result = _night(world, mind)
    inquiry = step(result.log_path, "choose")["inquiry"]
    assert research.queries == ["demo rehearsal schedules"]  # the named query never left the night
    assert inquiry["rejected"] == {"domain_screen": 1} and inquiry["spark_calls"] > 0
    assert inquiry["research_offered"] and inquiry["research_calls"] == 1
    record = summary(result.log_path)
    assert (record["paid_actions"], record["wonder_calls"]) == (2, 1)  # the inquiry's research, and the note
    assert record["cost_parts"]["mind"] > 0
