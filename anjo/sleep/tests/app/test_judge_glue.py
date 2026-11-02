"""The night job's judge ports read the cohort: one list of judges, one set of counters."""
from __future__ import annotations

from typing import Any

import pytest

from anjo.sleep.app import judge_glue, judge_night
from anjo.sleep.app.judge_cohort import CapReached


@pytest.fixture(autouse=True)
def _reset_ports() -> Any:
    yield
    judge_night.set_judge_ports(None)


def test_ok_when_the_cohort_counts_the_night(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    claimed: list[str] = []
    monkeypatch.setattr(judge_glue, "claim_night", lambda user_id: claimed.append(user_id))

    # Act
    verdict = judge_glue.CohortCaps().try_start_night("judge-1")

    # Assert
    assert verdict == "ok" and claimed == ["judge-1"]


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("your_nights_spent_today", "per_judge"), ("nights_spent_today", "global"), ("counters_unreadable", "global"),
     ("not_a_judge", "global")],
)
def test_a_refused_claim_maps_to_the_night_jobs_verdicts(
    monkeypatch: pytest.MonkeyPatch, reason: str, expected: str
) -> None:
    # Arrange
    def refuse(user_id: str) -> None:
        raise CapReached(429, reason)

    monkeypatch.setattr(judge_glue, "claim_night", refuse)

    # Act
    verdict = judge_glue.CohortCaps().try_start_night("judge-1")

    # Assert
    assert verdict == expected


def test_wiring_makes_the_cohort_the_night_jobs_judges(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(judge_glue, "night_allowed", lambda user_id: user_id == "judge-1")

    # Act
    judge_glue.wire_judge_ports()
    ports = judge_night.judge_ports()

    # Assert
    assert ports.is_judge("judge-1") and not ports.is_judge("someone-else")
    assert isinstance(ports.caps, judge_glue.CohortCaps)
