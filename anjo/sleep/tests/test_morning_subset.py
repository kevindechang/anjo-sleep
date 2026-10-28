# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The morning's probe set by setting (``ANJO_SLEEP_MORNING_SUBSET``): unset, the full morning as
before; ``daily-v1``, the daily subset, asked by the accelerated runner's mornings, shown in the live
comparison's plan, passed through to each arm and recorded in the morning step and the report; any
other value stops before anything runs. Fakes only."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from anjo.sleep.night.daily import morning_receipt
from anjo.sleep.night.deps import Prices
from anjo.sleep.probes.arms import ArmDeps
from anjo.sleep.probes.morning import DAILY, MORNING_SUBSET_ENV, Morning, MorningResult, morning_subset
from anjo.sleep.probes.score import Verdict
from anjo.sleep.sim import accelerated
from anjo.sleep.tests.sleep_fakes import FakeContexts, FakeLogprobs, FakeModel, FakeRegistry, StoreSpy, ckpt, probe, taste
from anjo.sleep.tools import sleep_objective_live as live

NIGHT = date(2026, 10, 24)
PRICES = Prices(spark_in_per_m=1.25, spark_out_per_m=4.25, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.96)
REAL = ["--wiring", "x:y", "--simulate", "x:z", "--morning", "x:w", "--days", "2", "--start", "2026-10-20",
        "--user", "u1", "--persona", "film_founder"]


class Judge:
    def verdict(self, p: Any, first: str, second: str) -> Verdict:
        return "A" if len(first) > len(second) else "B" if len(second) > len(first) else "tie"


def _build(root: Path) -> Any:
    registry = FakeRegistry({"p1": ckpt("p1", 1)})
    probes = [probe("q1", "new_work"), probe("q2", "vocab"), probe("q3", "tone_hold"), taste("t1")]

    def build(user: str, persona: str, night: date) -> Morning:
        deps = ArmDeps(model=FakeModel(), adapters=registry, contexts=FakeContexts(), evidence=lambda uid: StoreSpy([]),
                       logprobs=FakeLogprobs(lambda adapter, messages: "A"), donors={}, until=night)
        return Morning(arms=deps, sealed=lambda uid: probes, judge=Judge(), prices=PRICES, root=root)
    return build, registry


def test_the_setting_names_the_full_morning_or_the_daily_subset_and_nothing_else() -> None:
    assert morning_subset({}) is None and morning_subset({MORNING_SUBSET_ENV: " "}) is None
    assert morning_subset({MORNING_SUBSET_ENV: "daily-v1"}) is DAILY
    with pytest.raises(ValueError, match=MORNING_SUBSET_ENV):
        morning_subset({MORNING_SUBSET_ENV: "daily"})


@pytest.mark.parametrize(("value", "label"), [("daily-v2", "development"), ("daily-v1", "final")])
def test_an_unknown_subset_or_a_final_run_on_one_stops_the_runner_before_it_starts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], value: str, label: str,
) -> None:
    monkeypatch.setenv(MORNING_SUBSET_ENV, value)
    with pytest.raises(SystemExit):
        accelerated.main([*REAL, "--label", label])
    assert MORNING_SUBSET_ENV in capsys.readouterr().err


@pytest.mark.parametrize("subset", [DAILY, None])
def test_the_runners_morning_asks_the_set_named_and_its_step_records_it(tmp_path: Path, subset: Any) -> None:
    build, registry = _build(tmp_path)
    score = accelerated.morning_scorer(build, {"p1": "film_founder"}, subset)
    done = cast(MorningResult, score("p1", NIGHT, "development",
                                     cast(Any, SimpleNamespace(status="promoted", checkpoint=registry.adapters["p1"]))))
    record = json.loads(done.path.read_text(encoding="utf-8"))
    name = None if subset is None else subset.name
    assert record.get("subset") == done.subset == morning_receipt(done, 1.0).get("subset") == name
    assert record["probes"] == 4 and set(record["arms"]) == ({"1", "3"} if subset else {"1", "2C", "2F", "3", "4"})


def test_the_live_comparison_shows_the_set_passes_it_through_and_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANJO_DATA_DIR", raising=False)  # a plan needs no data dir
    monkeypatch.delenv("ANJO_SLEEP_CHANNEL", raising=False)
    monkeypatch.setenv(MORNING_SUBSET_ENV, "daily-v1")
    lines: list[str] = []
    accounts = ["--account", "seed=syn-a", "--account", "today=syn-c"]
    now = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
    assert live.main([*accounts, "--days", "3", "--night-usd", "2.0"], out=lines.append, now=lambda: now) == 0
    assert any(line.startswith("  morning: daily-v1 (ANJO_SLEEP_MORNING_SUBSET") for line in lines)
    assert live.arm_env("today", {MORNING_SUBSET_ENV: "daily-v1", "HOME": "/h"}, "t")[MORNING_SUBSET_ENV] == "daily-v1"
    monkeypatch.setenv(MORNING_SUBSET_ENV, "full")
    with pytest.raises(SystemExit):
        live.main([*accounts, "--days", "3"], out=lines.append, now=lambda: now)
    build, registry = _build(tmp_path)
    for night, subset in ((NIGHT, DAILY), (date(2026, 10, 25), None)):
        accelerated.morning_scorer(build, {"p1": "film_founder"}, subset)(
            "p1", night, "development", cast(Any, SimpleNamespace(status="promoted", checkpoint=registry.adapters["p1"])))
    table = live.report(tmp_path, {"seed": "p1", "today": "syn-c"}, lambda line: None)["seed"]
    assert table["scored_on"] == ["daily-v1", "full"]
