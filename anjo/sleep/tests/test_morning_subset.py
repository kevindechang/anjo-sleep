# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The morning's probe set by setting (``ANJO_SLEEP_MORNING_SUBSET``): unset, the full morning as
before; ``daily-v1``, the daily subset, asked by the accelerated runner's mornings, shown in the live
comparison's plan, passed through to each arm and recorded in the morning step and the report; any
other value stops before anything runs. Fakes only."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from anjo.sleep.harness import HARNESS_ENV
from anjo.sleep.night.daily import morning_receipt
from anjo.sleep.night.deps import Prices
from anjo.sleep.probes.arms import ArmDeps
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.probes.morning import (
    ARM_ORDER,
    ARMS_EXTRA_ENV,
    COMMITTED_ARMS,
    DAILY,
    HARNESS_ARM,
    MORNING_PROBES,
    MORNING_SUBSET_ENV,
    Morning,
    MorningResult,
    full_arms,
    morning_subset,
    run_morning,
)
from anjo.sleep.probes.score import Verdict
from anjo.sleep.sim import accelerated
from anjo.sleep.tests.accelerated_rig import FakeAccelerated
from anjo.sleep.tests.night_fakes import step
from anjo.sleep.tests.sleep_fakes import FakeContexts, FakeLogprobs, FakeModel, FakeRegistry, StoreSpy, ckpt, probe, taste
from anjo.sleep.tools import sleep_objective_live as live

NIGHT = date(2026, 10, 24)
START = date(2026, 10, 20)  # before the accelerated rig's wall clock: a life dated in the past
FOUNDER = load_person_file(Path(__file__).parents[1] / "personas" / "founder.json").visible
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


def test_the_full_morning_asks_the_committed_arms_until_the_extra_setting_names_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A full score from before 2A and 4M stays the same run: the extra arms join only on the setting,
    # and the reported-only 2H arm joins unless its switch is explicitly off.
    assert full_arms({HARNESS_ENV: "0"}) == COMMITTED_ARMS and "2A" not in full_arms({HARNESS_ENV: "0"})
    assert full_arms({}) == (*COMMITTED_ARMS, HARNESS_ARM) and "2A" not in full_arms({}) and "4M" not in full_arms({})
    assert full_arms({ARMS_EXTRA_ENV: "1"}) == (*ARM_ORDER, HARNESS_ARM) and {"2A", "4M"} <= set(full_arms({ARMS_EXTRA_ENV: "1"}))
    assert full_arms({ARMS_EXTRA_ENV: "0"}) == (*COMMITTED_ARMS, HARNESS_ARM)
    assert full_arms({ARMS_EXTRA_ENV: "1", HARNESS_ENV: "0"}) == ARM_ORDER
    monkeypatch.setenv(ARMS_EXTRA_ENV, "1")
    monkeypatch.setenv(HARNESS_ENV, "0")
    build, registry = _build(tmp_path)
    done = run_morning("p1", NIGHT, registry.adapters["p1"], morning=build("p1", "film_founder", NIGHT))
    record = json.loads(done.path.read_text(encoding="utf-8"))
    assert set(record["arms"]) == {"1", "2C", "2A", "2F", "3", "4", "4M"}  # 3x has no donor here
    assert {"4M_vs_2A", "4M_vs_4"} <= set(record["pairs"])


@pytest.mark.parametrize(("value", "label"), [("daily-v2", "development"), ("daily-v1", "final")])
def test_an_unknown_subset_or_a_final_run_on_one_stops_the_runner_before_it_starts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], value: str, label: str,
) -> None:
    monkeypatch.setenv(MORNING_SUBSET_ENV, value)
    with pytest.raises(SystemExit):
        accelerated.main([*REAL, "--label", label])
    assert MORNING_SUBSET_ENV in capsys.readouterr().err


@pytest.mark.parametrize("subset", [DAILY, None])
def test_the_runners_morning_asks_the_set_named_and_its_step_records_it(
    tmp_path: Path, subset: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(HARNESS_ENV, "0")  # the full set's registered six, before the reported-only 2H
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
    assert {name: list(scored["by_night"]) for name, scored in table["scores"].items()} == {
        "daily-v1": ["2026-10-24"], "full": ["2026-10-25"]}


@pytest.mark.parametrize("subset", [DAILY, None])
def test_each_scored_night_records_its_set_in_its_summary_and_its_morning_step_even_failed(
    tmp_path: Path, subset: Any,
) -> None:
    fake = FakeAccelerated(tmp_path)
    scored = accelerated.morning_scorer(lambda uid, persona, night: fake.morning(uid), {"u1": "founder"}, subset)

    def score(uid: str, night: date, label: str, result: Any) -> object:
        if night == START:
            raise TimeoutError("the judge timed out")
        return scored(uid, night, label, result)

    results = accelerated.run_accelerated("u1", FOUNDER, 2, START, rig=replace(fake.rig(), score=score, subset=subset))
    name = "full" if subset is None else subset.name
    summaries = [json.loads((r.log_path.parent / "summary.json").read_text(encoding="utf-8")) for r in results]
    assert [s[MORNING_PROBES] for s in summaries] == [name, name]
    failed, done = (step(r.log_path, "morning") for r in results)
    assert (failed["status"], done["status"]) == ("failed", "ok")
    assert failed.get("subset") == done.get("subset") == (None if subset is None else name)
