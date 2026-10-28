# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""What generating APD's lives is expected to cost (MIND's data): one call priced from the measured
life, a life's calls counted from its own items by the gates' code path, so a life with six items
is expected to cost many times one with none, and a plan whose expected spend passes the pool's
cap says so. Every model is a fake: no network, no spend."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS
from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import core_splits, fakes, gates, gen_batch, generate, pairs, plan
from anjo.sleep.mind.apd.types import LIFE_DAYS
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.model_api import ChatResult

LIVE = ["--live", "--prices", "1.25,4.25,1.5,2.61", "--judge-prices", "1.25,4.25,0.75,3.75"]


class Counted:
    def __init__(self, inner: Any) -> None:
        self.inner, self.calls = inner, 0

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        self.calls += 1
        result: ChatResult = self.inner(model, messages, tools, timeout_s, temperature=temperature)
        return result


def test_a_lifes_expected_calls_are_the_gates_own_reads_of_its_items() -> None:
    # Arrange: a development life with three items, written by the fakes, its gates run again
    spec = next(s for s in generate.development_specs() if s.prevalence == 3)
    made = gen_batch.generate_life(spec, backends.fake_backend(FIREWORKS))
    reader, auditor = Counted(fakes.FakeGateReader()), Counted(fakes.FakeGateAuditor())

    # Act
    gates.check(made.life, made.problems, reader=reader, auditor=auditor, model=FIREWORKS.model)
    calls = plan.expected_calls(plan.surface_days(spec))

    # Assert: the plan counts exactly the reads and audits the gates made
    assert made.attempts == 1
    assert plan.surface_days(spec) == tuple(made.life.days.index(item.surface_day) for item in made.problems.items)
    assert (calls.writes, calls.reads, calls.muse, calls.retries) == (
        backends.GEN_WRITE_CALLS, reader.calls, auditor.calls, 0)
    assert backends.item_reads(LIFE_DAYS) == backends.GEN_READS_PER_ITEM  # the hold's bound per item


def test_a_lifes_expected_cost_grows_with_its_items() -> None:
    empty, three, six = (plan.expected_usd(plan.expected_calls((LIFE_DAYS - 1,) * n)) for n in (0, 3, 6))
    assert plan.CALL_USD == pytest.approx(plan.MEASURED_LIFE_USD / backends.GEN_WRITE_CALLS)
    assert empty == pytest.approx(plan.MEASURED_LIFE_USD + plan.CALL_USD)  # its measured calls, and the audit
    assert three == pytest.approx((20 + 3 * 35 + 4) * plan.CALL_USD)
    assert six == pytest.approx((20 + 6 * 35 + 7) * plan.CALL_USD)
    assert six > 10 * empty


def test_a_pairs_expected_calls_are_its_writes_its_three_items_reads_and_its_audits() -> None:
    assert pairs.PAIR_EXPECTED == backends.GenCalls(pairs.PAIR_CALLS.writes,
                                                    pairs.PAIR_ITEMS * backends.item_reads(pairs.LAST_DAY),
                                                    pairs.PAIR_CALLS.muse, retries=0)


def test_the_generate_plan_prints_each_lifes_range_and_flags_a_split_over_the_pool_cap(tmp_path: Path) -> None:
    # Arrange
    specs = core_splits.core_specs("core-train")
    each = [plan.expected_calls(plan.surface_days(spec)) for spec in specs]
    expected = sum(plan.expected_usd(calls) for calls in each)
    spark = sum(calls.spark for calls in each) * plan.CALL_USD
    lines: list[str] = []

    # Act
    assert cli.main([*LIVE, "--data", str(tmp_path), "generate", "--split", "core-train"], out=lines.append) == 0

    # Assert
    text = "\n".join(lines)
    low, high = min(plan.expected_usd(c) for c in each), max(plan.expected_usd(c) for c in each)
    assert f"Expected: about ${expected:,.2f} for 120 lives, one attempt each: ${low:,.2f} to ${high:,.2f} a life" in text
    assert spark > CAPS[FIREWORKS.pool]
    assert (f"Over the cap: about ${spark:,.2f} of it is on {FIREWORKS.pool}, more than its whole "
            f"${CAPS[FIREWORKS.pool]:,.2f} cap") in text
    assert list(tmp_path.iterdir()) == []


def test_a_plan_under_the_cap_flags_nothing(tmp_path: Path) -> None:
    lines: list[str] = []
    assert cli.main([*LIVE, "--data", str(tmp_path), "--only", "1", "generate"], out=lines.append) == 0
    assert not any(line.startswith("Over the cap") for line in lines)
