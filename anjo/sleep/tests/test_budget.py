# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The spend ledger (spec §12.1, Task 4): worst-case holds, caps, all-or-none reserves, overages
that block a pool, idempotent job keys, Sleep's claim on MIND's Baseten balance, and one ledger
shared through its file."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep import budget
from anjo.sleep.budget import Bound, BudgetError, Ledger

HOUR = 3600.0


def _ledger(tmp_path: Path, **kwargs: Any) -> Ledger:
    return Ledger(tmp_path / "sleep" / budget.LEDGER_NAME, **kwargs)


def _bound(amount: float) -> Bound:
    """A job whose worst case is ``amount``: an hour's hard timeout at that rate."""
    return Bound(rate_per_hour=amount, timeout_s=HOUR)


def test_budget_refuses_over_cap(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    held = ledger.reserve("night-1", ("meta_usd",), _bound(100.0))
    assert held is not None
    assert ledger.reserve("night-2", ("meta_usd",), _bound(40.01)) is None
    assert ledger.used("meta_usd") == pytest.approx(100.0)
    exact = ledger.reserve("night-2", ("meta_usd",), _bound(40.0))
    assert exact is not None  # the cap itself is reachable
    assert ledger.remaining("meta_usd") == pytest.approx(0.0)
    assert ledger.reserve("night-3", ("meta_usd",), _bound(0.01)) is None


def test_multi_pool_reserve_is_all_or_none(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    assert ledger.reserve("judge nights", ("judge_usd",), _bound(95.0)) is not None
    before = ledger.path.read_text(encoding="utf-8")

    # Modal has room for 10 more, the judges' pool does not: neither pool is debited.
    assert ledger.reserve("a judge's night", ("modal_usd", "judge_usd"), _bound(10.0)) is None
    assert ledger.path.read_text(encoding="utf-8") == before
    assert ledger.used("modal_usd") == 0.0

    both = ledger.reserve("a judge's night", ("modal_usd", "judge_usd"), _bound(5.0))
    assert both is not None
    assert (ledger.used("modal_usd"), ledger.used("judge_usd")) == (pytest.approx(5.0), pytest.approx(100.0))


def test_pools_reserved_together_share_a_unit(tmp_path: Path) -> None:
    with pytest.raises(BudgetError, match="share a unit"):
        _ledger(tmp_path).reserve("mixed", ("colab_cu", "judge_usd"), _bound(1.0))


@pytest.mark.parametrize(
    "pools", [(), ("nope_usd",), ("meta_usd", "meta_usd")],
)
def test_a_malformed_reserve_raises(tmp_path: Path, pools: tuple[str, ...]) -> None:
    with pytest.raises(BudgetError):
        _ledger(tmp_path).reserve("job", pools, _bound(1.0))  # type: ignore[arg-type]
    with pytest.raises(BudgetError, match="names its job"):
        _ledger(tmp_path).reserve(" ", ("meta_usd",), _bound(1.0))


@pytest.mark.parametrize(
    ("rate", "timeout"),
    [(-1.0, HOUR), (float("nan"), HOUR), (0.0, HOUR), (True, HOUR), (1.0, 0.0), (1.0, float("inf"))],
)
def test_a_bound_is_a_positive_rate_over_a_finite_hard_timeout(rate: float, timeout: float) -> None:
    with pytest.raises(BudgetError):
        Bound(rate_per_hour=rate, timeout_s=timeout)


def test_a_hold_is_the_rate_over_the_whole_hard_timeout(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    ledger.reserve("train", ("colab_cu",), Bound(rate_per_hour=12.0, timeout_s=budget.TRAINING_TIMEOUT_S))

    assert ledger.used("colab_cu") == pytest.approx(24.0)


def test_settle_replaces_the_hold_once(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    held = ledger.reserve("train", ("colab_cu",), _bound(50.0))
    assert held is not None

    held.settle(37.5)

    assert ledger.used("colab_cu") == pytest.approx(37.5)
    with pytest.raises(BudgetError, match="settled"):
        held.settle(1.0)
    with pytest.raises(BudgetError, match="no hold"):
        _ledger(tmp_path / "other").settle(held, 1.0)
    for bad in (-1.0, float("nan"), float("inf")):
        fresh = ledger.reserve(f"bad {bad}", ("colab_cu",), _bound(1.0))
        assert fresh is not None
        with pytest.raises(BudgetError):
            fresh.settle(bad)


def test_a_settlement_above_its_hold_blocks_the_pool_until_it_is_cleared(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    held = ledger.reserve("serve", ("modal_usd",), _bound(1.0))
    assert held is not None

    held.settle(900.0)  # the money is spent: recorded in full, past the $500 cap

    assert ledger.used("modal_usd") == pytest.approx(900.0)
    assert ledger.remaining("modal_usd") == 0.0
    assert ledger.reserve("next", ("modal_usd",), _bound(0.01)) is None
    assert ledger.overages() == ("serve",)

    small = ledger.reserve("train", ("colab_cu",), _bound(10.0))
    assert small is not None
    small.settle(12.0)  # inside the cap, but above its own hold
    assert ledger.reserve("eval", ("colab_cu",), _bound(1.0)) is None
    assert ledger.reserve("judge", ("vertex_usd",), _bound(1.0)) is not None  # other pools go on
    ledger.clear("train")
    assert ledger.reserve("eval", ("colab_cu",), _bound(1.0)) is not None
    assert ledger.used("colab_cu") == pytest.approx(13.0)  # the overage still counts
    with pytest.raises(BudgetError, match="no overage"):
        ledger.clear("eval")


def test_a_job_key_is_one_hold_so_a_crashed_job_retries_into_it(tmp_path: Path) -> None:
    now = [datetime(2026, 10, 28, 2, 0, tzinfo=timezone.utc)]
    bound = Bound(rate_per_hour=10.0, timeout_s=budget.TRAINING_TIMEOUT_S)
    first = _ledger(tmp_path, clock=lambda: now[0]).reserve("p-1a2b3c4d-n3:train", ("colab_cu",), bound)
    assert first is not None
    del first  # the job dies without settling

    now[0] += timedelta(hours=5)  # long past its hard timeout: still spent, never freed
    retry = _ledger(tmp_path, clock=lambda: now[0])
    assert retry.used("colab_cu") == pytest.approx(20.0)
    again = retry.reserve("p-1a2b3c4d-n3:train", ("colab_cu",), bound)
    assert again is not None and again.job == "p-1a2b3c4d-n3:train"
    assert retry.used("colab_cu") == pytest.approx(20.0)  # the same hold, not a second one
    with pytest.raises(BudgetError, match="another bound"):
        retry.reserve("p-1a2b3c4d-n3:train", ("colab_cu",), _bound(5.0))

    again.settle(14.0)  # both attempts' compute, settled once under the key
    assert retry.used("colab_cu") == pytest.approx(14.0)
    with pytest.raises(BudgetError, match="ended"):
        retry.reserve("p-1a2b3c4d-n3:train", ("colab_cu",), bound)
    lines = retry.path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["op"] for line in lines] == ["reserve", "settle"]


def test_release_frees_a_hold_that_spent_nothing(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    held = ledger.reserve("judge night", ("meta_usd", "judge_usd"), _bound(30.0))
    assert held is not None

    held.release()

    assert (ledger.used("meta_usd"), ledger.used("judge_usd")) == (0.0, 0.0)
    with pytest.raises(BudgetError, match="released"):
        ledger.release("judge night")
    with pytest.raises(BudgetError, match="released"):
        held.settle(1.0)
    with pytest.raises(BudgetError, match="no hold"):
        ledger.release("never held")


def test_sleep_may_take_minds_unspent_baseten_balance_but_mind_keeps_its_own_cap(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)

    assert ledger.reserve("practice", ("baseten_sleep_usd",), _bound(75.0)) is not None  # past its own 60
    assert ledger.remaining("baseten_mind_usd") == pytest.approx(25.0)
    assert ledger.reserve("apd", ("baseten_mind_usd",), _bound(25.01)) is None
    assert ledger.reserve("apd", ("baseten_mind_usd",), _bound(25.0)) is not None
    assert ledger.reserve("one more", ("baseten_sleep_usd",), _bound(0.01)) is None

    fresh = _ledger(tmp_path / "fresh")
    assert fresh.remaining("baseten_mind_usd") == pytest.approx(40.0)
    assert fresh.reserve("apd", ("baseten_mind_usd",), _bound(40.01)) is None


def test_the_ledger_is_one_file_every_process_shares(tmp_path: Path) -> None:
    def take(index: int) -> bool:
        return _ledger(tmp_path).reserve(f"serve window {index}", ("modal_usd",), _bound(10.0)) is not None

    with ThreadPoolExecutor(max_workers=12) as pool:
        held = list(pool.map(take, range(60)))  # a raise in any worker re-raises here

    assert held.count(True) == 50  # $500 of $10 windows; the other ten were refused
    assert _ledger(tmp_path).used("modal_usd") == pytest.approx(500.0)
    lines = (tmp_path / "sleep" / budget.LEDGER_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 50 and all(json.loads(line)["op"] == "reserve" for line in lines)


def test_an_unreadable_ledger_raises_rather_than_reads_as_empty(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.path.parent.mkdir(parents=True)
    ledger.path.write_text("not json\n", encoding="utf-8")

    with pytest.raises(BudgetError, match="line 1"):
        ledger.reserve("job", ("meta_usd",), _bound(1.0))


_RESERVE = {
    "op": "reserve", "job": "j", "pools": ["meta_usd"], "amount": 1.0, "rate_per_hour": 1.0,
    "timeout_s": HOUR, "at": "2026-10-28T02:00:00+00:00",
}


@pytest.mark.parametrize("events", [
    [{**_RESERVE, "amount": -1.0, "rate_per_hour": -1.0}],
    [{**_RESERVE, "amount": float("nan")}],
    [{**_RESERVE, "amount": True}],
    [{**_RESERVE, "amount": 0.5}],  # not the rate over the timeout
    [{**_RESERVE, "pools": ["nope_usd"]}],
    [{**_RESERVE, "pools": ["meta_usd", "colab_cu"]}],
    [{**_RESERVE, "at": "yesterday"}],
    [{**_RESERVE, "extra": 1}],
    [{key: value for key, value in _RESERVE.items() if key != "timeout_s"}],
    [_RESERVE, _RESERVE],
    [{"op": "settle", "job": "j", "amount": 1.0, "at": _RESERVE["at"]}],
    [_RESERVE, {"op": "settle", "job": "j", "amount": -2.0, "at": _RESERVE["at"]}],
    [_RESERVE, {"op": "settle", "job": "j", "amount": float("inf"), "at": _RESERVE["at"]}],
    [_RESERVE, {"op": "release", "job": "j", "at": _RESERVE["at"]}, {"op": "release", "job": "j", "at": _RESERVE["at"]}],
    [_RESERVE, {"op": "clear", "job": "j", "at": _RESERVE["at"]}],
    [_RESERVE, {"op": "refund", "job": "j", "at": _RESERVE["at"]}],
])
def test_every_event_is_checked_on_read(tmp_path: Path, events: list[dict[str, Any]]) -> None:
    ledger = _ledger(tmp_path)
    ledger.path.parent.mkdir(parents=True)
    ledger.path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    with pytest.raises(BudgetError, match="line"):
        ledger.used("meta_usd")


def test_a_torn_last_line_is_left_out_and_cut_before_the_next_append(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    assert ledger.reserve("first", ("meta_usd",), _bound(3.0)) is not None
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write('{"op":"reserve","job":"torn"')  # a write killed mid-line

    assert ledger.used("meta_usd") == pytest.approx(3.0)
    assert ledger.reserve("second", ("meta_usd",), _bound(2.0)) is not None

    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["job"] for line in lines] == ["first", "second"]
    assert ledger.used("meta_usd") == pytest.approx(5.0)


def test_the_module_reserve_uses_the_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))

    assert budget.reserve("brief", ("vertex_usd",), _bound(2.0)) is not None
    assert (tmp_path / "sleep" / budget.LEDGER_NAME).exists()
    budget.release("brief")
    assert budget.default_ledger().used("vertex_usd") == 0.0
    monkeypatch.setenv("ANJO_DATA_DIR", "")
    with pytest.raises(BudgetError, match="ANJO_DATA_DIR"):
        budget.reserve("brief", ("vertex_usd",), _bound(2.0))


def test_the_caps_and_time_limits_are_section_12_1s() -> None:
    assert dict(budget.CAPS) == {
        "modal_usd": 500.0, "colab_cu": 100.0, "meta_usd": 140.0, "baseten_sleep_usd": 60.0,
        "baseten_mind_usd": 40.0, "fireworks_mind_usd": 90.0, "fireworks_sleep_usd": 20.0, "vertex_usd": 50.0,
        "vertex_mind_usd": 150.0, "judge_usd": 100.0,
    }
    assert budget.UNITS["colab_cu"] == "cu" and budget.BASETEN_TOTAL == 100.0
    assert (budget.TRAINING_TIMEOUT_S, budget.EVAL_TIMEOUT_S) == (7200, 3600)
    assert (budget.SERVING_IDLE_S, budget.SERVING_WINDOW_S) == (600, 10800)
