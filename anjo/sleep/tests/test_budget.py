# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The spend ledger (spec §12.1, Task 4): caps, all-or-none reserves, Sleep's claim on MIND's
Baseten balance, and one ledger shared through its file."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from anjo.sleep import budget
from anjo.sleep.budget import BudgetError, Ledger


def _ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "sleep" / budget.LEDGER_NAME)


def test_budget_refuses_over_cap(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    held = ledger.reserve("night-1", ("meta_usd",), 100.0)
    assert held is not None
    assert ledger.reserve("night-2", ("meta_usd",), 40.01) is None
    assert ledger.used("meta_usd") == pytest.approx(100.0)
    exact = ledger.reserve("night-2", ("meta_usd",), 40.0)
    assert exact is not None  # the cap itself is reachable
    assert ledger.remaining("meta_usd") == pytest.approx(0.0)
    assert ledger.reserve("night-3", ("meta_usd",), 0.01) is None


def test_multi_pool_reserve_is_all_or_none(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    assert ledger.reserve("judge nights", ("judge_usd",), 95.0) is not None
    before = ledger.path.read_text(encoding="utf-8")

    # Modal has room for 10 more, the judges' pool does not: neither pool is debited.
    assert ledger.reserve("a judge's night", ("modal_usd", "judge_usd"), 10.0) is None
    assert ledger.path.read_text(encoding="utf-8") == before
    assert ledger.used("modal_usd") == 0.0

    both = ledger.reserve("a judge's night", ("modal_usd", "judge_usd"), 5.0)
    assert both is not None
    assert (ledger.used("modal_usd"), ledger.used("judge_usd")) == (pytest.approx(5.0), pytest.approx(100.0))


def test_pools_reserved_together_share_a_unit(tmp_path: Path) -> None:
    with pytest.raises(BudgetError, match="share a unit"):
        _ledger(tmp_path).reserve("mixed", ("colab_cu", "judge_usd"), 1.0)


@pytest.mark.parametrize(
    ("pools", "est"),
    [((), 1.0), (("nope_usd",), 1.0), (("meta_usd", "meta_usd"), 1.0), (("meta_usd",), -1.0),
     (("meta_usd",), float("nan")), (("meta_usd",), 0.0), (("meta_usd",), True)],
)
def test_a_malformed_reserve_raises(tmp_path: Path, pools: tuple[str, ...], est: float) -> None:
    with pytest.raises(BudgetError):
        _ledger(tmp_path).reserve("job", pools, est)  # type: ignore[arg-type]


def test_settle_replaces_the_estimate_once(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    held = ledger.reserve("train", ("colab_cu",), 50.0)
    assert held is not None

    held.settle(62.5)  # an overrun is still recorded: the compute was spent

    assert ledger.used("colab_cu") == pytest.approx(62.5)
    with pytest.raises(BudgetError, match="settled once"):
        held.settle(1.0)
    with pytest.raises(BudgetError, match="no such reservation"):
        _ledger(tmp_path / "other").settle(held, 1.0)


def test_sleep_may_take_minds_unspent_baseten_balance_but_mind_keeps_its_own_cap(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)

    assert ledger.reserve("practice", ("baseten_sleep_usd",), 75.0) is not None  # past its own 60
    assert ledger.remaining("baseten_mind_usd") == pytest.approx(25.0)
    assert ledger.reserve("apd", ("baseten_mind_usd",), 25.01) is None
    assert ledger.reserve("apd", ("baseten_mind_usd",), 25.0) is not None
    assert ledger.reserve("one more", ("baseten_sleep_usd",), 0.01) is None

    fresh = _ledger(tmp_path / "fresh")
    assert fresh.remaining("baseten_mind_usd") == pytest.approx(40.0)
    assert fresh.reserve("apd", ("baseten_mind_usd",), 40.01) is None


def test_the_ledger_is_one_file_every_process_shares(tmp_path: Path) -> None:
    def take(_index: int) -> bool:
        return _ledger(tmp_path).reserve("serve window", ("modal_usd",), 10.0) is not None

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
        ledger.reserve("job", ("meta_usd",), 1.0)


def test_the_module_reserve_uses_the_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))

    assert budget.reserve("brief", ("vertex_usd",), 2.0) is not None
    assert (tmp_path / "sleep" / budget.LEDGER_NAME).exists()
    monkeypatch.setenv("ANJO_DATA_DIR", "")
    with pytest.raises(BudgetError, match="ANJO_DATA_DIR"):
        budget.reserve("brief", ("vertex_usd",), 2.0)


def test_the_caps_and_time_limits_are_section_12_1s() -> None:
    assert dict(budget.CAPS) == {
        "modal_usd": 500.0, "colab_cu": 300.0, "meta_usd": 140.0, "baseten_sleep_usd": 60.0,
        "baseten_mind_usd": 40.0, "vertex_usd": 50.0, "judge_usd": 100.0,
    }
    assert budget.UNITS["colab_cu"] == "cu" and budget.BASETEN_TOTAL == 100.0
    assert (budget.TRAINING_TIMEOUT_S, budget.EVAL_TIMEOUT_S) == (7200, 3600)
    assert (budget.SERVING_IDLE_S, budget.SERVING_WINDOW_S) == (600, 10800)
