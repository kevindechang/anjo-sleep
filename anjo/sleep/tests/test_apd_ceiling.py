# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""APD's run-level spend ceiling (``--max-usd``, Task 17): a paid command sums what its own holds
have settled before each new unit and starts nothing more once that reaches the ceiling; other
runs' and Sleep's spend on the same pools never counts. The ``--live`` plan shows it beside the
holds, and each finished unit prints what the run has settled per life and per arm-day. Every
model is a fake: no network, no spend."""
from __future__ import annotations

from pathlib import Path

import pytest

from anjo.sleep.budget import Ledger
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import generate
from anjo.sleep.mind.apd.ceiling import RunLedger
from anjo.sleep.mind.apd.pilot import DEV_ARMS
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.night.holds import usd

LIVE = ["--live", "--prices", "1.25,4.25,1.5,2.61", "--judge-prices", "1.25,4.25,0.75,3.75"]
TINY = "0.0001"  # above nothing, below what one fake life or one fake M run settles
BAD = ("0", "-1", "nan", "inf", "twenty")


def _main(argv: list[str]) -> tuple[int, str]:
    lines: list[str] = []
    code = cli.main(argv, out=lines.append)
    return code, "\n".join(lines)


def _events(data: Path) -> list[dict[str, object]]:
    return list(read_records(data / "fake-budget.jsonl", PLAIN))


def _reserved(data: Path) -> list[str]:
    return [str(e["job"]) for e in _events(data) if e["op"] == "reserve"]


def _settled(data: Path, prefix: str) -> float:
    return sum(float(str(e["amount"])) for e in _events(data) if e["op"] == "settle" and str(e["job"]).startswith(prefix))


# ── Stopping ──────────────────────────────────────────────────────────────────


def test_run_stops_before_new_unit_at_ceiling(tmp_path: Path) -> None:
    data = tmp_path / "pilot"
    code, text = _main(["--fake", "--data", str(data), "--only", "2", "generate", "--max-usd", TINY])
    assert code == cli.CEILING, text
    assert "Ceiling: this run settled $" in text and "after 1 of 2 life(s) generated" in text
    assert len(DataDir(data).lives("development")) == 1
    second = generate.development_specs()[1]
    assert not any(job.startswith(f"apd:gen:development:{second.seed}:") for job in _reserved(data))  # no new hold
    assert Ledger(data / "fake-budget.jsonl").open_holds() == ()

    assert _main(["--fake", "--data", str(data), "--only", "2", "generate"])[0] == 0  # the second, under $20
    lives = sorted(DataDir(data).lives("development"), key=lambda life: life.id)
    code, text = _main(["--fake", "--data", str(data), "--parallel", "1", "run", "--max-usd", TINY])
    assert code == cli.CEILING, text
    assert "after 2 of 4 life-arm run(s)" in text  # the first life's B ($0) and M; then nothing new
    assert "serving Glimmer" in text and "Stopped" not in text
    runs = {(life.id, arm) for life in lives for arm in DEV_ARMS if DataDir(data).run(life.id, arm) is not None}
    assert runs == {(lives[0].id, "B"), (lives[0].id, "M")}
    assert not any(f"run:{lives[1].id}:" in job for job in _reserved(data))
    assert Ledger(data / "fake-budget.jsonl").open_holds() == ()


def test_ceiling_counts_only_this_runs_settlements(tmp_path: Path) -> None:
    path = tmp_path / "budget.jsonl"
    mine, other = RunLedger(path), RunLedger(path)
    held = mine.reserve("apd:gen:a:spark:a1", ("fireworks_mind_usd",), usd(1.0))
    theirs = other.reserve("apd:gen:b:spark:a1", ("fireworks_mind_usd",), usd(1.0))
    session = mine.reserve("apd:colab:s1:a1", ("colab_cu",), usd(10.0))  # compute units, not dollars
    assert held is not None and theirs is not None and session is not None
    held.settle(0.4)
    theirs.settle(0.7)
    session.settle(3.0)
    assert mine.reserve("apd:gen:c:spark:a1", ("fireworks_mind_usd",), usd(1.0)) is not None  # open: not settled
    released = mine.reserve("apd:judge:muse:d:a1", ("meta_usd",), usd(1.0))
    assert released is not None
    released.release()
    sleep = Ledger(path).reserve("night:sleep:a1", ("baseten_sleep_usd",), usd(5.0))
    assert sleep is not None
    sleep.settle(5.0)
    assert mine.run_usd() == pytest.approx(0.4) and other.run_usd() == pytest.approx(0.7)
    assert Ledger(path).used("fireworks_mind_usd") == pytest.approx(2.1)  # the pool's total holds everyone's

    data = tmp_path / "pilot"  # earlier runs and Sleep spent far past the ceiling on the same ledger
    data.mkdir()
    earlier = Ledger(data / "fake-budget.jsonl")
    for job, pool, amount in (("apd:gen:earlier:spark:a1", "fireworks_mind_usd", 50.0),
                              ("apd:judge:muse:earlier:a1", "meta_usd", 5.0),
                              ("night:sleep:2026-10-28:a1", "baseten_sleep_usd", 30.0)):
        found = earlier.reserve(job, (pool,), usd(amount))  # type: ignore[arg-type]
        assert found is not None
        found.settle(amount)
    code, text = _main(["--fake", "--data", str(data), "--only", "1", "generate", "--max-usd", "0.01"])
    assert code == 0 and "Ceiling:" not in text, text
    assert 0 < _settled(data, "apd:gen:development:") < 0.01


# ── The plan ──────────────────────────────────────────────────────────────────


def test_live_plan_shows_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_backend(*args: object, **kwargs: object) -> None:
        raise AssertionError("a plan builds no live backend")

    monkeypatch.setattr(backends, "live_backend", no_backend)
    monkeypatch.setattr("anjo.sleep.budget.default_ledger", no_backend)
    for command in (["generate"], ["run"], ["timing"], ["judge"], ["run", "--max-usd", "5"]):
        code, text = _main([*LIVE, "--data", str(tmp_path), *command])
        assert code == 0 and text.endswith("Nothing has run. Add --yes to start it."), text
        lines = text.splitlines()
        ceiling = next(n for n, line in enumerate(lines) if line.startswith("Ceiling: "))
        assert ceiling == max(n for n, line in enumerate(lines) if line.startswith("Hold on ")) + 1  # beside the holds
        amount = "$5.00" if "--max-usd" in command else "$20.00"
        assert lines[ceiling].startswith(f"Ceiling: {amount} settled by this run (--max-usd)"), lines[ceiling]
        assert "Note: the ceiling stops new work, not a unit in flight" in lines[ceiling + 1]
        if command[0] == "run":
            assert "up to 8 at once" in lines[ceiling + 1]
    code, text = _main([*LIVE, "--data", str(tmp_path), "intervene"])
    assert code == 0 and "Ceiling" not in text  # only generate, run, timing and judge take one
    for bad in BAD:
        with pytest.raises(SystemExit):
            cli.main([*LIVE, "--data", str(tmp_path), "generate", "--max-usd", bad], out=lambda _line: None)
    assert list(tmp_path.iterdir()) == []  # nothing written, nothing held


# ── What a unit costs ─────────────────────────────────────────────────────────


def test_measured_cost_per_unit_printed(tmp_path: Path) -> None:
    data = tmp_path / "pilot"
    code, text = _main(["--fake", "--data", str(data), "--only", "2", "generate"])
    assert code == 0, text
    lines = [line for line in text.splitlines() if "per life generated" in line]
    assert len(lines) == 2 and "for 1 life generated" in lines[0] and "for 2 lives generated" in lines[1]
    spent = _settled(data, "apd:gen:")
    assert spent > 0 and f"${spent:.4f} settled" in lines[-1] and f"${spent / 2:.4f} per life generated" in lines[-1]

    code, text = _main(["--fake", "--data", str(data), "--parallel", "1", "run"])
    assert code == 0, text
    lines = [line for line in text.splitlines() if "per arm-day" in line]
    days = 2 * sum(len(life.days) for life in DataDir(data).lives("development"))  # B and M on each life
    ran = _settled(data, "apd:run:")
    assert len(lines) == 4 and "for 4 life-arm run(s)" in lines[-1] and f"{days} arm-days" in lines[-1]
    assert ran > 0 and f"${ran:.4f} settled" in lines[-1] and f"${ran / days:.5f} per arm-day" in lines[-1]
