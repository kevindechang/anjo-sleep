# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The morning backfill (``tools.morning_backfill``): a promoted night whose morning failed is scored
afterwards on its own adapter, pinned while a later one keeps serving, logged ``backfilled`` in the
night's log and read by the live comparison's report. Fakes only: no model, judge or GPU call."""
from __future__ import annotations

import fcntl
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import Ledger
from anjo.sleep.checkpoints import CheckpointRegistry, ref_record
from anjo.sleep.night.deps import Prices
from anjo.sleep.probes.arms import ArmDeps
from anjo.sleep.probes.morning import Morning
from anjo.sleep.probes.score import Verdict
from anjo.sleep.tests.sleep_fakes import FakeContexts, FakeLogprobs, FakeModel, StoreSpy, probe, taste
from anjo.sleep.tools import morning_backfill as backfill_tool
from anjo.sleep.tools import sleep_objective_live as live
from anjo.sleep.types import CheckpointRef

USER = "syn-seed"
NIGHTS = (date(2026, 10, 24), date(2026, 10, 25), date(2026, 10, 26))
PRICES = Prices(spark_in_per_m=1.25, spark_out_per_m=4.25, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.96)


def _ckpt(n: int) -> CheckpointRef:
    return CheckpointRef(f"p-abc-n{n}", USER, NIGHTS[n - 1], f"uri://{n}", None if n == 1 else f"p-abc-n{n - 1}")


class Judge:
    """Deterministic: the longer reply wins. ``fail`` raises on every call (a judge that is down)."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.calls, self.fail = 0, fail

    def verdict(self, p: Any, first: str, second: str) -> Verdict:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return "A" if len(first) > len(second) else "B" if len(second) > len(first) else "tie"


class World:
    """One synthetic account that promoted three nights, each morning failed; n3 serves."""

    def __init__(self, tmp_path: Path, *, synthetic: bool = True, judge: Judge | None = None) -> None:
        self.root, self.synthetic = tmp_path / "results", synthetic
        self.registry = CheckpointRegistry(tmp_path / "checkpoints")
        for n in (1, 2, 3):
            self.registry.promote(USER, _ckpt(n))
        self.model, self.judge = FakeModel(), judge or Judge()
        self.ledger = Ledger(tmp_path / "budget.jsonl")
        self.probes = [probe("q1", "new_work", person=USER), probe("q2", "vocab", person=USER),
                       probe("q3", "tone_hold", person=USER), taste("t1", person=USER)]
        for n, night in enumerate(NIGHTS, start=1):
            self.night(night, {"status": "promoted", "checkpoint": ref_record(_ckpt(n)), "accelerated": True,
                               "label": "development", "cost_usd": 1.0})

    def night(self, night: date, summary: dict[str, Any], *, morning: str = "failed") -> Path:
        folder = self.root / USER / "nights" / night.isoformat()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        steps = [{"step": "train", "status": "ok", "receipt": {"train_minutes": 3.0}},
                 {"step": "morning", "status": morning, "uses": "new", "system": None}]
        (folder / "log.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps), encoding="utf-8")
        return folder

    def morning(self, user: str, persona: str, night: date) -> Morning:
        deps = ArmDeps(model=self.model, adapters=self.registry, contexts=FakeContexts(),
                       evidence=lambda uid: StoreSpy([]), logprobs=FakeLogprobs(lambda adapter, messages: "A"),
                       donors={}, until=night)
        return Morning(arms=deps, sealed=lambda uid: self.probes, judge=self.judge, prices=PRICES, root=self.root)

    @property
    def ports(self) -> backfill_tool.Ports:
        return backfill_tool.Ports(morning=self.morning, is_synthetic=lambda user: self.synthetic,
                                   logs=lambda user: self.root / user / "nights", registry=self.registry,
                                   ledger=self.ledger)

    def run(self, night: date, *extra: str) -> tuple[int, list[str]]:
        said: list[str] = []
        code = backfill_tool.main(["--user", USER, "--persona", "film_founder", "--night", night.isoformat(), *extra],
                                  ports=self.ports, out=said.append)
        return code, said

    def steps(self, night: date) -> list[dict[str, Any]]:
        path = self.root / USER / "nights" / night.isoformat() / "log.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_the_plan_names_the_nights_adapter_its_holds_and_cost_and_runs_nothing(tmp_path: Path) -> None:
    world = World(tmp_path)
    code, said = world.run(NIGHTS[0])
    text = "\n".join(said)
    assert code == 0 and "p-abc-n1, the adapter the night promoted, pinned; p-abc-n3 keeps serving" in text
    assert "the full set: 4 probes, 24 replies" in text  # an accelerated night's morning asked every arm
    assert "holds: the judge $" in text and "expected: $" in text and "night-01.json" in text
    assert world.model.calls == [] and world.judge.calls == 0 and not (world.root / USER / "scores").exists()
    assert [s["status"] for s in world.steps(NIGHTS[0]) if s["step"] == "morning"] == ["failed"]  # nothing written


def test_go_scores_the_nights_own_adapter_while_a_later_one_keeps_serving(tmp_path: Path) -> None:
    world = World(tmp_path)
    code, said = world.run(NIGHTS[0], "--go")
    assert code == 0 and said[-1].startswith("scored: ")
    personal = {c["adapter"].id for c in world.model.calls if c["adapter"] is not None}
    assert personal == {"p-abc-n1"}  # arms 3 and 4 on the night's own adapter
    assert world.registry.last_promoted(USER) == _ckpt(3)  # what serves never moved
    record = json.loads((world.root / USER / "scores" / "night-01.json").read_text(encoding="utf-8"))
    assert (record["night"], record["checkpoint"], record["pinned"]) == ("2026-10-24", "p-abc-n1", True)
    assert "subset" not in record and record["arms"]["3"]["overall"]["score"] is not None
    last = [s for s in world.steps(NIGHTS[0]) if s["step"] == "morning"][-1]
    assert (last["status"], last["backfilled"], last["checkpoint"]) == ("ok", True, "p-abc-n1")
    assert last["judge_calls"] == world.judge.calls > 0
    assert world.run(NIGHTS[0], "--go")[1] == ["refused: the morning of 2026-10-24 already succeeded"]


def test_the_live_comparisons_report_reads_backfilled_mornings_in_night_order(tmp_path: Path) -> None:
    world = World(tmp_path)
    assert world.run(NIGHTS[2], "--go")[0] == 0  # the last night first: its file is night-01.json
    assert world.run(NIGHTS[0], "--go")[0] == 0
    said: list[str] = []
    table = live.report(world.root, {"seed": USER, "today": "syn-today"}, said.append)["seed"]  # seed runs with its control
    files = {json.loads(p.read_text())["night"]: json.loads(p.read_text())
             for p in (world.root / USER / "scores").glob("night-*.json")}
    assert [s["night"] for s in live.scores(world.root, USER)] == ["2026-10-24", "2026-10-26"]
    assert table["arm3_by_night"] == [files[n]["arms"]["3"]["overall"]["score"] for n in ("2026-10-24", "2026-10-26")]
    assert table["arm3_mean"] is not None and (table["nights"], table["promoted"]) == (3, 3)


@pytest.mark.parametrize("rule", ["not_promoted", "morning_ok", "not_synthetic", "lock_held", "scored"])
def test_each_refusal_comes_before_any_call_or_write(tmp_path: Path, rule: str) -> None:
    world = World(tmp_path, synthetic=rule != "not_synthetic")
    night = NIGHTS[0]
    if rule == "not_promoted":
        world.night(night, {"status": "rejected", "checkpoint": None, "accelerated": True})
    elif rule == "morning_ok":
        world.night(night, {"status": "promoted", "checkpoint": ref_record(_ckpt(1)), "accelerated": True},
                    morning="ok")
    elif rule == "scored":
        (world.root / USER / "scores").mkdir(parents=True)
        (world.root / USER / "scores" / "night-01.json").write_text(json.dumps({"night": night.isoformat()}))
    before = world.steps(night)
    with open(tmp_path / "night.lock", "a+b") as other:  # another run's lock, when the rule says so
        if rule == "lock_held":
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, said = world.run(night, "--go")
    expected = {"not_promoted": "wasn't promoted (rejected)", "morning_ok": "already succeeded",
                "not_synthetic": "isn't registered synthetic", "lock_held": "night lock is held",
                "scored": "already scored"}[rule]
    assert code == 1 and said[-1].startswith("refused: ") and expected in said[-1]
    assert world.model.calls == [] and world.judge.calls == 0 and world.steps(night) == before
    if rule == "not_synthetic":
        assert world.run(night, "--real")[0] == 0  # a real person's night only when asked for by name


def test_a_night_started_after_the_plan_holds_off_the_run(tmp_path: Path) -> None:
    world = World(tmp_path)
    chosen = backfill_tool.plan(USER, "film_founder", NIGHTS[0], world.ports)
    with open(tmp_path / "night.lock", "a+b") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(backfill_tool.Refused, match="another night run"):
            backfill_tool.backfill(chosen, world.ports)
    assert world.model.calls == [] and not (world.root / USER / "scores").exists()


def test_a_backfill_that_fails_is_logged_backfilled_and_raised(tmp_path: Path) -> None:
    world = World(tmp_path, judge=Judge(fail=RuntimeError("the judge is down")))
    chosen = backfill_tool.plan(USER, "film_founder", NIGHTS[1], world.ports)
    with pytest.raises(RuntimeError):
        backfill_tool.backfill(chosen, world.ports)
    last = [s for s in world.steps(NIGHTS[1]) if s["step"] == "morning"][-1]
    assert (last["status"], last["backfilled"], last["why"]) == ("failed", True, "morning failed: RuntimeError")
    assert not (world.root / USER / "scores").exists()


@pytest.mark.parametrize(("accelerated", "flag", "subset"), [(True, (), None), (True, ("--subset", "daily-v1"), "daily-v1"),
                                                             (False, (), "daily-v1")])
def test_the_probe_set_is_the_nights_own_unless_a_subset_is_named(
    tmp_path: Path, accelerated: bool, flag: tuple[str, ...], subset: str | None,
) -> None:
    world = World(tmp_path)
    world.night(NIGHTS[0], {"status": "promoted", "checkpoint": ref_record(_ckpt(1)), "accelerated": accelerated})
    assert world.run(NIGHTS[0], "--go", *flag)[0] == 0
    record = json.loads((world.root / USER / "scores" / "night-01.json").read_text(encoding="utf-8"))
    assert record.get("subset") == subset
    last = [s for s in world.steps(NIGHTS[0]) if s["step"] == "morning"][-1]
    assert last.get("subset") == subset
