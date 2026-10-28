# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The safety gate's switch (``ANJO_SLEEP_SAFETY_GATE``, off on the hackathon trunk): off, the night sends
no case to any model, still runs the floor, logs the gate ``skipped`` and promotes past it; on, the gate
is today's. Every page shows a skipped gate as skipped, never as passed. Fakes and placeholders only."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.demo.panel import night_record
from anjo.sleep.gates import GateThresholds
from anjo.sleep.gates.safety import SKIPPED_REASON, SafetyCases, SafetyGate
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night import cost
from anjo.sleep.night.nightlog import gate_record
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.ownership.changes import night_views, what_changed
from anjo.sleep.reports.results import Night, gate_cell
from anjo.sleep.tests.night_fakes import NIGHT, World, step, summary
from anjo.sleep.types import CheckpointRef, GateResult

CKPT = CheckpointRef("p-aaaaaaaaaaaa-n1", "p1", date(2026, 10, 28), "modal-volume://x", None)
LIMITS = GateThresholds(general_points=2.0, collapse_points=5.0, safety_regressions=0)
PLACEHOLDER = "PLACEHOLDER-TRIP"


class Chat:
    """The checkpoint trips on case ``trip``; every call is recorded."""

    def __init__(self, trip: int | None = None) -> None:
        self.trip = trip
        self.calls: list[tuple[str | None, int | None]] = []

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float,
                 seed: int | None = None) -> ChatResult:
        self.calls.append((None if adapter is None else adapter.id, seed))
        index = int(messages[-1]["content"].split()[-1])
        text = PLACEHOLDER if adapter is not None and index == self.trip else "a neutral reply"
        return ChatResult(text=text, tool_calls=(), model="m", input_tokens=1, output_tokens=1, finish_reason="stop")


class Floor:
    def __init__(self, failed: tuple[str, ...] = ()) -> None:
        self.failed = failed
        self.runs = 0

    def __call__(self) -> tuple[str, ...]:
        self.runs += 1
        return self.failed


def _cases(floor: Floor) -> SafetyCases:
    return SafetyCases(prompts=[f"neutral case {i}" for i in range(12)],
                       flag=lambda reply: "placeholder" if reply == PLACEHOLDER else None, floor=floor)


class Gates:
    """General and collapse as fakes (``failing`` fail), safety the real gate, switched by ``safety_on``."""

    def __init__(self, *, safety_on: bool, failing: frozenset[str] = frozenset(), trip: int | None = None) -> None:
        self.chat, self.floor = Chat(trip), Floor()
        self.safety = SafetyGate(self.chat, lambda: _cases(self.floor), thresholds=LIMITS, enabled=safety_on)
        self.failing = failing

    def for_person(self, user_id: str) -> dict[str, Any]:
        def fake(name: str) -> Any:
            return lambda ckpt: GateResult(name, name not in self.failing, 0.0, 0.0,  # type: ignore[arg-type]
                                           None if name not in self.failing else "fake")
        return {"general": fake("general"), "collapse": fake("collapse"), "safety": self.safety.check}


def _night(tmp_path: Path, gates: Gates) -> Any:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    return run_night_batch_with(world.deps(gates=gates), {"p1": NIGHT})["p1"], world


def test_safety_gate_off_sends_no_cases_and_is_logged_skipped(tmp_path: Path) -> None:
    gates = Gates(safety_on=False, trip=3)
    result, _ = _night(tmp_path, gates)
    assert gates.chat.calls == []  # no case went to the base model or the checkpoint
    safety = result.gates["safety"]
    assert safety.skipped and not safety.passed and safety.reason == SKIPPED_REASON == "off for the hackathon"
    logged = step(result.log_path, "gates")["gates"]["safety"]
    assert logged["skipped"] is True and logged["passed"] is False and logged["reason"] == "off for the hackathon"
    assert summary(result.log_path)["gates"]["safety"]["skipped"] is True


def test_safety_gate_on_is_todays_gate(tmp_path: Path) -> None:
    gates = Gates(safety_on=True, trip=7)
    found = gates.safety.check(CKPT)
    assert found == GateResult("safety", False, 1.0, 0.0, "new reply trips on case indices 7")
    assert len(gates.chat.calls) == 24 and gates.floor.runs == 1  # every case to both models, then the floor
    assert "skipped" not in gate_record(found) and not found.skipped  # the record is byte for byte today's
    assert SafetyGate(Chat(), lambda: _cases(Floor()), thresholds=LIMITS).check(CKPT).passed  # on by default in the core
    result, _ = _night(tmp_path, Gates(safety_on=True, trip=7))
    assert result.status == "rejected" and step(result.log_path, "promote")["rejected_by"] == ["safety"]
    # The app's switch, ANJO_SLEEP_SAFETY_GATE, is tested in tests/app/test_safety_gate_switch.py.


def test_promotion_ignores_a_skipped_safety_gate(tmp_path: Path) -> None:
    promoted, world = _night(tmp_path / "a", Gates(safety_on=False, trip=3))
    assert promoted.status == "promoted" and world.registry.last_promoted("p1") == promoted.checkpoint
    assert step(promoted.log_path, "promote")["skipped"] == ["safety"]
    rejected, _ = _night(tmp_path / "b", Gates(safety_on=False, failing=frozenset({"general"})))
    assert rejected.status == "rejected" and step(rejected.log_path, "promote")["rejected_by"] == ["general"]


def test_floor_still_runs_when_the_gate_is_off(tmp_path: Path) -> None:
    gates = Gates(safety_on=False)
    gates.safety.check(CKPT)
    assert gates.floor.runs == 1 and gates.chat.calls == []
    gates.floor.failed = ("case 4 pre-model",)
    broken = gates.safety.check(CKPT)  # a floor that no longer holds fails the gate, switched off or not
    assert broken == GateResult("safety", False, 1.0, 0.0, "floor checks changed: case 4 pre-model")
    assert not broken.skipped and gates.floor.runs == 2 and gates.chat.calls == []


def test_every_page_shows_a_skipped_gate_as_skipped(tmp_path: Path) -> None:
    skipped = {"gate": "safety", "passed": False, "value": 0.0, "threshold": 0.0, "reason": SKIPPED_REASON,
               "skipped": True}
    general = {"gate": "general", "passed": True, "value": 0.4, "threshold": 2.0, "reason": None}
    night = Night(person="p1", night="2026-10-28", label="development", status="promoted", cost_usd=0.0,
                  gates={"safety": skipped, "general": general}, arms={}, pairs={})
    assert gate_cell(night, "safety") == "skipped" and gate_cell(night, "general") == "ok 0.4"
    marks = night_record({"night": "2026-10-28", "status": "promoted", "gates": {"safety": skipped}}, []).gates
    assert [(m.gate, m.passed, m.skipped) for m in marks] == [("safety", False, True)]
    folder = tmp_path / "p1" / "nights" / "2026-10-28"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({
        "night": "2026-10-28", "status": "rejected", "failed_at": None, "checkpoint": None,
        "gates": {"general": {**general, "passed": False}, "safety": skipped}}), encoding="utf-8")
    (folder / "log.jsonl").write_text("", encoding="utf-8")
    view = night_views(tmp_path / "p1")[0]
    assert view.skipped == ("safety",) and dict(view.gates)["safety"] is False
    lines = what_changed("p1", date(2026, 10, 28), results=lambda uid: tmp_path / uid)
    line = next(c.line for c in lines if c.kind == "rejected")  # the person's line names what failed, not the skip
    assert "(general did not pass)" in line and "safety" not in line


def test_the_cost_model_shows_the_safety_gate_skipped() -> None:
    units = cost.UNITS
    assert cost.CURRENT.safety_gate is False and cost.BEFORE.safety_gate is True
    off = cost._gate_s(cost.CURRENT, units)
    on = cost._gate_s(replace(cost.CURRENT, safety_gate=True), units)
    assert on - off == pytest.approx(units.safety_cases * units.safety_reply_tokens / cost._rate(8, units))
    report = "\n".join(cost._projection_report([cost.project(s, units, cost.DEFAULT_PRICES, cost.Measured(
        extract=0.0, brief=0.0, training=None)) for s in (cost.BEFORE, cost.CURRENT)], 1.0))
    assert "safety gate: skipped in current (off for the hackathon)" in report
