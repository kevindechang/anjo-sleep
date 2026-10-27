# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""The morning probe run (Task 12a): every arm answers the sealed probes on the checkpoint that is
serving after the night, the pairs are judged in both orders, and one content-free score file is
written. Nothing reaches history or evidence (RF5), a scored night is never rerun, and every paid call
is held first. Fakes only."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.night.deps import Prices
from anjo.sleep.probes.arms import ARMS, ArmDeps
from anjo.sleep.probes.morning import Morning, MorningError, run_morning
from anjo.sleep.probes.score import Verdict
from anjo.sleep.reports import results
from anjo.sleep.stops import BudgetStop
from anjo.sleep.tests.sleep_fakes import (
    FakeContexts,
    FakeLogprobs,
    FakeModel,
    FakeRegistry,
    StoreSpy,
    ckpt,
    probe,
    taste,
)

NIGHT = date(2026, 11, 3)
PRICES = Prices(spark_in_per_m=0.2, spark_out_per_m=0.6, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.95)


class Judge:
    """Deterministic: the longer reply wins; calls are counted."""

    def __init__(self) -> None:
        self.calls = 0

    def verdict(self, p: Any, first: str, second: str) -> Verdict:
        self.calls += 1
        return "A" if len(first) > len(second) else "B" if len(second) > len(first) else "tie"


class Rig:
    def __init__(self, tmp_path: Path, *, donor: bool = True, ledger: Ledger | None = None) -> None:
        self.model, self.judge, self.store = FakeModel(), Judge(), StoreSpy([])
        self.registry = FakeRegistry({"p1": ckpt("p1", 1), **({"donor": ckpt("donor", 1)} if donor else {})})
        self.probes = [probe("q1", "new_work"), probe("q2", "vocab"), probe("q3", "tone_hold"), taste("t1")]
        deps = ArmDeps(model=self.model, adapters=self.registry, contexts=FakeContexts(),
                       evidence=lambda uid: self.store, logprobs=FakeLogprobs(lambda adapter, messages: "A"),
                       donors={"p1": "donor"}, until=NIGHT)
        self.morning = Morning(arms=deps, sealed=lambda uid: self.probes, judge=self.judge, prices=PRICES,
                               ledger=ledger, root=tmp_path / "results")

    def run(self, night: date = NIGHT, label: str = "development", serving: Any = "registry",
            promoted: bool = True) -> Any:
        found = self.registry.adapters["p1"] if serving == "registry" else serving
        return run_morning("p1", night, found, morning=self.morning, label=label, promoted=promoted)

    @property
    def scores(self) -> Path:
        return self.morning.root / "p1" / "scores"  # type: ignore[operator]


def test_morning_run_never_saved_to_history(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    done = rig.run()
    assert rig.store.writes == 0  # no evidence, no history (RF5)
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert files == ["results/p1/scores/night-01.json"]
    body = done.path.read_text(encoding="utf-8")
    assert "reply from" not in body and all(p.prompt not in body for p in rig.probes)  # no reply or probe text
    assert set(json.loads(body)) == {"night", "label", "checkpoint", "checkpoint_basis", "probes", "arms", "pairs", "unavailable",
                               "outcomes"}


def test_morning_rerun_refused_once_scored(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.run()
    asked, judged = len(rig.model.calls), rig.judge.calls
    with pytest.raises(MorningError, match="already scored"):
        rig.run()
    assert (len(rig.model.calls), rig.judge.calls) == (asked, judged)  # refused before a single call
    assert [p.name for p in rig.scores.iterdir()] == ["night-01.json"]
    rig.run(night=date(2026, 11, 4))  # the next night is scored as its own file
    assert sorted(p.name for p in rig.scores.iterdir()) == ["night-01.json", "night-02.json"]


def test_morning_scores_serving_checkpoint(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    serving = rig.registry.adapters["p1"]
    done = rig.run()
    personal = {c["adapter"].id for c in rig.model.calls if c["adapter"] is not None and c["adapter"].person_id == "p1"}
    assert personal == {serving.id}  # arms 3 and 4 ran on the serving checkpoint
    assert json.loads(done.path.read_text(encoding="utf-8"))["checkpoint"] == serving.id
    stale = replace(serving, id="p1-n0")
    other = Rig(tmp_path / "other")
    with pytest.raises(MorningError, match="serving"):
        other.run(serving=stale)
    assert other.model.calls == [] and other.judge.calls == 0


def test_every_arm_is_scored_on_both_dimensions(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    record = json.loads(rig.run(label="final").path.read_text(encoding="utf-8"))
    assert record["label"] == "final" and record["night"] == NIGHT.isoformat() and record["probes"] == 4
    assert set(record["arms"]) == set(ARMS)
    assert {d: v["score"] for d, v in record["arms"]["1"].items()} == {"knowledge": 0.5, "interaction": 0.5, "overall": 0.5}
    for arm in ARMS - {"1"}:
        assert set(record["arms"][arm]) == {"knowledge", "interaction", "overall"}
    assert set(record["pairs"]) == {"3_vs_3x", "4_vs_2C"}
    assert rig.judge.calls > 0 and record["unavailable"] == {} and record["checkpoint_basis"] == "promoted"


def test_the_score_file_is_what_the_results_table_reads(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.run()
    night = rig.morning.root / "p1" / "nights" / NIGHT.isoformat()  # type: ignore[operator]
    night.mkdir(parents=True)
    (night / "summary.json").write_text(json.dumps({"status": "promoted", "night": NIGHT.isoformat(),
                                                    "label": "development", "gates": {}, "cost_usd": 0.0}))
    text = results.table(["p1"], root=rig.morning.root)
    assert "arm 4" in text and "pair 4_vs_2C" in text and "/" in next(row for row in text.splitlines() if row.startswith(NIGHT.isoformat() + " | promoted"))


def test_a_donor_that_never_promoted_leaves_3x_unavailable_not_invented(tmp_path: Path) -> None:
    rig = Rig(tmp_path, donor=False)
    record = json.loads(rig.run().path.read_text(encoding="utf-8"))
    assert "3x" not in record["arms"] and record["unavailable"] == {"3x": "donor_unavailable"}  # a code, no text
    assert "3_vs_3x" not in record["pairs"] and "4_vs_2C" in record["pairs"]


def test_paid_calls_are_held_first_and_settled(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    Rig(tmp_path, ledger=ledger).run()
    events = read_records(ledger.path, PLAIN)
    reserves = [e for e in events if e["op"] == "reserve"]
    assert [e["pools"] for e in reserves] == [["meta_usd"], ["modal_usd", "baseten_sleep_usd"]]  # both held before a call
    assert [e["op"] for e in events].count("settle") == 2 and not ledger.open_holds()
    assert all(e["job"].startswith("morning:2026-11-03:") for e in reserves)


def test_no_room_in_a_pool_stops_before_any_call(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "meta_usd": 0.01})
    rig = Rig(tmp_path, ledger=ledger)
    with pytest.raises(BudgetStop):
        rig.run()
    assert rig.judge.calls == 0 and not ledger.open_holds() and not rig.scores.exists()


def test_a_probe_of_another_person_is_refused(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.probes = [probe("q9", "new_work", person="someone-else")]
    with pytest.raises(MorningError, match="another person"):
        rig.run()
    assert rig.model.calls == []


STAT_KEYS = {"n", "wins", "ties", "losses", "score", "decisive_share", "ci"}


def test_score_files_carry_the_statistics_the_claim_rules_need(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    record = json.loads(rig.run().path.read_text(encoding="utf-8"))
    for name, stats in [*((f"{a}_vs_1", record["arms"][a]) for a in ARMS - {"1"}), *record["pairs"].items()]:
        assert set(stats) == {"knowledge", "interaction", "overall"}, name
        for stat in stats.values():
            assert set(stat) == STAT_KEYS and stat["n"] == stat["wins"] + stat["ties"] + stat["losses"]
            low, high = stat["ci"]
            assert 0.0 <= low <= stat["score"] <= high <= 1.0
            assert stat["score"] == pytest.approx((stat["wins"] + 0.5 * stat["ties"]) / stat["n"])


def test_per_probe_outcomes_recompute_the_counts(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    record = json.loads(rig.run().path.read_text(encoding="utf-8"))
    ids = {p.id for p in rig.probes}
    for row in record["outcomes"]:
        assert set(row) == {"probe", "pair", "kind", "outcome", "flip"}
        assert row["probe"] in ids and row["outcome"] in {"win", "tie", "loss"} and isinstance(row["flip"], bool)
    assert {row["pair"] for row in record["outcomes"]} == {f"{a}_vs_1" for a in ARMS - {"1"}} | set(record["pairs"])
    mine = [r for r in record["outcomes"] if r["pair"] == "4_vs_2C" and r["kind"] != "pushback"]
    stat = record["pairs"]["4_vs_2C"]["overall"]
    assert [sum(r["outcome"] == o for r in mine) for o in ("win", "tie", "loss")] == [stat["wins"], stat["ties"], stat["losses"]]
    text = json.dumps(record)
    assert not any(p.prompt in text for p in rig.probes) and "reply from" not in text


def test_the_results_table_reproduces_the_ci_from_the_file_alone(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    record = json.loads(rig.run().path.read_text(encoding="utf-8"))
    night = rig.morning.root / "p1" / "nights" / NIGHT.isoformat()  # type: ignore[operator]
    night.mkdir(parents=True)
    (night / "summary.json").write_text(json.dumps({"status": "promoted", "night": NIGHT.isoformat(),
                                                    "label": "development", "gates": {}, "cost_usd": 0.0}))
    text = results.table(["p1"], root=rig.morning.root)
    low, high = record["arms"]["4"]["knowledge"]["ci"]
    assert f"{record['arms']['4']['knowledge']['score']:.2f} [{low:.2f},{high:.2f}]" in text


def test_a_night_that_promoted_nothing_scores_what_was_serving_and_says_so(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    record = json.loads(rig.run(serving=None, promoted=False).path.read_text(encoding="utf-8"))
    assert record["checkpoint_basis"] == "not_promoted" and record["checkpoint"] == rig.registry.adapters["p1"].id


def test_a_promoted_night_must_give_its_own_checkpoint(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    with pytest.raises(MorningError, match="promoted"):
        rig.run(serving=None, promoted=True)
    assert rig.model.calls == []
