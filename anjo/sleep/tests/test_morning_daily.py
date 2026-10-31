# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""The light morning run for daily nights: a fixed subset of probes and arms, committed before any
result (``DAILY``), picked from the sealed probes by their ids alone. The full set is unchanged and
stays for the final runs: a final run never takes a subset. Fakes only."""
from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.night.deps import Prices
from anjo.sleep.probes.arms import ArmDeps
from anjo.sleep.probes.morning import (
    COMMITTED_ARMS,
    DAILY,
    PAIRS,
    Morning,
    MorningError,
    ProbeSubset,
    run_morning,
    subset_probes,
)
from anjo.sleep.probes.score import Verdict
from anjo.sleep.tests.sleep_fakes import FakeContexts, FakeLogprobs, FakeModel, FakeRegistry, StoreSpy, ckpt, probe, taste
from anjo.sleep.types import ProbeKind

NIGHT = date(2026, 11, 3)
PRICES = Prices(spark_in_per_m=0.2, spark_out_per_m=0.6, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.95)
COUNTS: dict[ProbeKind, int] = {"new_work": 25, "vocab": 15, "tone_hold": 15, "taste_ab": 15, "pushback": 10}  # a life's registered probes
FULL_KEYS = {"night", "label", "checkpoint", "checkpoint_basis", "probes", "arms", "pairs", "unavailable", "outcomes",
             "seconds", "judge_model", "judge"}
# The pairs a full run without ANJO_SLEEP_ARMS_EXTRA can answer: the new arms aren't asked.
COMMITTED_PAIRS = {name for name, (a, b) in PAIRS.items() if a in COMMITTED_ARMS and b in COMMITTED_ARMS}


def sealed() -> list[Any]:
    found = []
    for kind, count in COUNTS.items():
        for i in range(count):
            found.append(taste(f"t{i:02d}", source=f"item-{i}") if kind == "taste_ab" else probe(f"{kind}-{i:02d}", kind, prompt=f"Probe {kind} {i}?"))
    return found


class Judge:
    def __init__(self, prefer_longer: bool = True) -> None:
        self.calls = 0
        self.prefer_longer = prefer_longer

    def verdict(self, p: Any, first: str, second: str) -> Verdict:
        self.calls += 1
        longer: Verdict = "A" if len(first) > len(second) else "B" if len(second) > len(first) else "tie"
        flipped: dict[Verdict, Verdict] = {"A": "B", "B": "A", "tie": "tie"}
        return longer if self.prefer_longer else flipped[longer]


def _morning(tmp_path: Path, probes: list[Any], judge: Judge) -> tuple[Morning, FakeModel]:
    model = FakeModel()
    deps = ArmDeps(model=model, adapters=FakeRegistry({"p1": ckpt("p1", 1), "donor": ckpt("donor", 1)}),
                   contexts=FakeContexts(), evidence=lambda uid: StoreSpy([]),
                   logprobs=FakeLogprobs(lambda adapter, messages: "A"), donors={"p1": "donor"}, until=NIGHT)
    return Morning(arms=deps, sealed=lambda uid: probes, judge=judge, prices=PRICES, root=tmp_path / "results"), model


def _asked(model: FakeModel) -> set[str]:
    return {c["messages"][-1]["content"] for c in model.calls}


def test_daily_morning_subset_is_fixed_and_full_run_unchanged(tmp_path: Path) -> None:
    # Committed now, before any result: these numbers change only in a new, dated subset.
    assert DAILY == ProbeSubset(name="daily-v1", arms=("1", "3"),
                                per_kind={"new_work": 5, "vocab": 3, "tone_hold": 3, "taste_ab": 3, "pushback": 0})
    probes = sealed()
    chosen = subset_probes(probes, DAILY)
    shuffled = probes[:]
    random.Random(4).shuffle(shuffled)
    assert [p.id for p in subset_probes(shuffled, DAILY)] == [p.id for p in chosen]  # by id alone, not order
    assert {k: sum(p.kind == k for p in chosen) for k in COUNTS} == dict(DAILY.per_kind)

    # Two daily mornings whose judges disagree ask the very same probes, of the same two arms.
    records = []
    for n, judge in enumerate((Judge(True), Judge(False))):
        morning, model = _morning(tmp_path / f"daily{n}", probes, judge)
        done = run_morning("p1", NIGHT, ckpt("p1", 1), morning=morning, subset=DAILY)
        record = json.loads(done.path.read_text(encoding="utf-8"))
        records.append(record)
        assert record["subset"] == "daily-v1" and record["probes"] == 14 and set(record["arms"]) == {"1", "3"}
        assert record["pairs"] == {} and {row["probe"] for row in record["outcomes"]} == {p.id for p in chosen}
        assert done.judge_calls == 2 * 11  # 3 against 1, both orders, on the 11 judged probes (taste_ab has no judge)
        assert {c["adapter"].person_id if c["adapter"] else None for c in model.calls} == {None, "p1"}  # arms 1 and 3
    assert records[0]["arms"]["3"] != records[1]["arms"]["3"]  # the results differed; the subset didn't

    # The full run is the run as it was: every arm, every probe, the same file.
    morning, model = _morning(tmp_path / "full", probes, Judge())
    full = json.loads(run_morning("p1", NIGHT, ckpt("p1", 1), morning=morning, label="final").path.read_text(encoding="utf-8"))
    assert set(full) == FULL_KEYS and full["probes"] == 80 and set(full["arms"]) == set(COMMITTED_ARMS)
    assert set(full["pairs"]) == COMMITTED_PAIRS
    assert {p.prompt for p in probes if p.kind != "taste_ab"} <= _asked(model)

    final, _ = _morning(tmp_path / "final", probes, Judge())
    with pytest.raises(MorningError, match="full set"):
        run_morning("p1", NIGHT, ckpt("p1", 1), morning=final, label="final", subset=DAILY)
