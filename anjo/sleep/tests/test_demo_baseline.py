"""The demo baseline picker (``tools.demo_baseline``): pooled development 4 vs 2C and 4 vs 2F on new_work."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from anjo.sleep.night.deps import Prices
from anjo.sleep.probes.arms import Arm, ArmDeps
from anjo.sleep.probes.morning import COMMITTED_ARMS, Morning, run_morning
from anjo.sleep.probes.score import Verdict
from anjo.sleep.tests.sleep_fakes import (
    FakeContexts,
    FakeLogprobs,
    FakeRegistry,
    StoreSpy,
    ckpt,
    probe,
    taste,
)
from anjo.sleep.tools import demo_baseline as pick

NIGHT = date(2026, 10, 28)
PRICES = Prices(
    spark_in_per_m=0.2, spark_out_per_m=0.6, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.95
)


@dataclass
class ArmSizedModel:
    """Reply length by arm so 4 vs 2C and 4 vs 2F differ though 2C and 2F share the base adapter."""

    lengths: dict[Arm, int]
    judged_probes: int
    calls: int = 0

    def complete(
        self,
        adapter: Any,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        seed: int | None = None,
    ) -> Any:
        from anjo.sleep.tests.sleep_fakes import ChatResult

        arm = COMMITTED_ARMS[self.calls // self.judged_probes]
        self.calls += 1
        text = "x" * self.lengths[arm]
        words = sum(len(m["content"].split()) for m in messages)
        name = adapter.id if adapter else "base"
        return ChatResult(
            text=text,
            tool_calls=(),
            model=name,
            input_tokens=words,
            output_tokens=3,
            finish_reason="stop",
        )


class Judge:
    def verdict(self, p: Any, first: str, second: str) -> Verdict:
        return "A" if len(first) > len(second) else "B" if len(second) > len(first) else "tie"


def _score_root(root: Any, lengths: dict[Arm, int]) -> None:
    for person in pick.pooled_lives():
        probes = [probe("nw0", "new_work", person=person), taste("t0", person=person)]
        judged = sum(1 for p in probes if p.kind != "taste_ab")
        model = ArmSizedModel(lengths, judged)
        registry = FakeRegistry({person: ckpt(person, 1), "donor": ckpt("donor", 1)})
        deps = ArmDeps(
            model=model,
            adapters=registry,
            contexts=FakeContexts(),
            evidence=lambda uid: StoreSpy([]),
            logprobs=FakeLogprobs(lambda adapter, messages: "A"),
            donors={person: "donor"},
            until=NIGHT,
        )
        morning = Morning(
            arms=deps,
            sealed=lambda uid, probes=probes: probes,
            judge=Judge(),
            prices=PRICES,
            root=root,
        )
        run_morning(person, NIGHT, ckpt(person, 1), morning=morning, label="development")


def test_demo_baseline_picks_2C_when_4_beats_2C_more_than_2F(tmp_path: Any) -> None:
    """Arm 4 wins more against 2F than against 2C, so 2C is the stronger notes-only baseline."""
    lengths = {"1": 1, "2C": 100, "2F": 10, "3": 1, "3x": 1, "4": 50}
    _score_root(tmp_path, lengths)
    got = pick.recommend(tmp_path)
    assert got["baseline"] == "2C"
    assert "2C stronger" in got["reason"]


def test_demo_baseline_picks_2F_when_2F_stronger_on_new_work(tmp_path: Any) -> None:
    lengths = {"1": 1, "2C": 10, "2F": 100, "3": 1, "3x": 1, "4": 50}
    _score_root(tmp_path, lengths)
    got = pick.recommend(tmp_path)
    assert got["baseline"] == "2F"
    assert "2F stronger" in got["reason"]


def test_demo_baseline_inconclusive_defaults_to_2C(tmp_path: Any) -> None:
    lengths = {"1": 1, "2C": 50, "2F": 50, "3": 1, "3x": 1, "4": 50}
    _score_root(tmp_path, lengths)
    tied = pick.recommend(tmp_path)
    assert tied["baseline"] == "2C" and "inconclusive" in tied["reason"]

    empty = pick.recommend(tmp_path / "missing")
    assert empty["baseline"] == "2C" and "missing" in empty["reason"].lower()
