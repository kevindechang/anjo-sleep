"""What a run returns and how it reads on the screen."""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from anjo.sleep.probes.score import PairScore
from anjo.sleep.types import CheckpointRef, Target

Say = Callable[[str], None]

NO_CRISIS_GATE: Final = (
    "NO CRISIS GATE: this example has no crisis gate, because Anjo's crisis case lists stay "
    "private. It runs no gates at all, so no checkpoint it trains is ever marked promoted."
)
FAKE_NOTICE: Final = (
    "FAKE RUN: Modal, Meta and Baseten are faked, so every reply and score below comes from "
    "fakes and means nothing. It only shows the flow."
)
LIVE_NOTICE: Final = (
    "LIVE RUN: this calls paid services with your own keys: Baseten (Muse Spark Open), the Meta "
    "API (Muse Spark, the judge) and Modal (a GPU to train, a GPU to serve)."
)
STEP_TITLES: Final = {
    "load": "1/7 Load the persona and the day into plain files",
    "probes": "2/7 Write and hash the probes",
    "extract": "3/7 Extract the day's evidence on Spark Open, then fold it into the brief",
    "choose": "4/7 Her chooser picks the night's target",
    "practice": "5/7 Practice, build the dataset, train a LoRA on Modal",
    "arms": "6/7 Run the arms on the probes",
    "chat": "7/7 Serve base plus adapter on vLLM and open a chat",
}
DISCLAIMER: Final = (
    "Intervals are descriptive: a handful of probes, one person. This is a demonstration, not "
    "the pre-registered test."
)


@dataclass(frozen=True)
class ScoreRow:
    arm_a: str
    arm_b: str
    wins: int
    ties: int
    losses: int
    n: int
    score: float
    low: float
    high: float


def score_row(arm_a: str, arm_b: str, result: PairScore) -> ScoreRow:
    return ScoreRow(
        arm_a=arm_a, arm_b=arm_b, wins=result.wins, ties=result.ties, losses=result.losses,
        n=result.n, score=result.score, low=result.ci[0], high=result.ci[1],
    )


@dataclass(frozen=True)
class ExampleReport:
    stages: tuple[str, ...]
    probe_sha256: str
    extracted_items: int
    receipt: dict[str, Any]
    checkpoint: CheckpointRef | None
    dataset_examples: int
    arms: tuple[str, ...]
    table: tuple[ScoreRow, ...]


def heading(say: Say, stage: str) -> None:
    title = STEP_TITLES.get(stage)
    if title is not None:
        say("")
        say(f"== {title}")


def format_receipt(receipt: dict[str, Any], targets: Sequence[Target]) -> list[str]:
    """Her chooser's receipt, as recorded (ids, kinds, strengths, statuses and reason codes, no
    text), then in plain words with the target labels the receipt itself leaves out."""
    labels = {target.id: target.label for target in targets}
    winner = receipt["winner"]
    lines = ["Chooser receipt (public default policy):"]
    for candidate in receipt["candidates"]:
        reason = candidate["reason"] or "-"
        lines.append(
            f"  {candidate['want_id']}  {candidate['action']}  strength {candidate['strength']:.2f}"
            f"  {candidate['status']}  reason {reason}  ({labels.get(candidate['target_id'], '?')})"
        )
    lines.append(f"  winner: {winner if winner else 'nothing (staying quiet is a normal choice)'}")
    lines.append(f"  receipt: {json.dumps(receipt, sort_keys=True)}")
    return lines


def format_table(rows: Sequence[ScoreRow]) -> list[str]:
    lines = ["  A vs B     W-T-L     score   interval        probes"]
    for row in rows:
        lines.append(
            f"  {row.arm_a:>3} vs {row.arm_b:<3}  {row.wins}-{row.ties}-{row.losses}"
            f"{'':<4}{row.score:.2f}    [{row.low:.2f}, {row.high:.2f}]   {row.n}"
        )
    lines.append(DISCLAIMER)
    return lines
