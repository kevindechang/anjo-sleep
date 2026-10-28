"""The dry run's estimate of what replay would spend (Task 17b), before any of it runs: the
labels' Spark Open dollars from their prompts' sizes, each reply at its cap, beside the Spark Open
hold of the labels and the branches' resolver calls, on the provider's pool; and the most A100
minutes and compute units the branches and the 17b arms can keep Glimmer busy (their ``colab_cu``
holds, ``replay.spend``)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.mind.apd.types import Life
from anjo.sleep.mind.replay.resolve import LABEL_MAX_TOKENS, label_messages, later_evidence
from anjo.sleep.mind.replay.spend import glimmer_seconds
from anjo.sleep.mind.replay.types import REPLAY_K, StartState
from anjo.sleep.mind.spark_provider import BASETEN, SparkProvider
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import spark_usd, spark_worst_usd
from anjo.sleep.night.mind import MIND_SPARK_CALLS
from anjo.sleep.train.runners.colab import A100_CU_PER_HOUR, POOL

ARMS_PER_LIFE: Final = 3  # M, M+replay and M+shuffled


@dataclass(frozen=True)
class SpendEstimate:
    """``label_tokens`` are the labels' prompt tokens; the dollars are ``None`` with no prices.
    Each ``*_hold_*`` is one hold, taken and settled before the next is taken."""

    label_calls: int
    label_tokens: int
    label_usd: float | None
    label_hold_usd: float | None
    branch_spark_hold_usd: float | None
    branch_runs: int
    arm_runs: int
    branch_hold_cu: float
    arm_hold_cu: float
    gpu_minutes: float
    compute_units: float


def _cu(seconds: float) -> float:
    return A100_CU_PER_HOUR * seconds / 3600.0


def estimate(
    starts: Sequence[StartState], lives: Sequence[Life], held_out: Sequence[Life], prices: Prices | None, *,
    k: int = REPLAY_K,
) -> SpendEstimate:
    """Replay's spend before it runs: every start state labelled and branched (at most), and each
    held-out life run under the three arms."""
    by_id = {life.id: life for life in lives}
    asked = [start for start in starts if later_evidence(start, by_id[start.life_id])]
    reads = [sum(estimate_tokens(str(m["content"])) for m in label_messages(s, by_id[s.life_id])) for s in asked]
    seconds = glimmer_seconds(len(starts) * k + ARMS_PER_LIFE * sum(len(life.days) for life in held_out))
    return SpendEstimate(
        label_calls=len(asked), label_tokens=sum(reads),
        label_usd=None if prices is None else sum(spark_usd(prices, n, LABEL_MAX_TOKENS) for n in reads),
        label_hold_usd=None if prices is None else spark_worst_usd(prices, len(starts)),
        branch_spark_hold_usd=None if prices is None else spark_worst_usd(prices, k * MIND_SPARK_CALLS),
        branch_runs=len(starts) * k, arm_runs=ARMS_PER_LIFE * len(held_out),
        branch_hold_cu=_cu(glimmer_seconds(k)),
        arm_hold_cu=_cu(glimmer_seconds(max((len(life.days) for life in held_out), default=0))),
        gpu_minutes=seconds / 60.0, compute_units=_cu(seconds),
    )


def _usd(value: float | None) -> str:
    return "unpriced (no list prices given)" if value is None else f"${value:,.2f}"


def describe(found: SpendEstimate, provider: SparkProvider = BASETEN) -> str:
    return "\n".join((
        f"Replay spend plan (Glimmer on Colab vLLM, Spark Open on {provider.label} ({provider.model}); "
        "each step held before it runs):",
        f"- Spark Open: {found.label_calls} label calls reading about {found.label_tokens:,} tokens, about "
        f"{_usd(found.label_usd)}. On {provider.pool} the labels hold at most {_usd(found.label_hold_usd)}, and each "
        f"start state's branches (their resolver calls) at most {_usd(found.branch_spark_hold_usd)}, each "
        "settled at what it cost.",
        f"- Glimmer: {found.branch_runs} branch nights and {found.arm_runs} arm runs, at most "
        f"{found.gpu_minutes:,.0f} A100 minutes ({found.compute_units:,.1f} compute units) in all. On {POOL} each "
        f"start state's branches hold at most {found.branch_hold_cu:.1f} and each arm run at most "
        f"{found.arm_hold_cu:.1f}, each settled at the time it took.",
    ))
