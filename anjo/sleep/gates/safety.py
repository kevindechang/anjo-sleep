"""The safety gate (Task 5): zero regressions against the base model on Anjo's crisis cases. The
legal floor, so it runs every case on every night and is never cut for cost.

The case lists and the crisis checks stay in the private app; the core never holds a case's words.
The app's adapter (``anjo.sleep.app.safety_gate``) loads them at run time and hands this gate:

- ``prompts``: every case's text, in a fixed order, which the gate refers to by index only;
- ``flag``: Anjo's reply check (``reply_crisis_reason``), the reason a reply trips it or ``None``;
- ``floor``: the checks that must hold exactly as on main whatever the weights are: the pre-model
  crisis path on every case, and the reply check on every known trip. It returns the labels of
  the checks that failed (an index and a kind, never words).

``check`` sends every prompt to the base model and to the checkpoint with the registered sampling
and a seed per case, each reply at the client's full cap (the check reads the whole reply: a method
or a dose after a capped reply's end would go unseen), runs ``flag`` on both replies, and counts a regression wherever the
checkpoint's reply trips the check and the base model's reply to the same case doesn't. Every
failed floor check counts too. The night passes only with none (``max_regressions`` 0).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from anjo.sleep.gates import EVAL_SEED, TEMPERATURE, GateThresholds, in_order, listed, load_thresholds
from anjo.sleep.probes.clients import ModelClient
from anjo.sleep.serving import config
from anjo.sleep.types import CheckpointRef, GateResult


@dataclass(frozen=True)
class SafetyCases:
    """What the app loads for one run: the prompts by index, the reply check and the floor."""

    prompts: Sequence[str]
    flag: Callable[[str], str | None]
    floor: Callable[[], tuple[str, ...]]


class SafetyGate:
    """``SafetyGate`` port. The base model's flags are kept for the run."""

    def __init__(
        self, client: ModelClient, cases: Callable[[], SafetyCases], *,
        thresholds: GateThresholds | None = None, workers: int = config.CONCURRENT_REQUESTS,
    ) -> None:
        self._client = client
        self._workers = workers
        self._load = cases
        self._cases: SafetyCases | None = None
        self._threshold = (thresholds or load_thresholds()).safety_regressions
        self._base: list[bool] | None = None

    def _tripped(self, adapter: CheckpointRef | None, index: int) -> bool:
        assert self._cases is not None
        reply = self._client.complete(
            adapter, [{"role": "user", "content": self._cases.prompts[index]}],
            temperature=TEMPERATURE, seed=EVAL_SEED + index,
        ).text
        return self._cases.flag(reply) is not None

    def check(self, ckpt: CheckpointRef) -> GateResult:
        if self._cases is None:
            self._cases = self._load()
        if not self._cases.prompts:
            raise ValueError("the safety gate runs on every case, and none were loaded")
        floor = self._cases.floor()
        indices = list(range(len(self._cases.prompts)))
        if self._base is None:
            self._base = in_order(lambda i: self._tripped(None, i), indices, self._workers)
        mine = in_order(lambda i: self._tripped(ckpt, i), indices, self._workers)
        regressions = [i for i, b, m in zip(indices, self._base, mine, strict=True) if m and not b]
        count = len(regressions) + len(floor)
        passed = count <= self._threshold
        parts = [
            *([f"new reply trips on case indices {listed(regressions)}"] if regressions else []),
            *([f"floor checks changed: {listed(list(floor))}"] if floor else []),
        ]
        return GateResult("safety", passed, float(count), float(self._threshold),
                          None if passed else "; ".join(parts))
