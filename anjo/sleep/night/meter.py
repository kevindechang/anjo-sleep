"""What a night spends (Task 5): Wonder's calls and dollars, and the cost parts of the night.

``WonderMeter`` holds the night's Wonder allowance. Every Wonder or sub-agent call is debited,
retries and failed calls included, so the second offer at step 10 sees what step 4 really used.

``SpendMeter`` prices one person's night for ``NightResult.cost_parts`` at list prices (§5.5): the
Spark Open tokens of extraction, the brief and practice's scenes, the training GPU's minutes, and
the serving GPU's time spent on that person's practice and gates. ``mind`` stays 0 until Task 17
runs thinker cycles.

Both are shared by calls running side by side (``parallel.Calls``): every counter changes, and is
read together with the others, under the meter's lock, so the totals are exact whatever order the
calls end in.
"""
from __future__ import annotations

import math
import threading
from collections.abc import Callable
from typing import Any, Final, get_args

from anjo.sleep.model_api import ChatResult
from anjo.sleep.types import CostPart, NightAllowance

COST_PARTS: Final[tuple[CostPart, ...]] = get_args(CostPart)
ExtractChat = Callable[[list[dict[str, Any]]], ChatResult]


class WonderSpent(RuntimeError):
    """A Wonder call was asked for after the night's calls or dollars ran out."""


class WonderMeter:
    """The night's Wonder calls and dollars, debited per call."""

    def __init__(self, calls_left: int, usd_left: float) -> None:
        if calls_left < 0 or usd_left < 0 or not math.isfinite(usd_left):
            raise ValueError("an allowance is not negative")
        self._calls_left = calls_left
        self._usd_left = usd_left
        self.calls = 0
        self.usd = 0.0
        self._lock = threading.Lock()

    @property
    def calls_left(self) -> int:
        return self._calls_left

    @property
    def usd_left(self) -> float:
        return self._usd_left

    def can_call(self) -> bool:
        with self._lock:
            return self._calls_left > 0 and self._usd_left > 0

    def require(self) -> None:
        """Before a call: raise ``WonderSpent`` when none is left (the study records ``budget``)."""
        # Checked, then debited after the call: no reservation, which is exact while homework calls one at a time.
        if not self.can_call():
            raise WonderSpent("Wonder's calls or dollars for the night are spent")

    def debit(self, usd: float) -> None:
        """After every call, a retry or a failed one included: one call and what it cost."""
        if usd < 0 or not math.isfinite(usd):
            raise ValueError("a call's cost is not negative")
        with self._lock:
            self.calls += 1
            self.usd += usd
            self._calls_left = max(0, self._calls_left - 1)
            self._usd_left = max(0.0, self._usd_left - usd)

    def allowance(self, actions_left: int) -> NightAllowance:
        with self._lock:
            return NightAllowance(actions_left=actions_left, calls_left=self._calls_left, usd_left=self._usd_left)


class SpendMeter:
    """One person's cost parts. ``part`` is the step the next Spark Open call is charged to."""

    def __init__(self, *, spark_in_per_m: float, spark_out_per_m: float) -> None:
        self._in = spark_in_per_m / 1_000_000
        self._out = spark_out_per_m / 1_000_000
        self._usd: dict[CostPart, float] = dict.fromkeys(COST_PARTS, 0.0)
        self._lock = threading.Lock()
        self.part: CostPart = "extract"  # set by the night between steps, never while a step's calls run

    def charge(self, result: ChatResult) -> ChatResult:
        cost = result.input_tokens * self._in + result.output_tokens * self._out
        with self._lock:
            self._usd[self.part] += cost
        return result

    def add(self, part: CostPart, usd: float) -> None:
        if usd < 0 or not math.isfinite(usd):
            raise ValueError("a cost is not negative")
        with self._lock:
            self._usd[part] += usd

    def wrap_extract(self, inner: ExtractChat) -> ExtractChat:
        return lambda messages: self.charge(inner(messages))

    def wrap_chat(self, inner: Any) -> Any:
        """The brief's and practice's ``ChatFn`` (model, messages, tools, timeout, temperature), and
        full practice's cap (``max_tokens``), passed on only when a caller sets one."""

        def chat(model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None,
                 max_tokens: int | None = None) -> ChatResult:
            capped = {} if max_tokens is None else {"max_tokens": max_tokens}
            return self.charge(inner(model, messages, tools, timeout_s, temperature=temperature, **capped))

        return chat

    def parts(self) -> dict[CostPart, float]:
        with self._lock:
            return dict(self._usd)
