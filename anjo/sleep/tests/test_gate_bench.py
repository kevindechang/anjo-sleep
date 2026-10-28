# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The gate bench (``python -m anjo.sleep.tools.gate_bench``): plan-first. Without ``--go`` it prints the
slice, the levels, the GPU's price and KV room and the budget hold, and sends nothing. With ``--go`` it
runs the same requests the gates send at each concurrency and prints seconds, tokens a second and
dollars, never a prompt's or a reply's words."""
from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest

from anjo.sleep.model_api import ChatResult
from anjo.sleep.serving import config, gpus
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.sets.loader import CollapsePrompt, GeneralItem
from anjo.sleep.tools import gate_bench

SECRET = "zebra-quartz"  # a word in every prompt: it must never be printed


class _Never:
    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the plan sends nothing and holds nothing")


class FakeClient:
    """Answers like Glimmer's client: each reply echoes its prompt, so a printed word would show."""

    def __init__(self) -> None:
        self.days: list[date] = []
        self.calls = 0
        self._lock = threading.Lock()
        self.day: date | None = None

    def on(self, day: date) -> FakeClient:
        dated = FakeClient()
        dated.__dict__.update(self.__dict__)
        dated.day = day
        self.days.append(day)
        return dated

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float,
                 seed: int | None = None, max_tokens: int | None = None) -> ChatResult:
        with self._lock:
            self.calls += 1
        text = "NO" if max_tokens is not None and max_tokens <= 8 else f"echo {messages[-1]['content']}"
        return ChatResult(text=text, tool_calls=(), model="m", input_tokens=40, output_tokens=200,
                          finish_reason="stop")

    def continuation_logprobs(self, adapter: Any, messages: list[dict[str, str]], continuation: str) -> Sequence[float]:
        with self._lock:
            self.calls += 1
        return (-1.0, -float(len(continuation) % 3))


def _general() -> tuple[GeneralItem, ...]:
    return tuple(GeneralItem(id=f"g{i}", source="mmlu", subject="s", question=f"{SECRET} question {i}?",
                             choices=(f"{SECRET} a", "b", "cc", "ddd"), answer=1) for i in range(200))


def _collapse() -> tuple[CollapsePrompt, ...]:
    return tuple(CollapsePrompt(id=f"c{i}", domain="cooking", prompt=f"{SECRET} everyday ask {i}") for i in range(60))


def test_gate_bench_plan_prints_hold_and_sends_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    never = _Never()
    assert gate_bench.main([], environ={}, client=never, wake=never, ledger=never) == 0
    plan = capsys.readouterr().out
    assert "nothing is sent" in plan and "--go" in plan
    assert "8, 16, 32" in plan
    assert "general" in plan and "collapse" in plan
    assert "safety: off" in plan  # the night's safety gate is off by default; a slice is optional
    bound_h = (gate_bench.BENCH_MAX_S + gate_bench.WAKE_S + config.SCALEDOWN_S) / 3600
    assert f"${gpus.L40S.usd_per_hour * bound_h:.2f}" in plan and "modal_usd" in plan  # the hold, at the L40S's price
    assert "5.78 GiB" in plan and "measured" in plan  # the L40S's KV room
    assert "ANJO_SLEEP_SERVE_MAX_INPUTS" in plan  # 32 at once needs the server and this host to take it

    assert gate_bench.main(["--gpu", "H100"], environ={}, client=never, wake=never, ledger=never) == 0
    h100 = capsys.readouterr().out
    assert f"${gpus.H100.usd_per_hour * bound_h:.2f}" in h100 and "estimate" in h100 and "FP8 natively" in h100
    assert gate_bench.main(["--gpu", "A100-80GB"], environ={}, client=never, wake=never, ledger=never) == 0
    assert "weight-only" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        gate_bench.main(["--safety-cases", "200"], environ={}, client=never, wake=never, ledger=never)  # needs --safety


def test_gate_bench_go_prints_numbers_never_words(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeClient()
    woke = WakeResult(seconds=1.5, ok=True)
    args = ["--go", "--concurrency", "4,8", "--safety", "anjo.sleep.tests.test_gate_bench:_safety_cases"]
    assert gate_bench.main(args, environ={}, client=fake, wake=lambda: woke, ledger=None,
                           general=_general(), collapse=_collapse()) == 0
    out = capsys.readouterr().out
    assert SECRET not in out and "echo" not in out
    lines = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    measured = [line for line in lines if "gate" in line]
    assert {(line["concurrency"], line["gate"]) for line in measured} == {
        (c, g) for c in (4, 8) for g in ("general", "collapse", "safety")}
    general = next(line for line in measured if line["gate"] == "general")
    assert general["requests"] == 50 * 4 - 0  # every 4th of the 200 items, each choice one call
    collapse = next(line for line in measured if line["gate"] == "collapse")
    assert collapse["requests"] == 120 and collapse["generated_tokens"] == 60 * 200 + 60 * 200
    assert all(line["seconds"] >= 0 and "usd" in line for line in measured)
    nights = [line for line in lines if line.get("line") == "night"]
    assert {line["concurrency"] for line in nights} == {4, 8}
    assert all({"general", "collapse"} <= set(line["seconds_by_gate"]) for line in nights)
    # Each level is dated a day of its own, so no level reads another's prompts from vLLM's prefix cache.
    assert len(set(fake.days)) == 2


def test_gate_bench_refuses_more_at_once_than_this_host_sends(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        gate_bench.main(["--go", "--concurrency", str(config.CONCURRENT_REQUESTS * 2)], environ={},
                        client=FakeClient(), wake=lambda: WakeResult(0.0, True), ledger=None,
                        general=_general(), collapse=_collapse())


def _safety_cases() -> Any:
    from anjo.sleep.gates.safety import SafetyCases

    return SafetyCases(prompts=[f"{SECRET} case {i}" for i in range(300)], flag=lambda reply: None, floor=lambda: ())
