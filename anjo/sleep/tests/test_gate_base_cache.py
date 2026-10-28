# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The base model's gate answers, kept across nights: the base is fixed, so the same serving
fingerprint (revision, precision, rendering, the gates' day, the reply cap), items, seeds,
temperature and max_tokens give the same answers. Any change is a miss and a recompute. The
adapter's side is asked every night and never stored. Fakes only."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.gates import EVAL_SEED, GateThresholds, collapse
from anjo.sleep.gates.base_cache import BaseCache, JsonlBaseStore
from anjo.sleep.gates.collapse import CollapseGate
from anjo.sleep.gates.general import GeneralGate
from anjo.sleep.gates.safety import SafetyCases, SafetyGate
from anjo.sleep.model_api import ChatResult
from anjo.sleep.sets.loader import CollapsePrompt, GeneralItem
from anjo.sleep.types import CheckpointRef, GateResult

LIMITS = GateThresholds(general_points=2.0, collapse_points=5.0, safety_regressions=0)
SERVING = "glimmer@rev-a fp8 vllm-0.28.0 low 1024 2026-10-28"
NIGHT_1 = CheckpointRef("p-aaaaaaaaaaaa-n1", "p1", date(2026, 10, 28), "modal-volume://x", None)
NIGHT_2 = CheckpointRef("p-aaaaaaaaaaaa-n2", "p1", date(2026, 10, 29), "modal-volume://y", None)
TRIP = "TRIP-PLACEHOLDER"
ADAPTER_MARK = "ADAPTER-SAID"


def _reply(text: str) -> ChatResult:
    return ChatResult(text=text, tool_calls=(), model="m", input_tokens=1, output_tokens=1, finish_reason="stop")


class Glimmer:
    """One server for every gate: log-probabilities, replies and the collapse judge. Records who
    was asked what: ``("base" | "adapter", "logprob" | "reply" | "judge")``."""

    def __init__(self, items: Sequence[GeneralItem] = ()) -> None:
        self.by_question = {item.question: item for item in items}
        self.calls: list[tuple[str, str]] = []

    def _who(self, adapter: CheckpointRef | None) -> str:
        return "base" if adapter is None else "adapter"

    def continuation_logprobs(self, adapter: Any, messages: list[dict[str, str]], continuation: str) -> Sequence[float]:
        self.calls.append((self._who(adapter), "logprob"))
        item = self.by_question[messages[-1]["content"]]
        return (-0.1,) if continuation == item.choices[item.answer] else (-3.0,)

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None,
                 max_tokens: int | None = None) -> ChatResult:
        content = messages[-1]["content"]
        if content.startswith("# collapse_judge_v1"):
            self.calls.append((self._who(adapter), "judge"))
            return _reply("NO")
        self.calls.append((self._who(adapter), "reply"))
        return _reply(f"{ADAPTER_MARK} on topic" if adapter is not None else "on topic")

    def count(self, who: str, kind: str) -> int:
        return sum(1 for call in self.calls if call == (who, kind))


def _items(n: int = 6) -> list[GeneralItem]:
    return [GeneralItem(f"g{i:03d}", "mmlu", "s", f"question {i}?", ("a", "b", "c", "d"), i % 4) for i in range(n)]


def _prompts(n: int = 5) -> list[CollapsePrompt]:
    return [CollapsePrompt(f"c{i:02d}", "cooking", f"prompt #{i}") for i in range(n)]


def _cases(flag: Callable[[str], str | None] | None = None, prompts: Sequence[str] | None = None) -> SafetyCases:
    return SafetyCases(prompts=list(prompts or [f"neutral case {i}" for i in range(4)]),
                       flag=flag or (lambda reply: "trip" if reply == TRIP else None), floor=lambda: ())


def _cache(path: Path, serving: str = SERVING) -> BaseCache:
    return BaseCache(JsonlBaseStore(path, codec=PLAIN), serving=serving)


def _night(path: Path, ckpt: CheckpointRef, *, serving: str = SERVING, items: Sequence[GeneralItem] | None = None,
           safety: SafetyCases | None = None) -> tuple[Glimmer, dict[str, GateResult]]:
    """One night's run: fresh gates (nothing kept in memory from another night) over one cache file."""
    found = list(items if items is not None else _items())
    server = Glimmer(found)
    cache = _cache(path, serving)
    gates: dict[str, Any] = {
        "general": GeneralGate(server, items=found, thresholds=LIMITS, cache=cache),
        "collapse": CollapseGate(server, ["recycling"], prompts=_prompts(), thresholds=LIMITS, workers=1, cache=cache),
        "safety": SafetyGate(server, lambda: safety or _cases(), thresholds=LIMITS, workers=1, cache=cache),
    }
    return server, {name: gate.check(ckpt) for name, gate in gates.items()}


def test_base_gate_answers_are_cached_across_nights(tmp_path: Path) -> None:
    path = tmp_path / "gate_base.jsonl"
    first, first_results = _night(path, NIGHT_1)
    assert first.count("base", "logprob") == 24 and first.count("base", "reply") == 5 + 4  # night 1 asks the base
    second, second_results = _night(path, NIGHT_2)
    assert second.count("base", "logprob") == 0 and second.count("base", "reply") == 0  # night 2 reads it back
    assert second.count("base", "judge") == 5  # the collapse judge still reads every one of the adapter's replies
    assert {name: (r.passed, r.value) for name, r in second_results.items()} == {
        name: (r.passed, r.value) for name, r in first_results.items()}
    # The base's safety replies are kept, never its flags: tonight's reply check reads them afresh.
    stricter = _cases(flag=lambda reply: "trip" if "on topic" in reply else None)
    third, third_results = _night(path, NIGHT_2, safety=stricter)
    assert third.count("base", "reply") == 0
    assert third_results["safety"].passed  # both sides trip now: kept flags would have called the adapter's trips new


@pytest.mark.parametrize("change", ["serving", "items", "seed", "temperature", "max_tokens", "case"])
def test_cache_misses_on_any_key_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    path = tmp_path / "gate_base.jsonl"
    _night(path, NIGHT_1)
    kwargs: dict[str, Any] = {}
    if change == "serving":  # another revision, precision, rendering, gate day or reply cap
        kwargs["serving"] = SERVING.replace("rev-a", "rev-b")
    elif change == "items":
        kwargs["items"] = [*_items()[:-1], GeneralItem("g005", "mmlu", "s", "question 5, reworded?", ("a", "b", "c", "d"), 1)]
    elif change == "seed":
        monkeypatch.setattr(collapse, "EVAL_SEED", EVAL_SEED + 1)
    elif change == "temperature":
        monkeypatch.setattr(collapse, "TEMPERATURE", 0.7)
    elif change == "max_tokens":
        monkeypatch.setattr(collapse, "VERDICT_MAX_TOKENS", 16)
    else:
        kwargs["safety"] = _cases(prompts=["neutral case 0", "neutral case 1", "neutral case 2", "a new case 3"])
    server, _ = _night(path, NIGHT_2, **kwargs)
    asked = {"general": server.count("base", "logprob"), "base replies": server.count("base", "reply")}
    if change in ("serving",):
        assert asked == {"general": 24, "base replies": 9}  # every gate's base is recomputed
    elif change == "items":
        assert asked == {"general": 24, "base replies": 0}  # the general set changed; the others still hit
    elif change == "case":
        assert asked == {"general": 0, "base replies": 4}  # the safety corpus changed
    else:
        assert asked == {"general": 0, "base replies": 5}  # collapse's sampling changed
    monkeypatch.undo()
    again, _ = _night(path, NIGHT_2)  # the first key is still kept
    assert again.count("base", "logprob") == 0 and again.count("base", "reply") == 0


def test_adapter_answers_are_never_cached(tmp_path: Path) -> None:
    path = tmp_path / "gate_base.jsonl"
    _night(path, NIGHT_1)
    for ckpt in (NIGHT_1, NIGHT_2):  # warm cache, the same checkpoint again and a new one
        server, _ = _night(path, ckpt)
        assert server.count("adapter", "logprob") == 24  # every choice of every item
        assert server.count("adapter", "reply") == 5 + 4  # every collapse prompt and every safety case
    records = read_records(path, PLAIN)
    assert sorted(r["gate"] for r in records) == ["collapse", "general", "safety"]  # one base side per gate, once
    assert ADAPTER_MARK not in path.read_text(encoding="utf-8")
    assert all(NIGHT_1.id not in str(r) and NIGHT_2.id not in str(r) for r in records)


def test_an_unpinned_server_keeps_nothing(tmp_path: Path) -> None:
    path = tmp_path / "gate_base.jsonl"
    found = _items()
    for _ in range(2):
        server = Glimmer(found)
        GeneralGate(server, items=found, thresholds=LIMITS, cache=None).check(NIGHT_1)
        assert server.count("base", "logprob") == 24
    assert not path.exists()


def test_a_kept_answer_of_the_wrong_shape_is_recomputed(tmp_path: Path) -> None:
    path = tmp_path / "gate_base.jsonl"
    _night(path, NIGHT_1)
    general_key = next(r["key"] for r in read_records(path, PLAIN) if r["gate"] == "general")
    JsonlBaseStore(path, codec=PLAIN).put(general_key, "general", [0, 1])  # too few answers: never trusted
    server, _ = _night(path, NIGHT_2)
    assert server.count("base", "logprob") == 24
    assert _night(path, NIGHT_2)[0].count("base", "logprob") == 0  # the recomputed answers are kept again
