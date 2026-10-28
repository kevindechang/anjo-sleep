"""The collapse gate (Task 5): does training drag her into the person's field when the question
is about something else?

The registered collapse set holds 60 everyday prompts; the gate keeps those whose domain is not in
the person's ``field_domains`` (``sets.loader.collapse_for``). The base model and the checkpoint
each answer every prompt with the registered sampling and a seed per prompt. The base model then
judges each reply with ``collapse_judge_v1.md`` at temperature 0: YES when the reply was dragged
into the field. An answer that is neither YES nor NO counts as dragged, so an unclear judge errs
toward rejecting. The night is rejected when the checkpoint's dragged rate is more than 5.0
points above the base model's.

The base side (the base model's replies and its verdicts on them) is kept across nights when the
app passes a ``cache`` (``base_cache``), keyed on the prompts, the person's fields, the judge
prompt, the seeds, both temperatures and the verdict's cap; the checkpoint's replies, and the
verdicts on them, are asked every night.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Protocol

from anjo.sleep.context.template import render
from anjo.sleep.gates import EVAL_SEED, TEMPERATURE, GateThresholds, in_order, listed, load_thresholds
from anjo.sleep.gates.base_cache import BaseCache, base_side, digest
from anjo.sleep.model_api import ChatResult
from anjo.sleep.serving import config
from anjo.sleep.sets.loader import CollapsePrompt, collapse_for, load_collapse
from anjo.sleep.types import CheckpointRef, GateResult

PROMPT_PATH: Final = Path(__file__).parent / "prompts" / "collapse_judge_v1.md"
JUDGE_TEMPERATURE: Final = 0.0
# YES or NO, read from its first word: at temperature 0 the first tokens are the same at any cap.
VERDICT_MAX_TOKENS: Final = 8
RULE: Final = "dragged unless the verdict's first word is NO"


class CappedClient(Protocol):
    """``ModelClient`` that caps one reply's tokens (``VllmClient``)."""

    def complete(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *, temperature: float,
        seed: int | None = None, max_tokens: int | None = None,
    ) -> ChatResult: ...


class CollapseGate:
    """``check(ckpt)`` for one person's collapse gate; the base side is kept per run."""

    def __init__(
        self, client: CappedClient, field_domains: Sequence[str], *,
        prompts: Sequence[CollapsePrompt] | None = None, thresholds: GateThresholds | None = None,
        workers: int = config.CONCURRENT_REQUESTS, cache: BaseCache | None = None,
    ) -> None:
        if not field_domains:
            raise ValueError("the collapse gate needs the person's field domains")
        self._client = client
        self._fields = ", ".join(field_domains)
        self._prompts = collapse_for(tuple(field_domains), tuple(prompts) if prompts is not None else load_collapse())
        if not self._prompts:
            raise ValueError("no collapse prompt lies outside the person's field")
        self._threshold = (thresholds or load_thresholds()).collapse_points
        self._template = PROMPT_PATH.read_text(encoding="utf-8")
        self._base: list[bool] | None = None
        self._workers = workers
        self._cache = cache

    def _dragged(self, adapter: CheckpointRef | None, index: int) -> bool:
        prompt = self._prompts[index]
        reply = self._client.complete(
            adapter, [{"role": "user", "content": prompt.prompt}], temperature=TEMPERATURE,
            seed=EVAL_SEED + index,
        ).text
        verdict = self._client.complete(None, [{"role": "user", "content": render(self._template, {
            "FIELDS": self._fields, "QUESTION": prompt.prompt, "REPLY": reply,
        })}], temperature=JUDGE_TEMPERATURE, seed=EVAL_SEED, max_tokens=VERDICT_MAX_TOKENS).text.strip().upper()
        return not verdict.startswith("NO")

    def _key(self) -> dict[str, Any]:
        return {"prompts": [[p.id, p.prompt] for p in self._prompts], "fields": self._fields,
                "judge": digest(self._template), "seed": EVAL_SEED, "temperature": TEMPERATURE,
                "judge_temperature": JUDGE_TEMPERATURE, "verdict_max_tokens": VERDICT_MAX_TOKENS, "rule": RULE}

    def _valid(self, kept: list[Any]) -> bool:
        return len(kept) == len(self._prompts) and all(isinstance(flag, bool) for flag in kept)

    def _rate(self, flags: list[bool]) -> float:
        return 100.0 * sum(flags) / len(flags)

    def check(self, ckpt: CheckpointRef) -> GateResult:
        indices = list(range(len(self._prompts)))
        if self._base is None:
            self._base = base_side(self._cache, "collapse", self._key,
                                   lambda: in_order(lambda i: self._dragged(None, i), indices, self._workers), self._valid)
        mine = in_order(lambda i: self._dragged(ckpt, i), indices, self._workers)
        above = self._rate(mine) - self._rate(self._base)
        passed = above <= self._threshold
        newly = [p.id for p, b, m in zip(self._prompts, self._base, mine, strict=True) if m and not b]
        reason = None if passed else (
            f"dragged into the field {above:.1f} points above base (over {self._threshold:g}): {listed(newly)}"
        )
        return GateResult("collapse", passed, above, self._threshold, reason)
