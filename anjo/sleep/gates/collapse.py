"""The collapse gate (Task 5): does training drag her into the person's field when the question
is about something else?

The registered collapse set holds 60 everyday prompts; the gate keeps those whose domain is not in
the person's ``field_domains`` (``sets.loader.collapse_for``). The base model and the checkpoint
each answer every prompt with the registered sampling and a seed per prompt. The base model then
judges each reply with ``collapse_judge_v1.md`` at temperature 0: YES when the reply was dragged
into the field. An answer that is neither YES nor NO counts as dragged, so an unclear judge errs
toward rejecting. The night is rejected when the checkpoint's dragged rate is more than 5.0
points above the base model's.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from anjo.sleep.context.template import render
from anjo.sleep.gates import EVAL_SEED, TEMPERATURE, GateThresholds, in_order, listed, load_thresholds
from anjo.sleep.probes.clients import ModelClient
from anjo.sleep.sets.loader import CollapsePrompt, collapse_for, load_collapse
from anjo.sleep.types import CheckpointRef, GateResult

PROMPT_PATH: Final = Path(__file__).parent / "prompts" / "collapse_judge_v1.md"
JUDGE_TEMPERATURE: Final = 0.0


class CollapseGate:
    """``check(ckpt)`` for one person's collapse gate; the base side is kept per run."""

    def __init__(
        self, client: ModelClient, field_domains: Sequence[str], *,
        prompts: Sequence[CollapsePrompt] | None = None, thresholds: GateThresholds | None = None,
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

    def _dragged(self, adapter: CheckpointRef | None, index: int) -> bool:
        prompt = self._prompts[index]
        reply = self._client.complete(
            adapter, [{"role": "user", "content": prompt.prompt}], temperature=TEMPERATURE,
            seed=EVAL_SEED + index,
        ).text
        verdict = self._client.complete(None, [{"role": "user", "content": render(self._template, {
            "FIELDS": self._fields, "QUESTION": prompt.prompt, "REPLY": reply,
        })}], temperature=JUDGE_TEMPERATURE, seed=EVAL_SEED).text.strip().upper()
        return not verdict.startswith("NO")

    def _rate(self, flags: list[bool]) -> float:
        return 100.0 * sum(flags) / len(flags)

    def check(self, ckpt: CheckpointRef) -> GateResult:
        indices = list(range(len(self._prompts)))
        if self._base is None:
            self._base = in_order(lambda i: self._dragged(None, i), indices)
        mine = in_order(lambda i: self._dragged(ckpt, i), indices)
        above = self._rate(mine) - self._rate(self._base)
        passed = above <= self._threshold
        newly = [p.id for p, b, m in zip(self._prompts, self._base, mine, strict=True) if m and not b]
        reason = None if passed else (
            f"dragged into the field {above:.1f} points above base (over {self._threshold:g}): {listed(newly)}"
        )
        return GateResult("collapse", passed, above, self._threshold, reason)
