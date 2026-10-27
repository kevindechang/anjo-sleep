"""The second-family judge (Task 2, spec 5.1): Gemini 3.8 Flash on Vertex, a diagnostic.

It judges final pairs again so that agreement with Muse Spark can be reported per life and per
dimension. It sees synthetic people's pairs only: anyone else is refused before a prompt is
built. The Vertex client is injected through a port (``VertexClient``); this module imports no
Vertex SDK and makes no network call itself.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from anjo.sleep.probes.clients import VertexClient
from anjo.sleep.probes.score import Verdict, judge_prompt, parse_verdict
from anjo.sleep.types import Probe

MODEL_ID: Final = "gemini-3.8-flash"  # verified Oct 27 against the Vertex model list (spec 5.1)
TEMPERATURE: Final = 0.0


class NotSyntheticError(PermissionError):
    """The second judge was asked about a person who isn't synthetic."""


@dataclass(frozen=True)
class SecondJudge:
    """A ``PairwiseJudge`` on Vertex. ``is_synthetic`` is the app's check (cleared sessions)."""

    client: VertexClient
    brief_for: Callable[[str], str]
    is_synthetic: Callable[[str], bool]
    model: str = MODEL_ID

    def verdict(self, probe: Probe, first: str, second: str) -> Verdict:
        if not self.is_synthetic(probe.person_id):
            raise NotSyntheticError(f"{probe.person_id} is not synthetic; Vertex sees synthetic people only")
        prompt = judge_prompt(probe, self.brief_for(probe.person_id), first, second)
        return parse_verdict(self.client.generate(self.model, prompt, temperature=TEMPERATURE))
