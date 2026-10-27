# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""The Gemini second judge (Task 2): synthetic people only, through an injected client."""
from __future__ import annotations

import subprocess
import sys

import pytest

from anjo.sleep.probes.score import score, BootstrapConfig
from anjo.sleep.probes.arms import ArmReply
from anjo.sleep.probes.second_judge import MODEL_ID, NotSyntheticError, SecondJudge
from anjo.sleep.tests.sleep_fakes import probe


class FakeVertex:
    def __init__(self, answer: str = '{"verdict": "A", "reason": "r"}') -> None:
        self.answer = answer
        self.calls: list[tuple[str, str, float]] = []

    def generate(self, model, prompt, *, temperature):
        self.calls.append((model, prompt, temperature))
        return self.answer


def _judge(client, synthetic=lambda pid: True) -> SecondJudge:
    return SecondJudge(client=client, brief_for=lambda pid: f"BRIEF({pid})", is_synthetic=synthetic)


def test_second_judge_uses_gemini_flash_at_temperature_zero():
    client = FakeVertex()
    verdict = _judge(client).verdict(probe(), "one", "two")
    assert verdict == "A"
    model, prompt, temperature = client.calls[0]
    assert model == MODEL_ID and temperature == 0.0
    assert "BRIEF(p1)" in prompt and "one" in prompt and "two" in prompt


def test_second_judge_refuses_anyone_who_is_not_synthetic():
    client = FakeVertex()
    with pytest.raises(NotSyntheticError):
        _judge(client, synthetic=lambda pid: False).verdict(probe(), "a", "b")
    assert client.calls == []


def test_second_judge_plugs_into_score_in_both_orders():
    client = FakeVertex('{"verdict": "A", "reason": "r"}')
    out = score([probe()], [ArmReply("x", 1, ())], [ArmReply("y", 1, ())], _judge(client),
                bootstrap=BootstrapConfig(50, 1, 0.95))
    assert len(client.calls) == 2 and out.ties == 1


def test_importing_the_second_judge_loads_no_vertex_sdk():
    script = "import sys, anjo.sleep.probes.second_judge; print([m for m in sys.modules if 'vertex' in m or m.startswith('google')])"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"
