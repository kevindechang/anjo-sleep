# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The safety gate's switch on the sleep host: ``ANJO_SLEEP_SAFETY_GATE`` is off unless it says ``on``,
and any other value stops the night before it starts."""
from __future__ import annotations

import pytest

from anjo.sleep.app.safety_gate import GATE_ENV, gate_on


def test_the_safety_gate_switch_is_off_unless_on() -> None:
    assert GATE_ENV == "ANJO_SLEEP_SAFETY_GATE"
    assert gate_on({}) is False and gate_on({GATE_ENV: "off"}) is False and gate_on({GATE_ENV: " on "}) is True
    with pytest.raises(ValueError, match=GATE_ENV):
        gate_on({GATE_ENV: "1"})
