# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The final test's registered judge (``probes.registered_judge``): read from ``thresholds.json``, and a
final morning under any other refused; a development morning is never checked."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anjo.sleep import fireworks_api
from anjo.sleep.probes.morning import MorningError
from anjo.sleep.probes.registered_judge import registered_model, require_registered


def test_thresholds_register_deepseek_v4p1_flash() -> None:
    assert registered_model() == fireworks_api.DEEPSEEK_V4P1_FLASH


def test_a_final_morning_runs_only_under_the_registered_judge() -> None:
    require_registered(fireworks_api.DEEPSEEK_V4P1_FLASH, "final")
    for other in ("muse-spark-1.3", "unknown", fireworks_api.GLM_5P3_FLASH):
        with pytest.raises(MorningError, match="registered judge"):
            require_registered(other, "final")
        require_registered(other, "development")  # a development morning may use the development judge


def test_a_judge_this_code_cant_name_refuses_every_final_morning(tmp_path: Path) -> None:
    named = tmp_path / "thresholds.json"
    named.write_text(json.dumps({"judging": {"judge": "Some Other Judge with judge_v1.md"}}), encoding="utf-8")
    with pytest.raises(MorningError, match="can't name"):
        require_registered(fireworks_api.DEEPSEEK_V4P1_FLASH, "final", named)
    with pytest.raises(MorningError, match="can't be read"):
        require_registered(fireworks_api.DEEPSEEK_V4P1_FLASH, "final", tmp_path / "missing.json")
