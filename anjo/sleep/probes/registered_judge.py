"""The final test's registered judge (Task 12b): a final-labelled morning runs under it or not at all.

``prereg/thresholds.json`` registers the judge in words (``judging.judge``: "DeepSeek V4.1 Flash
(Fireworks) with judge_v1.md; ..."). ``registered_model`` reads that entry and names the model id the
code calls (``REGISTERED``), and ``require_registered`` refuses a final morning whose ``Morning``
names any other (``Morning.judge_model``): ``ANJO_SLEEP_JUDGE`` unset selects Muse Spark
(``app.morning.make_judge``), a development judge only since 2026-11-01. A thresholds entry this code
can't name refuses every final morning, never guesses. A development morning is never checked here.
The runners that score a final morning call it before any call: the accelerated runner before its
first day (``Rig.preflight``) and again at each morning (``morning_scorer``), the backfill in its plan
and the daily runner at its morning (``night/daily.py``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from anjo.sleep import fireworks_api
from anjo.sleep.probes.morning import MorningError

THRESHOLDS: Final = Path(__file__).parents[1] / "prereg" / "thresholds.json"
FINAL: Final = "final"
# How thresholds.json words each judge this code can call, and the model id it calls.
REGISTERED: Final = {"DeepSeek V4.1 Flash (Fireworks)": fireworks_api.DEEPSEEK_V4P1_FLASH}


def registered_model(path: Path = THRESHOLDS) -> str:
    """The model id of the judge ``thresholds.json`` registers; refused when it names none this code calls."""
    try:
        words = str(json.loads(path.read_text(encoding="utf-8"))["judging"]["judge"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise MorningError(f"the registered judge can't be read from thresholds.json ({type(error).__name__})") from None
    named = [model for name, model in REGISTERED.items() if words.startswith(name)]
    if len(named) != 1:
        raise MorningError("thresholds.json registers a judge this code can't name: no final morning runs")
    return named[0]


def require_registered(judge_model: str, label: str, path: Path = THRESHOLDS) -> None:
    """Refuse a final morning (``label``) judged by anything but the registered judge; any other label passes."""
    if label != FINAL:
        return
    wanted = registered_model(path)
    if judge_model != wanted:
        raise MorningError(f"a final morning is judged by the registered judge ({wanted}), not {judge_model}: "
                           "set ANJO_SLEEP_JUDGE=deepseek")
