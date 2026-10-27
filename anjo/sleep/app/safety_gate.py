"""The safety gate's cases (Task 5): Anjo's crisis corpus and reply check, loaded at run time.

The core gate (``anjo.sleep.gates.safety``) never holds a case's words. This adapter reads the
existing private sources, ``ALL_CASES`` (the pre-model crisis corpus) and ``REPLY_TRIPS`` (the
replies the reply check must trip on), and hands the gate the prompts by index, the reply check
(``reply_crisis_reason``) and the floor: the existing checks that must still hold exactly as on
main. It reads those modules and calls them; it never changes them.
"""
from __future__ import annotations

from anjo.sleep.gates.safety import SafetyCases


def _floor() -> tuple[str, ...]:
    """Labels (an index and a kind, never words) of every existing check that no longer holds."""
    from anjo.core.crisis_reply import reply_crisis_reason
    from anjo.evals.crisis_cases import ALL_CASES
    from anjo.evals.crisis_recall import classify
    from anjo.evals.crisis_reply_cases import REPLY_TRIPS

    failed = [f"case {index} pre-model" for index, case in enumerate(ALL_CASES) if not case.allows(classify(case.text))]
    failed += [
        f"reply trip {index}" for index, (_language, text, reason) in enumerate(REPLY_TRIPS)
        if reply_crisis_reason(text) != reason
    ]
    return tuple(failed)


def load_cases() -> SafetyCases:
    """Every case's text in corpus order, Anjo's reply check and the floor."""
    from anjo.core.crisis_reply import reply_crisis_reason
    from anjo.evals.crisis_cases import ALL_CASES

    return SafetyCases(prompts=tuple(case.text for case in ALL_CASES), flag=reply_crisis_reason, floor=_floor)
