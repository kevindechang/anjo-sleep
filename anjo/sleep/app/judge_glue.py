"""Joins the judge cohort (``judge_cohort``) to the judge night job (``judge_night``).

The night job asks two things through ``JudgePorts``: is this account a judge allowed a night, and
may it start one now. Both answers come from the cohort, so there is one list of judges and one set
of daily counters.
"""
from __future__ import annotations

from anjo.sleep.app.judge_cohort import CapReached, claim_night, night_allowed
from anjo.sleep.app.judge_night import JudgePorts, StartVerdict, set_judge_ports

_PER_JUDGE_REASON = "your_nights_spent_today"


class CohortCaps:
    """``JudgeCaps`` over the cohort's daily counters: ``ok`` counts the night."""

    def try_start_night(self, user_id: str) -> StartVerdict:
        try:
            claim_night(user_id)
        except CapReached as refused:
            return "per_judge" if refused.reason == _PER_JUDGE_REASON else "global"
        return "ok"


def wire_judge_ports() -> None:
    """At startup on the sleep host: judges are the cohort's, and its caps decide each night."""
    set_judge_ports(JudgePorts(is_judge=night_allowed, caps=CohortCaps()))
