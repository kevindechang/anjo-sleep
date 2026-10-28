"""Tonight's nap or deep sleep for one person: the ``sleep`` step, after ``notice`` and before the probes.

It gathers what ``sleep_pressure.decide`` reads: the turns and the trainable items since her last
deep sleep, the open targets she is most at risk on, the days since that sleep and her settled spend
this month (``past.py``), and asks the pressure port (``NightDeps.sleep_pressure``: the stand-in, or
MIND's core's once it ships). The step's record, which the summary carries as ``sleep``, adds
tonight's own turn count (``turns``, None when nothing counts them), which later nights add up.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Final

from anjo.sleep.night.deps import NightDeps
from anjo.sleep.night.nightlog import summaries
from anjo.sleep.night.past import days_since_deep, last_deep, month_spent, turns_since
from anjo.sleep.night.sleep_pressure import SleepDecision, SleepState, decide, high_risk
from anjo.sleep.ports import EvidenceStore, TargetStore

NAP_WHY: Final = "napped: no probes, practice, training, gates, promotion or morning tonight"


def tonight(
    deps: NightDeps, uid: str, night: date, window: tuple[datetime, datetime], as_of: datetime,
    store: EvidenceStore, targets: TargetStore,
) -> tuple[SleepDecision, dict[str, Any]]:
    """The decision and the step's record. Needs ``deps.sleep_policy``."""
    if deps.sleep_policy is None:
        raise ValueError("a night decides its sleep only under a policy")
    history = summaries(deps.logs(uid))
    since = last_deep(history, night)
    turns = None if deps.turns is None else deps.turns(uid, window)
    fresh = [item for item in store.trainable(night) if item.kind != "practice" and (since is None or item.day > since)]
    spent = 0.0 if deps.ledger is None else month_spent(deps.ledger.settlements(), uid, night)
    state = SleepState(
        new_turns=turns_since(history, since, night) + (turns or 0), new_evidence=len(fresh),
        open_high_risk=high_risk(targets.open_targets(as_of), as_of),
        days_since_deep=days_since_deep(history, night), month_spent_usd=spent,
    )
    made = decide(state, deps.sleep_policy, pressure=deps.sleep_pressure(state), force_deep=deps.settings.force_deep)
    return made, {**made.record(), "turns": turns}
