"""Tonight's nap or deep sleep for one person: the ``sleep`` step, right after ``extract``, and what a nap
puts off.

``tonight`` gathers what ``sleep_pressure.decide`` reads: the turns and the trainable items since her
last deep sleep, the open targets she is most at risk on, the days since that sleep, her settled
spend this month (``past.py``), the month's nights left, Wonder's room left and whether tonight's
practice is already paid for (a rerun), and asks the pressure port (``NightDeps.sleep_pressure``: the
stand-in, or MIND's core's once it ships). The step's record, which the summary carries as ``sleep``,
adds tonight's own turn count (``turns``, None when nothing counts them), which later nights add up.

A nap wakes no Glimmer. It runs the CPU steps (the sweep, ``close_day``, ``extract``, the chooser,
a Wonder study, the brief) and skips the probes and the GPU steps (``NAP_WHY``). What needs Glimmer
waits for her next deep sleep (``DEFER_WHY``), deferred, not lost: MIND's notice pass, its
``inquire`` (not offered, so the question stays live) and the note (held back: its target stays
unsettled and its results unconsumed, so the next deep night's chooser picks it up). That night's
notice pass reads every item since her last night that woke (``past.notice_from``).

A nap at the month's cap (``capped``) skips its paid steps too (``CAPPED_WHY``). With no room even for a
nap (``at_cap``) the night is a capped nap whatever her day held, so it is known before extract, which
it skips: its record keeps the day's window (``extract_deferred``) and the next night with room
extracts every day so deferred first (``deferred_days``). No study is offered and the brief waits; the
next night with room folds every item since (``Briefs.unfolded``).

A deep night is priced on her own (``past.measured_deep``): the settled cost of her last deep nights at
tonight's settings, MIND's core (the policy's), the thought channel (``NightDeps.channel``) and practice
fresh or a rerun's, which every ``sleep`` record keeps (``core``, ``channel``, ``practice_fresh``).
"""
from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any, Final

from anjo.sleep.mind import receipt
from anjo.sleep.night.deps import NightDeps
from anjo.sleep.night.nightlog import summaries
from anjo.sleep.night.past import (
    days_since_deep,
    gpu_batches,
    last_deep,
    measured_deep,
    month_spent,
    notice_from,
    turns_since,
)
from anjo.sleep.night.sleep_pressure import SleepDecision, SleepState, decide, fits, high_risk
from anjo.sleep.night.state import GPU_STEPS, PersonNight
from anjo.sleep.ports import EvidenceStore, TargetStore
from anjo.sleep.types import Choice, Evidence

NAP_WHY: Final = "napped: no probes, practice, training, gates, promotion or morning tonight"
DEFER_WHY: Final = "napped: deferred to her next waking, so no Glimmer wakes tonight"
NAPLESS: Final = frozenset({"probes", *GPU_STEPS})
CAPPED_WHY: Final = "capped: no room under the month's cap, so the nap's paid steps wait for a night with room"
CAPPED_STEPS: Final = frozenset({"brief"})  # and no study offered (``run._offered``)


def tonight(
    deps: NightDeps, uid: str, night: date, window: tuple[datetime, datetime], as_of: datetime,
    store: EvidenceStore, targets: TargetStore, wonder_spent: float = 0.0,
) -> tuple[SleepDecision, dict[str, Any], date]:
    """The decision, the step's record and the first day tonight's notice pass reads. Needs
    ``deps.sleep_policy``. ``wonder_spent`` is Wonder's spend in its rolling window."""
    if deps.sleep_policy is None:
        raise ValueError("a night decides its sleep only under a policy")
    history = summaries(deps.logs(uid))
    since = last_deep(history, night)
    turns = None if deps.turns is None else deps.turns(uid, window)
    fresh = [item for item in store.trainable(night) if item.kind != "practice" and (since is None or item.day > since)]
    settings = {"core": deps.sleep_policy.core, "channel": deps.channel is not None,
                "practice_fresh": not deps.transcripts(uid).for_night(night)}
    measured = () if deps.ledger is None else measured_deep(
        history, deps.ledger.settlements(), uid, night, settings, batches=gpu_batches(history),
        units=deps.ledger.settlements("cu"))
    state = SleepState(
        new_turns=turns_since(history, since, night) + (turns or 0), new_evidence=len(fresh),
        open_high_risk=high_risk(targets.open_targets(as_of), as_of), days_since_deep=days_since_deep(history, night),
        **_month(deps, uid, night, history, wonder_spent), practice_fresh=settings["practice_fresh"],
        measured_deep_usd=measured,
    )
    made = decide(state, deps.sleep_policy, pressure=deps.sleep_pressure(state), force_deep=deps.settings.force_deep)
    return made, {**made.record(), **settings, "turns": turns}, notice_from(history, night)


def _month(deps: NightDeps, uid: str, night: date, history: Sequence[Mapping[str, Any]],
           wonder_spent: float) -> dict[str, Any]:
    """The month's side of tonight's state: her settled spend, the nights left after tonight and Wonder's room."""
    spent = 0.0 if deps.ledger is None else month_spent(
        deps.ledger.settlements(), uid, night, batches=gpu_batches(history), units=deps.ledger.settlements("cu"))
    return {"month_spent_usd": spent, "nights_left": calendar.monthrange(night.year, night.month)[1] - night.day,
            "wonder_left_usd": max(0.0, deps.settings.wonder_usd_month - wonder_spent)}


def at_cap(deps: NightDeps, p: PersonNight) -> bool:
    """No room under the month's cap even for a nap: tonight is a capped nap whatever her day held."""
    if deps.sleep_policy is None:
        return False
    month = _month(deps, p.uid, p.night, summaries(deps.logs(p.uid)), p.wonder_month_spent)
    state = SleepState(new_turns=0, new_evidence=0, open_high_risk=0, days_since_deep=None, **month)
    return not fits(state, deps.sleep_policy, deps.sleep_policy.nap_usd)


def deferred_days(history: Sequence[Mapping[str, Any]], night: date) -> list[tuple[date, tuple[datetime, datetime], datetime]]:
    """The days capped naps left unextracted since her last night that extracted, oldest first: each
    day, its window and its clock (``as_of``)."""
    found = []
    for summary in sorted(history, key=lambda s: str(s.get("night")), reverse=True):
        if str(summary.get("night")) >= night.isoformat():
            continue
        sleep = summary.get("sleep")
        deferred = sleep.get("extract_deferred") if isinstance(sleep, Mapping) else None
        if not isinstance(deferred, Mapping):
            break
        start, end = (datetime.fromisoformat(at) for at in deferred["window"])
        found.append((date.fromisoformat(str(summary["night"])), (start, end), datetime.fromisoformat(deferred["as_of"])))
    return found[::-1]


def sleep_step(deps: NightDeps, p: PersonNight) -> Mapping[str, Any]:
    """The ``sleep`` step: a nap stops the night's training and anything that wakes Glimmer."""
    made, record, p.notice_from = tonight(deps, p.uid, p.night, p.window, p.as_of, p.store, p.targets,
                                          p.wonder_month_spent)
    if made.kind == "nap":
        p.due, p.status = False, "napped"
    if p.capped:  # extract waited for a night with room: the window it reads then
        window = [p.window[0].isoformat(), p.window[1].isoformat()]
        record = {**record, "extract_deferred": {"window": window, "as_of": p.as_of.isoformat()}}
    p.capped = p.capped or (made.kind == "nap" and made.capped)
    return record


def mind_on(deps: NightDeps) -> bool:
    """MIND's inquiry runs (Task 17): the setting on and its port real."""
    return deps.settings.mind_inquiry and deps.inquiry.available


def put_off(deps: NightDeps, p: PersonNight, step: str) -> str | None:
    """Why tonight skips ``step``, or None when it runs: a nap runs no probes or GPU step, and
    puts MIND's notice pass, which wakes Glimmer, off to her next deep sleep. Before extract it marks a
    night with no room even for a nap ``capped`` (``at_cap``), which then skips extract too."""
    if step == "extract":
        p.capped = at_cap(deps, p)
        return CAPPED_WHY if p.capped else None
    if not p.napped:
        return None
    if step in NAPLESS:
        return NAP_WHY
    if p.capped and step in CAPPED_STEPS:
        return CAPPED_WHY
    if step == "notice" and deps.settings.mind_inquiry and deps.notice.available:
        return DEFER_WHY
    return None


def unnoticed(p: PersonNight) -> list[Evidence]:
    """The items tonight's notice pass reads: tonight's, and after naps every one since her last
    night that woke (the pass leaves out what her workspace already knows)."""
    if p.notice_from is None or p.notice_from >= p.night:
        return [item for item in p.store.for_day(p.night) if not item.held_out]
    start = p.notice_from
    return [item for item in p.store.stored(p.night) if item.day >= start]


def deferred_note(choice: Choice, target_id: str) -> dict[str, Any]:
    """A nap's note: not written (the writer is Glimmer), its target left for the next deep sleep."""
    return {"outcome": "deferred", "target_id": target_id, "why": DEFER_WHY, "receipt": receipt.record(choice, None)}
