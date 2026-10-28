"""Toy lives with deterministic latents: the core's fakes until real lives and latents arrive.

Torch-free. An event's latent is a sum of parts plus a little noise: its **role** (what kind of
thing was said: a booking, a cancellation, a delivery, a plan, a constraint, small talk, done, the
person raising a problem), shared by every life like word meanings, and per-life random parts: the
**thread** it belongs to, the **party** it involves (a supplier, a crew member) and an
**attribute** (a plan's look, a constraint's room). So binding by thread and remembering a party
must be done per person, and can't be memorized across lives.

Each life holds, over ``DAYS`` days, background threads that end well and the history of a party
who cancelled twice, in threads that are done; often a second party who delivered twice. Then,
each on its own draw:

- **The planted dependency (``gap``):** a new booking, the same in its own words whoever it is
  with. With the party who cancelled, it fails two days later: only a memory of that party across
  closed threads tells. Its decoy is the same booking with the party who delivered, or with a
  party never seen.
- **A contradiction:** a plan with attribute ``+a``, then three days later a constraint with ``-a``
  on the same thread, raised two days after. Its decoy is a constraint with an unrelated attribute.

``pairs`` builds counterfactual autobiographies from the same generator: one life written twice,
identical in every draw except how the booked party's two earlier bookings ended (cancelled, or
delivered), cut at the new booking. The background threads before it are the unrelated part.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Final

import numpy as np

from anjo.sleep.mind.core.types import EventLatent, LabelledStream, ThreadLabel

DIM: Final = 32
DAYS: Final = 10
DAY0: Final = date(2026, 10, 1)
ROLES: Final = ("commit", "cancel", "deliver", "plan", "constraint", "chat", "done", "raise")
NOISE: Final = 0.15
BOOKING: Final = 0.7  # lives with a new booking (a gap, or its decoy)
CONTRADICTION: Final = 0.5  # lives with a plan and a constraint (a contradiction, or its decoy)
SECOND_PARTY: Final = 0.7  # lives with a party who delivered twice
_ROLE_SEED: Final = 20_261_028


def _roles(dim: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(_ROLE_SEED + dim)
    return {role: rng.normal(size=dim).astype(np.float32) for role in ROLES}


@dataclass
class _Draft:
    """One life being written: its events (day, order, id, latent) and labels."""

    life_id: str
    rng: np.random.Generator
    dim: int
    events: list[tuple[int, float, str, np.ndarray]]
    labels: list[_Label]

    def vector(self, role: str, *parts: np.ndarray) -> np.ndarray:
        vec = _roles(self.dim)[role] + sum(parts, np.zeros(self.dim, dtype=np.float32))
        out: np.ndarray = (vec + NOISE * self.rng.normal(size=self.dim)).astype(np.float32)
        return out

    def part(self) -> np.ndarray:
        out: np.ndarray = self.rng.normal(size=self.dim).astype(np.float32)
        return out

    def say(self, day: int, role: str, *parts: np.ndarray, key: str | None = None) -> str:
        event_id = f"{self.life_id}-{key or f'e{len(self.events)}'}"
        self.events.append((day, float(self.rng.uniform()), event_id, self.vector(role, *parts)))
        return event_id



@dataclass(frozen=True)
class _Label:
    """A label whose times are read from its events once the life is complete (a later event on
    the same day moves the times of the day's events)."""

    id: str
    family: str
    event_ids: tuple[str, ...]
    origin_ids: tuple[str, ...]
    knowable_id: str
    outcome_id: str
    failed: bool
    missing: bool = False

    def resolve(self, times: dict[str, datetime]) -> ThreadLabel:
        return ThreadLabel(
            id=self.id, family=self.family, event_ids=self.event_ids, origin_ids=self.origin_ids,
            knowable_at=times[self.knowable_id], outcome_at=times[self.outcome_id], failed=self.failed,
            missing=self.missing,
        )


def _times(events: list[tuple[int, float, str, np.ndarray]]) -> dict[str, datetime]:
    """Each day's events from 08:00, 90 minutes apart, in their drawn order."""
    out: dict[str, datetime] = {}
    for day in sorted({e[0] for e in events}):
        todays = sorted((e for e in events if e[0] == day), key=lambda e: (e[1], e[2]))
        start = datetime.combine(DAY0 + timedelta(days=day), time(8), tzinfo=timezone.utc)
        out.update({e[2]: start + timedelta(minutes=90 * n) for n, e in enumerate(todays)})
    return out


def _background(draft: _Draft) -> None:
    for n in range(int(draft.rng.integers(2, 4))):
        thread, party = draft.part(), draft.part()
        start = int(draft.rng.integers(0, DAYS - 3))
        end = int(draft.rng.integers(start + 1, DAYS))
        draft.say(start, "plan", thread, party, key=f"bg{n}-plan")
        for m, day in enumerate(sorted(draft.rng.choice(np.arange(start, end + 1), size=2))):
            draft.say(int(day), "chat", thread, key=f"bg{n}-chat{m}")
        draft.say(end, "done", thread, key=f"bg{n}-done")


def _history(draft: _Draft, party: np.ndarray, outcome: str, days: tuple[int, int], name: str) -> list[str]:
    """Two earlier bookings with ``party``, each ending in ``outcome`` and closed."""
    ids = []
    for n, day in enumerate(days):
        thread = draft.part()
        draft.say(day, "commit", thread, party, key=f"{name}{n}")
        ids.append(draft.say(day + 1, outcome, thread, party, key=f"{name}{n}-end"))
        draft.say(day + 1, "done", thread, key=f"{name}{n}-done")
    return ids


def _booking(draft: _Draft, party: np.ndarray, record: list[str], fails: bool, *, cut: bool = False) -> str:
    """The new booking (its id); ``cut`` stops at the booking, for a transplant pair."""
    day = int(draft.rng.integers(5, 7))
    thread = draft.part()
    booked = draft.say(day, "commit", thread, party, key="booking")
    if cut:
        return booked
    ends = draft.say(day + 2, "cancel" if fails else "deliver", thread, party, key="booking-end")
    draft.labels.append(_Label(
        id=f"{draft.life_id}-gap", family="gap" if fails else "decoy", event_ids=(booked,),
        origin_ids=(booked, *record) if fails else (booked,), knowable_id=booked, outcome_id=ends,
        failed=fails, missing=fails,
    ))
    return booked


def _contradiction(draft: _Draft, conflicts: bool) -> None:
    thread, attribute = draft.part(), draft.part()
    start = int(draft.rng.integers(0, 3))
    plan = draft.say(start, "plan", thread, attribute, key="plan")
    other = -attribute if conflicts else draft.part()
    constraint = draft.say(start + 3, "constraint", thread, other, key="constraint")
    raised = draft.say(start + 5, "raise" if conflicts else "chat", thread, key="raised")
    draft.labels.append(_Label(
        id=f"{draft.life_id}-contradiction", family="contradiction" if conflicts else "decoy",
        event_ids=(plan, constraint), origin_ids=(plan, constraint), knowable_id=constraint, outcome_id=raised,
        failed=conflicts,
    ))


def _stream(draft: _Draft) -> LabelledStream:
    times = _times(draft.events)
    events = sorted(
        (EventLatent(eid, draft.life_id, times[eid], "turn", vec) for _, _, eid, vec in draft.events),
        key=lambda e: (e.t, e.event_id),
    )
    days = tuple(DAY0 + timedelta(days=n) for n in range(DAYS))
    return LabelledStream(draft.life_id, days, tuple(events), tuple(label.resolve(times) for label in draft.labels))


def _draft(seed: int, index: int, dim: int, booked: str | None = None) -> _Draft:
    """One life being written. ``booked`` (``"cancel"`` or ``"deliver"``) writes the transplant
    version instead: the booked party's earlier bookings end that way, and the life stops at the new
    booking. Both versions make the same draws, so they differ only in those two roles."""
    rng = np.random.default_rng([seed, index])
    draft = _Draft(f"toy{seed}-{index}", rng, dim, [], [])
    _background(draft)
    first, second = draft.part(), draft.part()
    record = _history(draft, first, booked or "cancel", (0, 2), "old")
    if rng.uniform() < SECOND_PARTY:
        _history(draft, second, "deliver", (1, 3), "other")
    else:
        second = draft.part()  # a party never seen
    has_booking, fails = rng.uniform() < BOOKING, rng.uniform() < 0.5
    has_contradiction, conflicts = rng.uniform() < CONTRADICTION, rng.uniform() < 0.5
    if booked is not None:
        _booking(draft, first, record, True, cut=True)
        return draft
    if has_booking:
        _booking(draft, first if fails else second, record, bool(fails))
    if has_contradiction:
        _contradiction(draft, conflicts=bool(conflicts))
    return draft


def life(seed: int, index: int, *, dim: int = DIM) -> LabelledStream:
    """One toy life; the same seed and index give the same life."""
    return _stream(_draft(seed, index, dim))


def lives(n: int, *, seed: int, dim: int = DIM) -> tuple[LabelledStream, ...]:
    return tuple(life(seed, i, dim=dim) for i in range(n))


# ── Counterfactual autobiographies ────────────────────────────────────────────


@dataclass(frozen=True)
class TransplantPair:
    """One life written twice, cut at a new booking (``now``, the ``relevant`` event, the same in
    both): in ``problem`` the booked party's two earlier bookings were cancelled, in ``benign``
    delivered; every other event is identical. ``history`` are those earlier threads' events;
    ``unrelated`` are the background events before the booking."""

    person: str
    problem: tuple[EventLatent, ...]
    benign: tuple[EventLatent, ...]
    now: tuple[EventLatent, ...]
    relevant: str
    history: tuple[str, ...]
    unrelated: tuple[str, ...]


def pairs(n: int, *, seed: int, dim: int = DIM) -> tuple[TransplantPair, ...]:
    out = []
    for index in range(n):
        problem, benign = (_stream(_draft(seed, index, dim, booked=way)).events for way in ("cancel", "deliver"))
        relevant = f"toy{seed}-{index}-booking"
        cut = next(n for n, e in enumerate(problem) if e.event_id == relevant)
        if [e.event_id for e in benign] != [e.event_id for e in problem]:
            raise AssertionError("the two versions of a transplant pair must hold the same events")
        out.append(TransplantPair(
            person=f"toy{seed}-{index}", problem=problem[:cut], benign=benign[:cut], now=problem[cut:cut + 1],
            relevant=relevant, history=tuple(e.event_id for e in problem[:cut] if "-old" in e.event_id),
            unrelated=tuple(e.event_id for e in problem[:cut] if "-bg" in e.event_id),
        ))
    return tuple(out)
