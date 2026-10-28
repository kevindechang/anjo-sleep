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
- **A contradiction**, in one of two families. **Plain** (the default): a plan with attribute
  ``+a``, then three days later a constraint with ``-a`` on the same thread, raised two days after;
  its decoy is a constraint with an unrelated attribute. Resemblance tells it (``contra_bar``'s
  untrained ``unmatched`` reads 0.68 to 0.70 of it, ``match_age`` 0.98 to 0.99 of its revisions,
  on 256 lives). The core's own training and tests run on it: trained on the stance family instead,
  the core's planted dependency separated at 0.42 on one seed of three at the tests' patience
  (0.91 with 60 epochs of it, over 400), and on stance transplant pairs the open thread held a slot
  of its own in 6 / 1 / 3 of 16 (3 still at that patience; ``test_mind_core`` asks 8).
  **Stance** (``stance=True``, the contradiction benchmark's, ``contra_run``): a plan about
  attribute ``a`` of a thread, holding a stance (``+s`` or ``-s`` along one **stance axis** every
  life shares, like a role: yes or no, on or off), then a constraint on the same thread about ``a``
  with the opposite stance, raised later. Its decoy is a constraint on the same thread that agrees
  (the same attribute and stance), or one about something else (another attribute, either stance),
  half each. Every constraint restates its attribute in part (``share``) and adds a detail of its
  own (``spread``), so how much it resembles the plan says nothing of a conflict, and every gap
  (plan to constraint, constraint to its raising) is drawn the same way for conflicts and decoys, so
  timing says nothing either. Conflict is the sign of the stance against the plan's, on the same
  attribute: an oracle with the generating parts (``oracle_parts``) reads it from the latents.

``revision_lives`` hold the hard case instead: the plan changes legitimately (plain: the next day,
to ``-a``; stance: the same attribute, the stance flipped, a drawn one or two days later), and the
constraint either follows the change (no conflict, though against the first plan it reads like a
contradiction) or still assumes the old plan (a stale assumption: a conflict). Stance draws its gap
to the revision the same way for both.

``pairs`` builds counterfactual autobiographies from the same generator: one life written twice,
identical in every draw except how the booked party's two earlier bookings ended (cancelled, or
delivered), cut at the new booking. Each also holds an unrelated thread still open at the cut: a
plan and a conflicting constraint (plain), not yet raised (a thread able to fail, whose risk the history
must not move).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Sequence
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
OPEN_START: Final = 1  # a transplant's open thread: its plan on day 1, its conflicting constraint on day 4
STANCE: Final = 3.0  # the stance axis's length (a part's is about sqrt(DIM)): clear of the noise, small beside the rest
REVISION_DAYS: Final = (1, 2)  # a revision comes this many days after the plan, drawn
GAP_DAYS: Final = (1, 3)  # a constraint this many days after the plan it speaks to (or its revision), drawn
RAISE_DAYS: Final = (1, 2)  # and is raised this many days after, drawn
SHARE: Final = (0.5, 1.0)  # how much of its attribute a constraint restates, drawn
SPREAD: Final = (0.0, 1.0)  # how much detail of its own it adds, drawn
_ROLE_SEED: Final = 20_261_028
Parts = tuple[np.ndarray, np.ndarray, np.ndarray]  # a contradiction's thread, attribute and its constraint's own detail
_STANCE_SEED: Final = 20_261_029


def _roles(dim: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(_ROLE_SEED + dim)
    return {role: rng.normal(size=dim).astype(np.float32) for role in ROLES}


def stance_axis(dim: int = DIM) -> np.ndarray:
    """The stance axis every life shares, ``STANCE`` long: a plan's or a constraint's sign along it
    is the stance it holds."""
    vec = np.random.default_rng(_STANCE_SEED + dim).normal(size=dim)
    out: np.ndarray = (STANCE * vec / np.linalg.norm(vec)).astype(np.float32)
    return out


@dataclass
class _Draft:
    """One life being written: its events (day, order, id, latent) and labels."""

    life_id: str
    rng: np.random.Generator
    dim: int
    events: list[tuple[int, float, str, np.ndarray]]
    labels: list[_Label]
    stance: bool = False  # which contradiction family it writes
    parts: dict[str, Parts] = field(default_factory=dict)  # each stance contradiction's generating parts

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


def _contradiction(
    draft: _Draft, conflicts: bool, *, start: int | None = None, raised: bool = True, revised: bool = False,
) -> None:
    (_stance if draft.stance else _plain)(draft, conflicts, start=start, raised=raised, revised=revised)


def _plain(draft: _Draft, conflicts: bool, *, start: int | None, raised: bool, revised: bool) -> None:
    """A plan, then three days later a constraint on the same thread; ``raised`` (two days after)
    ends it, else it stays open (the transplant's unrelated thread, whose times aren't labelled).
    ``revised``: the plan changes the next day, and the constraint follows the change, or (a
    conflict) still assumes the old plan."""
    thread, attribute = draft.part(), draft.part()
    start = int(draft.rng.integers(0, 3)) if start is None else start
    plan = draft.say(start, "plan", thread, attribute, key="plan")
    if revised:
        changed = draft.say(start + 1, "plan", thread, -attribute, key="revision")
        constraint = draft.say(start + 3, "constraint", thread, attribute if conflicts else -attribute, key="constraint")
        ids: tuple[str, ...] = (plan, changed, constraint)
    else:
        constraint = draft.say(start + 3, "constraint", thread, -attribute if conflicts else draft.part(),
                               key="constraint")
        ids = (plan, constraint)
    if not raised:
        return
    raised_id = draft.say(start + 5, "raise" if conflicts else "chat", thread, key="raised")
    draft.labels.append(_Label(
        id=f"{draft.life_id}-contradiction", family="contradiction" if conflicts else "decoy",
        event_ids=ids, origin_ids=ids, knowable_id=constraint, outcome_id=raised_id, failed=conflicts,
    ))


def _stance(draft: _Draft, conflicts: bool, *, start: int | None, raised: bool, revised: bool) -> None:
    """A plan, then a constraint on the same thread; ``raised`` ends it, else it stays open (the
    transplant's unrelated thread, whose times aren't labelled). ``revised``: the plan changes first,
    and the constraint follows the change, or (a conflict) still assumes the old plan. Every draw is
    made before anything is said, the same whether it conflicts or not, so a pair's two versions
    differ only from the constraint on."""
    rng, axis = draft.rng, stance_axis(draft.dim)
    thread, attribute, other, detail = draft.part(), draft.part(), draft.part(), draft.part()
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    start = int(rng.integers(0, 3)) if start is None else start
    revision = int(rng.integers(REVISION_DAYS[0], REVISION_DAYS[1] + 1))
    gap = int(rng.integers(GAP_DAYS[0], GAP_DAYS[1] + 1))
    raise_gap = int(rng.integers(RAISE_DAYS[0], RAISE_DAYS[1] + 1))
    share, spread = float(rng.uniform(*SHARE)), float(rng.uniform(*SPREAD))
    unrelated, other_sign = rng.uniform() < 0.5, (1.0 if rng.uniform() < 0.5 else -1.0)
    ids = [draft.say(start, "plan", thread, attribute, sign * axis, key="plan")]
    day, held = start, sign
    if revised:
        day, held = start + revision, -sign
        ids.append(draft.say(day, "plan", thread, attribute, held * axis, key="revision"))
    if conflicts or revised or not unrelated:
        about, stance = attribute, (-held if conflicts else held)
    else:
        about, stance = other, other_sign
    constraint = draft.say(day + gap, "constraint", thread, share * about, spread * detail, stance * axis,
                           key="constraint")
    ids.append(constraint)
    if not raised:
        return
    raised_id = draft.say(day + gap + raise_gap, "raise" if conflicts else "chat", thread, key="raised")
    label = f"{draft.life_id}-contradiction"
    draft.parts[label] = (thread, attribute, spread * detail)
    draft.labels.append(_Label(
        id=label, family="contradiction" if conflicts else "decoy", event_ids=tuple(ids), origin_ids=tuple(ids),
        knowable_id=constraint, outcome_id=raised_id, failed=conflicts,
    ))


def _stream(draft: _Draft) -> LabelledStream:
    times = _times(draft.events)
    events = sorted(
        (EventLatent(eid, draft.life_id, times[eid], "turn", vec) for _, _, eid, vec in draft.events),
        key=lambda e: (e.t, e.event_id),
    )
    days = tuple(DAY0 + timedelta(days=n) for n in range(DAYS))
    return LabelledStream(draft.life_id, days, tuple(events), tuple(label.resolve(times) for label in draft.labels))


def _draft(
    seed: int, index: int, dim: int, booked: str | None = None, clash: bool | None = None, revise: bool = False,
    stance: bool = False,
) -> _Draft:
    """One life being written. ``booked`` (``"cancel"`` or ``"deliver"``) writes the transplant
    version instead: the booked party's earlier bookings end that way, and the life stops at the new
    booking. Both versions make the same draws, so they differ only in those two roles. ``clash``
    writes a contradiction pair's version: the life always holds a plan and a constraint, which
    conflicts or not; up to the constraint both versions are the same. ``revise`` writes a revision
    life (its own ids): the plan always changes, and the constraint follows it or is stale.
    ``stance`` writes its contradiction in the stance family; every draw before it is the same."""
    rng = np.random.default_rng([seed, index])
    draft = _Draft(f"{'rev' if revise else 'toy'}{seed}-{index}", rng, dim, [], [], stance=stance)
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
        _contradiction(draft, conflicts=True, start=OPEN_START, raised=False)  # open at the cut (day 5 or 6)
        _booking(draft, first, record, True, cut=True)
        return draft
    if has_booking:
        _booking(draft, first if fails else second, record, bool(fails))
    if clash is not None or revise:
        _contradiction(draft, conflicts=bool(conflicts) if clash is None else clash, revised=revise)
    elif has_contradiction:
        _contradiction(draft, conflicts=bool(conflicts))
    return draft


def life(seed: int, index: int, *, dim: int = DIM, stance: bool = False) -> LabelledStream:
    """One toy life; the same seed and index (and family) give the same life."""
    return _stream(_draft(seed, index, dim, stance=stance))


def lives(n: int, *, seed: int, dim: int = DIM, stance: bool = False) -> tuple[LabelledStream, ...]:
    return tuple(life(seed, i, dim=dim, stance=stance) for i in range(n))


def revision_lives(n: int, *, seed: int, dim: int = DIM, stance: bool = False) -> tuple[LabelledStream, ...]:
    """Lives whose plan always changes once: half follow the change, half keep the stale assumption."""
    return tuple(_stream(_draft(seed, i, dim, revise=True, stance=stance)) for i in range(n))


def oracle(knowable: EventLatent, own: Sequence[EventLatent], parts: Parts) -> float:
    """A contradiction as its generating parts read it from the latents: how far the constraint's
    stance lies from the latest version of the plan's (``own[-1]``), in units of a sign flip, when
    it speaks to the plan's attribute; 0.0 when it speaks to something else."""
    dim = knowable.vec.shape[0]
    roles, (thread, attribute, detail) = _roles(dim), parts
    about = float((knowable.vec - roles["constraint"] - thread - detail) @ attribute / (attribute @ attribute))
    if about < SHARE[0] / 2 or not own:
        return 0.0
    plan = own[-1].vec - roles["plan"]
    return abs(float(stance_axis(dim) @ (plan - knowable.vec + roles["constraint"]))) / (2 * STANCE**2)


def oracle_parts(n: int, *, seed: int, dim: int = DIM, revisions: bool = False) -> dict[str, Parts]:
    """The generating parts of each contradiction in ``lives(n, seed=seed, stance=True)`` (or
    ``revision_lives``), by label id, for an oracle that reads the latents with them."""
    return {label: parts for i in range(n)
            for label, parts in _draft(seed, i, dim, revise=revisions, stance=True).parts.items()}


def _shifted(stream: LabelledStream, person: str, days: int) -> LabelledStream:
    by = timedelta(days=days)
    events = tuple(replace(e, life_or_person=person, t=e.t + by) for e in stream.events)
    labels = tuple(replace(lab, knowable_at=lab.knowable_at + by, outcome_at=lab.outcome_at + by) for lab in stream.labels)
    return LabelledStream(person, tuple(d + by for d in stream.days), events, labels)


def chained(n: int, *, seed: int, links: int, dim: int = DIM) -> tuple[LabelledStream, ...]:
    """``n`` long lives, each ``links`` toy lives lived one after another by one person: many
    nights, for what builds up across them (a fast memory that is never reset saturates)."""
    out = []
    for i in range(n):
        person = f"toy{seed}-chain{i}"
        parts = [_shifted(life(seed, i * links + j, dim=dim), person, j * DAYS) for j in range(links)]
        out.append(LabelledStream(person, tuple(d for p in parts for d in p.days),
                                  tuple(e for p in parts for e in p.events), tuple(lab for p in parts for lab in p.labels)))
    return tuple(out)


# ── Counterfactual autobiographies ────────────────────────────────────────────


@dataclass(frozen=True)
class TransplantPair:
    """One life written twice, cut at a new booking (``now``, the ``relevant`` event, the same in
    both): in ``problem`` the booked party's two earlier bookings were cancelled, in ``benign``
    delivered; every other event is identical. ``history`` are those earlier threads' events and
    ``moved`` the ones among them that differ (how each ended); ``unrelated`` are the events of a
    thread still open at the cut (a plan and a conflicting constraint, not yet raised)."""

    person: str
    problem: tuple[EventLatent, ...]
    benign: tuple[EventLatent, ...]
    now: tuple[EventLatent, ...]
    relevant: str
    history: tuple[str, ...]
    moved: tuple[str, ...]
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
            moved=tuple(a.event_id for a, b in zip(problem, benign, strict=True) if not np.array_equal(a.vec, b.vec)),
            unrelated=tuple(e.event_id for e in problem[:cut] if e.event_id.endswith(("-plan", "-constraint"))
                            and "-bg" not in e.event_id),
        ))
    return tuple(out)


@dataclass(frozen=True)
class ContrastPair:
    """A minimal pair for contrastive training: one life written twice, the same up to a
    constraint that flips the plan's attribute in ``problem`` and names an unrelated one in
    ``benign``; ``label`` is the contradiction's label id in both."""

    problem: LabelledStream
    benign: LabelledStream
    label: str


def contradiction_pairs(
    n: int, *, seed: int, dim: int = DIM, revisions: bool = False, stance: bool = False,
) -> tuple[ContrastPair, ...]:
    """``n`` minimal pairs; with ``revisions``, every odd one is a revision pair instead: the same
    changed plan, with a constraint still assuming the old plan (the problem) or following the change."""
    out = []
    for i in range(n):
        revise = revisions and i % 2 == 1
        out.append(ContrastPair(_stream(_draft(seed, i, dim, clash=True, revise=revise, stance=stance)),
                                _stream(_draft(seed, i, dim, clash=False, revise=revise, stance=stance)),
                                f"{'rev' if revise else 'toy'}{seed}-{i}-contradiction"))
    return tuple(out)
