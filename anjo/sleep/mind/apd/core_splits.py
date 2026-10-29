"""MIND core's lives: ``core-train``, ``core-val`` and ``core-test`` (MIND's data).

**The splits.** Each is a pure function of its name and ``CORE_LABEL`` (``generate.draw_specs``),
fixed before any of its lives was written: ``CORE_LIVES`` lives, ``CORE_EMPTY`` of them with no
planted item, the rest planting 1 to ``MAX_PREVALENCE`` items. They are held out from each other by
field and template (``core_families``). Only ``core-train`` (and APD's development lives) carry
training labels; ``core-val`` and ``core-test`` items stay sealed (``labels``).

**Subsets.** ``b12`` is the 12 ``core-test`` lives the notebook baseline B runs on. It was drawn by
seed before any run (``draw_subset``), stratified by prevalence with at least ``B12_EMPTY`` lives
that plant nothing, and is pinned in ``B12_SEEDS``: a change to the split or the draw fails its
test instead of moving the subset. A subset names lives by their spec's seed; a life generated on a
later attempt keeps its spec (``gen_batch.spec_ids``). Pure.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final

from anjo.sleep.mind.apd import generate
from anjo.sleep.mind.apd.gen_batch import spec_ids
from anjo.sleep.mind.apd.generate import LifeSpec
from anjo.sleep.mind.apd.types import CORE_SPLITS, MAX_PREVALENCE, CoreSplit, Life

CORE_LABEL: Final = "core-v1"
CORE_LIVES: Final[dict[CoreSplit, int]] = {"core-train": 120, "core-val": 20, "core-test": 40}
CORE_EMPTY: Final[dict[CoreSplit, int]] = {"core-train": 15, "core-val": 3, "core-test": 6}
SUBSETS: Final[dict[str, CoreSplit]] = {"b12": "core-test"}
B12_SIZE: Final = 12
B12_EMPTY: Final = 2  # at least this many of b12 plant nothing
B12_SEEDS: Final = (
    24669217588998, 77366009082834, 85947257106673, 88458211401902, 160925515386209, 168367185272422,
    176842782688129, 216932522140416, 235428662265663, 246609441459095, 262351798521200, 280154879010824,
)  # drawn 2026-10-28, before any core-test life was generated
# Amended 2026-10-29, before any core-test life or pair existed: the three core-test reversal plant
# lines (wedding-salty-lamb, theatre-hoarse-lead, photo-green-cast) changed text under the same ids.
# Each said its problem was gone, which gave the whole item away on its own day, so the leak gate
# failed every life that drew one (``core_families``). The split, its draw, b12 and the pair specs
# are unchanged; a life's made record keeps its field's template texts (``texts_digest``).
# Amended again 2026-10-29, still before any core-test life or pair existed: five core-test templates
# pin the days their facts name (wedding-salty-lamb, theatre-hoarse-lead, and theatre-advance-sales,
# whose second week went on sale "only yesterday" days after its sales were counted, now Tuesday the
# 1st) or move a date out of the life's days (theatre-tech-cast to the 17th, theatre-candle-approval
# to the 19th, the pairs' opening), under the same ids. A weekday said on one scheduled day and again
# on another named two dates for one event, and a date inside the life was in the past on some days
# that named it as to come (``test_apd_core_dates``). The split, its draw, b12 and the pair specs are
# unchanged.
# Amended a third time 2026-10-29, still before any core-test life or pair existed: two core-test plant
# lines stated their own meaning beside the fact (wedding-rsvp-count: "the other 60 ... will be on the
# final list"; theatre-school-matinee: a full house booked for the dark Wednesday itself) and now state
# the raw fact only, under the same ids (``test_apd_core_reversals``, ``leak_v1.md``). The split, its
# draw, b12 and the pair specs are unchanged.


def core_specs(split: CoreSplit) -> tuple[LifeSpec, ...]:
    """The split's lives, in order."""
    if split not in CORE_SPLITS:
        raise ValueError(f"{split!r} is not one of the core's splits {CORE_SPLITS}")
    return generate.draw_specs(split, f"{split}:{CORE_LABEL}", CORE_LIVES[split], CORE_EMPTY[split])


def _rank(label: str, specs: Sequence[LifeSpec]) -> list[LifeSpec]:
    return sorted(specs, key=lambda spec: hashlib.sha256(f"{label}:{spec.seed}".encode()).hexdigest())


def _shares(counts: dict[int, int], places: int) -> dict[int, int]:
    """``places`` spread over the strata in proportion to ``counts``, by largest remainder (ties to
    the lower prevalence), never more than a stratum holds."""
    total = sum(counts.values())
    exact = {p: places * n / total for p, n in counts.items()}
    shares = {p: int(share) for p, share in exact.items()}
    for p in sorted(counts, key=lambda p: (-(exact[p] - shares[p]), p))[:places - sum(shares.values())]:
        shares[p] += 1
    return {p: min(n, counts[p]) for p, n in shares.items()}


def draw_subset(specs: Sequence[LifeSpec], size: int, empty: int, label: str) -> tuple[LifeSpec, ...]:
    """``size`` of ``specs``, stratified by prevalence: ``empty`` that plant nothing, the rest spread
    over the prevalences 1 to ``MAX_PREVALENCE`` in proportion to the split; within a stratum, the
    first by the hash of ``label`` and the seed. In ``specs``' order."""
    by_prevalence = {p: [s for s in specs if s.prevalence == p] for p in range(MAX_PREVALENCE + 1)}
    counts = {p: len(found) for p, found in by_prevalence.items() if p and found}
    shares = {0: min(empty, len(by_prevalence[0])), **_shares(counts, size - empty)}
    chosen = {spec.seed for p, n in shares.items() for spec in _rank(label, by_prevalence[p])[:n]}
    return tuple(spec for spec in specs if spec.seed in chosen)


def subset_specs(subset: str) -> tuple[LifeSpec, ...]:
    """The pinned subset's specs, in its split's order."""
    if subset not in SUBSETS:
        raise ValueError(f"no subset {subset!r}; the subsets are {sorted(SUBSETS)}")
    pinned = frozenset(B12_SEEDS)
    found = tuple(spec for spec in core_specs(SUBSETS[subset]) if spec.seed in pinned)
    if len(found) != len(pinned):
        raise ValueError(f"{subset}'s pinned seeds aren't all {SUBSETS[subset]} lives")
    return found


def in_subset(lives: Sequence[Life], subset: str) -> tuple[Life, ...]:
    """The lives on disk that belong to the subset, by their specs' ids."""
    ids = frozenset(life_id for spec in subset_specs(subset) for life_id in spec_ids(spec))
    return tuple(life for life in lives if life.id in ids)
