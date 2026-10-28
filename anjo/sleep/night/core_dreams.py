"""MIND's core as the sleep objective's dream source (``NightDeps.dreams``).

With ``ANJO_SLEEP_DREAMS=1`` (``sft_dpo_seed``) and the core on, the dataset step asks her Glimmer
to speak from the states tonight's dream kept (``Dreamt.receipt``'s ``boundaries`` and ``dreams``,
read by ``dreams.seeds_from_dream``). Each seed's thread is the text of the real events it rests on,
one a line, from her evidence that may train only (never a held-out item, nor one whose ancestry
can't train), and its ancestry is theirs; a state resting on none of them seeds nothing. For anyone
the core doesn't serve there is no seed; with the core off the source is ``NoDreams``, and the
dataset receipt says the dreams are not wired.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date

from anjo.sleep.night.deps import MindCore, NoCore
from anjo.sleep.night.dreams import DreamClient, DreamSeed, DreamSource, GlimmerDreams, NoDreams, seeds_from_dream
from anjo.sleep.ports import EvidenceStore
from anjo.sleep.provenance import union
from anjo.sleep.types import Evidence


def core_dreams(
    core: MindCore, evidence: Callable[[str], EvidenceStore], client: DreamClient, *, seed: int,
) -> DreamSource:
    """Her Glimmer's dreams from the core's dream seeds, or ``NoDreams`` with the core off."""
    if isinstance(core, NoCore):
        return NoDreams()

    def seeds(user_id: str, night: date) -> tuple[DreamSeed, ...]:
        dreamt = core.dreamt(user_id, night)
        if dreamt is None:
            return ()
        held = {item.id: item for item in evidence(user_id).trainable(night) if item.text.strip()}
        made = seeds_from_dream(dreamt.receipt, lambda ids: _describe([held[i] for i in ids if i in held]))
        return tuple(made_seed for made_seed in made if made_seed.thread)

    return GlimmerDreams(client, seeds, seed=seed)


def _describe(items: Sequence[Evidence]) -> tuple[str, frozenset[str] | None]:
    """The thread as the items say it, one a line, and their ancestry."""
    return "\n".join(f"- {item.text.strip()}" for item in items), union(item.ancestry for item in items)
