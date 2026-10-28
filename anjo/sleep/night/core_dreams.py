"""MIND's core as the sleep objective's dream source (``NightDeps.dreams``).

With ``ANJO_SLEEP_DREAMS=1`` (``sft_dpo_seed``) and the core on, the dataset step asks her Glimmer
to speak from the states tonight's dream kept (``Dreamt.receipt``'s ``boundaries`` and ``dreams``,
read by ``dreams.seeds_from_dream``). Each seed's thread is the text of the real events it rests on,
one a line, from her evidence that may train only (never a held-out item, nor one whose ancestry
can't train), and its ancestry is theirs; a state resting on none of them seeds nothing. For anyone
the core doesn't serve there is no seed; with the core off the source is ``NoDreams``, and the
dataset receipt says the dreams are not wired.

**Through the thought channel** (``ANJO_SLEEP_DREAMS_CHANNEL=1``, ``dreams_channel``; it needs the
channel on, ``ANJO_SLEEP_CHANNEL``): each seed's imagined state is read back from the core
(``MindCore.imagined``, by the id its receipt kept) and her reply is made through the channel over
it, in the night's arm and through the projector her night reads (``channel_night.channel_for``),
so each dream row carries that state's snapshot. A seed whose state can't be read back stops the
dreams (the dataset step fails), never a dream made without its state. Off, the dreams are as before.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Final

from anjo.sleep.night.channel_night import ChannelSetup, channel_for
from anjo.sleep.night.core_snapshot import CHANNEL_ENV, Channel
from anjo.sleep.night.deps import MindCore, NoCore
from anjo.sleep.night.dreams import DreamClient, DreamSeed, DreamSource, GlimmerDreams, NoDreams, seeds_from_dream
from anjo.sleep.ports import EvidenceStore
from anjo.sleep.provenance import union
from anjo.sleep.types import Evidence


DREAMS_CHANNEL_ENV: Final = "ANJO_SLEEP_DREAMS_CHANNEL"  # "1": each dream's reply through the thought channel


def dreams_channel(env: Mapping[str, str], channel: ChannelSetup | None) -> ChannelSetup | None:
    """The channel the dreams go through: the night's, under ``ANJO_SLEEP_DREAMS_CHANNEL=1``; None when
    it's unset or 0. Set with the channel off, the night stops at build."""
    raw = env.get(DREAMS_CHANNEL_ENV, "").strip()
    if raw in ("", "0"):
        return None
    if raw != "1":
        raise ValueError(f"{DREAMS_CHANNEL_ENV} is 1, or 0 or unset for off")
    if channel is None:
        raise ValueError(f"{DREAMS_CHANNEL_ENV}=1 needs the thought channel on ({CHANNEL_ENV})")
    return channel


def core_dreams(
    core: MindCore, evidence: Callable[[str], EvidenceStore], client: DreamClient, *, seed: int,
    channel: ChannelSetup | None = None,
) -> DreamSource:
    """Her Glimmer's dreams from the core's dream seeds, each reply through ``channel`` over its own
    imagined state when it is given; ``NoDreams`` with the core off."""
    if isinstance(core, NoCore):
        return NoDreams()

    def seeds(user_id: str, night: date) -> tuple[DreamSeed, ...]:
        dreamt = core.dreamt(user_id, night)
        if dreamt is None:
            return ()
        held = {item.id: item for item in evidence(user_id).trainable(night) if item.text.strip()}
        made = seeds_from_dream(dreamt.receipt, lambda ids: _describe([held[i] for i in ids if i in held]))
        return tuple(made_seed for made_seed in made if made_seed.thread)

    def through(user_id: str, night: date, dream_seed: DreamSeed) -> Channel:
        assert channel is not None
        reading = core.imagined(user_id, night, dream_seed.state_ref)
        if reading is None or reading.state_id != dream_seed.state_ref:
            raise ValueError("a dream's imagined state can't be read back for the thought channel")
        return channel_for(channel, reading)

    return GlimmerDreams(client, seeds, seed=seed, channel=None if channel is None else through)


def _describe(items: Sequence[Evidence]) -> tuple[str, frozenset[str] | None]:
    """The thread as the items say it, one a line, and their ancestry."""
    return "\n".join(f"- {item.text.strip()}" for item in items), union(item.ancestry for item in items)
