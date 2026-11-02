"""The judges' night (2026-11-02): a compressed night on a judge's own data, run in two phases.

Only a judge's night carries a profile (``NightSettings.demo``); Teo's and Kevin's nights carry None
and every step runs as before, to the byte. With a profile, ``run.py`` runs:

- **Phase 1**, all on the CPU side except MIND's own thinker calls: close the day, extract, MIND's
  notice, choose (and inquire), the brief, then three steps only a profile adds: ``homework`` (a
  chosen study starts in the background at ``choose`` and runs while the brief is folded and her
  prediction is committed; this step waits for it), ``self_predict`` (MIND's loop commits its
  prediction about her own practice now, before practice, rather than inside the practice step) and
  ``note`` (her note is written at the end of phase 1).
- **Phase 2**: practice, dataset, train, gates and promote, as on every night, then ``refine``: when
  the gates kept tonight's weights, the note is written again by the promoted adapter; when they
  rejected them, the phase-1 note gets the held-back line; otherwise the note stands.

The profile's numbers, each overridable on the judge host's environment (``from_env``):

- ``practice_scenes``: 8 (full practice's floor is 16 for every other night);
- ``token_cap``: about 5,000 training tokens (the registered night cap is 16,384);
- ``general_items`` and ``collapse_prompts``: the fast gate, 40 general items and 10 collapse prompts,
  a fixed subset of the registered sets (``fast_subset``): ordered by sha256 over the seed and the
  item's id, the first ``n``, kept in registered order. Its receipt records the counts, the seed and
  the chosen ids' sha256, so a verdict can be recomputed on the same items. The registered thresholds
  are kept, so a fast gate is stricter, never looser: one newly wrong item of 40 is 2.5 points (over
  the 2-point bound) and one more dragged prompt of 10 is 10 points (over the 5-point bound). The
  base model's answers to the subset are cached like the full sets' (``gates.base_cache``);
- ``train_steps``: 8 optimizer steps of plain SFT (the trainer's step plan, ``StepPlan``);
- ``probes``: off. A judge's night is never evaluated against sealed probes, so it writes none (a
  real person's probes are a Muse Spark call), and the step is logged ``skipped`` with that reason.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Final, TypeVar

from anjo.sleep.sets.loader import CollapsePrompt, GeneralItem, load_collapse, load_general

SUBSET_SEED: Final = 20261102  # fixed before any judge night ran; the fast gate's items follow from it
DEMO_MIN_SCENES: Final = 4  # full practice's floor under a profile (16 on every other night)
SCENES_ENV: Final = "ANJO_SLEEP_DEMO_SCENES"
TOKEN_CAP_ENV: Final = "ANJO_SLEEP_DEMO_TOKEN_CAP"
GENERAL_ENV: Final = "ANJO_SLEEP_DEMO_GENERAL"
COLLAPSE_ENV: Final = "ANJO_SLEEP_DEMO_COLLAPSE"
STEPS_ENV: Final = "ANJO_SLEEP_DEMO_STEPS"
PROBES_SKIPPED: Final = "a judge's night is never evaluated on sealed probes, so it writes none"
T = TypeVar("T")


class DemoProfileError(ValueError):
    """A profile setting outside what a judge's night may run."""


def _bounded(name: str, value: int, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise DemoProfileError(f"{name} is a whole number from {low} to {high}")
    return value


def ids_digest(ids: Sequence[str]) -> str:
    """One sha256 over a subset's ids, in sorted order: the record a verdict is recomputed from."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def fast_subset(items: Sequence[T], n: int, seed: int, id_of: Callable[[T], str]) -> tuple[T, ...]:
    """The first ``n`` of ``items`` ordered by sha256 over ``seed`` and each id, kept in their own
    order: the same items for the same set, size and seed, whoever asks."""
    ranked = sorted(items, key=lambda item: hashlib.sha256(f"{seed}\x1f{id_of(item)}".encode("utf-8")).hexdigest())
    chosen = {id_of(item) for item in ranked[:n]}
    return tuple(item for item in items if id_of(item) in chosen)


@dataclass(frozen=True)
class DemoProfile:
    """A judge's night: what it cuts, and by how much (the module's docstring has the why)."""

    practice_scenes: int = 8
    token_cap: int = 5000
    general_items: int = 40
    collapse_prompts: int = 10
    train_steps: int = 8
    probes: bool = False
    subset_seed: int = SUBSET_SEED

    def __post_init__(self) -> None:
        _bounded("practice_scenes", self.practice_scenes, DEMO_MIN_SCENES, 20)
        _bounded("token_cap", self.token_cap, 1000, 16384)
        _bounded("general_items", self.general_items, 10, 200)
        _bounded("collapse_prompts", self.collapse_prompts, 5, 60)
        _bounded("train_steps", self.train_steps, 1, 64)

    def general_set(self) -> tuple[GeneralItem, ...]:
        """The fast gate's general items, from the registered 200."""
        return _general(self.general_items, self.subset_seed)

    def collapse_set(self) -> tuple[CollapsePrompt, ...]:
        """The fast gate's collapse prompts, from the registered 60, before the person's fields are left out."""
        return _collapse(self.collapse_prompts, self.subset_seed)

    def gate_receipt(self) -> dict[str, object]:
        """What the gates step records of the fast gate: counts, the seed and each subset's ids' sha256."""
        general, collapse = self.general_set(), self.collapse_set()
        return {
            "general_items": len(general), "general_ids_sha256": ids_digest([i.id for i in general]),
            "collapse_prompts": len(collapse), "collapse_ids_sha256": ids_digest([p.id for p in collapse]),
            "seed": self.subset_seed, "thresholds": "registered",
        }


@cache
def _general(n: int, seed: int) -> tuple[GeneralItem, ...]:
    return fast_subset(load_general(), n, seed, lambda item: item.id)


@cache
def _collapse(n: int, seed: int) -> tuple[CollapsePrompt, ...]:
    return fast_subset(load_collapse(), n, seed, lambda prompt: prompt.id)


def from_env(environ: Mapping[str, str]) -> DemoProfile:
    """The profile, with any of its numbers the judge host's environment overrides. A value that
    isn't a whole number in range stops the night before it starts."""
    base = DemoProfile()
    chosen: dict[str, int] = {}
    for env, name in ((SCENES_ENV, "practice_scenes"), (TOKEN_CAP_ENV, "token_cap"), (GENERAL_ENV, "general_items"),
                      (COLLAPSE_ENV, "collapse_prompts"), (STEPS_ENV, "train_steps")):
        raw = environ.get(env, "").strip()
        if not raw:
            continue
        try:
            chosen[name] = int(raw)
        except ValueError:
            raise DemoProfileError(f"{env} is a whole number") from None
    if not chosen:
        return base
    return DemoProfile(**{**{name: getattr(base, name) for name in (
        "practice_scenes", "token_cap", "general_items", "collapse_prompts", "train_steps")}, **chosen})
