"""A continuum memory for MIND's core (``ANJO_SLEEP_MIND_CMS``, default off). Torch-free.

Behrouz, Razaviyayn, Zhong and Mirrokni, "Nested Learning: The Illusion of Deep Learning
Architectures" (NeurIPS 2025; arXiv 2512.24695), §7.1: a Continuum Memory System is a chain of
memory blocks, each updated at its own frequency. Block ``l`` takes one step every ``C(l)`` inputs,
the sum of its rated gradients over the chunk (their eq. 71):
``theta(l) <- theta(l) - sum_{t in chunk} eta_t(l) grad f(theta(l); x_t)``, and nothing in between.
Their independent variant reads every block at once and aggregates (eq. 74: a learnable weighted sum).

With the switch on (``CoreConfig.cms``, which training and the CLI set from the switch), the
core's associative memory is three linear levels read by one key, each a ``d x d`` matrix:

- ``step``: the fast weights as they are with the switch off (Titans: a surprise step on every event,
  with momentum and a forget gate, ``net.fast_update``).
- ``session``: one step per session. A session ends at an event that comes ``SESSION_GAP_DAYS`` or
  more after the one before it, and at the latest when she sleeps. Until then each event adds its
  rated gradient to the level's ``pending`` sum; the level itself stays as it was.
- ``night``: one step per Sleep, and only then (``api.Core.consolidate``, which the night runs):
  the rated gradients since the last consolidation. Nothing else moves it, so a night that does not
  run leaves it as it was.

The read is ``(M_step + w_session M_session + w_night M_night) k`` (eq. 74, the weights learned,
starting at 1). Every level descends the same error, that read's against the event's value: the
surprise ``mean((read - v)^2)``, as with the switch off the fast and the slow memory's together. The
rates are learned, a function of the event as the fast level's are (``eta <= theta_max``), one per
slow level. At Sleep the session's and the night's sums are applied and the fast level keeps only
``residue`` of itself, its momentum none (consolidate, then reset, as with the switch off). Shared
training runs a Sleep between each day and the next, so the night level's rate and weight are
learned where it acts.

The state carries the levels as one ``(LEVELS, d, d)`` array (``CoreState.levels``, absent with the
switch off, so an off state saves to the same bytes as before). The night level replaces the slow
memory of the switch-off night (``api.SLOW_MEMORY``): a state under the switch has none.

Not built: HOPE's self-modifying update (their §8.1, eqs. 83-88), where the key, value, rate and
retention projections are memories that learn in context from values they generate. Here the
person's states are kept for months across nights and read by a model trained on ten-day lives: a key
projection that drifts in context would address the night level with keys it was never written under,
and nothing in training covers that drift. The rates here are functions of the event, not memories.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Final

CMS_ENV: Final = "ANJO_SLEEP_MIND_CMS"  # "1": a core built or trained now carries the continuum memory
SESSION_GAP_DAYS: Final = 0.125  # three hours without an event end a session
NAMES: Final = ("step", "session", "night")  # fastest first
SESSION, SESSION_PENDING, NIGHT, NIGHT_PENDING = range(4)  # rows of ``CoreState.levels``
LEVELS: Final = 4
RATE_BIAS: Final = -1.0  # the slow levels' rates start at sigmoid(-1) of theta_max


def cms_on(env: Mapping[str, str]) -> bool:
    """The switch: ``ANJO_SLEEP_MIND_CMS=1`` builds the core with the continuum memory."""
    return env.get(CMS_ENV, "").strip() == "1"
