"""Puts Anjo Sleep in place on its own host (Task 4). ``anjo/dashboard/app.py`` calls
``register_sleep`` only when ``ANJO_SLEEP_HOST=1``; with it unset nothing here runs, and the app
is byte for byte main's.

``register_hooks`` adds, once each:

- the route override (``assignment.ROUTE_OVERRIDES``): a flagged person who chose the open-model
  talker is served by Glimmer on Modal;
- the menu override (``talker_choice.MENU_OVERRIDES``): that talker is on offer to a flagged
  person while it is off everyone else's menu;
- the ``sleep_modal`` runtime (``hosts.register_runtime``), a host off the N-PROC vocabulary
  (``client.OFF_VOCABULARY_HOSTS``);
- the prompt hook (``open_model_turn.TURN_HOOKS``): context C, or the cleared context;
- the brief stores C reads (Task 5): the served brief for served turns, the training brief for
  cleared ones, and the adapter source: each person's promoted checkpoint serves them;
- the delivery hold (``outreach.DELIVERY_HOLDS``): no pending note opens a cleared session;
- the sole-answer hook (``open_model_turn.SOLE_ANSWER_HOOKS``): no other model answers a turn in
  a cleared session that Glimmer can't, unless it is a crisis turn;
- the erasure hook (``erasure.ERASURE_HOOKS``): account erasure and a memory wipe also remove the
  person's remote adapters, night logs, registry and the rest outside their own folder (Task 9).
"""
from __future__ import annotations

from typing import Any

from anjo.core import erasure, open_model_turn, outreach, talker_choice
from anjo.core.anjo_model import assignment, client, hosts
from anjo.core.best_effort import best_effort
from anjo.sleep.app.api import router
from anjo.sleep.app.cleared_sessions import in_cleared_session
from anjo.sleep.app.demo_api import PUBLIC_PREFIXES as DEMO_PUBLIC
from anjo.sleep.app.demo_api import page_router as demo_page_router
from anjo.sleep.app.demo_api import router as demo_router
from anjo.sleep.app.delivery import held
from anjo.sleep.app.flags import apply_tombstones, require_sleep_host, sleep_enabled_for
from anjo.sleep.app.judge_pages import PUBLIC_PREFIXES as JUDGE_PAGES_PUBLIC
from anjo.sleep.app.judge_pages import page_router as judge_page_router
from anjo.sleep.app.ownership import remote_cleanup
from anjo.sleep.app.prompt import ContextSetup, set_context_setup, sleep_turn_hook
from anjo.sleep.app.runtime import HOST, SleepRuntime, set_adapter_source, sleep_route
from anjo.sleep.app.storage import brief_store, checkpoint_registry, glimmer_turns
from anjo.sleep.context.brief_store import ServedBriefs, TrainingBriefs
from anjo.sleep.serving.activity import track
from anjo.sleep.types import CheckpointRef


def _add(hooks: list[Any], hook: Any) -> None:
    if hook not in hooks:
        hooks.append(hook)


def _promoted_adapter(user_id: str) -> CheckpointRef | None:
    """The adapter that serves ``user_id``: their last promoted checkpoint, or none (the base)."""
    return checkpoint_registry().last_promoted(user_id)


def register_hooks() -> None:
    require_sleep_host()
    apply_tombstones()  # at startup: a deleted person's forced-id entry never comes back
    _add(assignment.ROUTE_OVERRIDES, sleep_route)
    _add(talker_choice.MENU_OVERRIDES, sleep_enabled_for)
    _add(open_model_turn.TURN_HOOKS, sleep_turn_hook)
    _add(outreach.DELIVERY_HOLDS, held)
    set_context_setup(ContextSetup(served_briefs=ServedBriefs(brief_store), training_briefs=TrainingBriefs(brief_store)))
    set_adapter_source(_promoted_adapter)
    _add(open_model_turn.SOLE_ANSWER_HOOKS, in_cleared_session)
    _add(erasure.ERASURE_HOOKS, remote_cleanup)
    if HOST not in hosts.RUNTIMES:
        hosts.register_runtime(HOST, SleepRuntime)
    client.OFF_VOCABULARY_HOSTS.add(HOST)


def unregister_hooks() -> None:
    """Take the hooks out again (tests). The runtime stays registered: only a Glimmer pin, which
    only the route override makes, ever reaches it."""
    added: list[tuple[list[Any], Any]] = [
        (assignment.ROUTE_OVERRIDES, sleep_route),
        (talker_choice.MENU_OVERRIDES, sleep_enabled_for),
        (open_model_turn.TURN_HOOKS, sleep_turn_hook),
        (outreach.DELIVERY_HOLDS, held),
        (open_model_turn.SOLE_ANSWER_HOOKS, in_cleared_session),
        (erasure.ERASURE_HOOKS, remote_cleanup),
    ]
    for hooks, hook in added:
        while hook in hooks:
            hooks.remove(hook)


def register_sleep(app: Any) -> None:
    """The hooks, the dev API under ``/api``, the live demo page (Task 13a): the page at
    ``/sleep/demo`` and its routes under ``/api/sleep/demo``, and the judge product's pages at
    ``/sleep/app``: the only paths opened to anonymous visitors (``app.state.public_prefixes``, read
    by the app's auth middleware)."""
    register_hooks()
    with best_effort("mark the host's Glimmer turns as tracked"):  # unmarked, the night's GPU stop never stops
        track(glimmer_turns())
    app.include_router(router, prefix="/api")
    app.include_router(demo_router, prefix="/api")
    app.include_router(demo_page_router)
    app.include_router(judge_page_router)  # the judge product's static pages, at /sleep/app
    app.state.public_prefixes = (*DEMO_PUBLIC, *JUDGE_PAGES_PUBLIC)
