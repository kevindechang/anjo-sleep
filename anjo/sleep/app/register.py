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
- the sole-answer hook (``open_model_turn.SOLE_ANSWER_HOOKS``): no other model answers a turn in
  a cleared session that Glimmer can't, unless it is a crisis turn.
"""
from __future__ import annotations

from typing import Any

from anjo.core import open_model_turn, talker_choice
from anjo.core.anjo_model import assignment, client, hosts
from anjo.sleep.app.api import router
from anjo.sleep.app.cleared_sessions import in_cleared_session
from anjo.sleep.app.flags import require_sleep_host, sleep_enabled_for
from anjo.sleep.app.prompt import sleep_turn_hook
from anjo.sleep.app.runtime import HOST, SleepRuntime, sleep_route


def _add(hooks: list[Any], hook: Any) -> None:
    if hook not in hooks:
        hooks.append(hook)


def register_hooks() -> None:
    require_sleep_host()
    _add(assignment.ROUTE_OVERRIDES, sleep_route)
    _add(talker_choice.MENU_OVERRIDES, sleep_enabled_for)
    _add(open_model_turn.TURN_HOOKS, sleep_turn_hook)
    _add(open_model_turn.SOLE_ANSWER_HOOKS, in_cleared_session)
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
        (open_model_turn.SOLE_ANSWER_HOOKS, in_cleared_session),
    ]
    for hooks, hook in added:
        while hook in hooks:
            hooks.remove(hook)


def register_sleep(app: Any) -> None:
    """The hooks, and the dev API under ``/api``."""
    register_hooks()
    app.include_router(router, prefix="/api")
