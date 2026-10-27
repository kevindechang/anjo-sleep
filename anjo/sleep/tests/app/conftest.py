# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The app adapters' tests run on the sleep host, as the dev deployment does; a test that needs
the flag off deletes it. The web app's fixtures come from the private repo's tests/conftest.py."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from tests.conftest import _app, auth_client, client, offline_embedder  # noqa: F401  (the web app's fixtures)


@pytest.fixture(autouse=True)
def sleep_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HOST", "1")


@pytest.fixture
def unmounted_app(_app: Any) -> Iterator[Any]:  # noqa: F811  (the imported fixture)
    """The session's web app as imported without the sleep host, whatever an earlier test, from any
    file and in any order, left registered on it; put back afterwards."""
    from anjo.sleep.app.api import router
    from anjo.sleep.app.demo_api import page_router
    from anjo.sleep.app.demo_api import router as demo_router

    sleeps = {id(router), id(demo_router), id(page_router)}  # what register_sleep includes
    routes, state = list(_app.router.routes), dict(_app.state._state)
    # FastAPI keeps an included router as one entry that wraps it, with no path of its own.
    _app.router.routes[:] = [r for r in routes if id(getattr(r, "original_router", r)) not in sleeps
                             and not str(getattr(r, "path", "")).startswith(("/sleep", "/api/sleep"))]
    _app.state._state.pop("public_prefixes", None)
    try:
        yield _app
    finally:
        _app.router.routes[:] = routes
        _app.state._state.clear()
        _app.state._state.update(state)
