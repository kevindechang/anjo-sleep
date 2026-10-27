# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The app adapters' tests run on the sleep host, as the dev deployment does; a test that needs
the flag off deletes it. The web app's fixtures come from the private repo's tests/conftest.py."""
from __future__ import annotations

import pytest

from tests.conftest import _app, auth_client, client, offline_embedder  # noqa: F401  (the web app's fixtures)


@pytest.fixture(autouse=True)
def sleep_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HOST", "1")
