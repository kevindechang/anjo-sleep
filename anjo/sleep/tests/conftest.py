# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Test isolation for anjo/sleep/tests, in the private Anjo repo and in the public one.

Inside the private repo these tests get the same isolation as ``tests/conftest.py``: importing
it sets its environment (``ANJO_NO_DOTENV``, the test secrets, ``ANJO_ENV=dev``) before any app
import, and its autouse ``isolate`` fixture gives every test a temp data root with provider keys
stripped. Its network guard, ``tests.network_guard``, loads through the private ``addopts``.

The public repo has no private Anjo and no ``tests/`` package. There the core tests run on their
own, and tests of the adapters in ``anjo/sleep/app`` (kept under ``anjo/sleep/tests/app``), which
need private Anjo, are not collected. Its ``addopts`` load no guard, so ``offline`` below refuses
non-loopback traffic in every test, and steps aside wherever the private guard is active.
"""
from __future__ import annotations

import importlib.util

import pytest

from anjo.sleep.tests.offline import guard_sockets, private_guard_active

PRIVATE_ANJO = importlib.util.find_spec("anjo.core") is not None

if PRIVATE_ANJO:
    from tests.conftest import isolate  # noqa: F401  (autouse: applies to every test here)
else:
    collect_ignore = ["app"]


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    if not private_guard_active():
        guard_sockets(monkeypatch)
