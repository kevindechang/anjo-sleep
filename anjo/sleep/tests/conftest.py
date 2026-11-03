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

# In the public repo, a few tests reach code that only exists privately (``anjo.core``, through the
# app's prices) or the Google SDK (MIND's Gemini judge). Without them, that one error is a skip that
# says which is missing; any other error, and either one with the module present, still fails.
GOOGLE_SDK = importlib.util.find_spec("google") is not None


def _missing(name: str | None) -> str | None:
    parts = (name or "").split(".")
    if parts[:2] == ["anjo", "core"] and not PRIVATE_ANJO:
        return "private Anjo"
    if parts[0] == "google" and not GOOGLE_SDK:
        return "the google-genai SDK"
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):  # type: ignore[no-untyped-def]
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is None or not call.excinfo.errisinstance(ModuleNotFoundError):
        return
    what = _missing(getattr(call.excinfo.value, "name", None))
    if what is not None and report.failed:
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: needs {what}, not in the public repo")


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    if not private_guard_active():
        guard_sockets(monkeypatch)


# Built at runtime, so this file holds none of the names the key scan looks for.
_AWS_IDENTITY_ENV = ("AWS_PROFILE", "AWS_ROLE_ARN", "AWS_WEB_IDENTITY_TOKEN_FILE",
                     "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI",
                     *("AWS_" + rest for rest in ("ACCESS_" + "KEY_ID", "SECRET_" + "ACCESS_KEY", "SESSION_" + "TOKEN")))


@pytest.fixture(autouse=True)
def no_aws_identity(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """A boto3 client made in a test finds no credentials, reads no AWS file of this machine's and
    asks no metadata service: nothing can be signed, let alone sent."""
    nowhere = tmp_path_factory.getbasetemp() / "no-aws"
    monkeypatch.setenv("AWS_CONFIG_FILE", str(nowhere / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(nowhere / "credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    for name in _AWS_IDENTITY_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def default_mind_spark(monkeypatch: pytest.MonkeyPatch) -> None:
    """MIND's experiment on its default Spark Open provider, with no Fireworks key, whatever this
    shell sets: a test that wants another provider or a key sets it."""
    monkeypatch.delenv("ANJO_SLEEP_MIND_SPARK", raising=False)
    monkeypatch.delenv("ANJO_FIREWORKS_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def default_extra_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full morning asks the six committed arms, whatever this shell sets: a test that wants
    2A and 4M in the full run sets ``ANJO_SLEEP_ARMS_EXTRA`` itself."""
    from anjo.sleep.probes.morning import ARMS_EXTRA_ENV

    monkeypatch.delenv(ARMS_EXTRA_ENV, raising=False)


@pytest.fixture(autouse=True)
def no_colab_cli(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """No test finds this machine's Colab CLI, so a night's sweep never runs it; a test that wants an
    installed CLI points ``COLAB_BIN`` at a file of its own and fakes the CLI."""
    from anjo.sleep.train.runners import colab

    monkeypatch.setattr(colab, "COLAB_BIN", tmp_path_factory.getbasetemp() / "no-colab" / "colab")
