"""A trial or judge serving app can run beside the others under its own name and secret."""
from __future__ import annotations

import importlib

import pytest

from anjo.sleep.serving import config


def test_the_app_name_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv(config.APP_NAME_ENV, "anjo-sleep-serve-judges")

    # Act
    name = config.serve_app_name(config.SERVING, deterministic=True)

    # Assert
    assert name == "anjo-sleep-serve-judges"


def test_without_the_override_names_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.delenv(config.APP_NAME_ENV, raising=False)

    # Act / Assert
    assert config.serve_app_name(config.SERVING, deterministic=False) == config.APP_NAME
    assert config.serve_app_name(config.SERVING, deterministic=True) == f"{config.APP_NAME}-det"


def test_the_secret_name_override_is_read_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv("ANJO_SLEEP_SERVE_SECRET_NAME", "anjo-sleep-serve-judges")

    # Act
    reloaded = importlib.reload(config)

    # Assert
    assert reloaded.SECRET_NAME == "anjo-sleep-serve-judges"
    monkeypatch.delenv("ANJO_SLEEP_SERVE_SECRET_NAME")
    assert importlib.reload(config).SECRET_NAME == "anjo-sleep-serve"
