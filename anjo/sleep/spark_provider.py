"""Sleep's configurable Spark Open provider and its accounting parameters."""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from anjo.sleep import baseten_api, fireworks_api
from anjo.sleep.budget import Pool
from anjo.sleep.model_api import ChatResult, Retries

SETTING: Final = "ANJO_SLEEP_SPARK"


@dataclass(frozen=True)
class SparkProvider:
    model: str
    chat: Callable[..., ChatResult]
    pool: Pool
    # None preserves the existing configured Baseten prices.
    price: tuple[float, float] | None
    # The policy a delegated call passes; None: the provider's calls never retry by themselves.
    retries: Retries | None = None


def selected(environ: Mapping[str, str]) -> str:
    """The provider ``environ`` names; a missing or blank setting is ``baseten``."""
    return environ.get(SETTING, "").strip().casefold() or "baseten"


def from_env(environ: Mapping[str, str]) -> SparkProvider:
    """Choose a provider; a missing or blank setting preserves the Baseten default."""
    name = selected(environ)
    if name == "baseten":
        return SparkProvider(baseten_api.SPARK_OPEN, baseten_api.chat, "baseten_sleep_usd", None)
    if name == "fireworks":
        return SparkProvider(fireworks_api.SPARK_OPEN, fireworks_api.chat, "fireworks_sleep_usd", (1.25, 4.25),
                             fireworks_api.SLEEP_RETRIES)
    raise ValueError(f"{SETTING} must be baseten or fireworks, not {name!r}")


def current(environ: Mapping[str, str] | None = None) -> SparkProvider:
    """The provider this process runs on: ``os.environ`` unless ``environ`` is given."""
    return from_env(os.environ if environ is None else environ)
