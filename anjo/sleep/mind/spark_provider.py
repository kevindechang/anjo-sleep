"""Which provider serves Spark Open to MIND's experiment (Tasks 17 and 17b): one per experiment.

Spark Open is one open-weight model under one licence (the Muse Model License), served by two
providers at the same list price. MIND's experiment, everything under ``mind/apd/`` and
``mind/replay/``, takes its provider from one setting, ``ANJO_SLEEP_MIND_SPARK``:

- ``fireworks`` (the default): ``fireworks_api``, every call held on ``fireworks_mind_usd``;
- ``baseten``: ``baseten_api``, every call held on ``baseten_mind_usd``, as before.

The lives' writer and extractor, the gates' readers, the screen, the resolver, 17b's labels and
the intervention test's writer all go through it. The two providers' snapshots may differ, so a
pilot never mixes them: its data dir pins the provider and model id at its first Spark Open
command and refuses the other after that (``DataDir.pin_spark``).

Only the commands read the setting, passing their environment to ``from_env``; this module reads
none. Sleep's night (practice, extraction, the brief,
the simulator and the night's own MIND step) never does: it stays on Baseten.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal

from anjo.sleep import baseten_api, fireworks_api
from anjo.sleep.budget import Pool
from anjo.sleep.context.brief import ChatFn as WriteFn
from anjo.sleep.model_api import ChatResult
from anjo.sleep.spark_inquiry import TIMEOUT_S, ChatFn, spark_open

ProviderName = Literal["fireworks", "baseten"]
SETTING_ENV: Final = "ANJO_SLEEP_MIND_SPARK"
DEFAULT: Final[ProviderName] = "fireworks"


def fireworks_spark_open(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
    """One Spark Open call on Fireworks at temperature 0, no tools, its reply capped:
    ``spark_inquiry.spark_open``'s twin."""
    return fireworks_api.chat(fireworks_api.SPARK_OPEN, messages, None, TIMEOUT_S, temperature=0.0,
                              max_tokens=max_tokens)


@dataclass(frozen=True)
class SparkProvider:
    """One provider of Spark Open: the id it serves under (and its replies report), the pool every
    call of MIND's experiment on it is held on, its capped calls (``chat``: the screen, the
    resolver, the labels and the wrong hypothesis) and its generation calls (``write``: the lives'
    writer and extractor, and the gates' readers)."""

    name: ProviderName
    label: str
    model: str
    pool: Pool
    chat: ChatFn
    write: WriteFn


BASETEN: Final = SparkProvider("baseten", "Baseten", baseten_api.SPARK_OPEN, "baseten_mind_usd", spark_open,
                               baseten_api.chat)
FIREWORKS: Final = SparkProvider("fireworks", "Fireworks", fireworks_api.SPARK_OPEN, "fireworks_mind_usd",
                                 fireworks_spark_open, fireworks_api.chat)
PROVIDERS: Final[Mapping[str, SparkProvider]] = MappingProxyType({p.name: p for p in (FIREWORKS, BASETEN)})


def named(name: str) -> SparkProvider:
    """The provider called ``name``, or ``ValueError`` naming the setting and its choices."""
    found = PROVIDERS.get(name.strip().casefold())
    if found is None:
        raise ValueError(f"{SETTING_ENV} is {', '.join(sorted(PROVIDERS))}, not {name!r}")
    return found


def from_env(environ: Mapping[str, str]) -> SparkProvider:
    """The provider ``ANJO_SLEEP_MIND_SPARK`` names in ``environ``; Fireworks when it's unset or blank."""
    return named(environ.get(SETTING_ENV, "").strip() or DEFAULT)
