"""Where replay runs, what it spends, and the holds it takes first (Task 17b), as Task 17's MIND
step does in the night (``night/mind.py``).

- **Glimmer on Colab only:** every APD arm's thinker is Glimmer on Colab vLLM at one precision
  (``mind/inquiry/ports.py``), and replay's branches and 17b arms are too. ``colab_client``
  accepts only a live session of ``apd.colab_vllm.serve`` whose client points at its own tunnel's
  loopback end: the night's SageMaker and Modal servers, a server merely named ``colab``, and a
  session that has ended never serve replay.
- **Held before it runs:** each start state's branches, and each held-out life's run of each arm,
  hold their Glimmer worst case on ``colab_cu`` (every thinker and ``world_query`` call of a night
  at the client's request timeout, at the A100 rate) and settle the wall time they took, while the
  VM serves them. When the session itself holds its whole worst case (``serve`` does), ``Holds``
  is built with ``gpu=False``, so the VM's time is held once. The labels hold their Spark Open
  worst case on the provider's pool (``Holds.spark_pool``: ``baseten_mind_usd``, or
  ``fireworks_mind_usd`` as ``mind.spark_provider`` sets it; one call per start state, at the most
  a call reads and writes), and so do the branches' resolver calls (``MIND_SPARK_CALLS`` a branch,
  as a night's step); each settles the dollars it metered. A pool without room stops replay
  before the call (``BudgetStop``). Without a ledger (tests and ``--fake``) nothing is held.

The dry run's estimate of all of it is ``replay.estimate``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final, Protocol
from urllib.parse import urlsplit

from anjo.sleep.budget import Bound, Ledger, Pool
from anjo.sleep.mind.apd.colab_vllm import ColabSession
from anjo.sleep.mind.replay.types import ReplayRefused
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import held, spark_usd, spark_worst_usd, usd
from anjo.sleep.night.mind import GLIMMER_CALL_S, MIND_GLIMMER_CALLS
from anjo.sleep.serving.client import VllmClient, is_loopback
from anjo.sleep.spark_inquiry import ChatFn
from anjo.sleep.train.runners.colab import A100_CU_PER_HOUR, POOL

COLAB: Final = "colab"
SPARK_POOL: Final = "baseten_mind_usd"


class GlimmerServer(Protocol):
    """What replay needs of a serving backend (``serving.backend.Serving`` has both)."""

    @property
    def name(self) -> str: ...

    def client(self) -> VllmClient: ...


def colab_client(server: GlimmerServer) -> VllmClient:
    """The client of Glimmer on Colab vLLM: a live ``ColabSession``'s, pointing at its own
    tunnel's loopback end. Any other server is refused before a call, whatever its name."""
    if not isinstance(server, ColabSession) or server.name != COLAB:
        raise ReplayRefused(f"replay serves Glimmer on Colab vLLM only, at one precision, never on {server.name}")
    if not server.live:
        raise ReplayRefused(f"Colab session {server.session} isn't serving")
    client = server.client()
    if client.base_url != server.base_url or not is_loopback(urlsplit(server.base_url).hostname or ""):
        raise ReplayRefused("replay reaches Colab vLLM only through its session's own loopback tunnel")
    return client


@dataclass(frozen=True)
class Holds:
    """Where replay's spend is held: the ledger, the list prices, and the clock that times a GPU
    phase (passed in, so nothing here reads one); Spark Open on the provider's ``spark_pool``."""

    ledger: Ledger
    prices: Prices
    timer: Callable[[], float]
    job: str = "replay"
    gpu: bool = True  # False when the serving session holds the VM's time itself
    spark_pool: Pool = SPARK_POOL


@dataclass
class SparkMeter:
    """Spark Open's calls through one ``ChatFn``, priced as they return."""

    prices: Prices
    usd: float = 0.0

    def wrap(self, chat: ChatFn) -> ChatFn:
        def metered(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
            result = chat(messages, max_tokens)
            self.usd += spark_usd(self.prices, result.input_tokens, result.output_tokens)
            return result
        return metered


def glimmer_seconds(nights: int) -> float:
    """The most ``nights`` of a MIND run can keep Glimmer busy: every call at its timeout."""
    return nights * MIND_GLIMMER_CALLS * GLIMMER_CALL_S


@contextmanager
def held_gpu(holds: Holds | None, key: str, nights: int) -> Iterator[None]:
    """``nights`` of Glimmer held on ``colab_cu``, settled at the wall time they took."""
    if holds is None or not holds.gpu:
        yield
        return
    started = holds.timer()
    bound = Bound(A100_CU_PER_HOUR, glimmer_seconds(nights))
    with held(holds.ledger, f"{holds.job}:{key}", (POOL,), bound,
              lambda: A100_CU_PER_HOUR * max(0.0, holds.timer() - started) / 3600.0):
        yield


@contextmanager
def held_spark(holds: Holds | None, key: str, calls: int, meter: SparkMeter | None) -> Iterator[None]:
    """``calls`` Spark Open calls held on the provider's pool, settled at what ``meter`` priced."""
    if holds is None:
        yield
        return
    if meter is None:
        raise ValueError("a held step meters what it spends")
    with held(holds.ledger, f"{holds.job}:{key}", (holds.spark_pool,), usd(spark_worst_usd(holds.prices, calls)),
              lambda: meter.usd):
        yield
