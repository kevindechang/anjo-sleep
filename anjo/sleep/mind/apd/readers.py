"""Which model reads the gates of MIND's core splits and the counterfactual pairs (MIND's data).

The leak and discoverability readers (``gates``) of a ``core-train``, ``core-val`` or ``core-test``
life, and of a pair's histories, are the **reader** ``--reader`` names: Spark Open (``spark``, on
the experiment's provider and its pool) or Gemini on Vertex (``gemini``: ``second_judge.MODEL_ID``
at low thinking, through the APD judge's Vertex client, held on ``vertex_mind_usd``). The auditor
stays Muse Spark, and each attempt is held, retried, halted and ceilinged by the same code
(``backend``, ``gen_batch``).

- **Verdicts only.** A reader returns a flag or a statement checked against the sealed item; its
  words never enter a life, so a Gemini reader leaves the life's provenance as its writer made
  it. Gemini never writes.
- **APD's lives stay Spark Open's.** The development and held-out lives are registered as read by
  Spark Open; another reader is refused there (``check``).
- **Recorded, never mixed.** Each life and pair records its reader (``store.DataDir.save_made``;
  a life with no record was read by Spark Open, as every life was before readers were recorded),
  and a split, or the pairs, hold one reader's lives only (``mixed``).

Pure.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from anjo.sleep.budget import Pool
from anjo.sleep.mind.apd.generators import APD_SPLITS
from anjo.sleep.mind.spark_provider import SparkProvider
from anjo.sleep.probes import second_judge

ReaderName = Literal["spark", "gemini"]


@dataclass(frozen=True)
class Reader:
    """A model that may read a core split's gates: its name on the command line, its model id and
    pool (``None``: Spark Open, on the experiment's provider and its pool), and its list prices in
    and out (USD per million tokens) on the day they were checked."""

    name: ReaderName
    model: str | None = None
    pool: Pool | None = None
    list_prices: tuple[float, float] | None = None

    def reads_with(self, provider: SparkProvider) -> str:
        """The model id its reads ask for, and its record names."""
        return self.model or provider.model

    def pool_on(self, provider: SparkProvider) -> Pool:
        """The pool its reads are held on."""
        return self.pool or provider.pool


SPARK_READER: Final = Reader("spark")
GEMINI_READER: Final = Reader("gemini", second_judge.MODEL_ID, "vertex_mind_usd", (0.75, 3.75))  # list, 2026-10-28
READERS: Final = MappingProxyType({found.name: found for found in (SPARK_READER, GEMINI_READER)})


def check(split: str, reader: Reader) -> None:
    """``ValueError`` for another reader than Spark Open on APD's registered splits."""
    if reader.model is not None and split in APD_SPLITS:
        raise ValueError(f"APD's {split} lives are read by Spark Open, as registered; --reader {reader.name} is for "
                         "the core splits and the pairs")


def mixed(ids: Iterable[str], readers: Mapping[str, str], name: ReaderName) -> tuple[str, ...]:
    """The ids among ``ids`` another reader than ``name`` read, by ``readers`` (an id's recorded
    reader; one with none was read by Spark Open)."""
    return tuple(found for found in ids if readers.get(found, SPARK_READER.name) != name)
