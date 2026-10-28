"""Which model writes MIND's core splits and the counterfactual pairs (MIND's data).

The writer and the extractor of a ``core-train``, ``core-val`` or ``core-test`` life, and of a
pair's histories, are the **generator** ``--generator`` names: Spark Open (``spark``, the default,
on the experiment's provider) or a cheaper model on Fireworks' serverless API (``gpt-oss-120b``,
asked for low reasoning effort, its reasoning counted as output; ``glm-5p3-flash``). Everything
else stays as it is: the gates read on Spark Open, and each attempt is held, retried, halted and
ceilinged by the same code (``backend``, ``gen_batch``).

- **APD's lives stay Spark Open's.** The development and held-out lives are registered as Spark
  Open's; a generator is refused there (``check``).
- **Recorded.** A life records the model that wrote it on every session, and the one that
  extracted it on every evidence item (the generator, both); ``generator_of`` names it.
- **Never mixed.** A split, and the pairs, hold one generator's lives only: a life another wrote is
  a mismatch, never a life to reuse (``mixed``).

Pure.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from anjo.sleep import fireworks_api
from anjo.sleep.mind.apd.types import Life
from anjo.sleep.mind.spark_provider import FIREWORKS, SparkProvider
from anjo.sleep.provenance import SPARK_OPEN_IDS

GeneratorName = Literal["spark", "gpt-oss-120b", "glm-5p3-flash"]
APD_SPLITS: Final = ("development", "held_out")  # registered on Spark Open


@dataclass(frozen=True)
class Generator:
    """A model that may write a core split: its name on the command line, its Fireworks id
    (``None``: Spark Open, on the experiment's provider) and the reasoning effort it is asked for."""

    name: GeneratorName
    model: str | None = None
    reasoning_effort: str | None = None

    def writes_with(self, provider: SparkProvider) -> str:
        """The model id its lives are written with, and record."""
        return self.model or provider.model


SPARK: Final = Generator("spark")
GPT_OSS: Final = Generator("gpt-oss-120b", fireworks_api.GPT_OSS_120B, reasoning_effort="low")
GLM: Final = Generator("glm-5p3-flash", fireworks_api.GLM_5P3_FLASH)
GENERATORS: Final = MappingProxyType({found.name: found for found in (SPARK, GPT_OSS, GLM)})
BY_MODEL: Final = MappingProxyType({found.model: found for found in (GPT_OSS, GLM) if found.model is not None})


def check(split: str, generator: Generator, provider: SparkProvider) -> None:
    """``ValueError`` for a generator on APD's registered splits, or beside Spark Open on another
    provider than Fireworks (each attempt is held on one pool)."""
    if generator.model is None:
        return
    if split in APD_SPLITS:
        raise ValueError(f"APD's {split} lives are Spark Open's, as registered; --generator {generator.name} "
                         "is for the core splits and the pairs")
    if provider.pool != FIREWORKS.pool:
        raise ValueError(f"--generator {generator.name} runs on Fireworks, beside Spark Open on Fireworks only")


def generator_of(life: Life) -> GeneratorName:
    """The generator that wrote ``life``, by the model id its sessions record."""
    models = {session.model for session in life.sessions}
    model = next(iter(models)) if len(models) == 1 else None
    if model in SPARK_OPEN_IDS:
        return "spark"
    found = BY_MODEL.get(model or "")
    if found is None:
        raise ValueError(f"{life.id} was written by {', '.join(sorted(models))}, no generator's")
    return found.name


def mixed(lives: Iterable[Life], model: str) -> tuple[str, ...]:
    """The lives among ``lives`` that ``model`` did not write, every session."""
    return tuple(life.id for life in lives if {session.model for session in life.sessions} != {model})
