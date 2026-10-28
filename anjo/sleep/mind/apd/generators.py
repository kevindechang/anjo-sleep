"""Which model writes MIND's core splits and the counterfactual pairs (MIND's data).

The writer and the extractor of a ``core-train``, ``core-val`` or ``core-test`` life, and of a
pair's histories, are the **generator** ``--generator`` names: Spark Open (``spark``, on the
experiment's provider) or a cheaper model on Fireworks' serverless API (``deepseek-v4p1-flash``,
the core splits' and the pairs' default (``core_default``), asked for no reasoning;
``gpt-oss-120b``, asked for low reasoning effort; ``glm-5p3-flash``), any reasoning counted as
output. The gates read on the command's reader (``readers``), and each attempt is held, retried,
halted and ceilinged by the same code (``backend``, ``gen_batch``).

- **Who may train the core.** ``trains`` records why a generator's lives may train the MIND core:
  Spark Open's licence (``provenance.CLEARED``), or DeepSeek V4.1 Flash's MIT licence, which Kevin
  cleared on 2026-10-28 for the MIND core only (not for Glimmer's weights, so it is not in
  ``provenance.CLEARED``). Each life's record carries it (``store.DataDir.save_made``), and so does
  the labels export. ``None``: no clearance is recorded.

- **APD's lives stay Spark Open's.** The development and held-out lives are registered as Spark
  Open's; a generator is refused there (``check``).
- **Recorded.** A life records the model that wrote it on every session, and the one that
  extracted it on every evidence item (the generator, both); ``generator_of`` names it.
- **Never mixed.** A split, and the pairs, hold one generator's lives only: a life another wrote is
  a mismatch, never a life to reuse (``mixed``).
- **Its answer only** (``answer``). A reasoning model may put its reasoning in the reply between
  think tags, or wrap its JSON in a fence with prose around it; the writer and the extractor read
  the answer, never the reasoning (which a reply's own reasoning field already keeps apart). A
  draft is never taken for the answer: a reply cut at its cap, one whose reasoning never closed,
  and one with more than one fenced block outside its reasoning are refused (``Refused``), each by
  its rule.

Pure.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from anjo.sleep import fireworks_api
from anjo.sleep.mind.apd.types import Life
from anjo.sleep.mind.spark_provider import FIREWORKS, SparkProvider
from anjo.sleep.provenance import SPARK_OPEN_IDS

GeneratorName = Literal["spark", "deepseek-v4p1-flash", "gpt-oss-120b", "glm-5p3-flash"]
APD_SPLITS: Final = ("development", "held_out")  # registered on Spark Open


@dataclass(frozen=True)
class Generator:
    """A model that may write a core split: its name on the command line, its Fireworks id
    (``None``: Spark Open, on the experiment's provider), the reasoning effort it is asked for, why
    its lives may train the MIND core (``None``: no clearance recorded) and its list prices in and
    out (USD per million tokens) on the day they were checked, when they are known."""

    name: GeneratorName
    model: str | None = None
    reasoning_effort: str | None = None
    trains: str | None = None
    list_prices: tuple[float, float] | None = None

    def writes_with(self, provider: SparkProvider) -> str:
        """The model id its lives are written with, and record."""
        return self.model or provider.model


SPARK: Final = Generator("spark", trains="Spark Open: the Muse Model License, cleared for training (provenance)")
DEEPSEEK: Final = Generator(
    "deepseek-v4p1-flash", fireworks_api.DEEPSEEK_V4P1_FLASH, reasoning_effort="none",  # its non-thinking mode
    trains="DeepSeek V4.1 Flash: MIT licence (huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), cleared by Kevin on "
           "2026-10-28 to train the MIND core only",
    list_prices=(0.30, 1.20))  # Fireworks serverless, 2026-10-28; its $0.006 cached input is not counted
GPT_OSS: Final = Generator("gpt-oss-120b", fireworks_api.GPT_OSS_120B, reasoning_effort="low")
GLM: Final = Generator("glm-5p3-flash", fireworks_api.GLM_5P3_FLASH)
GENERATORS: Final = MappingProxyType({found.name: found for found in (SPARK, DEEPSEEK, GPT_OSS, GLM)})
BY_MODEL: Final = MappingProxyType({found.model: found for found in (DEEPSEEK, GPT_OSS, GLM)
                                    if found.model is not None})


def core_default(split: str) -> Generator:
    """Who writes ``split`` unless ``--generator`` says otherwise: DeepSeek V4.1 Flash for a core
    split and the pairs (``core-test``'s), Spark Open for APD's own."""
    return SPARK if split in APD_SPLITS else DEEPSEEK


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


_THINK: Final = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCED: Final = re.compile(r"```[A-Za-z]*[ \t]*\n(.*?)```", re.DOTALL)
CUT: Final = "cut"
OPEN_THINK: Final = "open_think"
FENCES: Final = "fences"


class Refused(ValueError):
    """A reply that can't be taken for its answer, and the ``rule`` it broke. Its message is the
    code's own, never the reply's text."""

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(message)
        self.rule = rule


def answer(text: str, finish_reason: str | None = None) -> str:
    """A generator's reply as its answer: closed reasoning taken out (and reasoning the chat
    template opened, up to its close), and the one fenced answer taken out of the prose around it.
    A reply that is its answer already is unchanged. Refused: a reply cut at its cap (``CUT``),
    reasoning left open (``OPEN_THINK``), more than one fenced block outside it (``FENCES``)."""
    if finish_reason == "length":
        raise Refused(CUT, "a reply cut at its cap")
    body = _THINK.sub("", text)
    if "<think>" in body:
        raise Refused(OPEN_THINK, "a reply whose reasoning never closed")
    body = body.rsplit("</think>", 1)[-1].strip()
    if body.startswith("{"):
        return body
    fenced = _FENCED.findall(body)
    if len(fenced) > 1:
        raise Refused(FENCES, "a reply with more than one fenced block")
    return fenced[0].strip() if fenced else body


def mixed(lives: Iterable[Life], model: str) -> tuple[str, ...]:
    """The lives among ``lives`` that ``model`` did not write, every session."""
    return tuple(life.id for life in lives if {session.model for session in life.sessions} != {model})
