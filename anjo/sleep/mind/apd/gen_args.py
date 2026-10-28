"""``generate`` and ``generate-pairs``' writer and reader flags (MIND's data; ``__main__``).

``--generator`` names who writes and extracts (``generators``) and ``--reader`` who reads the gates
(``readers``). Unless they say otherwise, a core split and the pairs are written by DeepSeek V4.1
Flash and read by Gemini (``generators.core_default``, ``DEFAULT_READER``), APD's own splits by
Spark Open alone, which is all they accept. ``--generator-prices`` and ``--reader-prices`` give a
model's list prices (in,out USD per million tokens); a model whose prices were checked
(``list_prices``) takes those unless they are given, and live, one without them is refused.

A Gemini reader's calls go through one ``limiter.Limiter`` for the whole command
(``Backend.read_limit``): at most ``--vertex-concurrency`` at once (default 4), and live their
starts paced to ``--vertex-rpm`` a minute (default ``limiter.DEFAULT_PER_MINUTE``, a conservative
one; 0 leaves them unpaced), so lives side by side don't send Vertex more than its quota takes.
The fakes stand in for Vertex and aren't paced.
"""
from __future__ import annotations

import argparse
import dataclasses

from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import generators, readers, retry
from anjo.sleep.mind.apd.generators import GENERATORS, Generator
from anjo.sleep.mind.apd.judging import Rate
from anjo.sleep.mind.apd.limiter import DEFAULT_CONCURRENCY, DEFAULT_PER_MINUTE, Limiter
from anjo.sleep.mind.apd.readers import READERS, Reader
from anjo.sleep.mind.apd.types import CORE_SPLITS
from anjo.sleep.mind.spark_provider import SparkProvider

FAKE_GENERATOR_RATE = Rate(0.15, 0.6)  # the fakes' generator without list prices: list-like


def add(command: argparse.ArgumentParser) -> None:
    command.add_argument("--generator", choices=tuple(GENERATORS),
                         help="who writes and extracts a core split's lives or the pairs (default "
                              f"{generators.DEEPSEEK.name}; APD's own splits take spark only, their default)")
    command.add_argument("--generator-prices", help="with another --generator than spark: its in,out list prices (USD "
                         "per million tokens; default its checked list prices, if any); --prices prices Spark Open")
    command.add_argument("--reader", choices=tuple(READERS),
                         help="who reads a core split's or the pairs' gates (default gemini: Gemini on Vertex; APD's "
                              "own splits take spark only, their default)")
    command.add_argument("--reader-prices", help="with --reader gemini: its in,out list prices (USD per million "
                         "tokens; default its checked list prices)")
    command.add_argument("--vertex-concurrency", type=int, default=DEFAULT_CONCURRENCY,
                         help=f"with --reader gemini: the most reads on Vertex at once across the run (default "
                              f"{DEFAULT_CONCURRENCY})")
    command.add_argument("--vertex-rpm", type=float, default=DEFAULT_PER_MINUTE,
                         help=f"with --reader gemini: pace the reads' starts to this many a minute (default "
                              f"{DEFAULT_PER_MINUTE:g}, a conservative one: Vertex's shared quota has no fixed rate; "
                              f"0: unpaced)")


def split_of(args: argparse.Namespace) -> str:
    """The split a generation command writes: ``core-test``'s fields for the pairs."""
    if args.command == "generate-pairs":
        return "core-test"
    if getattr(args, "held_out", False):
        return "held_out"
    return getattr(args, "split", None) or "development"


def generator(args: argparse.Namespace) -> Generator:
    name = getattr(args, "generator", None)
    return generators.core_default(split_of(args)) if name is None else GENERATORS[name]


def reader(args: argparse.Namespace) -> Reader:
    name = getattr(args, "reader", None)
    if name is not None:
        return READERS[name]
    return readers.GEMINI_READER if split_of(args) in CORE_SPLITS else readers.SPARK_READER


def _two(text: str | None, flag: str) -> Rate | None:
    if text is None:
        return None
    parts = tuple(float(part) for part in text.split(","))
    if len(parts) != 2 or any(part < 0 for part in parts):
        raise SystemExit(f"{flag} holds two prices, in and out, each at least 0")
    return Rate(parts[0], parts[1])


def _listed(prices: tuple[float, float] | None) -> Rate | None:
    return None if prices is None else Rate(*prices)


def generator_rate(args: argparse.Namespace) -> Rate | None:
    """``--generator-prices``, or the generator's checked list prices; ``None`` for Spark Open."""
    found = generator(args)
    if found.model is None:
        return None
    return _two(getattr(args, "generator_prices", None), "--generator-prices") or _listed(found.list_prices)


def reader_rate(args: argparse.Namespace) -> Rate | None:
    """``--reader-prices``, or the reader's checked list prices; ``None`` for Spark Open."""
    found = reader(args)
    if found.model is None:
        return None
    return _two(getattr(args, "reader_prices", None), "--reader-prices") or _listed(found.list_prices)


def check(parser: argparse.ArgumentParser, args: argparse.Namespace, provider: SparkProvider) -> None:
    """A generator or reader other than Spark Open only on the core splits and the pairs, the
    generator beside Spark Open on Fireworks; live, each priced."""
    split, writes, reads = split_of(args), generator(args), reader(args)
    try:
        generators.check(split, writes, provider)
        readers.check(split, reads)
    except ValueError as error:
        parser.error(str(error))
    if not args.fake and writes.model is not None and generator_rate(args) is None:
        parser.error(f"--generator {writes.name} needs --generator-prices in,out to price its holds")
    if getattr(args, "vertex_concurrency", DEFAULT_CONCURRENCY) < 1:
        parser.error("--vertex-concurrency lets at least 1 read through at once")
    if getattr(args, "vertex_rpm", DEFAULT_PER_MINUTE) < 0:
        parser.error("--vertex-rpm paces at least 0 reads a minute (0: unpaced)")


def apply(args: argparse.Namespace, backend: backends.Backend) -> backends.Backend:
    """The backend writing with the command's generator and reading with its reader, at their
    prices (the fakes' list-like); live, a Gemini reader calls Vertex (``backend.gemini_chat``)."""
    if args.command not in ("generate", "generate-pairs"):
        return backend
    writes, reads = generator(args), reader(args)
    found = backend
    if writes.model is not None:
        found = dataclasses.replace(found, generator=writes,
                                    generator_rate=generator_rate(args) or (FAKE_GENERATOR_RATE if backend.fake else None))
    if reads.model is not None:
        limit = Limiter(args.vertex_concurrency, None if backend.fake else args.vertex_rpm or None)  # the command's one
        found = dataclasses.replace(found, read_by=reads, reader_rate=reader_rate(args), read_limit=limit,
                                    reader=found.reader if backend.fake else backends.gemini_chat)
    return found


def pace_note(args: argparse.Namespace) -> str:
    """The plan's note on how a generation command's Gemini reads are limited and its 429s waited out."""
    rpm = args.vertex_rpm
    paced = (", unpaced" if not rpm else f", their starts paced to {rpm:g} a minute"
             + (" (a conservative default: Vertex's shared quota has no fixed rate to read)"
                if rpm == DEFAULT_PER_MINUTE else ""))
    reads = ("" if reader(args).model is None
             else f"Gemini reads: at most {args.vertex_concurrency} at once across the run{paced}; ")
    return (f"{reads}a 429 waits a backoff, or its hint plus up to {retry.HINT_JITTER:.0%}, at most "
            f"{retry.MAX_WAIT_S:g} s before that jitter, and tries again; an attempt a call still ended cools down "
            f"{retry.ATTEMPT_COOL_DOWN_S:g} s before the next")
