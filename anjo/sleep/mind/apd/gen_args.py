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

``generate``'s ``--repair`` gives a core split's attempt that fails only a leak, a missed origin day
or the script audit up to ``repair.ROUNDS`` repair rounds (``repair``); APD's own splits are
regenerated whole, as registered. ``--keep-failed`` keeps each failed check's life under the data
dir's ``quarantine``, on ``core-train`` only: every other split's items are sealed. ``--keep-off``
gives a core split's writer its life's planted topics, to keep to their beats (``generate.KEEP_OFF``);
APD's own splits are written as registered, and the pairs follow most core lives on disk.
"""
from __future__ import annotations

import argparse
import dataclasses

from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import gen_batch, generators, pairs, readers, repair, retry
from anjo.sleep.mind.apd.generators import GENERATORS, Generator
from anjo.sleep.mind.apd.judging import Rate
from anjo.sleep.mind.apd.limiter import DEFAULT_CONCURRENCY, DEFAULT_PER_MINUTE, Limiter
from anjo.sleep.mind.apd.readers import READERS, Reader
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.apd.types import CORE_SPLITS
from anjo.sleep.mind.spark_provider import SparkProvider

FAKE_GENERATOR_RATE = Rate(0.15, 0.6)  # the fakes' generator without list prices: list-like
KEEP_FAILED_SPLIT = "core-train"  # the one split whose failed lives may be kept: its items train the core, unsealed


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


def add_repair(command: argparse.ArgumentParser) -> None:
    command.add_argument("--repair", action="store_true",
                         help=f"a core split's attempt failing only a leak, a missed origin day or the script audit "
                              f"has the days named written again, up to {repair.ROUNDS} rounds, each held as an attempt")
    command.add_argument("--keep-off", action="store_true",
                         help="a core split's writer is given its life's planted topics and told to keep them to "
                              "their beats (recorded in the life's made record)")
    command.add_argument("--keep-failed", action="store_true",
                         help=f"keep {KEEP_FAILED_SPLIT}'s failed attempts' lives under the data dir's quarantine "
                              f"({KEEP_FAILED_SPLIT} only)")


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
    if getattr(args, "repair", False) and split in generators.APD_SPLITS:
        parser.error(f"--repair is for the core splits; APD's {split} lives are regenerated whole, as registered")
    if getattr(args, "keep_off", False) and split in generators.APD_SPLITS:
        parser.error(f"--keep-off is for the core splits; APD's {split} lives are written as registered")
    if getattr(args, "keep_failed", False) and split != KEEP_FAILED_SPLIT:
        parser.error(f"--keep-failed is for {KEEP_FAILED_SPLIT} only: {split}'s items are sealed, and a kept life and "
                     "the auditor's words about it would sit outside its sealed files")
    if not args.fake and writes.model is not None and generator_rate(args) is None:
        parser.error(f"--generator {writes.name} needs --generator-prices in,out to price its holds")
    if getattr(args, "vertex_concurrency", DEFAULT_CONCURRENCY) < 1:
        parser.error("--vertex-concurrency lets at least 1 read through at once")
    if getattr(args, "vertex_rpm", DEFAULT_PER_MINUTE) < 0:
        parser.error("--vertex-rpm paces at least 0 reads a minute (0: unpaced)")


def apply(args: argparse.Namespace, backend: backends.Backend) -> backends.Backend:
    """The backend writing with the command's generator and reading with its reader, at their
    prices (the fakes' list-like); live, a Gemini reader calls Vertex (``backend.gemini_chat``).
    ``generate`` also takes ``--repair``, ``--keep-off`` and ``--keep-failed``."""
    if args.command not in ("generate", "generate-pairs"):
        return backend
    writes, reads = generator(args), reader(args)
    found = backend
    if writes.model is not None:
        found = dataclasses.replace(found, generator=writes,
                                    generator_rate=generator_rate(args) or (FAKE_GENERATOR_RATE if backend.fake else None))
    if args.command == "generate":
        found = dataclasses.replace(found, repair=getattr(args, "repair", False),
                                    keep_failed=getattr(args, "keep_failed", False),
                                    keep_off=getattr(args, "keep_off", False))
    if reads.model is not None:
        limit = Limiter(args.vertex_concurrency, None if backend.fake else args.vertex_rpm or None)  # the command's one
        found = dataclasses.replace(found, read_by=reads, reader_rate=reader_rate(args), read_limit=limit,
                                    reader=found.reader if backend.fake else backends.gemini_chat)
    return found


def repair_note(args: argparse.Namespace) -> tuple[str, ...]:
    """The plan's note on ``--repair``, when it is given."""
    if not getattr(args, "repair", False):
        return ()
    return (f"--repair: an attempt failing only a leak, a missed origin day or the script audit takes up to "
            f"{repair.ROUNDS} repair rounds, each writing at most {repair.REPAIR_DAYS} days again, extracting every "
            f"later day again, and held before it starts at most as an attempt is (a full read again; only "
            f"changed or unread windows are paid for), so the worst case can reach {1 + repair.ROUNDS} times the one "
            f"shown. A round costs a share of the first check that grows the earlier its first day falls: on the "
            f"fakes, about 63% for day 6 of a six-item life",)


def keep_off_warning(args: argparse.Namespace, data: DataDir) -> tuple[str, ...]:
    """A warning when a core split's ``generate`` differs on ``--keep-off`` from most core lives on
    disk (a tie warns neither way): it goes on every core split or none, so the splits compare."""
    split = split_of(args)
    if args.command != "generate" or split not in CORE_SPLITS:
        return ()
    found, on = gen_batch.kept_off(data), bool(getattr(args, "keep_off", False))
    if found.majority is None or found.majority == on:
        return ()
    return (f"--keep-off is {'on' if on else 'off'} for this {split} run, unlike most {found}; it goes on every "
            "core split or none, so core-train, core-val and core-test stay comparable",)


def pairs_keep_off_note(args: argparse.Namespace, data: DataDir) -> tuple[str, ...]:
    """The plan's note on the pairs' ``keep_off`` (``pairs.pairs_keep_off``): which, and why."""
    if args.command != "generate-pairs":
        return ()
    on, why = pairs.pairs_keep_off(data)
    return (f"the pairs are written {'with' if on else 'without'} keep_off (their writer is "
            f"{'' if on else 'not '}given their topics), {why}",)


def pace_note(args: argparse.Namespace) -> str:
    """The plan's note on how a generation command's Gemini reads are limited and its 429s waited out."""
    rpm = args.vertex_rpm
    paced = (", unpaced" if not rpm else f", their starts paced to {rpm:g} a minute"
             + (" (a conservative default: Vertex's shared quota has no fixed rate to read)"
                if rpm == DEFAULT_PER_MINUTE else ""))
    reads = ("" if reader(args).model is None
             else f"Gemini reads: at most {args.vertex_concurrency} at once across the run{paced}; ")
    return (f"{reads}a 429 waits a backoff, or its hint plus up to {retry.HINT_JITTER:.0%}, at most "
            f"{retry.MAX_WAIT_S:g} s before that jitter, and tries again, a call at most {retry.RATE_LIMIT_ATTEMPTS} "
            f"tries in all (its timeouts and 5xx included) and an attempt at most "
            f"{retry.RATE_LIMIT_BUDGET_S / 60:g} min of 429 waits; an attempt a call still ended cools down "
            f"{retry.ATTEMPT_COOL_DOWN_S:g} s before the next")
