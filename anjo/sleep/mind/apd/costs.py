"""What one generation attempt that passes is expected to cost, priced by each call's token shape
(MIND's data; the plan prints it).

**The shapes.** A call's input is its prompt (the prompt file, at ``CHARS_PER_TOKEN``) and its
payload; its output is its reply, and for a Gemini read the thoughts it is billed for too.

- **A write** (``life_v1``) reads the earlier days and writes one; **an extraction**
  (``extract_open_v1``) reads that day and lists its evidence (``EXTRACT_SHARE`` of the day).
- **A leak read** sees one day, or two adjacent days, of one view: the person's messages
  (``DAY_TOKENS`` a day) or the evidence and targets the arms see (``EVIDENCE_SHARE`` of that), and
  answers a flag. An item surfacing on day index ``n`` has ``n`` single days and ``n - 1`` pairs
  in each view (``backend.item_reads``). **A discoverability read** sees the ``n`` days before the
  surface and answers a statement and its days.
- **Muse Spark** identifies each found item against its sealed item and audits the whole life.

**The day's size is not guessed from the prompt but taken from the one life measured**
(2026-10-28: no items, one attempt on Fireworks, $0.28 settled for its writes and extractions at
$1.25/$4.25): ``DAY_TOKENS`` is the day size that makes those 20 calls cost that (``calibrated_day``).
If Spark Open's replies carried reasoning it is an upper figure. ``READ_OUT`` and
``GEMINI_THOUGHTS`` are assumed. ``calls.breakdown`` prints the real shapes once a life is logged;
the constants are to be replaced by those.

Pure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from anjo.sleep.mind.apd.judging import Rate
from anjo.sleep.mind.apd.types import LIFE_DAYS

PROMPTS: Final = Path(__file__).parent / "prompts"
CHARS_PER_TOKEN: Final = 4.0
PAYLOAD_TOKENS: Final = 40  # a call's JSON keys, dates and beats, besides its days
MEASURED_LIFE_USD: Final = 0.28  # a development life with no items, one attempt, on Fireworks, 2026-10-28
MEASURED_AT: Final = Rate(1.25, 4.25)  # Spark Open's Fireworks list prices that day
EXTRACT_SHARE: Final = 0.5  # an extraction's reply against the day it reads (assumed)
EVIDENCE_SHARE: Final = 0.35  # the evidence view of a day against its messages (assumed)
SEALED_ITEM_TOKENS: Final = 300  # an item as the auditor sees it (assumed)
READ_OUT: Final = {"leak": 8, "discover": 80, "identify": 8, "audit": 60}  # the JSON each answers (assumed)
GEMINI_THOUGHTS: Final = 256  # a low-thinking Gemini read's thoughts, billed as output (assumed; the judge reserves 512)


def prompt_tokens(name: str) -> float:
    return len((PROMPTS / f"{name}.md").read_text(encoding="utf-8")) / CHARS_PER_TOKEN


@dataclass(frozen=True)
class Tokens:
    """Calls and their tokens in and out, summed."""

    calls: float = 0.0
    tokens_in: float = 0.0
    tokens_out: float = 0.0

    def __add__(self, other: Tokens) -> Tokens:
        return Tokens(self.calls + other.calls, self.tokens_in + other.tokens_in, self.tokens_out + other.tokens_out)

    def usd(self, rate: Rate) -> float:
        return rate.usd(round(self.tokens_in), round(self.tokens_out))


def _writes(day: float, days: int = LIFE_DAYS) -> Tokens:
    """A writer and an extraction call for each of ``days`` days of ``day`` tokens."""
    write, extract = prompt_tokens("life_v1") + PAYLOAD_TOKENS, prompt_tokens("extract_open_v1") + PAYLOAD_TOKENS
    tokens_in = sum(write + d * day for d in range(days)) + days * (extract + day)
    return Tokens(2 * days, tokens_in, days * day * (1 + EXTRACT_SHARE))


def calibrated_day() -> float:
    """The day size at which the measured life's writes and extractions cost what it settled."""
    fixed = _writes(0.0).usd(MEASURED_AT)
    per_day = (_writes(1000.0).usd(MEASURED_AT) - fixed) / 1000.0  # a life's writes cost grows with its day
    return (MEASURED_LIFE_USD - fixed) / per_day


DAY_TOKENS: Final = round(calibrated_day())


@dataclass(frozen=True)
class AttemptTokens:
    """One attempt's tokens: the writes and extractions, the gate reads and Muse Spark's calls."""

    writes: Tokens
    reads: Tokens
    muse: Tokens


def item_reads(surface: int, *, thinks: bool, day: float = DAY_TOKENS) -> Tokens:
    """One item surfacing on day index ``surface``: every leak window of both views and its
    discoverability read, each with ``GEMINI_THOUGHTS`` more out when the reader ``thinks``."""
    if surface < 1:
        raise ValueError("an item surfaces after at least one day")
    thoughts = GEMINI_THOUGHTS if thinks else 0
    windows, day_units = 2 * surface - 1, 3 * surface - 2  # per view: n singles and n - 1 pairs
    leak = Tokens(2 * windows, 2 * windows * (prompt_tokens("leak_v1") + PAYLOAD_TOKENS)
                  + day_units * day * (1 + EVIDENCE_SHARE), 2 * windows * (READ_OUT["leak"] + thoughts))
    discover = Tokens(1, prompt_tokens("discover_v1") + PAYLOAD_TOKENS + surface * day,
                      READ_OUT["discover"] + thoughts)
    return leak + discover


def _muse(items: int, day: float, audits: int = 1) -> Tokens:
    identify = Tokens(items, items * (prompt_tokens("identify_v1") + SEALED_ITEM_TOKENS + READ_OUT["discover"]),
                      items * READ_OUT["identify"])
    audit = Tokens(audits, audits * (prompt_tokens("audit_v1") + LIFE_DAYS * day + SEALED_ITEM_TOKENS * items),
                   audits * READ_OUT["audit"])
    return identify + audit


def life_tokens(surfaces: tuple[int, ...], *, thinks: bool, day: float = DAY_TOKENS) -> AttemptTokens:
    """A life whose items surface on ``surfaces`` (day indexes)."""
    reads = Tokens()
    for surface in surfaces:
        reads = reads + item_reads(surface, thinks=thinks, day=day)
    return AttemptTokens(_writes(day), reads, _muse(len(surfaces), day))


def pair_tokens(surfaces: tuple[int, ...], outcomes: int, *, thinks: bool, day: float = DAY_TOKENS) -> AttemptTokens:
    """A pair: its days written and extracted once, each history's ``outcomes`` messages extracted
    for it, and both histories gated: ``surfaces`` holds every item of both."""
    outcome = Tokens(2 * outcomes, 2 * outcomes * (prompt_tokens("extract_open_v1") + PAYLOAD_TOKENS + day / 6),
                     2 * outcomes * day / 6 * EXTRACT_SHARE)  # one message of a day's six or so
    reads = Tokens()
    for surface in surfaces:
        reads = reads + item_reads(surface, thinks=thinks, day=day)
    return AttemptTokens(_writes(day) + outcome, reads, _muse(len(surfaces), day, audits=2))
