"""The Nov 1 reveal and draw (Task 12b step 3-4), once, into the prereg ledger.

    python -m anjo.sleep.lives.draw_cli --seed-file PATH --registered-hash HEX --lives-sha HEX [--dry-run]

**The seed.** The seed file was sealed before night 1: its sha256, over the file's raw bytes, is the
ledger's ``seed_hash`` entry. The reveal checks exactly that: sha256(raw bytes) equals
``--registered-hash``, and the ledger holds a ``seed_hash`` entry with that hash. ``draw_lives`` takes
an ``int``; the file's bytes become it by ``seed_from_bytes``: one trailing newline is dropped, and the
rest is read as one big-endian unsigned integer (``int.from_bytes(body, "big")``). Every byte of the
secret counts, and the public hash alone gives nothing away.

**What it checks before anything is written:** the seed file against the registered hash, and a
``seed_hash`` entry with it; ``lives.json`` against ``--lives-sha`` and a ``lives`` entry with it; the
ledger's chain; that no ``seed_reveal`` or ``final_draw`` entry exists (the draw happens once).

**What it writes:** a ``seed_reveal`` entry (the hash, the seed's text and its integer), then a
``final_draw`` entry whose note is the draw as canonical JSON (the pair, the single life, the donor
map ``DONOR_RULE`` gives, the hashes it was drawn from) and whose sha256 is that note's. It prints the
draw as JSON. ``--dry-run`` checks the same and writes nothing. ``registered_draw`` reads the one
``final_draw`` entry back for the later steps (expansion, probes), checking its hash.
"""
from __future__ import annotations

import argparse
import base64
import json
import string
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from anjo.sleep.lives.draw import DONOR_RULE, LIVES_PATH, LivesError, draw_lives, load_pairs
from anjo.sleep.lives.prep import canonical
from anjo.sleep.prereg import ledger

SEED_HASH_KIND: Final = "seed_hash"
LIVES_KIND: Final = "lives"
REVEAL_KIND: Final = "seed_reveal"
DRAW_KIND: Final = "final_draw"
_PRINTABLE: Final = frozenset(string.printable) - frozenset("\r\n\t\x0b\x0c")


class DrawRefused(RuntimeError):
    """A check before the draw failed; nothing was written."""


@dataclass(frozen=True)
class FinalDraw:
    pair: tuple[str, str]
    single: str
    donors: dict[str, str]

    @property
    def lives(self) -> tuple[str, str, str]:
        return (self.pair[0], self.pair[1], self.single)


def seed_from_bytes(raw: bytes) -> int:
    """The sealed seed file's bytes as ``draw_lives``' seed: one trailing newline dropped, the rest
    one big-endian unsigned integer."""
    body = raw[:-1] if raw.endswith(b"\n") else raw
    if not body:
        raise DrawRefused("the seed file is empty")
    return int.from_bytes(body, "big")


def _seed_text(raw: bytes) -> str:
    body = raw[:-1] if raw.endswith(b"\n") else raw
    try:
        text = body.decode("ascii")
    except UnicodeDecodeError:
        text = None
    if text is not None and set(text) <= _PRINTABLE:
        return text
    return "base64:" + base64.b64encode(body).decode("ascii")


def _has(entries: Sequence[ledger.LedgerEntry], kind: str, sha: str) -> bool:
    return any(e.kind == kind and e.sha256 == sha for e in entries)


def _check(seed_file: Path, registered_hash: str, lives_sha: str, lives_path: Path) -> tuple[bytes, list[ledger.LedgerEntry]]:
    raw = seed_file.read_bytes()
    found = ledger.sha256_bytes(raw)
    if found != registered_hash.lower():
        raise DrawRefused("the seed file does not match the registered hash")
    entries = ledger.read_entries()
    ledger.verify_entries(entries)
    if not _has(entries, SEED_HASH_KIND, found):
        raise DrawRefused("the ledger has no seed_hash entry with this hash")
    if ledger.sha256_file(lives_path) != lives_sha.lower():
        raise DrawRefused("lives.json does not match --lives-sha")
    if not _has(entries, LIVES_KIND, lives_sha.lower()):
        raise DrawRefused("the ledger has no lives entry with --lives-sha")
    done = sorted({e.kind for e in entries if e.kind in (REVEAL_KIND, DRAW_KIND)})
    if done:
        raise DrawRefused(f"the ledger already holds {', '.join(done)}: the draw happens once")
    return raw, entries


def draw_record(seed: int, seed_sha: str, lives_sha: str, lives_path: Path = LIVES_PATH) -> dict[str, Any]:
    """The draw ``seed`` gives, as the ``final_draw`` note records it."""
    drawn = draw_lives(seed, lives_path, expected_sha256=lives_sha)
    pair_of = {lf.id: p.pair_id for p in load_pairs(lives_path) for lf in p.lives}
    return {
        "pair": [drawn.pair[0].id, drawn.pair[1].id],
        "pair_id": drawn.pair[0].pair_id,
        "single": drawn.single.id,
        "single_pair_id": pair_of[drawn.single.id],
        "donors": dict(sorted(drawn.donors.items())),
        "donor_rule_sha256": ledger.sha256_bytes(DONOR_RULE.encode()),
        "seed_sha256": seed_sha,
        "lives_sha256": lives_sha,
    }


def run(seed_file: Path, registered_hash: str, lives_sha: str, *, dry_run: bool = False,
        lives_path: Path = LIVES_PATH) -> dict[str, Any]:
    """Check, draw, and (unless ``dry_run``) register the reveal and the draw. Raises ``DrawRefused``."""
    raw, _ = _check(seed_file, registered_hash, lives_sha, lives_path)
    seed = seed_from_bytes(raw)
    seed_sha = ledger.sha256_bytes(raw)
    try:
        record = draw_record(seed, seed_sha, lives_sha.lower(), lives_path)
    except LivesError as error:
        raise DrawRefused(str(error)) from None
    if dry_run:
        return {**record, "dry_run": True}
    reveal = ledger.register(REVEAL_KIND, seed_sha, canonical({"seed_text": _seed_text(raw), "seed_int": str(seed)}))
    note = canonical(record)
    entry = ledger.register(DRAW_KIND, ledger.sha256_bytes(note.encode()), note)
    return {**record, "dry_run": False, "reveal_seq": reveal.seq, "draw_seq": entry.seq}


def registered_draw(path: Path | None = None) -> FinalDraw:
    """The one ``final_draw`` entry, its note checked against its hash and the donor rule."""
    entries = ledger.read_entries(path)
    ledger.verify_entries(entries)
    found = [e for e in entries if e.kind == DRAW_KIND]
    if len(found) != 1:
        raise DrawRefused(f"the ledger holds {len(found)} final_draw entries, not one")
    entry = found[0]
    if ledger.sha256_bytes(entry.note.encode()) != entry.sha256:
        raise DrawRefused("the final_draw note does not match its hash")
    record = json.loads(entry.note)
    pair = (str(record["pair"][0]), str(record["pair"][1]))
    single = str(record["single"])
    donors = {str(k): str(v) for k, v in record["donors"].items()}
    if donors != {pair[0]: pair[1], pair[1]: pair[0], single: pair[0]}:
        raise DrawRefused("the final_draw donor map is not the registered rule's")
    return FinalDraw(pair=pair, single=single, donors=donors)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reveal the sealed seed and draw the final lives, once.")
    ap.add_argument("--seed-file", type=Path, required=True)
    ap.add_argument("--registered-hash", required=True, help="the ledger's seed_hash entry (sha256 of the raw file)")
    ap.add_argument("--lives-sha", required=True, help="the ledger's lives entry (sha256 of lives.json)")
    ap.add_argument("--dry-run", action="store_true", help="check everything and write nothing")
    args = ap.parse_args(argv)
    try:
        out = run(args.seed_file, args.registered_hash, args.lives_sha, dry_run=args.dry_run)
    except (DrawRefused, ledger.LedgerError, OSError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(out, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
