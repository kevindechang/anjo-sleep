"""Hash everything that must be registered before night 1 (spec Task 2, "Before night 1").

    python -m anjo.sleep.prereg.register_night1               # print the list
    python -m anjo.sleep.prereg.register_night1 --register    # also write ledger entries

Optional inputs: `--sealed PATH` (repeatable, a sealed probe file) and `--seed-hash HEX`
(the sha256 of the sealed seed). The seed itself is never read or printed here.
Pushing to `prereg` is a separate step (`ledger.push_prereg`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from anjo.sleep.lives.draw import DONOR_RULE
from anjo.sleep.personas.model import (
    canonical_json,
    load_person_file,
    to_held_out_dict,
    to_visible_dict,
)
from anjo.sleep.prereg import ledger

SLEEP = Path(__file__).resolve().parents[1]
PERSONAS = ("film_student", "film_student_donor")
FILES: tuple[tuple[str, str], ...] = (
    ("probe_generator", "probes/generate.py"),
    ("prompt_probe", "probes/prompts/probe_v1.md"),
    ("prompt_judge", "probes/prompts/judge_v1.md"),
    ("prompt_neutral", "probes/prompts/neutral_v1.md"),
    ("prompt_brief", "context/prompts/brief_v1.md"),
    ("thresholds", "prereg/thresholds.json"),
    ("set_general", "sets/general.jsonl"),
    ("set_collapse", "sets/collapse.jsonl"),
    ("lives", "lives/lives.json"),
)
_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Item:
    kind: str
    label: str
    sha256: str | None  # None when the input is missing

    @property
    def missing(self) -> bool:
        return self.sha256 is None


def compute_items(
    root: Path = SLEEP, sealed: tuple[Path, ...] = (), seed_hash: str | None = None
) -> list[Item]:
    items: list[Item] = []
    for kind, rel in FILES:
        p = root / rel
        items.append(Item(kind, rel, ledger.sha256_file(p) if p.exists() else None))
    for name in PERSONAS:
        p = root / "personas" / f"{name}.json"
        if not p.exists():
            items += [Item("persona_visible", name, None), Item("persona_held_out", name, None)]
            continue
        pf = load_person_file(p)
        items.append(
            Item("persona_visible", name, ledger.sha256_bytes(canonical_json(to_visible_dict(pf.visible))))
        )
        items.append(
            Item("persona_held_out", name, ledger.sha256_bytes(canonical_json(to_held_out_dict(pf.held_out))))
        )
    items.append(Item("donor_rule", "DONOR_RULE", ledger.sha256_bytes(DONOR_RULE.encode())))
    for sp in sealed:
        items.append(Item("sealed_probes", sp.name, ledger.sha256_file(sp)))
    if seed_hash is not None:
        if not _HEX.match(seed_hash):
            raise ValueError("--seed-hash must be 64 lowercase hex characters")
        items.append(Item("seed_hash", "sealed until Nov 1", seed_hash))
    return items


def register_items(items: list[Item]) -> list[ledger.LedgerEntry]:
    """Write ledger entries; skip a (kind, label, sha) already registered. Refuse if any input is missing."""
    missing = [i for i in items if i.missing]
    if missing:
        raise ledger.LedgerError("missing inputs: " + ", ".join(f"{i.kind}:{i.label}" for i in missing))
    done = {(e.kind, e.sha256, e.note) for e in ledger.read_entries()}
    out = []
    for i in items:
        note = f"{i.label} (before night 1)"
        assert i.sha256 is not None
        if (i.kind, i.sha256, note) in done:
            continue
        out.append(ledger.register(i.kind, i.sha256, note))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--register", action="store_true", help="write ledger entries")
    ap.add_argument("--sealed", action="append", default=[], type=Path)
    ap.add_argument("--seed-hash")
    args = ap.parse_args(argv)
    items = compute_items(sealed=tuple(args.sealed), seed_hash=args.seed_hash)
    for i in items:
        print(f"{i.sha256 or 'MISSING' + ' ' * 57}  {i.kind:18} {i.label}")
    if args.register:
        entries = register_items(items)
        print(json.dumps({"registered": len(entries), "ledger": str(ledger.ledger_path())}))
    return 1 if any(i.missing for i in items) and args.register else 0


if __name__ == "__main__":
    sys.exit(main())
