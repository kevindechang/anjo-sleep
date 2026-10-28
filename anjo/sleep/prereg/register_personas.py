"""Register new development personas as dated ledger entries (never re-registers anything).

    python -m anjo.sleep.prereg.register_personas --persona PATH [--persona PATH ...]
        [--sealed PATH ...] [--retire PERSON_ID --reason TEXT] [--register]

Hashes only what it is given: each persona's visible and held-out parts (canonical JSON), the
sealed probe files, one `donor_pairs` entry per dev pair whose two files are both given, and a
`persona_retired` entry for `--retire`. It only appends, and it never pushes: run `push_prereg`
separately. Without `--register` it just prints the list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from anjo.sleep.personas.model import (
    PersonFile,
    canonical_json,
    load_person_file,
    to_held_out_dict,
    to_visible_dict,
)
from anjo.sleep.prereg import ledger
from anjo.sleep.prereg.register_night1 import Item, register_items

SUFFIX = "personas-v2"
# The registered development pairs: each is the other's arm 3x donor.
DEV_PAIRS: tuple[tuple[str, str], ...] = (
    ("founder", "founder_donor"),
    ("researcher", "researcher_donor"),
    ("film_founder", "film_founder_donor"),
)


class RegisterPersonasError(ValueError):
    """The given files break a pairing or input rule."""


def pairing(a: str, b: str) -> dict[str, str]:
    """The symmetric donor mapping for one pair, as a plain dict."""
    return {a: b, b: a}


def pairing_sha(a: str, b: str) -> str:
    return ledger.sha256_bytes(canonical_json(pairing(a, b)))


def _check_pair(files: dict[str, PersonFile], a: str, b: str) -> None:
    pa, pb = files[a], files[b]
    if pa.visible.field != pb.visible.field:
        raise RegisterPersonasError(f"{a} and {b} must share a field")
    ta = {t.id: t for t in pa.held_out.taste_ab}
    tb = {t.id: t for t in pb.held_out.taste_ab}
    shared = ta.keys() & tb.keys()
    if not shared or any(
        ta[k].options != tb[k].options or ta[k].pick == tb[k].pick for k in shared
    ):
        raise RegisterPersonasError(f"{a} and {b} must share taste items with opposite picks")


def compute_items(
    persona_paths: tuple[Path, ...] = (),
    sealed: tuple[Path, ...] = (),
    retire: str | None = None,
    reason: str | None = None,
) -> list[Item]:
    if not (persona_paths or sealed or retire):
        raise RegisterPersonasError("nothing to register")
    files: dict[str, PersonFile] = {}
    items: list[Item] = []
    for path in persona_paths:
        pf = load_person_file(path)
        pid = pf.visible.person_id
        if pid in files:
            raise RegisterPersonasError(f"{pid} given twice")
        files[pid] = pf
        items.append(
            Item("persona_visible", pid,
                 ledger.sha256_bytes(canonical_json(to_visible_dict(pf.visible))))
        )
        items.append(
            Item("persona_held_out", pid,
                 ledger.sha256_bytes(canonical_json(to_held_out_dict(pf.held_out))))
        )
    for a, b in DEV_PAIRS:
        given = [x in files for x in (a, b)]
        if any(given) and not all(given):
            raise RegisterPersonasError(f"give both files of the pair {a}/{b}")
        if all(given):
            _check_pair(files, a, b)
            items.append(Item("donor_pairs", f"{a}+{b}", pairing_sha(a, b)))
    unpaired = set(files) - {x for pair in DEV_PAIRS for x in pair}
    if unpaired:
        raise RegisterPersonasError(f"not in a registered pair: {sorted(unpaired)}")
    for sp in sealed:
        items.append(Item("sealed_probes", sp.name, ledger.sha256_file(sp)))
    if retire:
        if not reason or not reason.strip():
            raise RegisterPersonasError("--retire needs --reason")
        body = canonical_json({"person_id": retire, "reason": reason})
        items.append(Item("persona_retired", retire, ledger.sha256_bytes(body)))
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persona", action="append", default=[], type=Path)
    ap.add_argument("--sealed", action="append", default=[], type=Path)
    ap.add_argument("--retire")
    ap.add_argument("--reason")
    ap.add_argument("--register", action="store_true", help="write ledger entries")
    args = ap.parse_args(argv)
    items = compute_items(tuple(args.persona), tuple(args.sealed), args.retire, args.reason)
    for i in items:
        print(f"{i.sha256}  {i.kind:18} {i.label}")
    if args.register:
        entries = register_items(items, suffix=SUFFIX)
        print(json.dumps({"registered": len(entries), "ledger": str(ledger.ledger_path())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
