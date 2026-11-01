"""Hash what the final test registers at the freeze, before the seed is revealed (spec Task 12b step 2).

    python -m anjo.sleep.prereg.register_freeze               # print the list
    python -m anjo.sleep.prereg.register_freeze --register    # also write ledger entries

The list is ``register_night1``'s files hashed again as they stand at the freeze, plus the final
analysis (``reports/final.py``), the life expansion (its prompt and module), the draw, and the final
preparation commands (``draw_cli``, ``expand_cli``, ``seal_cli`` and ``lives.prep``, which carry the
registered expansion and probe seeds), each noted "at the freeze"; the donor rule; and one
``final_recipe`` entry: the night recipe's settings as this process's environment holds them
(``RECIPE_ENV``; an unset one is recorded as null): its sha256 is that JSON's, its note the JSON and
"(at the freeze)". Run
it in the environment the final lives will run in. It only appends, never twice for the same hash
and note, and it never pushes: ``ledger.push_prereg`` is a separate step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from anjo.sleep.lives.draw import DONOR_RULE
from anjo.sleep.lives.prep import SPARK_SETTING
from anjo.sleep.prereg import ledger
from anjo.sleep.prereg.register_night1 import FILES, SLEEP, Item, register_items

SUFFIX: Final = "at the freeze"
RECIPE_KIND: Final = "final_recipe"
FREEZE_FILES: Final[tuple[tuple[str, str], ...]] = (
    *FILES,
    ("final_report", "reports/final.py"),
    ("prompt_expand", "sim/prompts/expand_v1.md"),
    ("life_expansion", "sim/expand.py"),
    ("lives_draw", "lives/draw.py"),
    ("final_draw_cli", "lives/draw_cli.py"),
    ("final_prep", "lives/prep.py"),
    ("final_expand_cli", "sim/expand_cli.py"),
    ("final_seal_cli", "probes/seal_cli.py"),
)
RECIPE_ENV: Final = (
    "ANJO_SLEEP_OBJECTIVE", "ANJO_SLEEP_CAPACITY", "ANJO_SLEEP_DAY_DREAMS", "ANJO_SLEEP_HARNESS",
    "ANJO_SLEEP_SDFT", "ANJO_SLEEP_TRAIN_RUNNER", "ANJO_SLEEP_CONCURRENCY", "ANJO_SLEEP_JUDGE",
    "ANJO_SLEEP_ADAPTER_FOLD", "ANJO_SLEEP_DREAMS", "ANJO_SLEEP_PROTECT", "ANJO_SLEEP_SERVE_DETERMINISTIC",
    "ANJO_SLEEP_ARMS_EXTRA", "ANJO_SLEEP_MORNING_SUBSET", "ANJO_SLEEP_PRESSURE", SPARK_SETTING,
)


def recipe(environ: Mapping[str, str]) -> dict[str, str | None]:
    """The night recipe's settings, each as set (stripped) or None when unset or blank."""
    return {name: (environ.get(name, "").strip() or None) for name in RECIPE_ENV}


def recipe_note(environ: Mapping[str, str]) -> str:
    return json.dumps(recipe(environ), sort_keys=True, separators=(",", ":"))


def compute_items(environ: Mapping[str, str], root: Path = SLEEP) -> list[Item]:
    items = [Item(kind, rel, ledger.sha256_file(root / rel) if (root / rel).exists() else None)
             for kind, rel in FREEZE_FILES]
    items.append(Item("donor_rule", "DONOR_RULE", ledger.sha256_bytes(DONOR_RULE.encode())))
    note = recipe_note(environ)
    items.append(Item(RECIPE_KIND, note, ledger.sha256_bytes(note.encode())))
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--register", action="store_true", help="write ledger entries")
    args = ap.parse_args(argv)
    items = compute_items(os.environ)
    for i in items:
        print(f"{i.sha256 or 'MISSING' + ' ' * 57}  {i.kind:18} {i.label}")
    if not args.register:
        return 0
    try:
        entries = register_items(items, SUFFIX)
    except ledger.LedgerError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"registered": len(entries), "ledger": str(ledger.ledger_path())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
