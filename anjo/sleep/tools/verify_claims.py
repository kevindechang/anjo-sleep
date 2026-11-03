"""Check every headline claim of this repo from its own files, offline, in a few seconds.

    git fetch origin prereg
    python -m anjo.sleep.tools.verify_claims

It needs only the standard library and this checkout:

1. **The ledger.** Reads ``ledger.jsonl`` from the ``prereg`` branch (``git show origin/prereg:...``,
   or ``--ledger <file>``) and checks its hash chain: each entry commits to the one before it, so an
   edited, removed or reordered entry fails (``anjo.sleep.prereg.ledger.verify_entries``).
2. **The freeze** (ledger seq 54 to 71). Hashes each file the freeze registered (the probe and judge
   prompts, the pass marks in ``prereg/thresholds.json``, the gate sets, the draw and the final report
   code) and checks it is unchanged.
3. **Sleep final** (ledger seq 92 to 94). Re-derives each ``results/<life>/scores/final.json``'s
   registered hash (``reports/final.py``: sha256 of ``{"final": <file text>, "life_id": <id>}`` as
   compact sorted JSON) and finds it in a ``final_score`` entry naming that life. Then prints the
   claim rows: own adapter vs another person's (3 v 3x) and vs the base model (3 v 1).
4. **MIND held-out** (ledger seq 84 to 88). Hashes the pre-registration, the tools manifest, the
   seeds and the scores in ``mind/cog/heldout/`` against their entries, checks every tool against
   the manifest, then prints P1 to P5.

Exit 0 when every check holds, 1 when any fails, and each line says which.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from anjo.sleep.prereg.ledger import LedgerEntry, LedgerError, verify_entries

SLEEP: Final = Path(__file__).resolve().parents[1]
RESULTS: Final = SLEEP / "results"
HELDOUT: Final = SLEEP / "mind" / "cog" / "heldout"
LEDGER_REF: Final = "origin/prereg:ledger.jsonl"
FREEZE_SUFFIX: Final = " (at the freeze)"
MIND_FILES: Final = (  # (ledger kind, file under mind/cog/heldout/)
    ("mind_prereg", "prereg_mind_heldout.md"),
    ("mind_tools", "tools/MANIFEST.sha256"),
    ("mind_seeds", "seeds.json"),
    ("mind_scores", "score/SCORES.json"),
)
DIMENSIONS: Final = ("knowledge", "interaction")
ROWS: Final = (("3_vs_3x", "own adapter vs another person's"), ("3_vs_1", "own adapter vs the base model"))


@dataclass(frozen=True)
class Check:
    ok: bool
    line: str


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def final_digest(life_id: str, text: str) -> str:
    """The hash ``reports/final.py`` registered for one ``final.json``."""
    body = json.dumps({"final": text, "life_id": life_id}, sort_keys=True, separators=(",", ":"))
    return sha256(body.encode("utf-8"))


def read_ledger(path: Path | None, run: Callable[[Sequence[str]], str]) -> list[LedgerEntry]:
    text = path.read_text(encoding="utf-8") if path is not None else run(["git", "show", LEDGER_REF])
    return [LedgerEntry(**json.loads(line)) for line in text.splitlines() if line.strip()]


def _git(args: Sequence[str]) -> str:
    try:
        return subprocess.run(list(args), cwd=SLEEP, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise LedgerError(f"can't read {LEDGER_REF} (run `git fetch origin prereg` first): {error}") from None


def check_ledger(entries: Sequence[LedgerEntry]) -> Check:
    try:
        count = verify_entries(entries)
    except LedgerError as error:
        return Check(False, f"ledger chain: BROKEN ({error})")
    return Check(True, f"ledger chain: {count} entries, every hash links to the one before")


def check_freeze(entries: Sequence[LedgerEntry], root: Path = SLEEP) -> list[Check]:
    """Every file the freeze registered, by its path under ``anjo/sleep/``, still has that hash."""
    checks: list[Check] = []
    for e in entries:
        name = e.note.removesuffix(FREEZE_SUFFIX)
        if name == e.note or "." not in Path(name).name:  # not a freeze entry, or not a file (DONOR_RULE)
            continue
        path = root / name
        ok = path.is_file() and sha256(path.read_bytes()) == e.sha256
        state = "unchanged since the freeze" if ok else ("MISSING" if not path.is_file() else "CHANGED since the freeze")
        checks.append(Check(ok, f"freeze {name}: ledger seq {e.seq}, {state}"))
    if not checks:
        checks.append(Check(False, "freeze: no registered files found in the ledger"))
    return checks


def check_sleep(entries: Sequence[LedgerEntry], results: Path = RESULTS) -> tuple[list[Check], dict[str, Any] | None]:
    """Each life a ``final_score`` entry names has its ``final.json``, and that file hashes to the entry.

    The expected files come from the ledger, so a deleted score file fails, and so does a
    ``final.json`` no entry registered.
    """
    registered: dict[str, list[LedgerEntry]] = {}
    for e in entries:
        if e.kind == "final_score":
            registered.setdefault(e.note.split()[0], []).append(e)
    checks: list[Check] = []
    finals: list[dict[str, Any]] = []
    for life, scored in sorted(registered.items()):
        path = results / life / "scores" / "final.json"
        if not path.is_file():
            checks.append(Check(False, f"Sleep final.json {life[:8]}: MISSING (registered at ledger seq {scored[-1].seq})"))
            continue
        text = path.read_text(encoding="utf-8")
        hit = next((e for e in scored if e.sha256 == final_digest(life, text)), None)
        checks.append(Check(hit is not None, f"Sleep final.json {life[:8]}: "
                            + (f"ledger seq {hit.seq}" if hit else "CHANGED since its ledger entry")))
        finals.append(json.loads(text)["dimensions"])
    for path in sorted(results.glob("*/scores/final.json")):
        if path.parents[1].name not in registered:
            checks.append(Check(False, f"Sleep final.json {path.parents[1].name[:8]}: NO ledger entry registers it"))
    if not registered:
        return [Check(False, "Sleep final: no final_score entries in the ledger")], None
    if finals:
        same = all(f == finals[0] for f in finals)
        checks.append(Check(same, f"Sleep final: the {len(finals)} files carry {'the same' if same else 'DIFFERENT'} claim rows"))
    return checks, (finals[0] if finals else None)


def check_mind(entries: Sequence[LedgerEntry], heldout: Path = HELDOUT) -> tuple[list[Check], dict[str, Any] | None]:
    by_kind: dict[str, list[LedgerEntry]] = {}
    for e in entries:
        by_kind.setdefault(e.kind, []).append(e)
    checks: list[Check] = []
    for kind, name in MIND_FILES:
        path = heldout / name
        if not path.exists():
            checks.append(Check(False, f"MIND {name}: missing"))
            continue
        digest = sha256(path.read_bytes())
        hit = next((e for e in by_kind.get(kind, []) if e.sha256 == digest), None)
        checks.append(Check(hit is not None, f"MIND {name}: " + (f"ledger seq {hit.seq}" if hit else "NO matching ledger entry")))
    manifest = heldout / "tools" / "MANIFEST.sha256"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            tool = heldout / "tools" / Path(name.strip().lstrip("*")).name
            ok = tool.exists() and sha256(tool.read_bytes()) == digest
            checks.append(Check(ok, f"MIND tool {tool.name}: {'matches the manifest' if ok else 'DIFFERS from the manifest'}"))
    scores = heldout / "score" / "SCORES.json"
    return checks, (json.loads(scores.read_text(encoding="utf-8")) if scores.exists() else None)


def _ci(row: dict[str, Any]) -> str:
    lo, hi = row["ci"]
    return f"95% CI {lo:+.4f} to {hi:+.4f}"


def sleep_table(dimensions: dict[str, Any]) -> list[str]:
    lines = ["", "Sleep final (3 held-out simulated people, 2 nights each; share of blind pairwise wins, ties half):"]
    for key, label in ROWS:
        for dim in DIMENSIONS:
            row = dimensions[dim][key]
            stats = row["stats"]
            lo, hi = stats["ci"]
            lines.append(f"  {label:<34} {dim:<12} {stats['score']:.3f}  (95% CI {lo:.3f} to {hi:.3f})  {row['outcome']}")
    return lines


def mind_table(scores: dict[str, Any]) -> list[str]:
    p4 = scores["P4"]["a_utility_per_life"]
    p5 = scores["P5_no_bar"]["log_loss_diff"]
    p3 = scores["P3"]["aurc"]
    return [
        "", "MIND held-out (60 simulated people, 12 weeks each; log loss, negative = MIND better):",
        f"  P1 vs online logistic regression     {scores['P1']['mean']:+.4f}  ({_ci(scores['P1'])})  {'pass' if scores['P1']['pass'] else 'fail'}",
        f"  P2 vs best Gemini setup (answered)   {scores['P2']['mean']:+.4f}  ({_ci(scores['P2'])})  {'pass' if scores['P2']['pass'] else 'fail'}",
        f"  P3 uncertainty ranks its errors      {p3['diff']:+.4f}  ({_ci(p3)}, AURC)  {'pass' if scores['P3']['pass'] else 'fail'}",
        f"  P4 asking questions pays off         {p4['mean']:+.3f}  ({_ci(p4)}, utility per life)  {'pass' if scores['P4']['pass'] else 'fail'}",
        f"  P5 the overnight step (no bar)       {p5['mean']:+.4f}  ({_ci(p5)})  {'harmful' if p5['ci'][0] > 0 else 'not harmful'}",
    ]


def main(argv: Sequence[str] | None = None, *, out: Callable[[str], None] = print,
         run: Callable[[Sequence[str]], str] = _git) -> int:
    parser = argparse.ArgumentParser(prog="python -m anjo.sleep.tools.verify_claims", description=__doc__.split("\n")[0])
    parser.add_argument("--ledger", type=Path, help=f"a ledger.jsonl file (default: {LEDGER_REF})")
    args = parser.parse_args(argv)
    try:
        entries = read_ledger(args.ledger, run)
    except (LedgerError, OSError, ValueError, TypeError) as error:
        out(f"FAIL  {error}")
        return 1
    sleep_checks, dimensions = check_sleep(entries, RESULTS)
    mind_checks, scores = check_mind(entries, HELDOUT)
    checks = [check_ledger(entries), *check_freeze(entries, SLEEP), *sleep_checks, *mind_checks]
    for check in checks:
        out(f"{'ok  ' if check.ok else 'FAIL'}  {check.line}")
    if dimensions is not None:
        for line in sleep_table(dimensions):
            out(line)
    if scores is not None:
        for line in mind_table(scores):
            out(line)
    failed = sum(not c.ok for c in checks)
    out("")
    out(f"{len(checks) - failed} of {len(checks)} checks hold." + ("" if not failed else " Some FAILED; see above."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
