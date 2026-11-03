"""Fresh before/after replies for the judge landing (Kevin approved ~$0.50 of Modal, 2026-11-03).

One held-out synthetic life (P07a, 13f17861), its night-2 morning (2026-10-28). The first N new_work
probes in sealed-file order (no picking), each answered by arm 1 (base, no notes), 2C (base + the
overnight notes) and 3 (her adapter, no notes). Replies are fresh samples, not the scored ones.
Writes JSON lines to OUT. Synthetic people only. Run from the freeze checkout with the final's env.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

from anjo.sleep.app.context_builder import RECALL, anjo_context_builder
from anjo.sleep.app.storage import brief_store, checkpoint_registry, evidence_store
from anjo.sleep.context.brief_store import ServedBriefs
from anjo.sleep.probes import arms
from anjo.sleep.probes.generate import load_sealed
from anjo.sleep.serving import backend
from anjo.sleep.app.cleared_sessions import is_synthetic

USER = "13f17861-04d4-4638-b3bf-2e96582f8853"
PERSONA = "P07a"
NIGHT = date(2026, 10, 28)
ARMS = ("1", "2C", "3")
N = 4


def main(sealed_dir: str, out: str) -> int:
    if not is_synthetic(USER):
        print("REFUSED: not a synthetic account", file=sys.stderr)
        return 2
    probes = [replace(p, person_id=USER) for p in load_sealed(Path(sealed_dir) / f"{PERSONA}.jsonl")]
    picked = [p for p in probes if p.kind == "new_work"][:N]
    serving = backend.from_env()
    t0 = time.monotonic()
    woke = serving.wake()
    print(f"wake: {woke} after {time.monotonic() - t0:.0f}s", flush=True)
    client = serving.client().on(NIGHT)
    deps = arms.ArmDeps(
        model=client, adapters=checkpoint_registry(), logprobs=client, donors={}, until=NIGHT,
        contexts=anjo_context_builder(ServedBriefs(brief_store), until=NIGHT, settings=RECALL),
        evidence=evidence_store,
    )
    with open(out, "w", encoding="utf-8") as fh:
        for probe in picked:
            row: dict[str, object] = {"probe_id": probe.id, "kind": probe.kind, "prompt": probe.prompt}
            for arm in ARMS:
                question = arms.prepare(arm, USER, probe, deps=deps)
                answer = arms.ask(question, deps=deps)
                row[arm] = answer.text
                row[f"{arm}_adapter"] = None if question.adapter is None else question.adapter.id
                if arm == "2C":
                    row["2C_context"] = question.context
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"done {probe.id} at {time.monotonic() - t0:.0f}s", flush=True)
    print(f"finished in {time.monotonic() - t0:.0f}s; Modal scales down after its idle window", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
