"""Apply a frozen Platt map (prereg §1) to a raw Gemini run's records. usage: SRC.jsonl DST.jsonl RUN A B"""
import sys
from dataclasses import replace
from pathlib import Path

from anjo.sleep.mind.cog.gemini import apply_platt
from anjo.sleep.mind.cog.run import load_records, save_records

src, dst, run, a, b = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
if dst.exists():
    raise SystemExit(f"{dst} exists")
out = [replace(r, run=run, predictor=run, queries=tuple(replace(q, p=apply_platt(q.p, (a, b))) for q in r.queries))
       for r in load_records(src)]
save_records(dst, out)
print(f"{run}: {len(out)} lives, a={a} b={b}")
