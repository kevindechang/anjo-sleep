"""MIND held-out draw (prereg §2). usage: mind_draw.py C COMMIT_DATE OUT.json
C is the prereg-repo commit sha returned by the push that carried the prereg entries, COMMIT_DATE its
committer date (ISO 8601, from the GitHub API). The beacon is drand's default chain, round R = the first
round whose time is at least 10 minutes after COMMIT_DATE; its randomness is unknown when C is pushed.
Seed i of world w = 100000 + int(sha256(f"{randomness}|{C}|mind-cog-heldout|{w}|{i}"), 16) % 900000,
keeping the first 200 distinct values in draw order; the Gemini subset is the first 60.
"""
import hashlib
import json
import math
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

CHAIN = "8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce"  # drand default (mainnet)
GENESIS, PERIOD, DELAY_S = 1595431050, 30, 600
N_LIVES, N_GEMINI, LO, SPAN = 200, 60, 100000, 900000
USED = set(range(0, 10200)) | set(range(20000, 20200))  # training, dev, OOD


def beacon_round(commit_date: str) -> int:
    t = datetime.fromisoformat(commit_date.replace("Z", "+00:00")).timestamp() + DELAY_S
    return math.ceil((t - GENESIS) / PERIOD) + 1


def fetch(round_: int) -> dict:
    if time.time() < GENESIS + (round_ - 1) * PERIOD:
        raise SystemExit(f"round {round_} is still in the future: wait")
    url = f"https://api.drand.sh/{CHAIN}/public/{round_}"
    with urllib.request.urlopen(url, timeout=20) as resp:  # noqa: S310 (fixed https URL)
        body = json.loads(resp.read())
    if body.get("round") != round_ or len(body.get("randomness", "")) != 64:
        raise SystemExit(f"unexpected beacon reply: {body}")
    return body


def draw(randomness: str, commit: str, world: str) -> list[int]:
    out: list[int] = []
    i = 0
    while len(out) < N_LIVES:
        h = hashlib.sha256(f"{randomness}|{commit}|mind-cog-heldout|{world}|{i}".encode()).hexdigest()
        s = LO + int(h, 16) % SPAN
        if s not in out:
            out.append(s)
        i += 1
    if USED & set(out):
        raise SystemExit("a drawn seed was used before")
    return out


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    commit, commit_date, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise SystemExit("C must be a 40-char lowercase commit sha")
    if out.exists():
        raise SystemExit(f"{out} exists: one draw")
    r = beacon_round(commit_date)
    b = fetch(r)
    body = {"C": commit, "commit_date": commit_date, "drand_chain": CHAIN, "drand_round": r,
            "drand_randomness": b["randomness"], "drand_signature": b.get("signature")}
    for world in ("main", "indep"):
        seeds = draw(b["randomness"], commit, world)
        body[world], body[f"{world}_gemini"] = seeds, seeds[:N_GEMINI]
    out.write_text(json.dumps(body, indent=1) + "\n")
    print(hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
