"""verify_claims: every registered file re-hashes to its ledger entry, and any edit fails the run."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from anjo.sleep.prereg.ledger import GENESIS, LedgerEntry, _body, _hash_body
from anjo.sleep.tools import verify_claims as vc

LIFE = "13f17861-04d4-4638-b3bf-2e96582f8853"
ROW = {"outcome": "win", "stats": {"score": 0.779, "ci": [0.713, 0.842]}}
DIMENSIONS = {d: {"3_vs_3x": ROW, "3_vs_1": {**ROW, "outcome": "inconclusive"}} for d in vc.DIMENSIONS}
SCORES = {
    "P1": {"mean": -0.0028, "ci": [-0.004, -0.0016], "pass": True},
    "P2": {"mean": -0.0526, "ci": [-0.0605, -0.0449], "pass": True},
    "P3": {"aurc": {"diff": -0.0713, "ci": [-0.0813, -0.0598]}, "pass": True},
    "P4": {"a_utility_per_life": {"mean": 0.396, "ci": [-0.226, 1.006]}, "pass": False},
    "P5_no_bar": {"log_loss_diff": {"mean": 0.0043, "ci": [0.0034, 0.0053]}},
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chain(rows: Sequence[tuple[str, str, str]]) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []
    prev = GENESIS
    for seq, (kind, digest, note) in enumerate(rows):
        body = _body({"seq": seq, "ts": "2026-11-03T00:00:00Z", "kind": kind, "sha256": digest,
                      "note": note, "prev_hash": prev})
        prev = _hash_body(body)
        entries.append(LedgerEntry(**body, entry_hash=prev))
    return entries


def _write_ledger(path: Path, entries: Sequence[LedgerEntry]) -> Path:
    path.write_text("".join(json.dumps(e.__dict__) + "\n" for e in entries), encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny anjo/sleep with one frozen file, one final.json and a MIND held-out folder."""
    sleep = tmp_path / "sleep"
    (sleep / "prereg").mkdir(parents=True)
    thresholds = sleep / "prereg" / "thresholds.json"
    thresholds.write_text('{"bar": 0.5}\n')
    final = sleep / "results" / LIFE / "scores" / "final.json"
    final.parent.mkdir(parents=True)
    final_text = json.dumps({"dimensions": DIMENSIONS, "life_id": LIFE}, sort_keys=True, indent=1)
    final.write_text(final_text)
    heldout = sleep / "mind" / "cog" / "heldout"
    (heldout / "tools").mkdir(parents=True)
    (heldout / "score").mkdir()
    tool = heldout / "tools" / "score.py"
    tool.write_text("print(1)\n")
    (heldout / "tools" / "MANIFEST.sha256").write_text(f"{_sha(tool.read_bytes())}  score.py\n")
    (heldout / "prereg_mind_heldout.md").write_text("bars\n")
    (heldout / "seeds.json").write_text("[1]\n")
    (heldout / "score" / "SCORES.json").write_text(json.dumps(SCORES))
    rows = [("thresholds", _sha(thresholds.read_bytes()), "prereg/thresholds.json (at the freeze)"),
            ("donor_rule", "0" * 64, "DONOR_RULE (at the freeze)")]
    rows += [(kind, _sha((heldout / name).read_bytes()), name) for kind, name in vc.MIND_FILES]
    rows.append(("final_score", vc.final_digest(LIFE, final_text), f"{LIFE} final.json attempt=1"))
    _write_ledger(tmp_path / "ledger.jsonl", _chain(rows))
    monkeypatch.setattr(vc, "SLEEP", sleep)
    monkeypatch.setattr(vc, "RESULTS", sleep / "results")
    monkeypatch.setattr(vc, "HELDOUT", heldout)
    return tmp_path


def _run(repo: Path, ledger: Path | None = None) -> tuple[int, str]:
    lines: list[str] = []
    rc = vc.main(["--ledger", str(ledger or repo / "ledger.jsonl")], out=lines.append)
    return rc, "\n".join(lines)


def test_every_check_holds_on_untouched_files(repo: Path) -> None:
    rc, out = _run(repo)

    assert rc == 0, out
    assert "FAIL" not in out
    assert "freeze prereg/thresholds.json: ledger seq 0, unchanged since the freeze" in out
    assert "DONOR_RULE" not in out  # a constant, not a file
    assert f"Sleep final.json {LIFE[:8]}: ledger seq 6" in out
    assert "MIND tool score.py: matches the manifest" in out
    assert "own adapter vs another person's    knowledge    0.779" in out
    assert "P5 the overnight step (no bar)       +0.0043" in out and "harmful" in out


@pytest.mark.parametrize("edit", [
    "sleep/prereg/thresholds.json",
    f"sleep/results/{LIFE}/scores/final.json",
    "sleep/mind/cog/heldout/score/SCORES.json",
    "sleep/mind/cog/heldout/tools/score.py",
])
def test_an_edited_registered_file_fails(repo: Path, edit: str) -> None:
    path = repo / edit
    path.write_text(path.read_text().replace("0.779", "0.879") + " ")

    rc, out = _run(repo)

    assert rc == 1
    assert "FAIL" in out and "Some FAILED" in out


def test_an_edited_ledger_entry_breaks_the_chain(repo: Path) -> None:
    ledger = repo / "ledger.jsonl"
    lines = ledger.read_text().splitlines()
    first = json.loads(lines[0])
    first["note"] = "prereg/thresholds.json (edited later)"
    ledger.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")

    rc, out = _run(repo)

    assert rc == 1
    assert "ledger chain: BROKEN" in out


def test_without_the_prereg_branch_it_says_how_to_fetch_it(repo: Path) -> None:
    def missing(_: Sequence[str]) -> str:
        raise vc.LedgerError("can't read origin/prereg:ledger.jsonl (run `git fetch origin prereg` first)")

    lines: list[str] = []
    rc = vc.main([], out=lines.append, run=missing)

    assert rc == 1
    assert "git fetch origin prereg" in lines[0]


def test_a_deleted_registered_score_file_fails(repo: Path) -> None:
    (repo / "sleep" / "results" / LIFE / "scores" / "final.json").unlink()

    rc, out = _run(repo)

    assert rc == 1
    assert f"Sleep final.json {LIFE[:8]}: MISSING (registered at ledger seq 6)" in out


def test_an_unregistered_score_file_fails(repo: Path) -> None:
    extra = repo / "sleep" / "results" / "0121c7fa-0000" / "scores" / "final.json"
    extra.parent.mkdir(parents=True)
    extra.write_text("{}")

    rc, out = _run(repo)

    assert rc == 1
    assert "Sleep final.json 0121c7fa: NO ledger entry registers it" in out
