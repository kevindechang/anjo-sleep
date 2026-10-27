"""Append-only, hash-chained pre-registration ledger.

Stored at `$ANJO_DATA_DIR/sleep/ledger.jsonl`. Each entry commits to the one before it, so
editing, deleting or reordering any old entry makes `verify()` fail. `push_prereg` appends
entries to the `prereg` branch of the public repo through the GitHub contents API.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

GENESIS = "0" * 64
DATA_DIR_ENV = "ANJO_DATA_DIR"
TOKEN_ENV = "ANJO_SLEEP_PREREG_TOKEN"
PREREG_REPO = "kevindechang/anjo-sleep"
PREREG_BRANCH = "prereg"
PREREG_PATH = "ledger.jsonl"
GITHUB_API = "https://api.github.com"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LedgerError(RuntimeError):
    """The ledger is unusable or was tampered with."""


@dataclass(frozen=True)
class LedgerEntry:
    seq: int
    ts: str
    kind: str
    sha256: str
    note: str
    prev_hash: str
    entry_hash: str


def _body(e: Mapping[str, Any]) -> dict[str, Any]:
    return {k: e[k] for k in ("seq", "ts", "kind", "sha256", "note", "prev_hash")}


def _hash_body(body: Mapping[str, Any]) -> str:
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def ledger_path() -> Path:
    base = os.environ.get(DATA_DIR_ENV)
    if not base:
        raise LedgerError(f"{DATA_DIR_ENV} is not set")
    return Path(base) / "sleep" / "ledger.jsonl"


def _parse(lines: Sequence[str]) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []
    for n, line in enumerate(lines, 1):
        try:
            raw = json.loads(line)
            entries.append(LedgerEntry(**raw))
        except (ValueError, TypeError) as exc:
            raise LedgerError(f"line {n} is not a ledger entry") from exc
    return entries


def verify_entries(entries: Sequence[LedgerEntry]) -> int:
    """Check the chain; return the entry count or raise LedgerError."""
    prev = GENESIS
    for i, e in enumerate(entries):
        if e.seq != i:
            raise LedgerError(f"entry {i}: sequence broken")
        if e.prev_hash != prev:
            raise LedgerError(f"entry {i}: chain broken")
        if _hash_body(_body(asdict(e))) != e.entry_hash:
            raise LedgerError(f"entry {i}: contents were edited")
        prev = e.entry_hash
    return len(entries)


def read_entries(path: Path | None = None) -> list[LedgerEntry]:
    p = path or ledger_path()
    if not p.exists():
        return []
    return _parse([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])


def verify(path: Path | None = None) -> int:
    """Raise LedgerError if any entry was edited, removed or reordered."""
    return verify_entries(read_entries(path))


def register(
    kind: str,
    sha256: str,
    note: str,
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> LedgerEntry:
    if not kind.strip():
        raise LedgerError("kind is required")
    if not _SHA256.match(sha256):
        raise LedgerError("sha256 must be 64 lowercase hex characters")
    p = path or ledger_path()
    existing = read_entries(p)
    verify_entries(existing)  # never extend a broken chain
    body = {
        "seq": len(existing),
        "ts": (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": kind,
        "sha256": sha256,
        "note": note,
        "prev_hash": existing[-1].entry_hash if existing else GENESIS,
    }
    entry = LedgerEntry(**body, entry_hash=_hash_body(body))
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(entry), sort_keys=True, ensure_ascii=False) + "\n")
    return entry


def register_file(kind: str, file: str | Path, note: str, **kw: Any) -> LedgerEntry:
    return register(kind, sha256_file(file), note, **kw)


# ---------------------------------------------------------------- publishing


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: dict[str, Any]


class HttpClient(Protocol):
    def request(
        self, method: str, url: str, *, headers: Mapping[str, str], json_body: Any | None
    ) -> HttpResponse: ...


class PushError(RuntimeError):
    """Publishing failed. The message never contains the token."""


class UrllibClient:
    """Real client for the orchestrator. Tests inject a fake instead."""

    def request(
        self, method: str, url: str, *, headers: Mapping[str, str], json_body: Any | None
    ) -> HttpResponse:
        data = json.dumps(json_body).encode() if json_body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=dict(headers))
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                return HttpResponse(resp.status, json.loads(resp.read() or b"{}"))
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read() or b"{}")
            except ValueError:
                body = {}
            return HttpResponse(exc.code, body)


def _token() -> str:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise PushError(f"{TOKEN_ENV} is not set")
    return token


def _line(e: LedgerEntry) -> str:
    return json.dumps(asdict(e), sort_keys=True, ensure_ascii=False)


def push_prereg(entries: Sequence[LedgerEntry], client: HttpClient | None = None) -> str:
    """Append entries to the `prereg` branch ledger file; return the new commit sha."""
    http: HttpClient = client or UrllibClient()
    headers = {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"{GITHUB_API}/repos/{PREREG_REPO}/contents/{PREREG_PATH}"
    got = http.request("GET", f"{url}?ref={PREREG_BRANCH}", headers=headers, json_body=None)
    if got.status == 404:
        remote_lines: list[str] = []
        remote_sha: str | None = None
    elif got.status == 200:
        text = base64.b64decode(got.body.get("content", "")).decode("utf-8")
        remote_lines = [ln for ln in text.splitlines() if ln.strip()]
        remote_sha = got.body.get("sha")
    else:
        raise PushError(f"could not read the {PREREG_BRANCH} ledger (status {got.status})")

    remote = _parse(remote_lines)
    have = {e.entry_hash for e in remote}
    merged = [*remote, *[e for e in entries if e.entry_hash not in have]]
    verify_entries(merged)  # the pushed file must itself be a valid chain
    if len(merged) == len(remote):
        raise PushError("nothing new to push")
    payload: dict[str, Any] = {
        "message": f"chore: register {len(merged) - len(remote)} pre-registration entries",
        "content": base64.b64encode(
            ("\n".join(_line(e) for e in merged) + "\n").encode()
        ).decode(),
        "branch": PREREG_BRANCH,
    }
    if remote_sha:
        payload["sha"] = remote_sha
    put = http.request("PUT", url, headers=headers, json_body=payload)
    if put.status not in (200, 201):
        raise PushError(f"push to {PREREG_BRANCH} failed (status {put.status})")
    return str(put.body["commit"]["sha"])
