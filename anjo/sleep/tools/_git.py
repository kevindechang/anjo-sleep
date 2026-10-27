"""A small git runner shared by the boundary check and the publish pipeline."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path


class GitError(RuntimeError):
    """A git command failed. The message carries git's stderr, never the command's input."""


def run(
    repo: Path,
    *args: str,
    input_bytes: bytes | None = None,
    env: Mapping[str, str] | None = None,
) -> bytes:
    """Run ``git -C repo args`` and return stdout as bytes; raise GitError on failure."""
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_bytes,
        capture_output=True,
        env=dict(env) if env is not None else None,
        check=False,
    )
    if done.returncode != 0:
        stderr = done.stderr.decode("utf-8", "replace").strip()
        raise GitError(f"git {args[0] if args else ''} failed: {stderr}")
    return done.stdout


def text(repo: Path, *args: str, env: Mapping[str, str] | None = None) -> str:
    """Run git and return stdout as stripped text."""
    return run(repo, *args, env=env).decode("utf-8", "replace").strip()


def ok(repo: Path, *args: str) -> bool:
    """Run git and say whether it exited 0 (for questions like ``merge-base --is-ancestor``)."""
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    return done.returncode == 0


def blob_sizes(repo: Path, ids: list[str]) -> dict[str, int]:
    """Object sizes by id, from one ``git cat-file --batch-check`` call."""
    if not ids:
        return {}
    raw = run(
        repo, "cat-file", "--batch-check=%(objectname) %(objectsize)",
        input_bytes="".join(f"{i}\n" for i in ids).encode(),
    ).decode()
    return {name: int(size) for name, size in (line.split() for line in raw.splitlines())}


def merge_base(repo: Path, base: str, head: str = "HEAD") -> str:
    return text(repo, "merge-base", base, head)


def toplevel(path: Path) -> Path:
    return Path(text(path, "rev-parse", "--show-toplevel"))
