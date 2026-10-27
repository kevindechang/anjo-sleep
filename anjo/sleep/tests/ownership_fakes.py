# mypy: disallow-untyped-defs=False
"""Fakes for the remote stores a delete reaches (Task 9): a Modal volume and an S3 bucket, in memory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeVolume:
    """A Modal volume: adapter folders at its root."""

    folders: set[str] = field(default_factory=set)

    def listdir(self, path: str) -> list[Any]:
        return [type("Entry", (), {"path": name})() for name in sorted(self.folders)]

    def remove_file(self, path: str, recursive: bool = False) -> None:
        assert recursive and path.startswith("/")
        self.folders.discard(path[1:])


@dataclass
class FakeS3:
    """A bucket: keys under adapters/<name>/, listed a page of 2 at a time."""

    keys: set[str] = field(default_factory=set)

    def list_objects_v2(self, *, Bucket: str, Prefix: str, ContinuationToken: str | None = None) -> dict[str, Any]:
        found = sorted(k for k in self.keys if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = found[start:start + 2]
        more = start + 2 < len(found)
        return {"Contents": [{"Key": k} for k in page], "IsTruncated": more,
                **({"NextContinuationToken": str(start + 2)} if more else {})}

    def delete_objects(self, *, Bucket: str, Delete: dict[str, Any]) -> dict[str, Any]:
        for item in Delete["Objects"]:
            self.keys.discard(item["Key"])
        return {}
