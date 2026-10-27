"""Remote copies of a person's adapters (Task 9, RF2): the SageMaker bucket's ``adapters/`` and
the Modal adapters volume, each behind ``RemoteAdapters`` with its client passed in.

An adapter name is checked first (``p-<hash>-n<night>``), so a name read from a damaged registry can
never turn into a path outside ``adapters/``. A name that is not there counts as already removed.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any, Final

from anjo.sleep.serving import config

ADAPTERS_PREFIX: Final = "adapters/"
S3_BATCH: Final = 1000  # the most keys one delete_objects call takes


class RemoteDeleteError(RuntimeError):
    """The store refused to remove an adapter's files."""


class S3Adapters:
    """The adapters bucket: every object under ``adapters/<name>/``."""

    name = "sagemaker bucket"

    def __init__(self, bucket: str, s3: Any) -> None:
        self._bucket = bucket
        self._s3 = s3

    def _keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        token: str | None = None
        while True:
            extra = {} if token is None else {"ContinuationToken": token}
            page = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=prefix, **extra)
            keys += [str(item["Key"]) for item in page.get("Contents", [])]
            if not page.get("IsTruncated"):
                return keys
            token = page["NextContinuationToken"]

    def _delete(self, keys: list[str]) -> None:
        for start in range(0, len(keys), S3_BATCH):
            batch = [{"Key": key} for key in keys[start:start + S3_BATCH]]
            done = self._s3.delete_objects(Bucket=self._bucket, Delete={"Objects": batch, "Quiet": True})
            if done.get("Errors"):
                raise RemoteDeleteError(f"the bucket kept {len(done['Errors'])} adapter files")

    def remove(self, adapters: Iterable[str]) -> int:
        removed = 0
        for adapter in sorted(config.check_adapter_name(item) for item in adapters):
            keys = self._keys(f"{ADAPTERS_PREFIX}{adapter}/")
            if keys:
                self._delete(keys)
                removed += 1
        return removed


class ModalAdapters:
    """The adapters volume: one folder per adapter, at its root."""

    name = "modal volume"

    def __init__(self, volume: Any) -> None:
        self._volume = volume

    def remove(self, adapters: Iterable[str]) -> int:
        wanted = {config.check_adapter_name(item) for item in adapters}
        present = {PurePosixPath(entry.path).name for entry in self._volume.listdir("/")}
        found = sorted(wanted & present)
        for adapter in found:
            self._volume.remove_file(f"/{adapter}", recursive=True)
        return len(found)
