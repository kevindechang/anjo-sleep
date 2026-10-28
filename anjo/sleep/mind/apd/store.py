"""The APD runner's files (Task 17): one data directory, written as the pilot goes.

::

    <data>/lives/<life>.json          a life, as the arms and the gates read it
    <data>/sealed/<life>.json         its problem file, sealed: only the judge and the report read it
    <data>/runs/<life>.<arm>.json     each finished run (``ArmRun``), written once
    <data>/judging/<judge>.jsonl      each judge's replies, by the hash of the prompt it was sent
    <data>/audit/pending.jsonl        the blind audit's questions (``judging``)
    <data>/audit/answers.jsonl        and Claude's answers, written by hand

A run is written once, when it ends, so ``run`` resumes by skipping every run already on disk; a
run cut short starts again from its first night. Every file is written whole, then renamed.

The codec is JSON with tags for what JSON lacks (dates, tuples, sets, maps with other keys, and
frozen dataclasses). It rebuilds only dataclasses from Anjo Sleep's own type modules, through
their constructors, so every rebuilt value is checked again; anything else is refused.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final

from anjo.sleep.mind.apd.types import ArmName, ArmRun, Life, ProblemFile, Split

_MODULES: Final = (
    "anjo.sleep.types", "anjo.sleep.mind.apd.types", "anjo.sleep.mind.inquiry.types",
    "anjo.sleep.mind.inquiry.steps", "anjo.sleep.mind.inquiry.ops",
)
_SAFE_ID: Final = re.compile(r"^[\w.+-]+$")


class StoreError(ValueError):
    """A file that isn't what the runner wrote."""


# ── The codec ─────────────────────────────────────────────────────────────────


def encode(value: Any) -> Any:
    """``value`` as JSON-ready data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, date):
        return {"$date": value.isoformat()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        if cls.__module__ not in _MODULES:
            raise StoreError(f"{cls.__name__} isn't one of Anjo Sleep's types")
        fields = {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value) if f.init}
        return {"$type": f"{cls.__module__}:{cls.__qualname__}", "fields": fields}
    if isinstance(value, tuple):
        return {"$tuple": [encode(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        return {"$set": sorted((encode(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))}
    if isinstance(value, Mapping):
        return {"$map": [[encode(key), encode(item)] for key, item in value.items()]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    raise StoreError(f"can't store a {type(value).__name__}")


def _dataclass(name: str) -> type:
    module, _, qualname = name.partition(":")
    if module not in _MODULES or not qualname.isidentifier():
        raise StoreError(f"{name} isn't one of Anjo Sleep's types")
    cls = getattr(importlib.import_module(module), qualname, None)
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise StoreError(f"{name} isn't one of Anjo Sleep's types")
    return cls


def decode(data: Any) -> Any:
    """What ``encode`` wrote, rebuilt."""
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, list):
        return [decode(item) for item in data]
    if not isinstance(data, dict) or len(data) not in (1, 2):
        raise StoreError("not a stored value")
    if "$type" in data:
        fields = data.get("fields")
        if not isinstance(fields, dict):
            raise StoreError("not a stored value")
        return _dataclass(str(data["$type"]))(**{key: decode(value) for key, value in fields.items()})
    (tag, inner), = data.items()
    if tag == "$datetime":
        return datetime.fromisoformat(str(inner))
    if tag == "$date":
        return date.fromisoformat(str(inner))
    if tag == "$tuple" and isinstance(inner, list):
        return tuple(decode(item) for item in inner)
    if tag == "$set" and isinstance(inner, list):
        return frozenset(decode(item) for item in inner)
    if tag == "$map" and isinstance(inner, list):
        return {decode(key): decode(value) for key, value in inner}
    raise StoreError("not a stored value")


def dumps(value: Any) -> str:
    return json.dumps(encode(value), ensure_ascii=False, sort_keys=True)


def loads(text: str, kind: type) -> Any:
    value = decode(json.loads(text))
    if not isinstance(value, kind):
        raise StoreError(f"expected a {kind.__name__}")
    return value


# ── The data directory ────────────────────────────────────────────────────────


def check_id(name: str) -> str:
    if not _SAFE_ID.match(name):
        raise StoreError("an id names a file: letters, digits, '.', '_', '+' and '-' only")
    return name


def write_text(path: Path, text: str) -> None:
    """Write ``path`` whole: to a temporary file first, then renamed over it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class DataDir:
    """The pilot's files under ``root``."""

    root: Path

    def _life(self, life_id: str) -> Path:
        return self.root / "lives" / f"{check_id(life_id)}.json"

    def _sealed(self, life_id: str) -> Path:
        return self.root / "sealed" / f"{check_id(life_id)}.json"

    def _run(self, life_id: str, arm: ArmName) -> Path:
        return self.root / "runs" / f"{check_id(life_id)}.{check_id(arm)}.json"

    @property
    def judging(self) -> Path:
        return self.root / "judging"

    @property
    def audit(self) -> Path:
        return self.root / "audit"

    def save_life(self, life: Life, problems: ProblemFile) -> None:
        """A life and its sealed problem file, the problem file first."""
        if problems.life_id != life.id:
            raise StoreError("a problem file is its own life's")
        write_text(self._sealed(life.id), dumps(problems))
        write_text(self._life(life.id), dumps(life))

    def has_life(self, life_id: str) -> bool:
        return self._life(life_id).exists() and self._sealed(life_id).exists()

    def life(self, life_id: str) -> Life:
        found: Life = loads(self._life(life_id).read_text(encoding="utf-8"), Life)
        if found.id != life_id:
            raise StoreError(f"{life_id}'s file holds another life")
        return found

    def lives(self, split: Split) -> tuple[Life, ...]:
        """Every life of ``split`` on disk, by id."""
        folder = self.root / "lives"
        found = [self.life(path.stem) for path in sorted(folder.glob("*.json"))] if folder.exists() else []
        return tuple(life for life in found if life.split == split)

    def sealed(self, life_id: str) -> ProblemFile:
        """A life's sealed problem file: the judge and the report read it, and nothing else."""
        found: ProblemFile = loads(self._sealed(life_id).read_text(encoding="utf-8"), ProblemFile)
        if found.life_id != life_id:
            raise StoreError(f"{life_id}'s sealed file holds another life's problems")
        return found

    def save_run(self, run: ArmRun) -> None:
        path = self._run(run.life_id, run.arm)
        if path.exists():
            raise StoreError(f"{run.arm} on {run.life_id} is already written")
        write_text(path, dumps(run))

    def run(self, life_id: str, arm: ArmName) -> ArmRun | None:
        path = self._run(life_id, arm)
        if not path.exists():
            return None
        found: ArmRun = loads(path.read_text(encoding="utf-8"), ArmRun)
        if (found.life_id, found.arm) != (life_id, arm):
            raise StoreError(f"{path.name} holds another run")
        return found
