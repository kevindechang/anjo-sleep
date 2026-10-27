"""Where a person's sleep files live and how they are written (Task 3).

Everything is under ``data/users/<id>/sleep/``, so account erasure and a factory reset remove it
with the rest of the person's directory, and the export inventory covers it as ``sleep/**``.
Each line is encrypted with ``encrypt_db``, the same at-rest encryption as chat history, and a
line without it is refused rather than read. Writers hold the account's writer guard and file
lock, so erasure waits for a write in flight and a write after erasure is refused.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from anjo.core.crypto import decrypt_db, encrypt_db
from anjo.core.erasure import _user_dir  # the directory account erasure removes
from anjo.core.user_data_guard import user_data_guard
from anjo.core.user_file_lock import user_lock
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.checkpoints import ROOT_DIR, CheckpointRegistry
from anjo.sleep.context.brief_store import JsonlBriefStore
from anjo.sleep.evidence.extractions import ExtractionLog
from anjo.sleep.evidence.jsonl import Guard
from anjo.sleep.evidence.store import JsonlEvidenceStore
from anjo.sleep.evidence.targets import EventLogTargetStore
from anjo.sleep.mind.inquiry.log import JsonlWorkspaceLog
from anjo.sleep.practice.transcripts import JsonlTranscriptStore

SLEEP_DIR: Final = "sleep"
EVIDENCE_FILE: Final = "evidence.jsonl"
TARGETS_FILE: Final = "targets.jsonl"
CLEARED_FILE: Final = "cleared_sessions.jsonl"
EXTRACTIONS_FILE: Final = "extractions.jsonl"
BRIEFS_FILE: Final = "briefs.jsonl"  # the served and training briefs (Task 5)
PRACTICE_FILE: Final = "practice.jsonl"  # the night's practice transcripts (Task 5)
WORKSPACE_FILE: Final = "workspace.jsonl"  # MIND's workspace log (Task 17)
_ENCRYPTED: Final = "enc1:"  # encrypt_db's prefix


def _text(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{where} returned {type(value).__name__}, not text")
    return value


class EncryptedCodec:
    def encode(self, text: str) -> str:
        return _text(encrypt_db(text), "encrypt_db")

    def decode(self, line: str) -> str:
        if not line.startswith(_ENCRYPTED):
            raise ValueError("a sleep record is not encrypted")
        return _text(decrypt_db(line), "decrypt_db")


CODEC: Final = EncryptedCodec()


def sleep_dir(user_id: str) -> Path:
    return Path(_user_dir(user_id)) / SLEEP_DIR


def writer_guard(user_id: str) -> Guard:
    @contextmanager
    def guard() -> Iterator[None]:
        with user_data_guard(user_id), user_lock(_user_dir(user_id)):
            yield

    return guard


def evidence_store(user_id: str, *, hosted_in_training: bool = False) -> JsonlEvidenceStore:
    require_sleep_host()
    return JsonlEvidenceStore(
        sleep_dir(user_id) / EVIDENCE_FILE, user_id, codec=CODEC, guard=writer_guard(user_id),
        hosted_in_training=hosted_in_training,
    )


def target_store(user_id: str) -> EventLogTargetStore:
    require_sleep_host()
    return EventLogTargetStore(
        sleep_dir(user_id) / TARGETS_FILE, user_id, codec=CODEC, guard=writer_guard(user_id),
        workspace=workspace_log(user_id),
    )


def extraction_log(user_id: str) -> ExtractionLog:
    require_sleep_host()
    return ExtractionLog(
        sleep_dir(user_id) / EXTRACTIONS_FILE, user_id, codec=CODEC, guard=writer_guard(user_id),
    )


def brief_store(user_id: str) -> JsonlBriefStore:
    require_sleep_host()
    return JsonlBriefStore(sleep_dir(user_id) / BRIEFS_FILE, user_id, codec=CODEC, guard=writer_guard(user_id))


def transcript_store(user_id: str) -> JsonlTranscriptStore:
    require_sleep_host()
    return JsonlTranscriptStore(sleep_dir(user_id) / PRACTICE_FILE, user_id, codec=CODEC, guard=writer_guard(user_id))


def workspace_log(user_id: str) -> JsonlWorkspaceLog:
    require_sleep_host()
    return JsonlWorkspaceLog(sleep_dir(user_id) / WORKSPACE_FILE, user_id, codec=CODEC, guard=writer_guard(user_id))


def data_root() -> Path:
    """The app's data dir (``ANJO_DATA_DIR`` on the host), read when called."""
    from anjo.core import paths

    return Path(paths.DATA_ROOT)


def checkpoint_registry() -> CheckpointRegistry:
    """Every person's promoted checkpoints: ``<data dir>/sleep/checkpoints``."""
    return CheckpointRegistry(data_root() / SLEEP_DIR / ROOT_DIR)
