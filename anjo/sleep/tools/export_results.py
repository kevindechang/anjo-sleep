"""Copy results about synthetic people out of the data dir for publishing (spec §12.4 step 3).

The layout it expects, and the only one it reads::

    $ANJO_DATA_DIR/sleep/results/
        <user_id>/
            scores/   any files: per-night and final scores
            charts/   any files: rendered charts
            nights/   any files, usually nights/<night>/summary.json and log.jsonl
            reports/  any files: written reports

Nothing else under ``$ANJO_DATA_DIR/sleep/`` is read: sealed probes, checkpoints, evidence
and the registry live beside ``results/`` and never pass through here.

Every ``<user_id>`` folder must be on the synthetic list. Results about a real person,
Kevin's included, must live outside ``sleep/results/`` (for example
``sleep/private_results/``); his aggregate scores are published only with his OK, as one
file he has read, added by hand. A non-synthetic folder, a file outside the four kind
folders, a symlink, a weights file (.safetensors .gguf .bin .pt) or a file over 5 MB makes
``export`` raise ``ExportRefused`` before it writes anything, so a refusal is never a
silent skip and never leaves a partial copy.

The copy lands at ``out/<user_id>/<kind>/...``; the publish script points ``out`` at
``anjo/sleep/results/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

RESULT_KINDS: frozenset[str] = frozenset({"scores", "charts", "nights", "reports"})
REFUSED_EXTENSIONS: tuple[str, ...] = (".safetensors", ".gguf", ".bin", ".pt")
MAX_FILE_BYTES = 5 * 1024 * 1024


class ExportRefused(RuntimeError):
    """The results tree holds something that must not be published."""


def results_root(data_dir: Path) -> Path:
    return data_dir / "sleep" / "results"


def _refuse_non_synthetic(root: Path, synthetic_ids: frozenset[str]) -> None:
    entries = sorted(root.iterdir())
    loose = [entry for entry in entries if entry.is_symlink() or not entry.is_dir()]
    if loose:
        raise ExportRefused(
            f"{len(loose)} entry(ies) at the top of {root} are not user folders; "
            "only <user_id>/ folders belong there"
        )
    unknown = [entry for entry in entries if entry.name not in synthetic_ids]
    if unknown:
        # The ids are withheld: they may be real people.
        raise ExportRefused(
            f"{len(unknown)} user folder(s) under {root} are not on the synthetic list; "
            "move real people's results out of sleep/results/"
        )


def _check_file(path: Path, rel: str) -> None:
    if path.is_symlink():
        raise ExportRefused(f"{rel}: a symlink; only regular files are exported")
    if not path.is_file():
        return
    parts = rel.split("/")
    if len(parts) < 3 or parts[1] not in RESULT_KINDS:
        raise ExportRefused(
            f"{rel}: outside the results layout (<user_id>/{{{','.join(sorted(RESULT_KINDS))}}}/...)"
        )
    if path.name.lower().endswith(REFUSED_EXTENSIONS):
        raise ExportRefused(f"{rel}: refused extension (weights never leave the data dir)")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ExportRefused(f"{rel}: over 5 MB")


def _collect(root: Path) -> list[tuple[Path, str]]:
    """Every exportable file under ``root`` as (source, relative path); raises on anything else."""
    found: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        _check_file(path, rel)
        if path.is_file() and not path.is_symlink():
            found.append((path, rel))
    return found


def export(data_dir: Path, out: Path, synthetic_ids: frozenset[str]) -> list[Path]:
    """Copy the synthetic people's results to ``out`` and return the written paths."""
    root = results_root(data_dir)
    if not root.is_dir():
        return []
    _refuse_non_synthetic(root, synthetic_ids)
    files = _collect(root)
    written: list[Path] = []
    for source, rel in files:
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        written.append(target)
    return written


def read_synthetic_ids(path: Path) -> frozenset[str]:
    """One id per line; blank lines and ``#`` comments are ignored."""
    ids = {line.strip() for line in path.read_text().splitlines()}
    return frozenset(i for i in ids if i and not i.startswith("#"))
