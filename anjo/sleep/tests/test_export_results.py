"""Results export: only synthetic people's results leave the data dir (spec §12.4 step 3, RF4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from anjo.sleep.tools import export_results
from anjo.sleep.tools.export_results import ExportRefused

SYNTHETIC = frozenset({"synth-film-student", "synth-donor"})


def _put(data_dir: Path, rel: str, content: bytes = b"{}\n") -> Path:
    path = data_dir / "sleep" / "results" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _written(out: Path) -> set[str]:
    return {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}


def test_export_copies_only_the_results_layout(tmp_path: Path) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    _put(data, "synth-film-student/scores/night-02.json", b'{"score": 1}\n')
    _put(data, "synth-film-student/charts/headline.svg", b"<svg/>\n")
    _put(data, "synth-film-student/nights/2026-10-28/summary.json")
    _put(data, "synth-film-student/nights/2026-10-28/log.jsonl")
    _put(data, "synth-donor/reports/final.md", b"# report\n")
    # Things next to the results tree (sealed probes, checkpoints) are never read.
    sealed = data / "sleep" / "sealed" / "probes.jsonl"
    sealed.parent.mkdir(parents=True)
    sealed.write_text("never exported\n")

    written = export_results.export(data, out, SYNTHETIC)

    assert _written(out) == {
        "synth-film-student/scores/night-02.json",
        "synth-film-student/charts/headline.svg",
        "synth-film-student/nights/2026-10-28/summary.json",
        "synth-film-student/nights/2026-10-28/log.jsonl",
        "synth-donor/reports/final.md",
    }
    assert sorted(written) == sorted(out / rel for rel in _written(out))
    assert (out / "synth-film-student/scores/night-02.json").read_bytes() == b'{"score": 1}\n'


def test_export_refuses_non_synthetic_user(tmp_path: Path) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    _put(data, "synth-donor/scores/a.json")
    _put(data, "someone-real/scores/a.json")

    with pytest.raises(ExportRefused, match="not on the synthetic list"):
        export_results.export(data, out, SYNTHETIC)

    # Refusal happens before anything is written, so nothing partial is left to commit.
    assert not out.exists() or _written(out) == set()


@pytest.mark.parametrize(
    "rel",
    [
        "synth-donor/notes.txt",  # a file outside any kind folder
        "synth-donor/transcripts/day1.json",  # a kind the layout does not name
        "loose.json",  # a file at the top of the results tree
    ],
)
def test_export_refuses_anything_outside_the_layout(tmp_path: Path, rel: str) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    _put(data, rel)

    with pytest.raises(ExportRefused):
        export_results.export(data, out, SYNTHETIC)


@pytest.mark.parametrize("rel", ["synth-donor/scores/adapter.safetensors", "synth-donor/charts/x.pt"])
def test_export_refuses_weights(tmp_path: Path, rel: str) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    _put(data, rel)

    with pytest.raises(ExportRefused, match="refused extension"):
        export_results.export(data, out, SYNTHETIC)


def test_export_refuses_files_over_5_mb(tmp_path: Path) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    _put(data, "synth-donor/charts/huge.svg", b"x" * (5 * 1024 * 1024 + 1))

    with pytest.raises(ExportRefused, match="5 MB"):
        export_results.export(data, out, SYNTHETIC)


def test_export_refuses_symlinks(tmp_path: Path) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}\n")
    link = data / "sleep" / "results" / "synth-donor" / "scores" / "link.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)

    with pytest.raises(ExportRefused, match="symlink"):
        export_results.export(data, out, SYNTHETIC)


def test_export_with_no_results_tree_writes_nothing(tmp_path: Path) -> None:
    assert export_results.export(tmp_path / "data", tmp_path / "out", SYNTHETIC) == []
