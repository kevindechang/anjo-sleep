# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""A life's template texts (MIND's data): each life's and pair's made record keeps a digest of the
template texts its field draws from (``core_families.texts_digest``), and a generation run refuses
to add to lives or pairs on disk written from other texts, as it refuses a mix of writers. A record
from before the digest was kept does not block. Every model is a fake: no network, no spend."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import core_families
from anjo.sleep.mind.apd.store import DataDir


def _run(data: Path, *argv: str) -> tuple[int, str]:
    lines: list[str] = []
    code = cli.main(["--fake", "--data", str(data), *argv], out=lines.append)
    return code, "\n".join(lines)


def test_a_lifes_and_a_pairs_made_record_keep_their_fields_template_texts(tmp_path: Path) -> None:
    # Act
    data = DataDir(tmp_path / "data")
    assert _run(data.root, "--only", "1", "generate", "--split", "core-train")[0] == 0
    assert _run(data.root, "generate-pairs", "--n", "1")[0] == 0

    # Assert: the field's digest, the same for every life of the field, whatever it drew
    (life,) = data.lives("core-train")
    assert (data.made(life.id) or {})["templates"] == core_families.texts_digest("core-train", life.field)
    (pair,) = data.pairs()
    assert (data.made(pair.id) or {})["templates"] == core_families.texts_digest("core-test", pair.failed.field)
    assert len(core_families.texts_digest("core-train", life.field)) == core_families.TEXTS_DIGEST_CHARS


def test_the_digest_moves_with_any_text_its_field_draws_and_no_other(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    truck, furniture = (core_families.texts_digest("core-train", name) for name in ("food truck", "furniture making"))
    pairs_field = core_families.PAIR_TEMPLATES[0].field
    paired = core_families.texts_digest("core-test", pairs_field)
    pool = tuple(dataclasses.replace(t, disambiguating=t.disambiguating + " Again.") if t.id == "truck-cold-reviews-v2"
                 else t for t in core_families._CORE_POOL)
    pair = dataclasses.replace(core_families.PAIR_TEMPLATES[0], ask=core_families.PAIR_TEMPLATES[0].ask + " Again?")

    # Act
    monkeypatch.setattr(core_families, "_CORE_POOL", pool)
    monkeypatch.setattr(core_families, "PAIR_TEMPLATES", (pair, *core_families.PAIR_TEMPLATES[1:]))

    # Assert
    assert core_families.texts_digest("core-train", "food truck") != truck
    assert core_families.texts_digest("core-train", "furniture making") == furniture
    assert core_families.texts_digest("core-test", pairs_field) != paired


def test_a_run_refuses_lives_written_from_other_texts_and_not_older_records(tmp_path: Path) -> None:
    # Arrange: one core-train life on disk, its record from other texts
    data = DataDir(tmp_path / "data")
    assert _run(data.root, "--only", "1", "generate", "--split", "core-train")[0] == 0
    (life,) = data.lives("core-train")
    record = data.made(life.id) or {}
    data.save_made(life.id, {**record, "templates": "0" * core_families.TEXTS_DIGEST_CHARS})

    # Act
    refused = _run(data.root, "--only", "2", "generate", "--split", "core-train")
    other_split = _run(data.root, "--only", "1", "generate", "--split", "core-val")
    data.save_made(life.id, {key: value for key, value in record.items() if key != "templates"})  # before digests
    older = _run(data.root, "--only", "2", "generate", "--split", "core-train")

    # Assert: refused, naming why; nothing more written; another split and a record without a digest don't block
    assert refused[0] == cli.REFUSED_CODE and "other template texts" in refused[1]
    assert other_split[0] == 0 and len(data.lives("core-val")) == 1
    assert older[0] == 0 and len(data.lives("core-train")) == 2


def test_a_pairs_run_refuses_pairs_written_from_other_texts(tmp_path: Path) -> None:
    data = DataDir(tmp_path / "data")
    assert _run(data.root, "generate-pairs", "--n", "1")[0] == 0
    (pair,) = data.pairs()
    data.save_made(pair.id, {**(data.made(pair.id) or {}), "templates": "0" * core_families.TEXTS_DIGEST_CHARS})
    code, text = _run(data.root, "generate-pairs", "--n", "2")
    assert code == cli.REFUSED_CODE and "other template texts" in text and len(data.pairs()) == 1
