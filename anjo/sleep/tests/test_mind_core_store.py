"""One person's mind on disk: one versioned unit a night, holding her core state, the thought
channel's version and her adapter together, replayed and rolled back as one (no torch)."""
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest

from anjo.sleep.mind.core import store

N1, N2, N3 = date(2026, 10, 28), date(2026, 10, 29), date(2026, 10, 30)


def _store(tmp_path: Path, person: str = "teo") -> store.MindStore:
    return store.MindStore(tmp_path / store.STORE_DIR, person)


def test_each_night_keeps_one_unit_with_everything_that_makes_her_mind(tmp_path: Path) -> None:
    held = _store(tmp_path)
    assert held.head() is None and held.before(N1) is None
    first = held.put_state(b"state one")
    assert held.put_state(b"state one") == first == store.state_id(b"state one")  # the same bytes, the same id
    assert first == hashlib.sha256(b"state one").hexdigest()[:16]  # the thought channel's state_sha form
    one = held.keep(N1, first, model="core-a", projector=None, lora="ad-1")
    two = held.keep(N2, held.put_state(b"state two"), model="core-a", projector="proj-1", lora="ad-2")
    assert (one.version, one.parent, two.version, two.parent) == (1, None, 2, 1)
    assert held.state(two.state) == b"state two" and held.head() == two
    assert held.units() == (one, two)
    assert _store(tmp_path).units() == (one, two)  # read back from disk


def test_a_rerun_of_a_night_starts_from_the_night_before(tmp_path: Path) -> None:
    held = _store(tmp_path)
    one = held.keep(N1, held.put_state(b"1"), model="m", projector=None, lora=None)
    held.keep(N2, held.put_state(b"2"), model="m", projector=None, lora=None)
    assert held.before(N2) == one  # a rerun of N2 must not observe N2's events twice
    again = held.keep(N2, held.put_state(b"2 again"), model="m", projector=None, lora=None)
    assert again.parent == one.version and held.before(N3) == again


def test_a_rollback_restores_a_night_as_one_unit(tmp_path: Path) -> None:
    held = _store(tmp_path)
    one = held.keep(N1, held.put_state(b"1"), model="m", projector="p1", lora="ad-1")
    two = held.keep(N2, held.put_state(b"2"), model="m", projector="p2", lora="ad-2")
    back = held.rollback(one.version)
    assert (back.state, back.projector, back.lora, back.night) == (one.state, "p1", "ad-1", N1)
    assert back.rolled_back_from == two.version and held.head() == back
    assert held.before(N3) == back
    with pytest.raises(store.StoreError, match="no version"):
        held.rollback(99)


def test_a_state_that_doesnt_match_its_id_is_refused(tmp_path: Path) -> None:
    held = _store(tmp_path)
    state_id = held.put_state(b"real")
    (held.folder / store.STATES / f"{state_id}.npz").write_bytes(b"tampered")
    with pytest.raises(store.StoreError, match="doesn't match"):
        held.state(state_id)
    with pytest.raises(store.StoreError, match="unknown"):
        held.state("0" * store.ID_HEX)


def test_erase_removes_only_that_persons_mind(tmp_path: Path) -> None:
    teo, other = _store(tmp_path, "teo"), _store(tmp_path, "_other-1")
    teo.keep(N1, teo.put_state(b"t"), model="m", projector=None, lora=None)
    other.keep(N1, other.put_state(b"o"), model="m", projector=None, lora=None)
    assert teo.erase() is True and not teo.folder.exists() and other.head() is not None
    assert teo.erase() is False
    with pytest.raises(ValueError, match="plain name"):
        store.MindStore(tmp_path, "../escape")


def test_a_unit_records_ids_and_versions_only(tmp_path: Path) -> None:
    held = _store(tmp_path)
    unit = held.keep(N1, held.put_state(b"1"), model="m", projector=None, lora="ad-1")
    assert set(store.unit_record(unit)) == {"version", "night", "state", "model", "projector", "lora", "parent",
                                            "rolled_back_from"}
