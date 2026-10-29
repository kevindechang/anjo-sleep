"""Mapped development preparation: real collector, fake stores and paid calls only."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.mind.collect import collect_life, load_frozen_labels
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.nightlog import POINTS, SUMMARY
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.prereg import ledger as prereg
from anjo.sleep.reports import mind
from anjo.sleep.tests.test_mind_collect import COEFFICIENTS, NIGHT, PERSONA
from anjo.sleep.tests.test_mind_deciders import FakeChat, _points, _world
from anjo.sleep.tools import mind_dev_prepare as tool

USER = "6f8063cb-7e48-4dc8-8d4e-26665047418e"
PRICES = (1.25, 4.25)


@pytest.fixture
def fixture_life(tmp_path):
    world, homework = _world(tmp_path / "source")
    result = run_night_batch_with(
        world.deps(homework=homework, estimator=COEFFICIENTS, is_synthetic=lambda _: True),
        {"p1": NIGHT},
    )["p1"]
    pre, _ = _points(result.log_path)
    for target in pre.raw.targets:
        world.targets("p1").upsert(replace(target, person_id="p1"))
    life = tmp_path / "results" / USER
    folder = life / "nights" / NIGHT.isoformat()
    folder.mkdir(parents=True)
    (folder / SUMMARY).write_text(json.dumps({"night": str(NIGHT), "label": "development"}))
    # Rewrite fixture account ids throughout events and snapshots, preserving all state.
    def account_ids(value):
        if isinstance(value, dict):
            return {key: USER if key == "person_id" else account_ids(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [account_ids(item) for item in value]
        return value

    (folder / POINTS).write_text("".join(
        json.dumps(account_ids(json.loads(line))) + "\n"
        for line in (result.log_path.parent / POINTS).read_text().splitlines()))
    target_path = tmp_path / "source" / "p1" / "targets.jsonl"
    events = tuple(json.loads(json.dumps(r).replace('"p1"', json.dumps(USER)))
                   for r in read_records(target_path, PLAIN))
    evidence = ({"id": "return", "person_id": USER, "day": str(NIGHT),
                 "kind": "correction", "text": "they replied", "source": "message:1",
                 "held_out": False, "created_at": pre.raw.as_of.isoformat(),
                 "target_id": pre.raw.targets[0].id, "ancestry": [], "model": None},)
    source = tool.Records(target_path, target_path.parent / "evidence.jsonl", events, evidence)
    budget = Ledger(tmp_path / "budget.jsonl")
    registration = tmp_path / "prereg.jsonl"
    calls = []

    def chat(model, messages, tools, timeout_s, *, temperature=None):
        assert model == "muse-spark" and temperature == 0
        assert "D0" not in messages[0]["content"]
        assert budget.open_holds()
        calls.append(messages)
        return ChatResult('{"relevant": true}', (), "muse-spark-1.3", 100, 20, "stop")

    kwargs = dict(is_synthetic=lambda _: True, records=lambda _: source, prices=PRICES,
                  budget=budget, registration=registration, chat=chat)
    args = ["--results", str(life.parent), "--life", f"film_founder={USER}"]
    return life, args, kwargs, calls


def test_plan_counts_cost_without_writes_or_calls(fixture_life):
    life, args, kwargs, calls = fixture_life
    lines = []
    before = set(life.parent.parent.rglob("*"))
    assert tool.main(args, out=lines.append, **kwargs) == 0
    text = "\n".join(lines)
    assert "rows (events)" in text and "1 rows" in text
    assert "pairs need labels" in text and "muse-spark-1.3" in text
    assert "expected $" in text and "budget hold $" in text
    assert "meta_usd + judge_usd" in text
    assert not calls
    assert set(life.parent.parent.rglob("*")) == before


def test_go_exports_collects_registers_and_reuses(fixture_life):
    life, args, kwargs, calls = fixture_life
    assert tool.main([*args, "--go"], **kwargs) == 0
    assert all((life / name).is_file() for name in
               ("targets.jsonl", "evidence.jsonl", "relevance.json"))
    record = collect_life(life / "nights", PERSONA,
                          chat=FakeChat({"action": "silence", "target_id": None}))
    assert record.expectations and record.returns and record.nights
    assert record.labels.digest == load_frozen_labels(life / "relevance.json").digest
    entries = prereg.read_entries(kwargs["registration"])
    assert entries[0].sha256 == record.labels.digest
    report = mind.mind_report([record], registered={record.life_id: entries[0].sha256},
                              coefficients=COEFFICIENTS, resamples=20)
    assert report.lives[0].life_id == "film_founder"
    budget_rows = read_records(kwargs["budget"].path, PLAIN)
    settle = next(row for row in budget_rows if row["op"] == "settle")
    assert settle["amount"] == pytest.approx(len(calls) * (100 * 1.25 + 20 * 4.25) / 1e6)
    count = len(calls)
    frozen = (life / "relevance.json").read_bytes()
    assert tool.main([*args, "--go"], **kwargs) == 0
    assert len(calls) == count and (life / "relevance.json").read_bytes() == frozen
    assert len(prereg.read_entries(kwargs["registration"])) == 1


@pytest.mark.parametrize("mapping", [f"stranger={USER}", "film_founder=kevin-demo",
                                      "film_founder=u-4f2a1b2c", "film_founder=../escape"])
def test_unsafe_mapping_refused_before_source_read(tmp_path, mapping):
    def forbidden(_):
        pytest.fail("must refuse before reading a store")
    assert tool.main(["--results", str(tmp_path), "--life", mapping], prices=PRICES,
                     records=forbidden, is_synthetic=lambda _: True) == 1


def _snapshot_targets_owned_by(life, owner: str) -> None:
    for log in life.rglob("mind_points.jsonl"):
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]
        for row in rows:
            row["targets"] = [{**t, "person_id": owner} for t in row.get("targets", [])]
        log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_snapshot_targets_blanked_as_points_writes_them_are_planned(fixture_life):
    life, args, kwargs, calls = fixture_life
    _snapshot_targets_owned_by(life, "")  # points.py:124 writes person_id=""
    lines = []
    assert tool.main(args, out=lines.append, **kwargs) == 0
    assert "pairs need labels" in "\n".join(lines) and not calls


def test_snapshot_targets_of_another_person_are_refused(fixture_life):
    life, args, kwargs, calls = fixture_life
    _snapshot_targets_owned_by(life, "someone-else")
    assert tool.main(args, out=lambda _: None, **kwargs) == 1
    assert not calls


def test_non_synthetic_refused(fixture_life):
    _, args, kwargs, calls = fixture_life
    kwargs["is_synthetic"] = lambda _: False
    assert tool.main([*args, "--go"], **kwargs) == 1
    assert not calls


@pytest.mark.parametrize("rehash", [False, True])
def test_changed_frozen_labels_refused(fixture_life, rehash):
    life, args, kwargs, calls = fixture_life
    assert tool.main([*args, "--go"], **kwargs) == 0
    path = life / "relevance.json"
    document = json.loads(path.read_text())
    document["labels"][0][2] = False
    if rehash:
        from anjo.sleep.probes.relevance import freeze
        document["digest"] = freeze(document["prompt_sha256"],
                                     {(t, i): v for t, i, v in document["labels"]},
                                     frozenset()).digest
    path.write_text(json.dumps(document))
    before = path.read_bytes()
    count = len(calls)
    assert tool.main([*args, "--go"], **kwargs) == 1
    assert path.read_bytes() == before and len(calls) == count


def test_budget_refusal_makes_no_calls_or_artifacts(fixture_life):
    life, args, kwargs, calls = fixture_life
    kwargs["budget"] = Ledger(life.parent.parent / "empty.jsonl", caps={**CAPS, "meta_usd": 0})
    assert tool.main([*args, "--go"], **kwargs) == 1
    assert not calls and not (life / "relevance.json").exists()


def test_host_reads_decode_events_and_use_evidence_export_without_guard(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from anjo.sleep.evidence.jsonl import append_records

    class Codec:
        def encode(self, text):
            return "encrypted:" + text

        def decode(self, line):
            assert line.startswith("encrypted:")
            return line.removeprefix("encrypted:")

    codec = Codec()
    targets_path = tmp_path / "targets.jsonl"
    evidence_path = tmp_path / "evidence.jsonl"
    events = [{"seq": 0, "op": "fixture"}]
    append_records(targets_path, codec, events)
    exported = [{"id": "fixture-export"}]
    names = {
        tool.STORAGE + "target_store": lambda _: SimpleNamespace(path=targets_path),
        tool.STORAGE + "evidence_store": lambda _: SimpleNamespace(
            path=evidence_path, export=lambda: {"evidence": exported}),
        tool.STORAGE + "CODEC": codec,
    }
    # writer_guard is deliberately absent: touching it fails, rather than creating a lock.
    monkeypatch.setattr(tool, "_named", names.__getitem__)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    source = tool.host_records(USER)
    assert source.targets == tuple(events) and source.evidence == tuple(exported)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_failed_call_settles_unknown_usage_and_leaves_no_labels(fixture_life):
    life, args, kwargs, _ = fixture_life

    def failed(*args, **kwargs):
        raise RuntimeError("provider failed after request")

    kwargs["chat"] = failed
    assert tool.main([*args, "--go"], **kwargs) == 1
    events = read_records(kwargs["budget"].path, PLAIN)
    settled = next(e for e in events if e["op"] == "settle")
    assert settled["amount"] > 0
    assert not kwargs["budget"].open_holds()
    assert not (life / "relevance.json").exists()
