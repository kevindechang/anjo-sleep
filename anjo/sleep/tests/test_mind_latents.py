# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's event latents: each event read through Glimmer's own hidden states, mean-pooled over its
tokens at fixed layers, cached by content hash, and computed on the Modal training app under a hold
on its pool. ``modal`` and the container are fakes here; no GPU, no network."""
from __future__ import annotations

import fcntl
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anjo.sleep.budget import DATA_DIR_ENV, Ledger
from anjo.sleep.mind.core import latents
from anjo.sleep.mind.sweep import MIND_KINDS, MindBusy, lock_path
from anjo.sleep.night.holds import hold
from anjo.sleep.serving import config
from anjo.sleep.train import latent_job, latent_runner

PIN = "c" * 40
T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def _events(texts: Sequence[str], person: str = "life-1") -> list[latents.EventInput]:
    return [latents.EventInput(f"e{i}", person, T0.replace(hour=9 + i), "turn", text) for i, text in enumerate(texts)]


def _write_events(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class Counting:
    """An encoder that records every text it is asked for."""

    def __init__(self, inner: latents.FakeEncoder) -> None:
        self.inner = inner
        self.spec = inner.spec
        self.seen: list[str] = []

    def encode(self, items: Sequence[latents.EncodeItem]) -> latents.Matrix:
        self.seen.extend(item.text for item in items)
        return self.inner.encode(items)


def test_latent_index_header_declares_d_and_layers(tmp_path: Path) -> None:
    spec = latents.glimmer_spec(PIN, layers=(19, 35))
    events = _events(["the supplier cancelled", "camera confirmed", "rent is due"])
    matrix = np.random.default_rng(0).standard_normal((3, spec.d)).astype(np.float32)
    assert latents.write_latents(tmp_path / "out", spec, events, matrix, [False, True, False]) == [
        tmp_path / "out" / "life-1"]  # one folder per life, so a delete finds it by its id

    lines = (tmp_path / "out" / "life-1" / latents.INDEX_FILE).read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert [json.loads(line)["truncated"] for line in lines[1:]] == [False, True, False]  # cut to fit, per event
    assert json.loads(lines[1])["role"] == "user" and header["truncated"] == 1
    assert header["format"] == latents.FORMAT
    assert header["d"] == spec.d == 2 * latents.HIDDEN == 13312
    assert header["layers"] == [19, 35]
    assert (header["model"], header["revision"], header["pooling"]) == (config.MODEL, PIN, "mean_event_tokens")
    assert header["count"] == 3 and header["dtype"] == "float32"
    assert set(latents.load_run(tmp_path / "out")) == {"life-1"}
    loaded_header, loaded = latents.load_latents(tmp_path / "out" / "life-1")
    assert loaded_header == header
    assert [item.event_id for item in loaded] == ["e0", "e1", "e2"]
    assert loaded[1].life_or_person == "life-1" and loaded[1].t == events[1].t and loaded[1].kind == "turn"
    assert loaded[1].vec.shape == (spec.d,) and loaded[1].vec.dtype == np.float32
    np.testing.assert_array_equal(loaded[1].vec, matrix[1])
    with pytest.raises(latents.LatentError, match="D"):
        latents.write_latents(tmp_path / "bad", spec, events, matrix[:, :10])
    # The default: one global (NoPE, full-attention) layer about two thirds of the way down.
    default = latents.glimmer_spec(PIN)
    assert default.layers == (35,) == latents.DEFAULT_LAYERS and default.d == latents.HIDDEN
    assert all(layer in latents.GLOBAL_LAYERS for layer in default.layers)
    with pytest.raises(latents.LatentError, match="layer"):
        latents.glimmer_spec(PIN, layers=(52,))


def test_encoder_cache_skips_seen_events(tmp_path: Path) -> None:
    events = _events(["the supplier cancelled", "camera confirmed", "rent is due"])
    encoder = Counting(latents.FakeEncoder(d=16))
    first = latents.encode_events(events[:2], encoder, latents.LatentCache(tmp_path / "cache"))
    assert encoder.seen == ["the supplier cancelled", "camera confirmed"]
    assert (first.encoded, first.cached) == (2, 0)

    # A new cache over the same folder, as a rerun has: only the unseen event is encoded, and an
    # event whose content was seen under another id costs nothing either.
    repeat = latents.EventInput("e9", "life-2", T0, "turn", "camera confirmed")
    again = latents.encode_events([*events, repeat], encoder, latents.LatentCache(tmp_path / "cache"))
    assert encoder.seen[2:] == ["rent is due"]
    assert (again.encoded, again.cached) == (1, 3)
    np.testing.assert_array_equal(again.matrix[:2], first.matrix)
    np.testing.assert_array_equal(again.matrix[3], again.matrix[1])
    assert again.matrix.dtype == np.float32 and again.matrix.shape == (4, 16)

    # The same text in another context, said by another speaker, or read another way, is another latent.
    context = latents.EventInput("e1", "life-1", T0, "turn", "camera confirmed", context=(("user", "it was late"),))
    said = latents.EventInput("e1", "life-1", T0, "turn", "camera confirmed", role="assistant")
    latents.encode_events([context, said], encoder, latents.LatentCache(tmp_path / "cache"))
    other = Counting(latents.FakeEncoder(d=16, seed=1))
    latents.encode_events(events, other, latents.LatentCache(tmp_path / "cache"))
    assert encoder.seen[3:] == ["camera confirmed", "camera confirmed"] and len(other.seen) == 3
    assert sorted(path.name for path in (tmp_path / "cache").iterdir()) == ["life-1", "life-2"]  # each life's own


def test_fake_encoder_is_deterministic() -> None:
    items = [latents.EncodeItem(f"k{i}", "user", text) for i, text in enumerate(
        ["the supplier cancelled the camera", "the supplier delivered the camera", "rent is due friday", ""])]
    a, b = latents.FakeEncoder(d=32).encode(items), latents.FakeEncoder(d=32).encode(items)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4, 32) and a.dtype == np.float32
    assert latents.FakeEncoder(d=7).encode(items).shape == (4, 7)  # any D
    assert not np.array_equal(a, latents.FakeEncoder(d=32, seed=1).encode(items))
    assert len({row.tobytes() for row in a}) == 4  # every text its own latent, the empty one too
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    # Shared words pull latents together, so the core can learn from fakes before the real encoder runs.
    assert float(a[0] @ a[1]) > float(a[0] @ a[2])
    said = latents.FakeEncoder(d=32).encode([latents.EncodeItem("k", "assistant", items[0].text)])
    assert not np.array_equal(said[0], a[0])  # the same words from another speaker
    spec = latents.FakeEncoder(d=32).spec
    assert (spec.encoder, spec.d) == ("fake", 32)


class FakeVolumeCalls:
    """The Modal calls: an upload, a spawn, a wait (the container's work, done by
    ``latent_job.write_outputs`` with deterministic rows), a fetch, a cancel and a remove. Each call
    records the holds open on the ledger at that moment."""

    def __init__(self, ledger: Ledger, root: Path, *, wait_error: Exception | None = None,
                 remove_error: Exception | None = None) -> None:
        self.ledger, self.root, self.wait_error, self.remove_error = ledger, root, wait_error, remove_error
        self.log: list[tuple[str, tuple[str, ...]]] = []
        self.uploaded: dict[str, bytes] = {}
        self.spawned: list[tuple[str, dict[str, Any], float]] = []
        self.cancelled: list[str] = []
        self.marked: dict[str, str] = {}  # the call id a job's container left in its folder

    def _record(self, name: str) -> None:
        self.log.append((name, tuple(hold.job for hold in self.ledger.open_holds())))

    def upload(self, job: str, data: bytes) -> None:
        self._record("upload")
        self.uploaded[job] = data

    def spawn(self, job: str, spec_json: str, exec_s: float) -> str:
        self._record("spawn")
        self.spawned.append((job, json.loads(spec_json), exec_s))
        return "fc-latents1"

    def wait(self, call_id: str, timeout_s: float) -> Mapping[str, object]:
        self._record("wait")
        if self.wait_error is not None:
            raise self.wait_error
        job, spec, _ = self.spawned[-1]
        rows = [json.loads(line) for line in self.uploaded[job].decode().splitlines()]
        matrix = np.stack([np.full(spec["d"], float(len(row["text"])), dtype=np.float32) for row in rows])
        stats = [{"key": row["key"], "tokens": 5, "event_tokens": 3, "truncated": "cancelled" in row["text"]}
                 for row in rows]
        latent_job.write_outputs(self.root / job, spec, stats, matrix)
        return {"events": len(rows), "tokens": 5 * len(rows), "seconds": 12.0, "load_seconds": 40.0, "d": spec["d"]}

    def fetch(self, job: str, dest: Path) -> None:
        self._record("fetch")
        for name in (latent_job.MATRIX_FILE, latent_job.INDEX_FILE):
            (dest / name).write_bytes((self.root / job / name).read_bytes())

    def cancel(self, call_id: str) -> None:
        self._record("cancel")
        self.cancelled.append(call_id)

    def remove(self, job: str) -> None:
        self._record("remove")
        if self.remove_error is not None:
            raise self.remove_error
        self.uploaded.pop(job, None)  # the events' text leaves the volume
        self.marked.pop(job, None)

    def find_call(self, job: str) -> str | None:
        return self.marked.get(job)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 60.0
        return self.now


EVENT_ROWS: list[dict[str, Any]] = [
    {"event_id": "e1", "life_or_person": "life-1", "t": "2026-10-01T09:00:00+00:00", "kind": "turn",
     "text": "the supplier cancelled"},
    {"id": "e2", "person_id": "life-1", "t": "2026-10-02", "kind": "fact", "text": "camera confirmed",
     "role": "assistant", "context": [{"role": "user", "content": "the shoot is on friday"}]},
]


def _data(tmp_path: Path) -> tuple[dict[str, str], Ledger, Path]:
    """A sleep data folder: the environment naming it, its ledger, and a run's folder in it."""
    return ({DATA_DIR_ENV: str(tmp_path / "data")}, Ledger(tmp_path / "data" / "sleep" / "budget.jsonl"),
            tmp_path / "data" / "sleep" / latents.LATENTS_DIR / latents.RUNS_DIR / "run-1")


def test_encode_plan_holds_before_any_call(tmp_path: Path) -> None:
    events = _write_events(tmp_path / "events.jsonl", EVENT_ROWS)
    env, ledger, out = _data(tmp_path)
    calls = FakeVolumeCalls(ledger, tmp_path / "volume")
    lines = list[str]()
    argv = ["encode", "--events", str(events), "--out", str(out), "--revision", PIN]

    assert latents.main([*argv, "--plan"], out=lines.append, calls=calls, ledger=ledger, clock=Clock(),
                        environ=env) == 0
    text = "\n".join(lines)
    hold = latent_runner.hold_bound().amount
    assert hold == pytest.approx(latent_job.FUNCTION_TIMEOUT_S / 3600 * latent_runner.A100_USD_PER_HOUR)
    assert f"hold ${hold:.2f}" in text and "modal_usd" in text and "2 to encode" in text
    assert calls.log == [] and not ledger.path.exists() and not out.exists()  # touched nothing

    assert latents.main(argv, out=lines.append, calls=calls, ledger=ledger, clock=Clock(), environ=env) == 0
    names = [name for name, _ in calls.log]
    assert names == ["upload", "spawn", "wait", "fetch", "remove"]  # finished: nothing to cancel
    assert calls.uploaded == {} and latent_runner.CallRecords(ledger.path.parent / latent_runner.CALLS_DIR).items() == []
    job = calls.spawned[0][0]
    held = (f"{latent_runner.HOLD_KIND}:{job}:a1",)
    assert calls.log[0][1] == held  # the hold was open before the first call touched Modal
    assert ledger.open_holds() == ()
    spent = ledger.settlements()[held[0]]
    assert 0 < spent <= hold
    _, spec, exec_s = calls.spawned[0]
    assert spec["layers"] == [35] and spec["revision"] == PIN and 0 < exec_s < latent_job.FUNCTION_TIMEOUT_S
    header, loaded = latents.load_latents(out / "life-1")
    assert header["d"] == latents.HIDDEN and [item.event_id for item in loaded] == ["e1", "e2"]
    assert loaded[1].t == datetime(2026, 10, 2, tzinfo=timezone.utc) and loaded[1].kind == "fact"
    rows = [json.loads(line) for line in (out / "life-1" / latents.INDEX_FILE).read_text().splitlines()[1:]]
    assert [(row["role"], row["truncated"]) for row in rows] == [("user", True), ("assistant", False)]
    assert any("tokens" in line and "seconds" in line for line in lines)  # the job's own count, printed

    # A rerun pays nothing: every event is cached (with its cut flag), so nothing is held and nothing is called.
    before = len(calls.log)
    assert latents.main(argv, out=lines.append, calls=calls, ledger=ledger, clock=Clock(), environ=env) == 0
    assert len(calls.log) == before and len(ledger.settlements()) == 1
    assert (out / "life-1" / latents.INDEX_FILE).read_text().count('"truncated": true') == 1

    # Latents stay in the sleep data folder, where a delete finds them.
    with pytest.raises(latents.LatentError, match="sleep data folder"):
        latents.main(["encode", "--events", str(events), "--out", str(tmp_path / "elsewhere"), "--revision", PIN,
                      "--plan"], out=lines.append, calls=calls, ledger=ledger, environ=env)
    with pytest.raises(latents.LatentError, match=DATA_DIR_ENV):
        latents.main([*argv, "--plan"], out=lines.append, calls=calls, ledger=ledger, environ={})
    root = out.parent.parent
    assert latents.erase_paths(root, "life-1") == [root / latents.CACHE_DIR / "life-1", out / "life-1"]
    assert all(path.exists() for path in latents.erase_paths(root, "life-1"))


def test_killed_encode_rerun_settles_both_attempts(tmp_path: Path) -> None:
    events = _write_events(tmp_path / "events.jsonl", EVENT_ROWS)
    env, ledger, out = _data(tmp_path)
    calls = FakeVolumeCalls(ledger, tmp_path / "volume")
    argv = ["encode", "--events", str(events), "--out", str(out), "--revision", PIN]
    spec = latents.glimmer_spec(PIN)
    items = latents.items_for(latents.read_events(events), spec)
    job = latent_runner.job_name(spec.as_json(), [item.key for item in items])
    # What a run killed hard after its spawn leaves: its hold open, its record, its events on the volume.
    killed = hold(ledger, f"{latent_runner.HOLD_KIND}:{job}", (latent_runner.POOL,), latent_runner.hold_bound())
    records = latent_runner.CallRecords(ledger.path.parent / latent_runner.CALLS_DIR)
    records.add(job, "fc-killed", ["life-1"])
    calls.uploaded[job] = b"the dead run's events"

    with open(lock_path(ledger, tmp_path), "a+b") as other:  # another MIND command is live on this host
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(MindBusy):
            latents.main(argv, out=print, calls=calls, ledger=ledger, clock=Clock(), environ=env)
        fcntl.flock(other, fcntl.LOCK_UN)
    assert calls.log == [] and len(ledger.open_holds()) == 1  # nothing swept, nothing held

    assert latents.main(argv, out=print, calls=calls, ledger=ledger, clock=Clock(), environ=env) == 0
    assert calls.cancelled == ["fc-killed"] and calls.uploaded == {}  # the dead call stopped, its text gone
    names = [name for name, _ in calls.log]
    assert names == ["cancel", "remove", "upload", "spawn", "wait", "fetch", "remove"]
    settled = ledger.settlements()
    assert settled[killed.job] == pytest.approx(latent_runner.hold_bound().amount)  # the dead run's worst case
    assert 0 < settled[f"{latent_runner.HOLD_KIND}:{job}:a2"] < settled[killed.job]  # the rerun's own attempt
    assert ledger.open_holds() == () and records.items() == []
    assert f"{latent_runner.HOLD_KIND}:" in MIND_KINDS  # MIND's own sweep settles them too


def test_a_call_killed_before_its_id_was_recorded_is_found_by_its_job(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    calls = FakeVolumeCalls(ledger, tmp_path / "volume")
    records = latent_runner.CallRecords(tmp_path / latent_runner.CALLS_DIR)
    job = "j-" + "a" * 16
    records.add(job, None, ["life-1"])  # the intent, recorded before the upload; killed after the spawn
    calls.uploaded[job] = b"events"
    calls.marked[job] = "fc-orphan"  # what the container wrote first (latent_job.mark_call)
    hold(ledger, f"{latent_runner.HOLD_KIND}:{job}", (latent_runner.POOL,), latent_runner.hold_bound())
    swept = latent_runner.sweep(records, calls, ledger)
    assert calls.cancelled == ["fc-orphan"] and calls.uploaded == {} and records.items() == []
    assert (swept.folders_removed, swept.holds_settled) == (1, 1) and ledger.open_holds() == ()
    # A call whose container hasn't started has no id to find: its folder goes, and it stops at its first read.
    records.add(job, None, ["life-1"])
    assert latent_runner.sweep(records, calls, ledger).folders_removed == 1 and calls.cancelled == ["fc-orphan"]


def test_the_container_marks_its_call_in_the_jobs_folder(tmp_path: Path) -> None:
    latent_job.mark_call(tmp_path / "j", "fc-01ABcd")
    assert latent_job.read_call((tmp_path / "j" / latent_job.CALL_FILE).read_bytes()) == "fc-01ABcd"
    assert latent_job.read_call(b"not json") is None and latent_job.read_call(b'{"call": "rm -rf"}') is None
    with pytest.raises(latent_job.LatentSpecError):
        latent_job.mark_call(tmp_path / "j", None)


def test_a_failed_remove_keeps_the_record_and_the_next_sweep_clears_it(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    calls = FakeVolumeCalls(ledger, tmp_path / "volume", remove_error=OSError("volume busy"))
    records = latent_runner.CallRecords(tmp_path / latent_runner.CALLS_DIR)
    encoder = latents.ModalEncoder(latents.glimmer_spec(PIN), calls=calls, ledger=ledger, clock=Clock(),
                                   scratch=tmp_path / "scratch", log=lambda line: None, lives=["life-1"])
    with pytest.raises(latent_runner.LatentJobError, match="volume"):
        encoder.encode([latents.EncodeItem("k1", "user", "hello")])
    assert ledger.open_holds() == () and [record.lives for record in records.items()] == [("life-1",)]
    assert latent_runner.stop_person_jobs(records, "life-2", calls=calls) == 0  # nobody else's
    calls.remove_error = None
    assert latent_runner.stop_person_jobs(records, "life-1", calls=calls) == 1  # a delete clears it
    assert records.items() == [] and calls.uploaded == {}


def test_a_call_that_runs_out_its_time_is_cancelled_and_settled(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    calls = FakeVolumeCalls(ledger, tmp_path / "volume", wait_error=TimeoutError("too slow"))
    encoder = latents.ModalEncoder(latents.glimmer_spec(PIN), calls=calls, ledger=ledger, clock=Clock(),
                                   scratch=tmp_path / "scratch", log=lambda line: None)
    with pytest.raises(latent_runner.LatentJobError):
        encoder.encode([latents.EncodeItem("k1", "user", "hello")])
    assert [name for name, _ in calls.log] == ["upload", "spawn", "wait", "cancel", "remove"]  # its text never stays
    assert calls.uploaded == {} and latent_runner.CallRecords(tmp_path / latent_runner.CALLS_DIR).items() == []
    assert ledger.open_holds() == ()
    assert 0 < sum(ledger.settlements().values()) <= latent_runner.hold_bound().amount


def test_fetched_latents_must_answer_the_keys_sent(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    calls = FakeVolumeCalls(ledger, tmp_path / "volume")
    original = calls.wait

    def wrong_order(call_id: str, timeout_s: float) -> Mapping[str, object]:
        job, spec, _ = calls.spawned[-1]
        rows = [json.loads(line) for line in calls.uploaded[job].decode().splitlines()]
        calls.uploaded[job] = b"".join(json.dumps(row).encode() + b"\n" for row in reversed(rows))
        return original(call_id, timeout_s)

    calls.wait = wrong_order  # type: ignore[method-assign]
    encoder = latents.ModalEncoder(latents.glimmer_spec(PIN), calls=calls, ledger=ledger, clock=Clock(),
                                   scratch=tmp_path / "scratch", log=lambda line: None)
    with pytest.raises(latent_runner.LatentJobError, match="keys"):
        encoder.encode([latents.EncodeItem("k1", "user", "hello"), latents.EncodeItem("k2", "user", "there")])
    assert ledger.open_holds() == ()


def test_events_file_is_checked_before_anything_is_planned(tmp_path: Path) -> None:
    bad = [
        [{"event_id": "e1", "life_or_person": "l", "t": "2026-10-01", "kind": "turn"}],  # no text
        [{"event_id": "e1", "life_or_person": "l", "t": "not a time", "kind": "turn", "text": "x"}],
        [{"event_id": "e1", "life_or_person": "l", "t": "2026-10-01", "kind": "turn", "text": "x"}] * 2,
        [{"event_id": "e1", "life_or_person": "../l", "t": "2026-10-01", "kind": "turn", "text": "x"}],  # a path
        [{"event_id": "e1", "life_or_person": "l", "t": "2026-10-01", "kind": "turn", "text": "x", "role": "tool"}],
        [{"event_id": "e1", "life_or_person": "l", "t": "2026-10-01", "kind": "turn", "text": "x", "context": "y"}],
    ]
    for number, rows in enumerate(bad):
        with pytest.raises(latents.LatentError):
            latents.read_events(_write_events(tmp_path / f"bad{number}.jsonl", rows))


# ── The container's side (train/latent_job.py): pure helpers, no torch ─────────────────────────────


def test_job_batches_pack_within_the_token_budget() -> None:
    lengths = [5, 300, 40, 300, 1, 2000, 299]
    batches = latent_job.batches(lengths, batch_tokens=1000)
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))
    for batch in batches:
        width = max(lengths[i] for i in batch)
        assert len(batch) * width <= 1000 or len(batch) == 1
    assert [5] in batches  # a sequence longer than the budget goes alone


def test_job_sequence_reads_the_event_as_its_speakers_turn_after_the_end_of_its_context() -> None:
    ids, start, stop, truncated = latent_job.sequence(
        bos=1, context=list(range(10, 21)), header=[7, 8], content=list(range(30, 36)), end=[9], max_context=4,
        max_event=3)
    assert ids == [1, 17, 18, 19, 20, 7, 8, 30, 31, 32, 9] and ids[start:stop] == [30, 31, 32] and truncated
    ids, start, stop, truncated = latent_job.sequence(
        bos=1, context=[], header=[7], content=[30, 31], end=[9], max_context=4, max_event=3)
    assert ids == [1, 7, 30, 31, 9] and ids[start:stop] == [30, 31] and not truncated
    assert set(latent_job.HEADERS) == set(latent_job.ROLES) == set(latents.ROLES)
    assert latent_job.HEADERS["assistant"] == "<|start|>assistant to=user<|message|>"  # as her replies are written


def test_job_pool_weights_cover_only_the_events_content_tokens() -> None:
    weights = latent_job.pool_weights(starts=[1, 2], stops=[3, 5], width=6)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(weights[0], [0, 0.5, 0.5, 0, 0, 0])
    np.testing.assert_allclose(weights[1], [0, 0, 1 / 3, 1 / 3, 1 / 3, 0])
    with pytest.raises(latent_job.LatentSpecError):
        latent_job.pool_weights(starts=[0], stops=[2], width=6)  # never the BOS


def test_job_refuses_a_spec_it_cant_read(tmp_path: Path) -> None:
    good = latents.glimmer_spec(PIN, layers=(3, 35)).as_json()
    assert latent_job.check_spec(good, PIN)["layers"] == [3, 35]
    for change, match in (({"revision": "d" * 40}, "pinned"), ({"layers": [52]}, "layer"),
                          ({"layers": [35, 3]}, "layer"), ({"d": 6656}, "d"), ({"encoder": "fake"}, "encoder"),
                          ({"pooling": "last"}, "pooling")):
        with pytest.raises(latent_job.LatentSpecError, match=match):
            latent_job.check_spec(json.dumps({**json.loads(good), **change}), PIN)


def test_job_outputs_round_trip(tmp_path: Path) -> None:
    spec = json.loads(latents.glimmer_spec(PIN).as_json())
    matrix = np.arange(2 * spec["d"], dtype=np.float32).reshape(2, spec["d"])
    stats = [{"key": "a", "tokens": 4, "event_tokens": 2, "truncated": False},
             {"key": "b", "tokens": 9, "event_tokens": 9, "truncated": True}]
    latent_job.write_outputs(tmp_path / "j", spec, stats, matrix)
    header, rows, read = latent_job.read_outputs(tmp_path / "j")
    assert header["d"] == spec["d"] and header["count"] == 2 and [row["key"] for row in rows] == ["a", "b"]
    np.testing.assert_array_equal(read, matrix)


# ── The training app declares the encoder (a fake ``modal``) ────────────────────────────────────────


def test_the_app_encodes_on_one_a100_with_weights_read_only_and_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys
    import types

    declared: dict[str, dict[str, Any]] = {}
    files: list[tuple[str, str]] = []

    class Image:
        def __getattr__(self, name: str) -> Any:
            def call(*args: Any, **kwargs: Any) -> Image:
                if name == "add_local_file":
                    files.append((str(args[0]), str(args[1])))
                return self
            return call

    class Volume:
        def __init__(self, name: str, ro: bool = False) -> None:
            self.name, self.ro = name, ro

        def read_only(self) -> Volume:
            return Volume(self.name, True)

    def function(**options: Any) -> Any:
        def wrap(fn: Any) -> Any:
            declared[fn.__name__] = options
            return fn
        return wrap

    fake = types.SimpleNamespace(
        Image=types.SimpleNamespace(debian_slim=lambda **kw: Image()),
        App=lambda *a, **kw: types.SimpleNamespace(function=function),
        Volume=types.SimpleNamespace(from_name=lambda name, **kw: Volume(name)),
        Secret=types.SimpleNamespace(from_name=lambda name: pytest.fail(f"the training app names a secret: {name}")),
    )
    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name, *a: fake if name == "modal" else real(name, *a))
    monkeypatch.delitem(sys.modules, "anjo.sleep.train.modal_app", raising=False)
    monkeypatch.setenv(config.REVISION_ENV, PIN)
    importlib.import_module("anjo.sleep.train.modal_app")

    options = declared[latent_job.FUNCTION]
    assert options["gpu"] == "A100-80GB" and options["timeout"] == latent_job.FUNCTION_TIMEOUT_S
    assert options["secrets"] == [] and options["scaledown_window"] == latent_job.SCALEDOWN_S
    volumes = options["volumes"]
    assert set(volumes) == {config.WEIGHTS_DIR, latent_job.VOLUME_DIR}
    assert volumes[config.WEIGHTS_DIR].ro and volumes[config.WEIGHTS_DIR].name == config.WEIGHTS_VOLUME
    assert volumes[latent_job.VOLUME_DIR].name == latent_job.VOLUME and not volumes[latent_job.VOLUME_DIR].ro
    assert ("train" in declared) and any(remote.endswith("/train/latent_job.py") for _, remote in files)
    rows = declared[latent_job.EMBED_FUNCTION]  # soft tokens' embedding rows: no GPU, weights read-only
    assert "gpu" not in rows and rows["secrets"] == [] and rows["timeout"] == latent_job.EMBED_TIMEOUT_S
    assert set(rows["volumes"]) == {config.WEIGHTS_DIR} and rows["volumes"][config.WEIGHTS_DIR].ro


def test_embedding_rows_are_read_by_offset_from_the_safetensors_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(latent_job, "VOCAB", 5)
    table: np.ndarray = np.arange(5 * latent_job.HIDDEN, dtype="<u2").reshape(5, latent_job.HIDDEN)  # bf16 bits
    key = latent_job.EMBED_KEYS[0]
    header = json.dumps({"other": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
                         key: {"dtype": "BF16", "shape": [5, latent_job.HIDDEN],
                               "data_offsets": [4, 4 + table.nbytes]}}).encode()
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"\0" * 4 + table.tobytes())
    (tmp_path / latent_job.SAFETENSORS_INDEX).write_text(
        json.dumps({"weight_map": {key: "model-00002-of-00002.safetensors"}}), encoding="utf-8")
    data = latent_job.embedding_rows(tmp_path, [3, 0, 3])
    assert np.array_equal(np.frombuffer(data, dtype="<u2").reshape(3, -1), table[[3, 0, 3]])
    for bad in ([], [5], [-1], [True]):
        with pytest.raises(latent_job.LatentSpecError, match="ids"):
            latent_job.embedding_rows(tmp_path, bad)
    with pytest.raises(latent_job.LatentSpecError, match="no input embedding"):
        latent_job.embedding_rows(tmp_path / "missing", [0])
