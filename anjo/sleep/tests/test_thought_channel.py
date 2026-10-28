"""The thought channel: MIND's state of one person as K soft tokens Glimmer reads (``mind.core.projector``),
trained with the person's LoRA in the night's own job (``train.channel_train``), its rows carrying the
core state they were made under (``night.core_snapshot``), pretrained across generated lives
(``mind.core.reader_data``, ``reader_targets``) and evaluated against zero, shuffled and pasted-text
arms (``mind.core.reader_eval``). Fakes only: no torch, no network, no paid call."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.mind.core import projector, reader_data, reader_eval, reader_targets
from anjo.sleep.mind.core.projector import Projector, Reading
from anjo.sleep.mind.core.state import CoreState
from anjo.sleep.mind.core.types import Predictions, SlotPrediction
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night import core_snapshot
from anjo.sleep.serving import soft_tokens
from anjo.sleep.serving.soft_tokens import SoftTokens
from anjo.sleep.tools import thought_channel, thought_channel_eval
from anjo.sleep.train import channel_train, lora

D, SLOTS, K = 8, 6, 4


def _state(seed: int = 0, *, person: str = "life-1") -> CoreState:
    rng = np.random.default_rng(seed)
    cites = tuple(((f"e{seed}-{k}", 0.9),) if k < 4 else () for k in range(SLOTS))
    return CoreState(person=person, model_id="core-abc", slots=rng.standard_normal((SLOTS, D)),
                     usage=np.full(SLOTS, 0.5), opened=np.zeros(SLOTS), memory=rng.standard_normal((D, D)) * 0.1,
                     momentum=np.zeros((D, D)), glob=rng.standard_normal(D), now=3.0, observed=5, cites=cites)


def _predictions(state: CoreState, choices: tuple[float, ...], speak: float = 0.8) -> Predictions:
    slots = tuple(SlotPrediction(slot=k, usage=0.5, risk=0.1 * k, hazard=(0.1,) * 7, missing=0.2, self_error=0.05,
                                 choice=choices[k], cites=state.cited(k)) for k in range(SLOTS))
    return Predictions(speak=speak, none=0.1, slots=slots)


def _anchors(k: int = K) -> Any:
    return soft_tokens.normalize(np.random.default_rng(7).standard_normal((k + 1, soft_tokens.HIDDEN)))


def _rms(rows: Any) -> Any:
    return np.sqrt((rows.astype(np.float64) ** 2).mean(axis=1))


# ── The projector ─────────────────────────────────────────────────────────────


def test_projector_outputs_post_norm_vectors_of_k() -> None:
    state = _state()
    reading = projector.read(state, _predictions(state, (0.1, 0.4, 0.2, 0.3, 0.9, 0.0)), k=K)
    # Slot 4 has the top choice but cites nothing: only citing slots are routed, by the core's own choice.
    assert reading.routed == (1, 3, 2)
    assert reading.gates[0] == pytest.approx(1.0) and reading.gates[2] >= projector.GATE_FLOOR
    proj = Projector.fresh(anchor_rows=_anchors(), k=K, d=D, width=16, seed=0)

    vectors = proj.project(reading)

    assert vectors.shape == (K, soft_tokens.HIDDEN) and vectors.dtype == np.float32
    assert np.allclose(_rms(vectors), 1.0, atol=1e-3)
    assert SoftTokens(vectors).k == K
    empty = proj.project(Reading.empty(K, D))  # the zero-tokens arm: same K, post-norm, no state
    assert empty.shape == (K, soft_tokens.HIDDEN) and np.allclose(_rms(empty), 1.0, atol=1e-3)
    other = _state(1)
    assert not np.allclose(proj.project(projector.read(other, _predictions(other, (0.1, 0.4, 0.2, 0.3, 0.9, 0.0)), k=K)),
                           vectors)
    again = Projector.load(proj.save())
    assert again.id == proj.id and np.array_equal(again.project(reading), vectors)
    # The projector the GPU trains reads the reading a training row carries, unchanged.
    assert np.array_equal(Reading.from_json(json.loads(json.dumps(reading.to_json()))).slot_x, reading.slot_x)
    assert channel_train.HIDDEN == soft_tokens.HIDDEN and channel_train.RMS_EPS == soft_tokens.RMS_EPS


def test_snapshot_is_the_row_field_and_the_zero_arm_keeps_the_state_id() -> None:
    state = _state()
    preds = _predictions(state, (0.1, 0.4, 0.2, 0.3, 0.0, 0.0))
    snap = core_snapshot.snapshot(state, preds, arm="state", k=K)
    zero = core_snapshot.snapshot(state, preds, arm="zero", k=K)

    back = core_snapshot.CoreSnapshot.from_json(json.loads(json.dumps(snap.to_json())))

    assert back.state_sha == snap.state_sha == zero.state_sha and back.core_model == "core-abc"
    assert np.array_equal(back.reading.slot_x, snap.reading.slot_x)
    assert zero.arm == "zero" and not zero.reading.slot_x.any() and zero.reading.speak == 0.0
    assert core_snapshot.channel_arm({}, synthetic=False) is None
    assert core_snapshot.channel_arm({core_snapshot.CHANNEL_ENV: "state"}, synthetic=False) == "state"
    assert core_snapshot.channel_arm({core_snapshot.CHANNEL_ENV: "zero"}, synthetic=True) == "zero"
    with pytest.raises(ValueError):  # the ablation never runs on a real person's night
        core_snapshot.channel_arm({core_snapshot.CHANNEL_ENV: "zero"}, synthetic=False)
    with pytest.raises(ValueError):
        core_snapshot.channel_arm({core_snapshot.CHANNEL_ENV: "loud"}, synthetic=True)


# ── Pretraining rows and their targets ────────────────────────────────────────


def _t(day: int, hour: int) -> str:
    return datetime(2026, 10, 1 + day, hour, tzinfo=timezone.utc).isoformat()


def _event(life: str, n: int, day: int, text: str) -> dict[str, Any]:
    return {"event_id": f"{life}-e{n}", "life_or_person": life, "seq": n, "t": _t(day, 8 + n % 8),
            "day": date(2026, 10, 1 + day).isoformat(), "kind": "lesson", "text": text}


LIFE = [
    _event("life-1", 0, 0, "Started prepping the night shoot at the harbour."),
    _event("life-1", 1, 0, "Rented the camera from Lumen Supply for the 14th."),
    _event("life-1", 2, 1, "Lumen Supply cancelled the camera order, rebooked."),
    _event("life-1", 3, 1, "Lunch was good."),
    _event("life-1", 4, 2, "Lumen Supply cancelled again this morning."),
    _event("life-1", 5, 2, "Tried a new grip setup."),
    _event("life-1", 6, 3, "Scouting the pier tonight."),
    _event("life-1", 7, 4, "Colour grading the test footage."),
    _event("life-1", 8, 5, "Can you help me plan the shoot day?"),
    _event("life-1", 9, 6, "Wrapped the shoot, the footage looks great."),
]
ITEM: dict[str, Any] = {"item_id": "life-1-i1", "family": "gap", "actionable": True, "template": "t1", "topic": "the camera",
        "resolution": "book a backup camera", "origin_ids": ["life-1-e1", "life-1-e2", "life-1-e4"],
        "disambiguating_ids": [], "knowable_event_id": "life-1-e4", "knowable_day": "2026-10-03",
        "resolvable_event_id": "life-1-e4", "resolvable_day": "2026-10-03", "surface_day": "2026-10-06",
        "cutoff_day": "2026-10-05"}
DECOY: dict[str, Any] = {**ITEM, "item_id": "life-1-d1", "family": "decoy", "actionable": False, "topic": "the grip",
         "origin_ids": ["life-1-e5"], "knowable_event_id": "life-1-e5", "knowable_day": "2026-10-03"}
QUIET = [_event("life-2", n, n, text) for n, text in enumerate(
    ["Drafting the brochure.", "Printed the proofs.", "Met the printer.", "Folded samples.", "Shipped the box."])]
TEST_MARKER = "SEALED-TEST-ONLY resolution"


def _export(folder: Path, split: str, lives: dict[str, list[dict[str, Any]]], labels: dict[str, list[dict[str, Any]]],
            *, sealed: bool = False) -> Path:
    (folder / "events").mkdir(parents=True)
    (folder / "labels").mkdir()
    for life, events in lives.items():
        (folder / "events" / f"{life}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    for life, items in labels.items():  # a sealed export never has these; one here must still never be read
        (folder / "labels" / f"{life}.json").write_text(json.dumps({"life_id": life, "items": items}), encoding="utf-8")
    manifest = {"schema": "mind-core-events-v1", "split": split, "sealed": sealed, "lives": [
        {"life_id": life, "field": "film", "first_day": events[0]["day"], "last_day": events[-1]["day"],
         "events": len(events)} for life, events in lives.items()]}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def _snap(seed: int) -> core_snapshot.CoreSnapshot:
    state = _state(seed)
    return core_snapshot.snapshot(state, _predictions(state, (0.1, 0.4, 0.2, 0.3, 0.0, 0.0)), arm="state", k=K)


def test_joint_training_row_holds_state_message_target_and_negative(tmp_path: Path) -> None:
    export = reader_data.TrainingExport.open(
        _export(tmp_path / "train", "core-train", {"life-1": LIFE, "life-2": QUIET}, {"life-1": [ITEM, DECOY], "life-2": []}))
    moments = reader_data.all_moments(export, seed=0)
    plans = reader_data.plan_rows(moments, seed=0, drop_share=0.0)
    positive = next(p for p in plans if p.moment.open_items)
    targets = {m.key: reader_data.Targets(plain=f"plain {m.key}", raised=f"raise {m.key}" if m.open_items else None)
               for m in (p for ps in moments.values() for p in ps)}
    negatives = {p.negative for p in plans if p.negative is not None and p.negative != reader_data.EMPTY_KEY}
    readings = {key: _snap(n) for n, key in enumerate(sorted({p.moment.key for p in plans} | negatives))}

    row = reader_data.training_row(positive, readings, targets)

    # The moment is after the problem became knowable and before the person raised it.
    assert positive.moment.open_items == ("life-1-i1",) and positive.moment.cite_ids == tuple(ITEM["origin_ids"])
    assert "2026-10-03" <= positive.moment.day.isoformat() < "2026-10-06"
    assert positive.moment.event_id not in {*ITEM["origin_ids"], *DECOY["origin_ids"]}
    assert row["prompt"][-1] == {"role": "user", "content": positive.moment.message}
    assert row["completion"] == [{"role": "assistant", "content": f"raise {positive.moment.key}"}]
    assert row["core"] == readings[positive.moment.key].to_json()
    # The negative: the same message under another moment's state, where no problem is open, and its plain reply:
    # first the same thread once settled (the person raised it on the 6th), so the channel learns liveness, not recall.
    negative = next(m for ms in moments.values() for m in ms if m.key == positive.negative)
    assert negative.open_items == () and negative.life_id == "life-1" and negative.day == date(2026, 10, 7)
    assert positive.negative is not None
    assert row["contrast"] == {"core": readings[positive.negative].to_json(), "plain": f"plain {positive.moment.key}"}
    # A quiet moment (here before anything is knowable) earns the plain reply, with no contrast.
    decoy = next(p for p in plans if not p.moment.open_items and p.moment.life_id == "life-1")
    quiet_row = reader_data.training_row(decoy, readings | {decoy.moment.key: _snap(9)}, targets)
    assert "contrast" not in quiet_row and quiet_row["completion"][0]["content"].startswith("plain")
    # State dropout: some positives take the empty reading as their negative, so zero tokens teach silence.
    dropped = reader_data.plan_rows(moments, seed=0, drop_share=1.0)
    assert all(p.negative == reader_data.EMPTY_KEY for p in dropped if p.moment.open_items)


def test_target_writer_never_sees_test_labels(tmp_path: Path) -> None:
    test_dir = _export(tmp_path / "test", "core-test", {"life-9": QUIET},
                       {"life-9": [{**ITEM, "resolution": TEST_MARKER}]}, sealed=True)
    unsealed_test = _export(tmp_path / "test2", "core-test", {"life-9": QUIET},
                            {"life-9": [{**ITEM, "resolution": TEST_MARKER}]})
    asked: list[str] = []

    def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        asked.append(json.dumps(messages))
        body = json.loads(messages[-1]["content"])
        reply = {"plain": "Sounds good.", "raise": "Sounds good. Lumen Supply cancelled twice: book a backup camera."
                 if body["problems"] else None}
        return ChatResult(text=json.dumps(reply), tool_calls=(), model="spark-open", input_tokens=100,
                          output_tokens=30, finish_reason="stop")

    for folder in (test_dir, unsealed_test, tmp_path / "pairs"):
        with pytest.raises(reader_data.ReaderDataError):
            reader_data.TrainingExport.open(folder)
    assert asked == []

    export = reader_data.TrainingExport.open(
        _export(tmp_path / "train", "core-train", {"life-1": LIFE, "life-2": QUIET}, {"life-1": [ITEM, DECOY], "life-2": []}))
    moments = reader_data.all_moments(export, seed=0)
    written = reader_targets.TargetWriter(chat).write(export, moments)

    assert asked and all(TEST_MARKER not in prompt and "life-9" not in prompt for prompt in asked)
    raised = [written[m.key] for ms in moments.values() for m in ms if m.open_items]
    assert raised and all(t.raised and "Lumen Supply" in t.raised for t in raised)
    assert all(written[m.key].raised is None for ms in moments.values() for m in ms if not m.open_items)


# ── The evaluation ────────────────────────────────────────────────────────────


def test_reader_eval_arms_share_prompts_and_seeds() -> None:
    cases = [*reader_eval.life_cases("life-1", LIFE), *reader_eval.life_cases("life-2", QUIET)]
    notes = {case.case_id: ("Rented the camera from Lumen Supply for the 14th.",) for case in cases}

    prompts = reader_eval.arm_prompts(cases, notes, seed=0)

    by_case: dict[str, dict[str, reader_eval.ArmPrompt]] = {}
    for prompt in prompts:
        by_case.setdefault(prompt.case_id, {})[prompt.arm] = prompt
    assert len(by_case) == len(cases) and all(set(arms) == set(reader_eval.ARMS) for arms in by_case.values())
    for case in cases:
        arms = by_case[case.case_id]
        assert len({arms[a].seed for a in reader_eval.ARMS}) == 1
        assert arms["state"].messages == arms["zero"].messages == arms["shuffled"].messages
        assert arms["text"].messages[-1] == arms["state"].messages[-1]  # the same message, the notes in the system turn
        assert "Lumen Supply" in arms["text"].messages[0]["content"]
        assert "Lumen Supply" not in arms["state"].messages[0]["content"]
        assert arms["state"].reading == case.reading_key and arms["zero"].reading == reader_eval.EMPTY
        assert arms["text"].reading is None
        shuffled_life = arms["shuffled"].reading.split(":")[0] if arms["shuffled"].reading else ""
        assert shuffled_life and shuffled_life != case.life_id  # another life's state


def test_transplant_eval_changes_only_the_state() -> None:
    shared = [_event("p1", n, n, text) for n, text in enumerate(
        ["Booked the camera from Lumen Supply for the shoot.", "Picked locations."])]
    failed = [*shared, _event("p1", 2, 2, "Lumen Supply cancelled."), _event("p1", 3, 3, "Lumen Supply cancelled again."),
              _event("p1", 4, 4, "Can you help me plan tomorrow's shoot?")]
    kept = [*shared, _event("p1", 2, 2, "Lumen Supply confirmed."), _event("p1", 3, 3, "Lumen Supply delivered."),
            _event("p1", 4, 4, "Can you help me plan tomorrow's shoot?")]
    record = {"pair_id": "p1", "slot_ids": ["p1-e0"], "final_ids": ["p1-e4"], "last_day": "2026-10-05",
              "decisions": {"failed": "speak", "kept": "wait"}}

    a, b = reader_eval.pair_cases(record, failed, kept)
    others = reader_eval.life_cases("life-1", LIFE)  # the shuffled arm's stand-ins are other people's states
    prompts = reader_eval.arm_prompts([a, b, *others], {a.case_id: (), b.case_id: ()}, seed=3)

    assert (a.kind, b.kind) == ("failed", "kept") and a.expect == "raise" and b.expect == "wait"
    assert a.message == b.message == "Can you help me plan tomorrow's shoot?" and a.day == b.day
    assert a.reading_key != b.reading_key  # the one thing that differs: the core state
    for arm in ("state", "zero", "shuffled"):
        pa = next(p for p in prompts if p.case_id == a.case_id and p.arm == arm)
        pb = next(p for p in prompts if p.case_id == b.case_id and p.arm == arm)
        assert pa.messages == pb.messages and pa.seed == pb.seed
        assert (pa.reading != pb.reading) == (arm == "state")
    keywords = reader_eval.keywords(["Booked the camera from Lumen Supply for the shoot."], ["Picked locations."])
    replies = {a.case_id: "Lumen Supply let you down twice; book a backup camera.", b.case_id: "Sure, here's a plan."}
    assert reader_eval.flips([(a, b)], replies, keywords={("p1"): keywords}) == (1, 1)
    assert reader_eval.flips([(a, b)], {a.case_id: "Sure.", b.case_id: "Sure."}, keywords={"p1": keywords}) == (0, 1)


# ── Plan first, hold before any call ──────────────────────────────────────────


class _Calls:
    """A fake Modal training call that checks a hold is open before anything reaches it."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger, self.seen, self.name = ledger, list[str](), ""

    def spawn(self, job: Any) -> str:
        assert self.ledger.open_holds(), "a call before any hold"
        assert "channel" in json.loads(job.config_json)  # the night's own job, with the channel block
        self.seen.append("spawn")
        self.name = job.name
        return "fc-call1"

    def wait(self, call_id: str, timeout_s: float) -> dict[str, Any]:
        self.seen.append("wait")
        files = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"x"}
        return {"adapter": self.name, "receipt": {"steps": 1, "channel": {"val_state_margin": 0.4,
                                                                           "val_zero_margin": 0.1}},
                "files": files, "projector": b"npz"}

    def cancel(self, call_id: str) -> None:
        self.seen.append("cancel")


def test_plan_holds_before_any_call(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"id": "r1", "prompt": [{"role": "user", "content": "hi"}],
                                "completion": [{"role": "assistant", "content": "hello"}]}) + "\n", encoding="utf-8")
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    calls = _Calls(ledger)
    lines: list[str] = []
    args = ["train", "--rows", str(rows), "--out", str(tmp_path / "out"), "--revision", "a" * 40]

    assert thought_channel.main(args, out=lines.append, ledger=ledger, calls=calls, records=tmp_path / "calls") == 0

    assert calls.seen == [] and ledger.open_holds() == () and not ledger.settlements()
    assert any("hold $" in line and "modal_usd" in line for line in lines)

    assert thought_channel.main([*args, "--yes"], out=lines.append, ledger=ledger, calls=calls,
                                records=tmp_path / "calls") == 0

    assert calls.seen[:2] == ["spawn", "wait"] and ledger.open_holds() == ()
    assert calls.name == lora.adapter_name(thought_channel.SHARED_PERSON, 0, 0)
    assert (tmp_path / "out" / "channel" / channel_train.PROJECTOR_FILE).read_bytes() == b"npz"
    assert not (tmp_path / "out" / lora.DATASET_FILE).exists()  # the rows went up, and no copy stays
    assert any("held-out margin: state +0.400" in line for line in lines)

    sent: list[str] = []
    eval_lines: list[str] = []
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(json.dumps(c.to_json()) + "\n" for c in reader_eval.life_cases("life-1", LIFE)),
                     encoding="utf-8")
    eval_args = ["--cases", str(cases), "--readings", str(tmp_path / "none.json"), "--projector", str(tmp_path / "p.npz"),
                 "--adapter", calls.name, "--out", str(tmp_path / "eval"), "--judge-prices", "1.25,4.25"]
    environ = {"ANJO_SLEEP_PRICES": "0.5,1.5,2.5,3.0"}

    assert thought_channel_eval.main(eval_args, out=eval_lines.append, ledger=ledger, environ=environ,
                                     client=lambda: sent.append("client")) == 0  # type: ignore[arg-type, return-value]

    assert sent == [] and ledger.open_holds() == ()
    assert any("hold $" in line and "modal_usd" in line for line in eval_lines)
    assert any("meta_usd" in line for line in eval_lines)


def test_a_night_or_apd_sweep_leaves_a_live_pretraining_hold_alone(tmp_path: Path) -> None:
    from anjo.sleep.budget import Bound
    from anjo.sleep.mind import sweep as mind_sweep
    from anjo.sleep.night import sweep as night_sweep
    from anjo.sleep.train.runners import modal as modal_runner

    # Arrange: a killed pretraining's leftovers (its recorded call, its open hold), then a rerun
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"id": "r1", "prompt": [{"role": "user", "content": "hi"}],
                                "completion": [{"role": "assistant", "content": "hello"}]}) + "\n", encoding="utf-8")
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    records = tmp_path / "calls"
    left = lora.adapter_name(thought_channel.SHARED_PERSON, 0, 7)
    modal_runner.CallRecords(records).add(left, "fc-left1")
    killed = ledger.reserve("thought_channel:train:2026-10-28:shared-reader:a1", ("modal_usd",), Bound(2.0, 3600.0))
    assert killed is not None
    seen: dict[str, Any] = {}

    class Calls(_Calls):
        def wait(self, call_id: str, timeout_s: float) -> dict[str, Any]:
            # a night command, and an APD command's stale-hold settle, start while the call runs
            with night_sweep.swept_night(self.ledger, lambda: (0, 0)) as swept:
                seen["night"] = swept.holds_settled
            seen["mind"] = mind_sweep.settle_stale(self.ledger)
            seen["open"] = [held.job for held in self.ledger.open_holds()]
            return super().wait(call_id, timeout_s)

        def cancel(self, call_id: str) -> None:
            self.seen.append(f"cancel {call_id}")

    calls = Calls(ledger)
    args = ["train", "--rows", str(rows), "--out", str(tmp_path / "out"), "--revision", "a" * 40, "--yes"]

    # Act
    assert thought_channel.main(args, out=print, ledger=ledger, calls=calls, records=records) == 0

    # Assert: the rerun's sweep cancelled the killed call and settled its hold at the full bound, first;
    # the night's and MIND's sweeps settled nothing while the live hold stood
    assert calls.seen[0] == "cancel fc-left1" and calls.seen[1:3] == ["spawn", "wait"]
    assert ledger.settlements()[killed.job] == pytest.approx(2.0)
    assert seen["night"] == 0 and seen["mind"] == 0
    assert [job.split(":")[:2] for job in seen["open"]] == [["thought_channel", "train"]]
    assert ledger.open_holds() == () and modal_runner.CallRecords(records).items() == []
    assert not any(kind.startswith(f"{thought_channel.HOLD_KIND}:") or f"{thought_channel.HOLD_KIND}:".startswith(kind)
                   for kind in (*night_sweep.NIGHT_KINDS, *mind_sweep.MIND_KINDS))


class _Served:
    """A fake served Glimmer: every arm's request recorded, every reply the same plain answer."""

    rendering = "fake/medium"

    def __init__(self) -> None:
        self.sent: list[tuple[str | None, bool, int | None]] = []

    def once(self) -> _Served:
        return self

    def with_load_wait(self, seconds: float) -> _Served:
        return self

    def with_soft_tokens(self, rows: Any) -> _Served:
        return self

    def on(self, day: date) -> _Served:
        assert day == reader_data.READER_DAY  # the date the reader was trained on
        return self

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None,
                 max_tokens: int | None = None, soft: SoftTokens | None = None) -> ChatResult:
        self.sent.append((None if adapter is None else adapter.id, soft is not None, seed))
        return ChatResult("Sure, here's a plan for tomorrow.", (), "glimmer", 40, 9, "stop")


def test_live_eval_replies_first_then_scores_from_sealed_problems(tmp_path: Path) -> None:
    from anjo.sleep.mind.apd import backend as backends
    from anjo.sleep.mind.apd import core_splits, gen_batch, labels, pairs
    from anjo.sleep.mind.apd.store import DataDir
    from anjo.sleep.mind.spark_provider import FIREWORKS

    # Arrange: one core-test life and one pair, exported as the cases step reads them (fake writers)
    data = DataDir(tmp_path / "data")
    spec = next(s for s in core_splits.core_specs("core-test") if s.prevalence >= 2)
    made = gen_batch.generate_life(spec, backends.fake_backend(FIREWORKS))
    data.save_life(made.life, made.problems)
    data.save_pair(pairs.generate_pair(pairs.pair_specs()[0], backends.fake_backend(FIREWORKS)).pair)
    labels.export(data, "core-test", tmp_path / "test")
    labels.export_pairs(data, tmp_path / "pairs")
    assert not (tmp_path / "test" / "labels").exists()  # core-test is sealed: the cases see events only
    cases_path = tmp_path / "cases.jsonl"
    assert thought_channel.main(["cases", "--export", str(tmp_path / "test"), "--out", str(cases_path),
                                 "--pairs", str(tmp_path / "pairs")], out=print) == 0
    cases = [reader_eval.Case.from_json(json.loads(line)) for line in cases_path.read_text().splitlines()]
    readings = {}
    for n, case in enumerate(cases):  # a reading per case, as the readings step writes them
        state = _state(n, person=case.life_id)
        seen = _predictions(state, (0.1, 0.4, 0.2, 0.3, 0.9, 0.0))
        readings[case.reading_key] = (core_snapshot.snapshot(state, seen, arm="state", k=K),
                                      projector.read(state, seen, k=K).cites)
    reader_data.write_readings(tmp_path / "readings.jsonl", readings)
    (tmp_path / "p.npz").write_bytes(Projector.fresh(anchor_rows=_anchors(), k=K, d=D, width=16).save())
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    served, asked = _Served(), []
    out = tmp_path / "eval"

    def judge(prompt: str) -> tuple[str, int, int]:
        written = (out / "replies.jsonl").read_text().splitlines()
        assert len(written) == 4 * len(cases), "the judge ran before every reply was written"
        asked.append(prompt)
        return '{"raises": 0, "other": false}', 300, 12

    adapter = lora.adapter_name(thought_channel.SHARED_PERSON, 0, 0)
    args = ["--cases", str(cases_path), "--readings", str(tmp_path / "readings.jsonl"), "--projector",
            str(tmp_path / "p.npz"), "--adapter", adapter, "--out", str(out), "--export", str(tmp_path / "test"),
            "--pairs", str(tmp_path / "pairs"), "--data", str(data.root), "--judge-prices", "1.25,4.25", "--yes"]
    environ = {"ANJO_SLEEP_PRICES": "0.5,1.5,2.5,3.0"}

    # Act
    assert thought_channel_eval.main(args, out=print, ledger=ledger, environ=environ, client=lambda: served,  # type: ignore[arg-type, return-value]
                                     wake=lambda: True, judge=judge, clock=iter(range(0, 10**6, 5)).__next__) == 0

    # Assert: four arms per case, the reader adapter on every arm but text, soft tokens on the reader's arms
    assert len(served.sent) == 4 * len(cases)
    assert sum(a == adapter and soft for a, soft, _ in served.sent) == 3 * len(cases)
    assert sum(a is None and not soft for a, soft, _ in served.sent) == len(cases)
    report = json.loads((out / "report.json").read_text())
    assert set(report["arms"]) == set(reader_eval.ARMS) and "carries nothing" in report["verdict"]
    assert report["arms"]["state"]["pairs"] == 1 and report["arms"]["state"]["raise_days"] >= 1
    assert report["arms"]["state"]["unrelated_pairs"] == 1 and report["arms"]["state"]["unrelated_stable"] == 1.0
    assert asked and all(len(p) <= thought_channel_eval.JUDGE_IN_TOKENS for p in asked)
    assert ledger.open_holds() == ()
    settled = ledger.settlements()
    assert any(job.startswith("thought_channel:eval_serve:") for job in settled)
    assert any(job.startswith("thought_channel:eval_judge:") for job in settled)

    # A rerun asks nothing again: the replies and the judge's answers are kept
    before = (len(served.sent), len(asked))
    assert thought_channel_eval.main(args, out=print, ledger=ledger, environ=environ, client=lambda: served,  # type: ignore[arg-type, return-value]
                                     wake=lambda: True, judge=judge) == 0
    assert (len(served.sent), len(asked)) == before
    shares = [arm["serving_usd_share"] for arm in json.loads((out / "report.json").read_text())["arms"].values()]
    assert all(share > 0 for share in shares)  # the first run's serving spend, still counted


def test_pretraining_steps_hold_before_the_first_target_call(tmp_path: Path) -> None:
    from anjo.sleep.mind.apd import backend as backends
    from anjo.sleep.mind.apd import core_splits, gen_batch, labels
    from anjo.sleep.mind.apd.store import DataDir
    from anjo.sleep.mind.spark_provider import FIREWORKS

    # Arrange: two core-train lives, exported with their labels (fake writers)
    data = DataDir(tmp_path / "data")
    for spec in [s for s in core_splits.core_specs("core-train") if s.prevalence >= 2][:2]:
        made = gen_batch.generate_life(spec, backends.fake_backend(FIREWORKS))
        data.save_life(made.life, made.problems)
    labels.export(data, "core-train", tmp_path / "train")
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    environ = {"ANJO_SLEEP_PRICES": "0.5,1.5,2.5,3.0"}
    calls: list[int] = []

    def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        assert ledger.open_holds(), "a Spark call before its hold"
        calls.append(max_tokens)
        body = json.loads(messages[-1]["content"])
        cited = " ".join(m["text"] for p in body["problems"] for m in p["earlier_messages"])
        reply = {"plain": "Sounds good, go for it.", "raise": f"Sounds good. Earlier you said: {cited}" if cited else None}
        return ChatResult(json.dumps(reply), (), "spark-open", 500, 80, "stop")

    plans = tmp_path / "work" / "plans.jsonl"
    targets = tmp_path / "work" / "targets.jsonl"
    step = ["targets", "--export", str(tmp_path / "train"), "--plans", str(plans), "--out", str(targets)]

    # Act and assert: the moments, then the targets' plan (no call, no hold), then the calls
    assert thought_channel.main(["moments", "--export", str(tmp_path / "train"), "--out", str(plans.parent)],
                                out=print) == 0
    lines: list[str] = []
    assert thought_channel.main(step, out=lines.append, ledger=ledger, chat=chat, environ=environ) == 0
    assert calls == [] and ledger.open_holds() == () and any("hold $" in line for line in lines)
    assert thought_channel.main([*step, "--yes"], out=print, ledger=ledger, chat=chat, environ=environ) == 0
    planned = reader_data.read_plans(plans)
    assert len(calls) == len(planned) and ledger.open_holds() == ()
    assert set(reader_data.read_targets(targets)) == {p.moment.key for p in planned}
    assert thought_channel.main([*step, "--yes"], out=print, ledger=ledger, chat=chat, environ=environ) == 0
    assert len(calls) == len(planned)  # nothing asked twice

    # The rows: each with the state it was made under, a raise row with its contrast state
    keys = {p.moment.key for p in planned} | {p.negative for p in planned if p.negative is not None
                                              and p.negative != reader_data.EMPTY_KEY}
    readings: dict[str, tuple[core_snapshot.CoreSnapshot, tuple[tuple[str, ...], ...]]] = {}
    for n, key in enumerate(sorted(keys)):
        state = _state(n, person=key.split(":")[0])
        seen = _predictions(state, (0.1, 0.4, 0.2, 0.3, 0.9, 0.0))
        readings[key] = (core_snapshot.snapshot(state, seen, arm="state", k=K), ())
    reader_data.write_readings(tmp_path / "work" / "readings.jsonl", readings)
    rows = tmp_path / "work" / "rows.jsonl"
    assert thought_channel.main(["rows", "--plans", str(plans), "--readings", str(tmp_path / "work" / "readings.jsonl"),
                                 "--targets", str(targets), "--out", str(rows)], out=print) == 0
    found = [json.loads(line) for line in rows.read_text().splitlines()]
    assert len(found) == len(planned) and all(row["core"]["arm"] == "state" for row in found)
    raised = [row for row in found if "contrast" in row]
    assert raised and all("Earlier you said" in row["completion"][0]["content"] for row in raised)


def test_the_nights_job_trains_the_projector_with_the_lora_as_one_unit(tmp_path: Path) -> None:
    from anjo.sleep.serving import config, framing
    from anjo.sleep.train.dataset import Dataset, Example

    class Runner:
        def __init__(self) -> None:
            self.inputs: dict[str, object] = {}

        def run(self, job: str, inputs: Any, *, timeout_s: float) -> dict[str, object]:
            self.inputs = dict(inputs)
            rows = [json.loads(line) for line in Path(str(inputs["dataset_path"])).read_text().splitlines()]
            assert all("core" in row for row in rows)  # each row carries the state it was made under
            out = Path(str(inputs["workdir"])) / "out"
            out.mkdir(parents=True, exist_ok=True)
            (out / "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "base_model_name_or_path": config.MODEL, "r": 16,
                framing.RENDERING_KEY: inputs["config"]["rendering"]}))
            (out / "adapter_model.safetensors").write_bytes(b"weights")
            projector_file = Path(str(inputs["workdir"])) / "channel" / channel_train.PROJECTOR_FILE
            projector_file.parent.mkdir(parents=True, exist_ok=True)
            projector_file.write_bytes(b"trained")
            return {"adapter_dir": str(out), "steps": 9, "epochs": 1, "train_minutes": 4.0, "gpu_minutes": 11.0,
                    "compute_units": 0.0, "examples": 2, "dropped_truncated": 0, "gpu": "A100-80GB",
                    "projector_path": str(projector_file), "channel": {"val_state_margin": 0.3}}

    state = _state()
    snap = core_snapshot.snapshot(state, _predictions(state, (0.1, 0.4, 0.2, 0.3, 0.9, 0.0)), arm="state", k=K)
    core = json.dumps(snap.to_json())
    data = Dataset(examples=tuple(Example(id=f"r{n}", prompt=(("user", "hi"),), completion="hey", context_dropped=True,
                                          tokens=10, new=True, core=core) for n in range(2)), tokens=20, dropped={})
    shared = tmp_path / "shared.projector.npz"
    shared.write_bytes(b"shared")
    block = channel_train.block(k=K, d=D, width=16, arm="state", exec_s=3600.0)
    runner = Runner()
    name = lora.adapter_name("p1", 1, 0)

    _, receipt = lora.train_lora_with_receipt(
        None, data, lora.LoraCfg(), runner, person_id="p1", night=date(2026, 11, 1), name=name,
        workdir=tmp_path / "work", adapter_cache=tmp_path / "cache", publish=lambda f, n: None, revision="a" * 40,
        channel=block, prev_projector=shared)

    assert runner.inputs["prev_projector"] == str(shared)  # warm from the shared pretraining's projector
    assert runner.inputs["config"]["channel"] == block  # type: ignore[index]
    assert Path(receipt.projector) == tmp_path / "cache" / f"{name}{lora.PROJECTOR_SUFFIX}"
    assert Path(receipt.projector).read_bytes() == b"trained"  # beside the adapter, never inside its folder
    assert json.loads(receipt.channel_receipt) == {"val_state_margin": 0.3}
    with pytest.raises(lora.TrainError):
        lora.train_lora_with_receipt(None, data, lora.LoraCfg(), runner, person_id="p1", night=date(2026, 11, 1),
                                     name=lora.adapter_name("p1", 1, 1), workdir=tmp_path / "w2",
                                     adapter_cache=tmp_path / "cache", publish=lambda f, n: None, revision="a" * 40,
                                     prev_projector=shared)
