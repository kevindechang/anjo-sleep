# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The adapter-strength read (``tools.adapter_scale``) and the scaled copy (``serving.scale``).

Fakes only: no model, judge, GPU or network call. The adapter files are real safetensors containers
written by the test, so the scaling is checked on bytes rather than on a mock.
"""
from __future__ import annotations

import json
import struct
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from anjo.sleep.budget import Bound, Ledger
from anjo.sleep.checkpoints import CheckpointRegistry
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.sweep import _settle_stale as sweep_stale
from anjo.sleep.probes.arms import ArmDeps, reply_seed
from anjo.sleep.probes.morning import Morning
from anjo.sleep.probes.score import Verdict
from anjo.sleep.serving import config
from anjo.sleep.serving import scale
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.tests.sleep_fakes import FakeContexts, FakeLogprobs, FakeModel, StoreSpy, probe
from anjo.sleep.tools import adapter_scale as tool
from anjo.sleep.types import CheckpointRef

USER = "syn-scale"
NIGHT = date(2026, 10, 24)
NAME = "p-abc12345-n1"
BASE = "base_model.model.model.layers.0.self_attn.q_proj"
PRICES = Prices(spark_in_per_m=1.25, spark_out_per_m=4.25, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.96)


def _safetensors(tensors: dict[str, np.ndarray], dtype: str = "F32") -> bytes:
    header: dict[str, dict[str, Any]] = {}
    blob = b""
    for name, values in tensors.items():
        if dtype == "BF16":
            raw = scale._f32_to_bf16(values.astype(np.float32)).tobytes()
        else:
            raw = values.astype("<f4" if dtype == "F32" else "<f2").tobytes()
        header[name] = {"dtype": dtype, "shape": list(values.shape),
                        "data_offsets": [len(blob), len(blob) + len(raw)]}
        blob += raw
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)  # safetensors pads the header to 8 bytes
    return struct.pack("<Q", len(encoded)) + encoded + blob


def _write_adapter(folder: Path, *, name: str = NAME, dtype: str = "F32") -> tuple[np.ndarray, np.ndarray]:
    a = np.array([[0.5, -1.0], [2.0, 0.25]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [-3.0, 0.5]], dtype=np.float32)
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": 2, "lora_alpha": 4, "lora_dropout": 0.05,
        "base_model_name_or_path": config.MODEL, "target_modules": ["q_proj"]}), encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(_safetensors(
        {f"{BASE}.lora_A.weight": a, f"{BASE}.lora_B.weight": b}, dtype))
    return a, b


def _tensors(folder: Path) -> dict[str, np.ndarray]:
    raw = (folder / "adapter_model.safetensors").read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    meta = json.loads(raw[8:8 + size])
    data = raw[8 + size:]
    return {name: np.frombuffer(data[slice(*item["data_offsets"])], dtype="<f4")
            for name, item in meta.items() if item["dtype"] == "F32"}


def _bf16(folder: Path) -> np.ndarray:
    raw = (folder / "adapter_model.safetensors").read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    meta = json.loads(raw[8:size + 8])
    data = raw[8 + size:]
    item = meta[f"{BASE}.lora_B.weight"]
    return scale._bf16_to_f32(np.frombuffer(data[slice(*item["data_offsets"])], dtype="<u2"))


def test_s_one_is_a_byte_for_byte_copy_under_a_distinct_name(tmp_path: Path) -> None:
    source = tmp_path / "adapters" / NAME
    _write_adapter(source)
    target = scale.write_scaled(source, tmp_path / "adapters", 1.0)
    assert target.name != source.name and config.ADAPTER_NAME.fullmatch(target.name)
    for name in config.ADAPTER_FILES:
        assert (target / name).read_bytes() == (source / name).read_bytes()
    assert (source / "adapter_model.safetensors").is_file()  # the original is untouched


def test_s_zero_zeroes_every_b_matrix_and_leaves_a_alone(tmp_path: Path) -> None:
    folder = tmp_path / "adapters" / NAME
    a, b = _write_adapter(folder)
    target = scale.write_scaled(folder, tmp_path / "adapters", 0.0)
    found = _tensors(target)
    assert np.array_equal(found[f"{BASE}.lora_B.weight"].reshape(b.shape), np.zeros_like(b))
    assert np.array_equal(found[f"{BASE}.lora_A.weight"].reshape(a.shape), a)
    assert (target / "adapter_config.json").read_bytes() == (folder / "adapter_config.json").read_bytes()


def test_every_scale_gets_its_own_name_and_folder(tmp_path: Path) -> None:
    folder = tmp_path / "adapters" / NAME
    _write_adapter(folder)
    digest = scale.content_digest(folder)
    names = [scale.scaled_name(NAME, value, digest=digest) for value in (0.25, 0.5, 0.75)]
    assert len(set(names)) == 3 and NAME not in names
    assert all(config.ADAPTER_NAME.fullmatch(name) for name in names)
    assert names == [scale.scaled_name(NAME, value, digest=scale.content_digest(folder))
                     for value in (0.25, 0.5, 0.75)]
    assert all(name != scale.scaled_name(NAME, value) for name, value in zip(names, (0.25, 0.5, 0.75),
                                                                             strict=True))
    for name, value in zip(names, (0.25, 0.5, 0.75), strict=True):
        assert scale.write_scaled(folder, tmp_path / "adapters", value).name == name


def test_a_bfloat16_b_tensor_is_scaled_in_its_own_dtype(tmp_path: Path) -> None:
    folder = tmp_path / "adapters" / NAME
    a, b = _write_adapter(folder, dtype="BF16")
    target = scale.write_scaled(folder, tmp_path / "adapters", 0.5)
    assert np.allclose(_bf16(target).reshape(b.shape), b * 0.5, rtol=0, atol=1e-2)


def test_a_delta_stored_outside_lora_b_is_refused_rather_than_left_at_full_strength(tmp_path: Path) -> None:
    folder = tmp_path / "adapters" / NAME
    _write_adapter(folder)
    a = np.array([[0.5, -1.0], [2.0, 0.25]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [-3.0, 0.5]], dtype=np.float32)
    (folder / "adapter_model.safetensors").write_bytes(_safetensors({
        f"{BASE}.lora_A.weight": a, f"{BASE}.lora_B.weight": b,
        f"{BASE}.lora_magnitude_vector": np.array([1.0, 1.0], dtype=np.float32)}))
    with pytest.raises(scale.ScaleError, match="lora_magnitude_vector"):
        scale.write_scaled(folder, tmp_path / "adapters", 0.5)


class Judge:
    """Deterministic, never fails: arm 4's reply wins whichever order it is shown in."""

    def __init__(self) -> None:
        self.calls = 0
        self.pairs: list[tuple[str, str]] = []

    def verdict(self, p: Any, first: str, second: str) -> Verdict:
        self.calls += 1
        self.pairs.append((first, second))
        if first.startswith("reply from"):
            return "A"
        if second.startswith("reply from"):
            return "B"
        return "tie"


class World:
    """One synthetic account whose stored 2C replies cover two knowledge probes and one tone probe."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.checkpoints = CheckpointRegistry(tmp_path / "checkpoints")
        self.checkpoint = CheckpointRef(NAME, USER, NIGHT, f"modal-volume://adapters/{NAME}", None)
        self.checkpoints.promote(USER, self.checkpoint)
        self.adapters = tmp_path / "adapters"
        _write_adapter(self.adapters / NAME)
        self.replies = tmp_path / "replies" / f"{NIGHT.isoformat()}.jsonl"
        self.replies.parent.mkdir(parents=True)
        self.texts = {"q1": "the notes-only answer to q1", "q2": "the notes-only answer to q2",
                      "q3": "a tone answer the read leaves out"}
        rows = [{"persona": "film_founder", "checkpoint": NAME},
                *({"probe": probe_id, "arm": "2C", "text": text} for probe_id, text in self.texts.items())]
        self.replies.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.model, self.judge = FakeModel(), Judge()
        self.morning_probes = [probe("q1", "new_work", person=USER), probe("q2", "vocab", person=USER),
                               probe("q3", "tone_hold", person=USER)]
        self.ledger = Ledger(tmp_path / "budget.jsonl")
        self.published: list[tuple[str, str]] = []
        self.wake: Callable[[], WakeResult] | None = None
        self.persona: Callable[[str], str | None] | None = None

    def morning(self, user: str, persona: str, night: date) -> Morning:
        deps = ArmDeps(model=self.model, adapters=self.checkpoints, contexts=FakeContexts(),
                       evidence=lambda uid: StoreSpy([]), logprobs=FakeLogprobs(lambda adapter, messages: "A"),
                       donors={}, until=night)
        return Morning(arms=deps, sealed=lambda uid: self.morning_probes, judge=self.judge, prices=PRICES,
                       root=self.root / "results", wake=self.wake)

    @property
    def ports(self) -> tool.Ports:
        return tool.Ports(morning=self.morning, is_synthetic=lambda user: True, registry=self.checkpoints,
                          ledger=self.ledger, adapters=self.adapters, persona=self.persona,
                          publish=lambda folder, name: self.published.append((name, str(folder))))

    def name(self, value: float) -> str:
        return scale.scaled_name(NAME, value, digest=scale.content_digest(self.adapters / NAME))

    def run(self, *extra: str, stored: bool = True) -> tuple[int, list[str]]:
        said: list[str] = []
        args = ["--user", USER, "--night", NIGHT.isoformat(), *extra]
        if stored:
            args += ["--replies", str(self.replies)]
        code = tool.main(args, ports=self.ports, out=said.append)
        return code, said

    @property
    def target(self) -> Path:
        return self.root / "results" / USER / "scores" / f"adapter-scale-{NIGHT.isoformat()}.json"


def test_the_plan_names_the_copies_calls_costs_and_holds_and_makes_no_call(tmp_path: Path) -> None:
    world = World(tmp_path)
    code, said = world.run("--scales", "0.25,0.5,0.75")
    text = "\n".join(said)
    assert code == 0
    assert "the B matrices" in text and "reused; only arm 4 is generated (8 replies" in text
    assert "1 is the control" in text
    assert "up to 16 calls" in text and "holds:" in text and "reserved before any upload" in text
    assert "expected: $" in text
    assert "writes:" in text and "label development, never the final" in text
    assert world.model.calls == [] and world.judge.calls == 0
    assert world.published == [] and not world.target.exists()
    assert sorted(path.name for path in world.adapters.iterdir()) == [NAME]  # no scaled copy yet


def test_go_generates_only_arm_four_and_judges_the_stored_two_c_replies(tmp_path: Path) -> None:
    world = World(tmp_path)
    code, said = world.run("--scales", "0.25,0.5,0.75", "--go")
    assert code == 0 and said[-1].startswith(f"read: {world.target} ")
    scales = (0.25, 0.5, 0.75, 1.0)  # the control is always added
    names = {world.name(value) for value in scales}
    assert len(world.model.calls) == 2 * len(scales)  # two knowledge probes: arm 4 only
    assert {call["adapter"].id for call in world.model.calls} == names
    assert len(world.published) == len(scales) and {name for name, _ in world.published} == names
    assert world.judge.calls == 2 * 2 * len(scales)  # both orders, two probes, every scale
    for probe_id in ("q1", "q2"):
        seen = sum(text == world.texts[probe_id] for pair in world.judge.pairs for text in pair)
        assert seen == 2 * len(scales)  # the stored 2C text is reused in both orders, never generated
    record = json.loads(world.target.read_text(encoding="utf-8"))
    assert (record["label"], record["kind"], record["baseline"]) == ("development", "adapter_scale", "2C")
    assert record["probes"] == 2 and record["replies_stored"] == 3
    assert record["probe_set"] == "stored-2C"
    assert record["replies_generated"] == 2 * len(scales)
    assert sorted(record["scales"]) == ["0.25", "0.5", "0.75", "1"]
    for value, stat in record["scales"].items():
        assert stat["adapter"] == world.name(float(value))
        assert len(stat["ci"]) == 2 and stat["n"] == 2 and 0.0 <= stat["score"] <= 1.0
        assert stat["score"] == 1.0  # arm 4 is the first side; a swapped pair would score 0
    assert world.run("--scales", "0.25", "--go")[1] == [
        f"refused: {NIGHT.isoformat()} already has an adapter-scale read ({world.target.name})"]


def test_missing_stored_replies_are_refused_before_anything(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.replies.unlink()
    code, said = world.run("--scales", "0.5", "--go")
    assert code == 1 and said[-1].startswith("refused: the stored 2C replies can't be read")
    assert world.model.calls == [] and world.judge.calls == 0 and not world.target.exists()


def test_the_plan_generates_two_c_when_no_store_is_named_and_makes_no_call(tmp_path: Path) -> None:
    world = World(tmp_path)
    code, said = world.run("--persona", "film_founder", "--scales", "0.25,0.5", stored=False)
    text = "\n".join(said)
    assert code == 0
    assert "2C replies: generated in this run" in text and "8 in all" in text
    assert world.model.calls == [] and world.judge.calls == 0
    assert world.published == [] and not world.target.exists()
    assert sorted(path.name for path in world.adapters.iterdir()) == [NAME]


def test_go_generates_two_c_with_the_mornings_own_seed_when_no_store_is_named(tmp_path: Path) -> None:
    world = World(tmp_path)
    code, said = world.run("--persona", "film_founder", "--scales", "0.25,0.5", "--go", stored=False)
    assert code == 0 and said[-1].startswith(f"read: {world.target} ")
    scales = (0.25, 0.5, 1.0)  # the control is always added
    assert len(world.model.calls) == 2 * len(scales) + 2  # arm 4 at every scale, then 2C once a probe
    baseline = world.model.calls[-2:]
    assert [call["adapter"] for call in baseline] == [None, None]
    assert [call["seed"] for call in baseline] == [reply_seed(0, "q1", "2C"), reply_seed(0, "q2", "2C")]
    arm_four = world.model.calls[:-2]
    assert {call["adapter"].id for call in arm_four} == {world.name(value) for value in scales}
    assert [call["seed"] for call in arm_four] == [
        reply_seed(0, probe_id, "4") for value in scales for probe_id in ("q1", "q2")]
    assert world.judge.calls == 2 * 2 * len(scales)
    assert all("reply from base" in pair for pair in world.judge.pairs)  # the generated 2C is judged
    record = json.loads(world.target.read_text(encoding="utf-8"))
    assert record["probe_set"] == "generated-2C" and record["replies_stored"] == 0
    assert record["replies_generated"] == 2 * (1 + len(scales)) and record["replies_arm4"] == 2 * len(scales)


def test_the_one_control_is_always_added_under_its_own_name(tmp_path: Path) -> None:
    world = World(tmp_path)
    code, said = world.run("--scales", "0.5")
    text = "\n".join(said)
    assert code == 0
    assert "by 0.5, 1:" in text and "1 is the control" in text
    assert world.name(1.0) != world.name(0.5) and world.name(1.0) != NAME
    assert world.model.calls == [] and world.published == [] and not world.target.exists()


def test_a_stale_hold_from_a_killed_run_is_settled_before_this_one_reserves(tmp_path: Path) -> None:
    world = World(tmp_path)
    stale = world.ledger.reserve(f"{tool.KEY}{NIGHT.isoformat()}:{USER}:arms:a1", ("modal_usd",),
                                 Bound(1.0, 60.0))
    assert stale is not None
    code, _said = world.run("--scales", "0.5", "--go")
    assert code == 0
    assert world.ledger.settlements()[stale.job] == pytest.approx(stale.bound.amount)
    assert world.ledger.open_holds() == ()


def test_the_night_sweep_already_settles_this_tools_prefix(tmp_path: Path) -> None:
    world = World(tmp_path)
    stale = world.ledger.reserve(f"{tool.KEY}{NIGHT.isoformat()}:{USER}:judge:a1", ("judge_usd",),
                                 Bound(1.0, 60.0))
    assert stale is not None
    assert sweep_stale(world.ledger) == 1  # the key starts with morning:, which the sweep settles
    assert world.ledger.settlements()[stale.job] == pytest.approx(stale.bound.amount)
    assert world.ledger.open_holds() == ()


def test_a_copy_already_on_the_volume_is_reused_not_uploaded(tmp_path: Path) -> None:
    world = World(tmp_path)
    attempted: list[str] = []

    def publish(folder: Path, name: str) -> None:
        assert len(world.ledger.open_holds()) == 2  # the judge's and the arms' holds, before any upload
        attempted.append(name)
        if name == world.name(0.5):
            raise config.ServeConfigError(f"adapter {name} is already on the volume; a night's adapter never changes")

    said: list[str] = []
    code = tool.main(["--user", USER, "--night", NIGHT.isoformat(), "--replies", str(world.replies),
                      "--scales", "0.5", "--go"], ports=replace(world.ports, publish=publish), out=said.append)
    assert code == 0 and said[-1].startswith(f"read: {world.target} ")
    assert attempted == [world.name(0.5), world.name(1.0)]  # a refusal for the first copy didn't stop the run


def test_the_hosts_account_map_names_the_persona_when_no_store_or_flag_does(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.persona = lambda user: "film_founder" if user == USER else None
    code, said = world.run("--scales", "0.5", "--go", stored=False)
    assert code == 0 and said[-1].startswith(f"read: {world.target} ")
    assert len(world.model.calls) == 2 * 2 + 2  # arm 4 at 0.5 and 1, then 2C once a probe
    assert json.loads(world.target.read_text(encoding="utf-8"))["persona"] == "film_founder"


def test_a_serve_config_error_is_a_clean_refusal_and_settles_the_holds(tmp_path: Path) -> None:
    world = World(tmp_path)

    def publish(folder: Path, name: str) -> None:
        raise config.ServeConfigError("an adapter folder holds exactly adapter_config.json, adapter_model.safetensors")

    said: list[str] = []
    code = tool.main(["--user", USER, "--night", NIGHT.isoformat(), "--replies", str(world.replies),
                      "--scales", "0.5", "--go"], ports=replace(world.ports, publish=publish), out=said.append)
    assert code == 1
    assert said[-1] == ("refused: an adapter folder holds exactly adapter_config.json, "
                        "adapter_model.safetensors")
    assert world.model.calls == [] and not world.target.exists()
    assert world.ledger.open_holds() == ()  # the holds reserved before the upload were closed on the way out


def test_a_glimmer_that_does_not_wake_is_refused_without_a_call_or_upload(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.wake = lambda: WakeResult(0.1, False)
    code, said = world.run("--scales", "0.5", "--go")
    assert code == 1
    assert said[-1] == "refused: the serving app did not wake: nothing was generated"
    assert world.model.calls == [] and world.published == [] and not world.target.exists()
    assert sorted(path.name for path in world.adapters.iterdir()) == [NAME]  # no scaled copy written
