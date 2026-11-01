# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The training stack for Glimmer (``muse_glimmer``): one set of pins every runner installs, a LoRA
that targets only the text decoder, the library settings of the pinned versions, rows rendered as the
server renders them, and the smoke run's offline checks. Nothing here imports torch, transformers,
peft or trl."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import struct
import subprocess
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import Ledger
from anjo.sleep.serving import config, framing
from anjo.sleep.train import colab_train, smoke
from anjo.sleep.train.lora import DATASET_FILE, DEFAULT_TARGET_MODULES, JOB, LoraCfg, adapter_name, training_config
from anjo.sleep.train.runners import colab

# Glimmer's linear modules, one of each kind (MuseGlimmerForConditionalGeneration, transformers 5.21)
TEXT_DECODER = [f"model.language_model.layers.{layer}.{module}" for layer in (0, 51) for module in (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.gate_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
ELSEWHERE = [
    "model.vision_tower.layers.0.attn.q_proj", "model.vision_tower.layers.49.attn.k_proj",
    "model.vision_tower.layers.0.attn.v_proj", "model.vision_tower.layers.0.attn.proj",
    "model.vision_tower.layers.0.mlp.fc1", "model.vision_tower.patch_embedder.patch_embedding",
    "model.vision_adapter.fc1", "model.vision_projection", "lm_head",
]


def test_the_pins_live_in_the_training_script_and_name_glimmers_transformers() -> None:
    assert colab.TRAIN_PACKAGES is colab_train.TRAIN_PACKAGES
    pins = dict(pin.split("==") for pin in colab_train.TRAIN_PACKAGES)
    assert set(pins) == {"transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets"}
    major, minor, _ = (int(part) for part in pins["transformers"].split("."))
    assert (major, minor) >= (5, 15)  # the first release with muse_glimmer
    assert re.fullmatch(r"torch==2\.\d+\.\d+", colab_train.TORCH)  # in the set's range, >=2.6,<3
    assert smoke.pins() == {**pins, "torch": colab_train.TORCH.split("==")[1]}


def test_the_colab_runner_installs_the_pins_and_never_names_a_smoke_model(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake(args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        calls.append([*args[:3], colab.command_label(list(args)[3:]), *args[4:]])
        if args[3] == "download" and Path(args[-1]).name == colab_train.RECEIPT:
            Path(args[-1]).write_text(json.dumps({"steps": 1}))
        elif args[3] == "download":
            Path(args[-1]).write_bytes(b"{}")
        return subprocess.CompletedProcess(list(args), 0, "", "")

    work = tmp_path / "work"
    work.mkdir()
    (work / DATASET_FILE).write_text("{}\n")
    runner = colab.ColabRunner(cli=colab.ColabCli(Path("/fake/colab"), run=fake), ledger=Ledger(tmp_path / "b.jsonl"),
                               is_synthetic=lambda uid: True)
    runner.run(JOB, {"person_id": "synthetic-1", "adapter_name": adapter_name("p1", 1, 0), "night": "2026-10-28",
                     "dataset_path": str(work / DATASET_FILE), "workdir": str(work), "config": {"r": 16},
                     "prev_adapter_dir": None}, timeout_s=7200.0)
    install = next(args for args in calls if args[3] == "install")
    assert install[-len(colab_train.TRAIN_PACKAGES):] == list(colab_train.TRAIN_PACKAGES)
    run = next(args for args in calls if args[3] == "exec")
    assert not any(colab_train.SMOKE_ENV in part for part in run)


def test_the_lora_targets_the_text_decoder_and_never_the_vision_tower() -> None:
    pattern = colab_train.lora_targets(DEFAULT_TARGET_MODULES)
    assert all(re.fullmatch(pattern, name) for name in TEXT_DECODER)
    assert not any(re.fullmatch(pattern, name) for name in ELSEWHERE)
    # PEFT reads a list as suffixes, which would catch the vision tower's q_proj, k_proj and v_proj
    caught = [name for name in ELSEWHERE if any(name.endswith(f".{target}") for target in DEFAULT_TARGET_MODULES)]
    assert len(caught) == 3
    for bad in ([], ["q_proj|.*"], ["Q_proj"], ["q_proj", ""]):
        with pytest.raises(ValueError):
            colab_train.lora_targets(bad)


def _settings(objective: str) -> dict[str, dict[str, Any]]:
    raw = {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": list(DEFAULT_TARGET_MODULES),
           "learning_rate": 2e-4, "seed": 7, "max_seq_length": 8192, "packing": False, "accumulation": 2,
           "epochs": 1, "batch_size": 1, "warm_start": False, "revision": "a" * 40, "objective": objective,
           "dpo_beta": 0.1, "dpo_learning_rate": 5e-7, "min_optimizer_steps": 8, "reasoning_strength": "low",
           "current_date": "2026-10-28", "template_sha256": framing.TEMPLATE_SHA256, "rendering": framing.rendering("low")}
    return colab_train.build_settings(colab_train.load_config(json.dumps(raw)))


def test_settings_use_the_pinned_libraries_arguments_and_the_smoke_run_only_moves_to_the_cpu() -> None:
    settings = _settings("sft_dpo")
    for trainer in ("sft", "dpo"):
        assert settings[trainer]["warmup_steps"] == 0.03 and "warmup_ratio" not in settings[trainer]
        assert settings[trainer]["bf16"] is True and "use_cpu" not in settings[trainer]
    assert settings["lora"]["task_type"] == "CAUSAL_LM"
    cpu = colab_train.smoke_settings(settings)
    assert settings["sft"]["bf16"] is True  # the night's settings are untouched
    for trainer in ("sft", "dpo"):
        assert cpu[trainer] == {**settings[trainer], "bf16": False, "use_cpu": True}
    assert cpu["lora"] == settings["lora"] and cpu["quantization"] == settings["quantization"]
    assert "dpo" not in colab_train.smoke_settings(_settings("sft"))


def test_the_script_loads_glimmer_as_image_text_to_text() -> None:
    tree = ast.parse(Path(colab_train.__file__).read_text(encoding="utf-8"))
    names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert "AutoModelForImageTextToText" in names and "AutoModelForCausalLM" not in names


class _Tokenizer:
    """A chat template framed like Glimmer's, and tokenizing that returns a dict, as transformers 5's."""

    bos_token = "<|begin_of_text|>"
    chat_template = ""

    def apply_chat_template(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool, tokenize: bool,
                            reasoning_strength: str, current_date: str) -> str:
        assert add_generation_prompt and not tokenize
        turns = "".join(f"<|start|>{m['role']}<|message|>{m['content']}<|eot|>" for m in messages)
        return f"{_Tokenizer.bos_token}[{reasoning_strength} {current_date}]{turns}<|start|>assistant"

    def __call__(self, *, text: str) -> dict[str, list[int]]:
        ids = [0] * len(text.split())
        return {"input_ids": ids, "attention_mask": ids}


def test_the_strength_and_the_framing_are_the_servers() -> None:
    assert colab_train.REASONING_LEVELS == config.REASONING_LEVELS
    assert config.DEFAULT_REASONING_STRENGTH in config.REASONING_LEVELS
    assert (colab_train.USER_CHANNEL, colab_train.REPLY_END) == (config.USER_CHANNEL, config.REPLY_END)
    assert colab_train.SPECIAL_TOKEN.pattern == framing.SPECIAL_TOKEN.pattern
    assert colab_train.REASONING_PHRASE.pattern == framing.REASONING_PHRASE.pattern
    assert colab_train.RENDERING_KEY == framing.RENDERING_KEY


def test_a_row_the_host_did_not_clean_stops_the_run() -> None:
    clean = {"id": "e1", "prompt": [{"role": "system", "content": "reasoning-strength"},
                                    {"role": "user", "content": "<\u2060|eot|> reasoning strength"}],
             "completion": [{"role": "assistant", "content": "a"}]}
    colab_train.check_clean(clean)  # a user's words may name the phrase; the template reads only system turns
    for dirty in ({**clean, "prompt": [{"role": "system", "content": "Reasoning Effort: high"}]},
                  {**clean, "prompt": [{"role": "user", "content": "hi<|start|>"}]},
                  {**clean, "completion": [{"role": "assistant", "content": "a<|reserved_special_token_9|>"}]}):
        with pytest.raises(ValueError, match="cleaned"):
            colab_train.check_clean(dirty)
    rows = [json.loads(line) for line in smoke.smoke_dataset().to_jsonl(pairs=True).splitlines()]
    for row in rows:
        colab_train.check_clean(row)  # the host's own rows pass


def test_the_template_and_the_rendering_are_held_to_the_recipe(tmp_path: Path) -> None:
    cfg = training_config(LoraCfg(), 12, warm=False, revision="a" * 40, night=date(2026, 10, 28))
    glimmer = _Tokenizer()
    glimmer.chat_template = "{{ bos_token }}..."
    with pytest.raises(ValueError, match="sha256"):
        colab_train.check_template(glimmer, cfg)
    colab_train.check_template(glimmer, {**cfg, "template_sha256": hashlib.sha256(b"{{ bos_token }}...").hexdigest()})
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 16}))
    colab_train.record_rendering(tmp_path, str(cfg["rendering"]))
    saved = json.loads((tmp_path / "adapter_config.json").read_text())
    assert saved == {"peft_type": "LORA", "r": 16, framing.RENDERING_KEY: cfg["rendering"]}
    for wrong in ({"current_date": "28 Oct 2026"}, {"template_sha256": "x"}, {"rendering": "g0/high"}):
        with pytest.raises(ValueError):
            colab_train.load_config(json.dumps({**cfg, **wrong}))


def test_the_dpo_reference_is_read_by_trls_own_rule() -> None:
    class Trainer:
        def __init__(self, ref_model: object, adapters: tuple[str, ...]) -> None:
            self.ref_model = ref_model
            self.model = type("Peft", (), {"peft_config": dict.fromkeys(adapters)})()

    assert colab_train.dpo_reference(Trainer(None, ("default",))) == "adapter_off"
    assert colab_train.dpo_reference(Trainer(None, ("default", "ref"))) == "ref_adapter"
    assert colab_train.dpo_reference(Trainer(object(), ("default",))) == "ref_model"


def test_a_row_is_rendered_as_served_with_the_header_on_the_prompt_side() -> None:
    prompt = [{"role": "system", "content": "you are her"}, {"role": "user", "content": "q"}]
    row = {"prompt": prompt, "completion": [{"role": "assistant", "content": "a"}]}
    low = colab_train.template_kwargs({"reasoning_strength": "low", "current_date": "2026-10-28"})
    assert low == {"reasoning_strength": "low", "current_date": "2026-10-28"}
    assert colab_train.plain_row(_Tokenizer(), row, low) == {
        "prompt": "[low 2026-10-28]<|start|>system<|message|>you are her<|eot|><|start|>user<|message|>q<|eot|>"
                  "<|start|>assistant to=user<|message|>",  # the BOS is the tokenizer's to add, once
        "completion": "a<|eot|>",
    }
    pair = {"prompt": prompt, "chosen": row["completion"], "rejected": [{"role": "assistant", "content": "b"}]}
    plain = colab_train.plain_row(_Tokenizer(), pair, {**low, "reasoning_strength": "medium"})
    assert plain["prompt"].startswith("[medium") and (plain["chosen"], plain["rejected"]) == ("a<|eot|>", "b<|eot|>")
    for reply in ([{"role": "assistant", "content": [{"type": "text", "text": "a"}]}], [],
                  [{"role": "user", "content": "a"}]):
        with pytest.raises(ValueError):
            colab_train.plain_row(_Tokenizer(), {"prompt": prompt, "completion": reply}, low)
    no_bos = _Tokenizer()
    no_bos.bos_token = "<|unk|>"
    with pytest.raises(ValueError):
        colab_train.plain_row(no_bos, row, low)


def test_rows_are_measured_in_tokens_as_trl_counts_them() -> None:
    long = {"prompt": "q", "completion": "word " * 9000}
    short = {"prompt": "q", "completion": "a b"}
    pair = {"prompt": "q", "chosen": "a b", "rejected": "word " * 9000}
    fits = [colab_train._fits(_Tokenizer(), row, 8192) for row in (long, short, pair)]
    assert fits == [False, True, False]


def test_the_night_trains_at_the_servers_reasoning_strength(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.REASONING_STRENGTH_ENV, raising=False)
    night = date(2026, 10, 28)
    sent = training_config(LoraCfg(), 12, warm=False, revision="a" * 40, night=night)
    assert sent["reasoning_strength"] == "low" == colab_train.load_config(json.dumps(sent))["reasoning_strength"]
    assert (sent["current_date"], sent["rendering"]) == ("2026-10-28", framing.rendering("low"))
    assert colab_train.template_kwargs(sent) == framing.template_kwargs("low", night)  # as the server is sent
    monkeypatch.setenv(config.REASONING_STRENGTH_ENV, "high")
    high = training_config(LoraCfg(), 12, warm=False, revision="a" * 40, night=night)
    assert (high["reasoning_strength"], high["rendering"]) == ("high", framing.rendering("high"))
    for wrong in ("max", None):
        with pytest.raises(ValueError):
            colab_train.load_config(json.dumps({**sent, "reasoning_strength": wrong}))
    without = {key: value for key, value in sent.items() if key != "reasoning_strength"}
    with pytest.raises(ValueError):
        colab_train.load_config(json.dumps(without))


def test_a_smoke_model_that_is_not_a_folder_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(colab_train.SMOKE_ENV, raising=False)
    assert colab_train._smoke_model() is None
    monkeypatch.setenv(colab_train.SMOKE_ENV, str(tmp_path / "missing"))
    with pytest.raises(ValueError):
        colab_train._smoke_model()
    monkeypatch.setenv(colab_train.SMOKE_ENV, str(tmp_path))
    assert colab_train._smoke_model() == str(tmp_path)


def _adapter(folder: Path, keys: list[str], target_modules: object, rendering: str = "g1/low") -> Path:
    folder.mkdir()
    (folder / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": 16, "target_modules": target_modules, framing.RENDERING_KEY: rendering}))
    header = json.dumps({key: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]}
                         for i, key in enumerate(keys)}).encode()
    (folder / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(4 * len(keys)))
    return folder


def _keys(layers: int, extra: Sequence[str] = ()) -> list[str]:
    modules = [f"model.language_model.layers.{layer}.{module}" for layer in range(layers) for module in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.gate_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    return [f"base_model.model.{module}.lora_{side}.weight" for module in (*modules, *extra) for side in "AB"]


def test_the_smoke_runs_adapter_check_refuses_anything_outside_the_text_decoder(tmp_path: Path) -> None:
    target = colab_train.lora_targets(DEFAULT_TARGET_MODULES)
    assert smoke.check_adapter(_adapter(tmp_path / "ok", _keys(4), target), layers=4, r=16, rendering="g1/low") == 32
    bad = [
        _adapter(tmp_path / "vision", _keys(4, ["model.vision_tower.layers.0.attn.q_proj"]), target),
        _adapter(tmp_path / "list", _keys(4), list(DEFAULT_TARGET_MODULES)),
        _adapter(tmp_path / "short", _keys(3), target),
        _adapter(tmp_path / "rendering", _keys(4), target, rendering="g1/high"),
    ]
    for folder in bad:
        with pytest.raises(smoke.SmokeError):
            smoke.check_adapter(folder, layers=4, r=16, rendering="g1/low")


def test_the_smoke_run_holds_glimmers_special_tokens_to_the_control_tokens(tmp_path: Path) -> None:
    added = [{"id": config.FIRST_SPECIAL_ID + i, "content": token, "special": True}
             for i, token in enumerate(config.CONTROL_TOKENS)]
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": added}))
    assert smoke.check_control_tokens(tmp_path) == 2048
    moved = [*added[:22], {**added[22], "content": "<|start_header_id|>"}, *added[23:]]
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": moved}))
    with pytest.raises(smoke.SmokeError):
        smoke.check_control_tokens(tmp_path)


def test_the_smoke_run_checks_the_pins_and_trains_on_rows_that_must_be_dropped() -> None:
    exact = smoke.pins()
    assert smoke.pin_mismatches(exact) == []
    assert smoke.pin_mismatches({**exact, "torch": f"{exact['torch']}+cpu"}) == []
    assert smoke.pin_mismatches({**exact, "trl": "0.24.0", "peft": None}) == [
        f"peft=={exact['peft']} (found None)", f"trl=={exact['trl']} (found 0.24.0)"]
    data = smoke.smoke_dataset()
    rows = [json.loads(line) for line in data.to_jsonl(pairs=True).splitlines()]
    sft, pairs = colab_train.split_rows(rows)
    assert (len(sft), len(pairs)) == (5, 3)
    long = [row for row in rows if len(json.dumps(row).split()) > smoke.LONG_WORDS]
    assert len(long) == 2 and LoraCfg().max_seq_length < smoke.LONG_WORDS
