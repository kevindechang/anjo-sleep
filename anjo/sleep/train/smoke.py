"""The training stack's CPU smoke run: ``colab_train.py``'s real code path, end to end, on a tiny
random-weight Glimmer, before a paid night finds a broken pin or API the hard way.

It is opt-in and needs a separate Python with the training stack, which the repo's own environment
never has (and pytest never runs this)::

    uv venv --python 3.13 <venv>
    uv pip install --python <venv>/bin/python <TRAIN_PACKAGES...> <TORCH>   # from colab_train.py
    python -m anjo.sleep.train.smoke --python <venv>/bin/python [--tokenizer <folder>]

``--tokenizer`` names a local folder with Glimmer's own ``tokenizer.json``, ``tokenizer_config.json``
and ``chat_template.jinja`` (no weights): the tiny model then trains on Glimmer's real tokenizer and
chat template. The run, all offline (the Hugging Face libraries are told so, and nothing is
downloaded):

1. checks that the training Python has exactly ``TRAIN_PACKAGES`` and ``TORCH`` and, with
   ``--tokenizer``, that its special tokens are ``config.CONTROL_TOKENS``, id for id;
2. builds the tiny model and its tokenizer there (``smoke_model.py``);
3. writes a night's dataset with ``Dataset.to_jsonl`` and its recipe with ``training_config``, the
   host's own code, including one example and one pair longer than ``max_seq_length``;
4. runs ``colab_train.py`` as the VM does, with ``ANJO_TRAIN_SMOKE_MODEL`` naming the tiny model
   (fp32 on the CPU, no 4-bit): night 1 cold with ``sft_dpo``, night 2 a warm start with ``sft``.
   The script checks every row's tokens against the chat template before it trains;
5. checks each receipt (both long rows dropped and counted, every row's tokens checked) and each
   adapter: ``target_modules`` is the text-decoder pattern, and every tensor in
   ``adapter_model.safetensors`` is a LoRA on one of the text decoder's projections, every
   projection of every layer covered, none in the vision tower.

It prints one line per step and exits 1 on the first failed check.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Final

from anjo.sleep.serving import config
from anjo.sleep.train import colab_train, smoke_model
from anjo.sleep.train.dataset import Dataset, Example, Pair
from anjo.sleep.train.lora import DATASET_FILE, DEFAULT_TARGET_MODULES, LoraCfg, training_config

NIGHT: Final = date(2026, 10, 28)
REVISION: Final = "0" * 40  # the recipe names a revision; the smoke run's model is a local folder
MIN_STEPS: Final = 2  # two optimizer steps for SFT on the four rows that fit
LONG_WORDS: Final = 9000  # one word is one token, so these rows exceed max_seq_length (8192)
RUN_TIMEOUT_S: Final = 1800.0
OFFLINE: Final = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
_LORA_KEY: Final = re.compile(
    rf"base_model\.model\.(?P<module>{colab_train.TEXT_DECODER}\.(?P<name>[a-z0-9_]+))\.lora_[AB]\.weight")


class SmokeError(RuntimeError):
    """A check the smoke run makes failed."""


def pins() -> dict[str, str]:
    """Every pinned package and its version: ``TRAIN_PACKAGES`` and ``TORCH``."""
    return dict(pin.split("==", 1) for pin in (*colab_train.TRAIN_PACKAGES, colab_train.TORCH))


def pin_mismatches(found: Mapping[str, str | None]) -> list[str]:
    """The pins the training Python doesn't meet; a local build tag (``+cpu``) is ignored."""
    return [f"{name}=={want} (found {found.get(name)})" for name, want in pins().items()
            if (found.get(name) or "").split("+", 1)[0] != want]


def _installed(python: str) -> dict[str, str | None]:
    script = (
        "import importlib.metadata as m, json, sys\n"
        "def v(n):\n    try:\n        return m.version(n)\n    except m.PackageNotFoundError:\n        return None\n"
        "print(json.dumps({n: v(n) for n in json.loads(sys.argv[1])}))\n"
    )
    done = _run([python, "-I", "-c", script, json.dumps(sorted(pins()))], env=None, label="the version check")
    found: dict[str, str | None] = json.loads(done.stdout.strip().splitlines()[-1])
    return found


def smoke_dataset() -> Dataset:
    """Four short examples (the last without its context, so with no system turn) and one too long
    to train; two pairs and one too long to train."""
    words = smoke_model.WORDS
    prompt = (("system", "you are her"), ("user", "what did i say about the garden"))
    examples = tuple(
        Example(id=f"e{i}", prompt=prompt if i < 3 else prompt[1:], completion=" ".join(words[5 + i:15 + i]),
                context_dropped=i == 3, tokens=40, new=True)
        for i in range(4)
    ) + (Example(id="e-long", prompt=prompt, completion=" ".join([words[13]] * LONG_WORDS),
                 context_dropped=False, tokens=LONG_WORDS, new=True),)
    pairs = tuple(
        Pair(id=f"p{i}", prompt=prompt, chosen="the tomatoes were late this year", rejected="no idea",
             chosen_from="glimmer", rejected_from="glimmer", tokens=40)
        for i in range(2)
    ) + (Pair(id="p-long", prompt=prompt, chosen="that is fine", rejected=" ".join([words[13]] * LONG_WORDS),
              chosen_from="glimmer", rejected_from="glimmer", tokens=LONG_WORDS),)
    return Dataset(examples=examples, tokens=sum(e.tokens for e in examples), dropped={}, pairs=pairs)


def check_control_tokens(folder: Path) -> int:
    """The special tokens in ``folder``'s ``tokenizer.json`` are ``config.CONTROL_TOKENS``, each at
    its id. Returns how many there are."""
    added = json.loads((folder / "tokenizer.json").read_text(encoding="utf-8"))["added_tokens"]
    found = {token["id"]: token["content"] for token in added if token["special"]}
    want = {config.FIRST_SPECIAL_ID + offset: token for offset, token in enumerate(config.CONTROL_TOKENS)}
    if found != want:
        wrong = sorted(set(found.items()) ^ set(want.items()))[:3]
        raise SmokeError(f"the tokenizer's special tokens are not CONTROL_TOKENS: {wrong}")
    return len(found)


def _safetensors_keys(path: Path) -> list[str]:
    """The tensor names in a safetensors file, read from its JSON header (no safetensors package)."""
    with path.open("rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(size))
    return sorted(key for key in header if key != "__metadata__")


def check_adapter(folder: Path, *, layers: int, r: int) -> int:
    """The adapter in ``folder`` targets only the text decoder: its ``target_modules`` is the
    text-decoder pattern and each of its tensors is a LoRA on one of the recipe's projections, all
    ``layers`` of them covered. Returns the number of LoRA modules."""
    meta = json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))
    expected = colab_train.lora_targets(DEFAULT_TARGET_MODULES)
    if meta.get("peft_type") != "LORA" or meta.get("r") != r:
        raise SmokeError(f"the adapter is not a rank-{r} LoRA")
    if meta.get("target_modules") != expected:
        raise SmokeError(f"target_modules is {meta.get('target_modules')!r}, not the text-decoder pattern")
    keys = _safetensors_keys(folder / "adapter_model.safetensors")
    matches = [_LORA_KEY.fullmatch(key) for key in keys]
    outside = [key for key, match in zip(keys, matches) if match is None]
    if outside or not keys:
        raise SmokeError(f"tensors outside the text decoder's projections: {outside[:3] or 'no tensors'}")
    modules = {match.group("module") for match in matches if match is not None}
    names = {match.group("name") for match in matches if match is not None}
    covered = {module.split(".layers.", 1)[1].split(".", 1)[0] for module in modules}
    if not names <= set(DEFAULT_TARGET_MODULES) or covered != {str(i) for i in range(layers)}:
        raise SmokeError(f"the LoRA covers {sorted(names)} on layers {sorted(covered)}")
    return len(modules)


def _run(args: Sequence[str], *, env: Mapping[str, str] | None, label: str) -> subprocess.CompletedProcess[str]:
    done = subprocess.run(list(args), capture_output=True, text=True, timeout=RUN_TIMEOUT_S, check=False,
                          env=None if env is None else dict(env))
    if done.returncode != 0:
        raise SmokeError(f"{label} failed:\n{done.stderr[-4000:]}")
    return done


def _night(python: str, base: Path, work: Path, cfg: Mapping[str, object], data: Dataset,
           prev: Path | None) -> dict[str, object]:
    """One night of ``colab_train.py`` on the tiny model, as the runner would start it."""
    work.mkdir(parents=True)
    dataset = work / DATASET_FILE
    dataset.write_text(data.to_jsonl(pairs=cfg["objective"] == "sft_dpo"), encoding="utf-8")
    inherited = {key: value for key, value in os.environ.items() if not key.startswith("ANJO_TRAIN_")}
    env = {
        **inherited, **OFFLINE, "HF_HOME": str(work / "hf"), "ANJO_TRAIN_CONFIG": json.dumps(cfg),
        "ANJO_TRAIN_DATASET": str(dataset), "ANJO_TRAIN_OUT": str(work / "out"),
        colab_train.SMOKE_ENV: str(base), **({"ANJO_TRAIN_PREV": str(prev)} if prev is not None else {}),
    }
    _run([python, "-I", str(Path(colab_train.__file__))], env=env, label=f"colab_train.py ({work.name})")
    receipt: dict[str, object] = json.loads((work / "out" / colab_train.RECEIPT).read_text(encoding="utf-8"))
    return receipt


def _expect(receipt: Mapping[str, object], **want: object) -> None:
    wrong = {key: receipt.get(key) for key, value in want.items() if receipt.get(key) != value}
    if wrong:
        raise SmokeError(f"the receipt has {wrong}, expected {dict((key, want[key]) for key in wrong)}")


def smoke(python: str, root: Path, tokenizer: Path | None = None) -> list[str]:
    """The whole run in ``root``, on Glimmer's tokenizer in ``tokenizer`` or the tiny one; returns
    the lines it reports."""
    lines = []
    missing = pin_mismatches(_installed(python))
    if missing:
        raise SmokeError(f"the training Python doesn't have the pins: {', '.join(missing)}")
    lines.append("pins: " + ", ".join(f"{name} {version}" for name, version in pins().items()))
    if tokenizer is not None:
        lines.append(f"tokenizer: Glimmer's, {check_control_tokens(tokenizer)} special tokens, "
                     "each one in CONTROL_TOKENS at its id")
    base = root / "base"
    _run([python, "-I", str(Path(smoke_model.__file__)), str(base), *([str(tokenizer)] if tokenizer else [])],
         env={**os.environ, **OFFLINE}, label="smoke_model.py")
    layers = smoke_model.LAYERS
    data = smoke_dataset()
    cold = LoraCfg(objective="sft_dpo", min_optimizer_steps=MIN_STEPS)
    first = _night(python, base, root / "night1", training_config(cold, len(data.examples), warm=False,
                                                                  revision=REVISION), data, None)
    _expect(first, examples=4, dropped_truncated=1, pairs=2, pairs_dropped_truncated=1, steps=MIN_STEPS,
            objective="sft_dpo", gpu="cpu", rows_checked=6)
    if not isinstance(first["dpo_steps"], int) or first["dpo_steps"] < 1:
        raise SmokeError("the DPO pass made no optimizer step")
    modules = check_adapter(root / "night1" / "out", layers=layers, r=cold.r)
    lines.append(f"night 1 (sft_dpo, cold): {first['steps']} SFT steps on {first['examples']} examples "
                 f"({first['dropped_truncated']} too long, dropped), {first['dpo_steps']} DPO steps on "
                 f"{first['pairs']} pairs ({first['pairs_dropped_truncated']} too long, dropped)")
    lines.append(f"tokens: all {first['rows_checked']} rows as the template renders them at reasoning strength "
                 f"{config.reasoning_strength()}: the prompt is the server's generation prompt plus the header "
                 f"{config.USER_CHANNEL!r}, the loss falls on the reply and {config.REPLY_END} only")
    lines.append(f"adapter: {modules} LoRA modules, {layers} text-decoder layers x "
                 f"{modules // layers} projections, none outside the text decoder")
    warm = LoraCfg(objective="sft", min_optimizer_steps=MIN_STEPS)
    second = _night(python, base, root / "night2", training_config(warm, len(data.examples), warm=True,
                                                                   revision=REVISION), data,
                    root / "night1" / "out")
    _expect(second, examples=4, dropped_truncated=1, steps=MIN_STEPS, dpo_steps=0, objective="sft", rows_checked=4)
    check_adapter(root / "night2" / "out", layers=layers, r=warm.r)
    lines.append(f"night 2 (sft, warm from night 1): {second['steps']} SFT steps, adapter checked the same way")
    for night in ("night1", "night2"):
        present = sorted(path.name for path in (root / night / "out").iterdir())
        if not set(config.ADAPTER_FILES) <= set(present):
            raise SmokeError(f"{night} wrote {present}, without the files the runner downloads")
    lines.append(f"target_modules: {colab_train.lora_targets(DEFAULT_TARGET_MODULES)}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs the training stack
    parser = argparse.ArgumentParser(prog="python -m anjo.sleep.train.smoke")
    parser.add_argument("--python", required=True, help="a Python with TRAIN_PACKAGES and TORCH installed")
    parser.add_argument("--tokenizer", type=Path, help="a folder with Glimmer's tokenizer and chat template")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="anjo-train-smoke-") as root:
        try:
            lines = smoke(args.python, Path(root), args.tokenizer)
        except (SmokeError, subprocess.TimeoutExpired) as error:
            print(f"smoke failed: {error}", file=sys.stderr)
            return 1
    print("\n".join([*lines, "smoke passed"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
