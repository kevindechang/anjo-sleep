# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""``sft_dpo_seed`` with the thought channel in one job (``train.sleep_job``, ``train.sleep_loss.Soft``):
the smoke run's tiny Glimmer (random weights, nothing downloaded) with a LoRA and a projector, trained
by the one loss on CPU. The projector learns from ``sft`` and ``dpo`` only; seeding's teacher reads it as
the night began; its warmup, its DPO tensors, its own rate under the stretch, and the deadline. Skipped
without torch, transformers and peft."""
from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from anjo.sleep.tests.test_sleep_objective import _data, _today, _transcript  # noqa: E402
from anjo.sleep.tests.test_sleep_objective import Transcripts as _Transcripts  # noqa: E402
from anjo.sleep.train import channel_train, colab_train, sleep_job, sleep_loss, smoke_model, terms  # noqa: E402
from anjo.sleep.train.lora import LoraCfg, training_config  # noqa: E402

K, D, WIDTH = 4, 8, 16
USER = ("<|start|>", "user", "<|message|>", "what", "did", "i", "say", "about", "the", "garden", "<|eot|>")
HEADER = ("<|start|>", "assistant")
DAY = ("<|start|>", "system", "<|message|>", "she", "remembers", "the", "tomatoes", "were", "late", "<|eot|>")
REPLY = ("the", "tomatoes", "were", "late", "so", "plant", "sooner", "<|eot|>")
OTHER = ("that", "is", "fine", "<|eot|>")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any]:
    from transformers import AutoModelForImageTextToText, AutoTokenizer  # type: ignore[import-not-found,unused-ignore]

    folder = tmp_path_factory.mktemp("smoke") / "tiny"
    assert smoke_model.main(folder) == 0
    return AutoModelForImageTextToText.from_pretrained(folder, dtype=torch.float32), AutoTokenizer.from_pretrained(folder)


def _reading(seed: int) -> channel_train.Reading:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((K - 1, channel_train.features(D))).astype(np.float32),
            np.array([1.0, 0.6, 0.3], np.float32), rng.standard_normal(2 * D).astype(np.float32), 0.8)


class _Night:
    """The tiny model with a fresh LoRA and a fresh projector on it, the rows as ids, and the run's ``Soft``."""

    def __init__(self, tiny: tuple[Any, Any], *, warmup: float = 0.0) -> None:
        model, tokenizer = tiny
        torch.manual_seed(0)
        self.model = peft.get_peft_model(copy.deepcopy(model), peft.LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=colab_train.lora_targets(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                     "down_proj"])))
        self.eot = tokenizer.convert_tokens_to_ids("<|eot|>")
        ids = {name: tuple(tokenizer.convert_tokens_to_ids(["<|begin_of_text|>", *words] if name in ("user", "day")
                                                           else list(words)))
               for name, words in (("user", (*USER, *HEADER)), ("day", (*DAY, *USER, *HEADER)), ("reply", REPLY),
                                   ("other", OTHER))}
        self.prompt, self.teacher, self.reply, self.other = ids["user"], ids["day"], ids["reply"], ids["other"]
        readings = {"sft:e/own": _reading(1), "seed:e/own": _reading(1), "pair:p/own": _reading(2),
                    "empty": channel_train.readings_of([], "state", K, D)["empty"]}
        anchors = channel_train.anchor_rows(tokenizer, self.model.get_input_embeddings(), K)
        params = channel_train.feature_stats(channel_train.fresh_params(K, D, WIDTH, anchors, 0, hidden=64),
                                             [readings["sft:e/own"], readings["pair:p/own"]])
        self.channel = channel_train.Channel(self.model, params, device=torch.device("cpu"), eot=self.eot)
        self.soft = sleep_loss.Soft(self.channel, readings,
                                    tuple(self.channel.projector_groups(projector_lr=1e-2, anchor_lr=1e-3)),
                                    tuple(self.channel.params[n] for n in channel_train.LAYERS), warmup)

    def plan(self, **changes: Any) -> Any:
        return replace(sleep_loss.Plan(lr=1e-2, min_steps=2, beta=0.1, seed=3, eot=self.eot, pad=0, dpo_lr=1e-2),
                       **changes)

    def sft(self) -> Any:
        return sleep_loss.Seq(self.prompt, self.reply, state="sft:e/own")

    def seeds(self, n: int = 4) -> tuple[Any, ...]:
        return tuple(sleep_loss.SeedSeq(self.prompt, self.teacher, self.reply, "seed:e/own") for _ in range(n))

    def lora(self) -> list[Any]:
        return [p.detach().clone() for p in self.model.parameters() if p.requires_grad]

    def frozen(self) -> dict[str, Any]:
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if not p.requires_grad}


@dataclass(frozen=True)
class _Spied(sleep_loss.Soft):
    """The run's ``Soft``, keeping how each term asked for its tokens: (as the night began, detached)."""

    asked: list[tuple[bool, bool]] = field(default_factory=list)

    def tokens(self, state: str | None, *, ref: bool = False, detach: bool = False) -> Any:
        self.asked.append((ref, detach))
        return super().tokens(state, ref=ref, detach=detach)


@contextmanager
def _steps() -> Iterator[list[tuple[list[float], list[Any]]]]:
    """Each optimizer step's group rates and the trainable model weights after it."""
    seen: list[tuple[list[float], list[Any]]] = []

    def hook(optimizer: Any, args: Any, kwargs: Any) -> None:
        tensors = [p.detach().clone() for g in optimizer.param_groups[:1] for p in g["params"]]
        seen.append(([float(g["lr"]) for g in optimizer.param_groups], tensors))

    from torch.optim.optimizer import register_optimizer_step_post_hook  # type: ignore[import-not-found,unused-ignore]

    handle = register_optimizer_step_post_hook(hook)
    try:
        yield seen
    finally:
        handle.remove()


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(np.array_equal(a[n], b[n]) for n in a)


def test_a_spliced_reply_scores_as_the_channels_own_scoring_and_samples_from_the_embeddings(tiny: tuple[Any, Any]) -> None:
    night = _Night(tiny)
    tokens = (night.channel, night.channel.soft(_reading(1)).detach())
    with torch.no_grad():
        spliced = sleep_loss.reply_logprobs(night.model, night.prompt, night.reply, tokens).sum()
        served = night.channel.logps([(night.prompt, night.reply, tokens[1])], grad=False)[0]
        plain = sleep_loss.reply_logprobs(night.model, night.prompt, night.reply).sum()
    assert torch.allclose(spliced, served, atol=1e-4) and not torch.allclose(spliced, plain, atol=1e-3)
    torch.manual_seed(0)
    drawn = sleep_loss.sample(night.model, night.prompt, 6, night.plan(), tokens)
    assert 0 < len(drawn) <= 6 and all(isinstance(t, int) for t in drawn)  # new tokens only, never the prompt's


def test_sft_and_dpo_teach_the_projector_and_seeding_never_does(tiny: tuple[Any, Any]) -> None:
    # Arrange
    seeding, joint = _Night(tiny), _Night(tiny)
    start, base = seeding.channel.arrays(), seeding.frozen()
    lora = seeding.lora()
    spied = _Spied(*(getattr(seeding.soft, name) for name in ("channel", "readings", "groups", "dpo_params")))

    # Act
    alone = sleep_loss.train_terms(seeding.model, sleep_loss.Batch(seeds=seeding.seeds()), seeding.plan(), soft=spied)
    both = sleep_loss.train_terms(joint.model, sleep_loss.Batch(sft=(joint.sft(),), seeds=joint.seeds()), joint.plan(),
                                  soft=joint.soft)

    # Assert: seeding moved the LoRA and not one projector array; with an SFT row the projector moved,
    # and the teacher (night-start adapter and projector) still never did
    assert _same(start, seeding.channel.arrays()) and alone["on_policy_samples"] > 0 and alone["rows"]["imitate"] > 0
    assert set(spied.asked) == {(True, False), (False, True)}  # the teacher as the night began, the student detached
    assert any(not torch.equal(a, b) for a, b in zip(lora, seeding.lora(), strict=True))
    assert not _same(start, joint.channel.arrays()) and both["teacher_drift"] == 0.0 == alone["teacher_drift"]
    assert both["seed_kl_after"] != both["seed_kl_before"] and both["grads_adapter_only"]
    assert all(torch.equal(p, base[n]) for n, p in seeding.model.named_parameters() if n in base)  # the base never moves
    assert alone["projector_only_updates"] == 0 and not alone["out_of_time"]


def test_the_projector_learns_alone_first_then_with_the_lora(tiny: tuple[Any, Any]) -> None:
    night = _Night(tiny, warmup=0.75)
    lora = night.lora()
    with _steps() as seen:
        done = sleep_loss.train_terms(night.model, sleep_loss.Batch(sft=(night.sft(),)), night.plan(min_steps=4),
                                      soft=night.soft)
    assert done["steps"] == 4 and done["projector_only_updates"] == 3
    held = [all(torch.equal(a, b) for a, b in zip(lora, after, strict=True)) for _, after in seen]
    assert held == [True, True, True, False]  # three updates of the projector alone, then the LoRA joins


def test_the_dpo_block_trains_the_projectors_mlp_against_the_night_start_reference(tiny: tuple[Any, Any]) -> None:
    night = _Night(tiny)
    start = night.channel.arrays()
    pair = sleep_loss.PairSeq(night.prompt, night.reply, night.other, state="pair:p/own")
    with _steps() as seen:
        done = sleep_loss.train_terms(night.model, sleep_loss.Batch(pairs=(pair,) * 2), night.plan(), soft=night.soft)
    after = night.channel.arrays()
    assert done["dpo"]["steps"] == len(seen) > 0 and all(len(rates) == 1 for rates, _ in seen)  # its own AdamW, one group
    assert all(not np.array_equal(start[n], after[n]) for n in channel_train.LAYERS)
    assert all(np.array_equal(start[n], after[n]) for n in (*channel_train.ANCHORS, *channel_train.SCALING))
    with torch.no_grad(), night.model.disable_adapter():  # the reference: the adapter off, the projector as it began
        ref = sleep_loss.reply_logprobs(night.model, night.prompt, night.reply, night.soft.tokens("pair:p/own", ref=True))
        now = sleep_loss.reply_logprobs(night.model, night.prompt, night.reply, night.soft.tokens("pair:p/own"))
    assert not torch.allclose(ref, now)


def test_the_stretch_keeps_each_group_at_its_own_peak_and_the_deadline_stops_the_loop(tiny: tuple[Any, Any]) -> None:
    night = _Night(tiny)
    clock = iter(range(10_000))
    with _steps() as seen:
        sleep_loss.train_terms(night.model, sleep_loss.Batch(sft=(night.sft(),)), night.plan(stretch_s=30.0),
                               soft=night.soft, timer=lambda: float(next(clock)))
    assert seen and all(rates[1] == pytest.approx(rates[0]) and rates[2] == pytest.approx(rates[0] / 10)
                        for rates, _ in seen)  # LoRA 1e-2, projector MLP 1e-2, anchors 1e-3, each scaled alike
    late = _Night(tiny)
    lora = late.lora()
    done = sleep_loss.train_terms(late.model, sleep_loss.Batch(sft=(late.sft(),), seeds=late.seeds(1)),
                                  late.plan(deadline=-1.0), soft=late.soft)
    assert done["steps"] == 0 and done["out_of_time"] and all(torch.equal(a, b) for a, b in zip(lora, late.lora(), strict=True))


class _Ids:
    """A stand-in tokenizer (``test_sleep_objective._Tokens``'s rule): role marks, then word lengths."""

    def apply_chat_template(self, messages: list[dict[str, str]], *, add_generation_prompt: bool, tokenize: bool,
                            return_dict: bool, **template: str) -> list[int]:
        ids = [1]
        for message in messages:
            ids += [{"system": 7, "user": 8}[message["role"]], *(len(w) for w in message["content"].split())]
        return [*ids, 9] if add_generation_prompt else ids

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": [5] if text == colab_train.USER_CHANNEL else [len(w) for w in text.split()]}

    def convert_tokens_to_ids(self, token: str) -> int:
        return 11


def test_the_rows_reach_the_job_naming_their_state_and_a_seeded_channel_recipe_is_accepted() -> None:
    reading = {"k": K, "d": D, "slot_x": "", "gates": "", "global_x": "", "speak": 0.5}
    core = json.dumps({"v": 1, "reading": reading})
    made = [_transcript(0)]
    data = _data(made)
    data = replace(data, examples=tuple(replace(e, core=core) for e in data.examples))
    seeds, _ = terms.seed_rows(data, _Transcripts(made), [_today(frozenset())], [], hosted_in_training=False)
    assert seeds and all(row.core == core for row in seeds)
    written = [json.loads(line) for line in replace(data, seeds=seeds).to_jsonl(terms=True).splitlines()]
    plain, _ = colab_train.split_rows(written)
    by_kind = sleep_job.term_rows(colab_train, written)
    assert by_kind["seed"][0]["core"] == plain[0]["core"] == json.loads(core)
    keys = [sleep_job.state_key(row) for row in (plain[0], by_kind["seed"][0], {"id": "x", "chosen": [], "core": core})]
    assert keys == [f"sft:{plain[0]['id']}/own", f"seed:{plain[0]['id']}/own", "pair:x/own"]  # one id, two keys
    assert sleep_job.state_key({"id": "y"}) is None
    cfg = {"max_seq_length": 8192, "terms": {"fold_rank": None, "dreams": False}, "channel": {"k": K}}
    template = {"reasoning_strength": "low", "current_date": "2026-10-24"}
    batch, _ = sleep_job.seed_batch(colab_train, sleep_loss, _Ids(), [(plain[0], {})], [], by_kind, cfg, template)
    assert batch.sft[0].state == keys[0] and batch.seeds[0].state == keys[1]
    short, too_long = sleep_job.seed_batch(colab_train, sleep_loss, _Ids(), [], [], by_kind,
                                           {**cfg, "max_seq_length": len(batch.seeds[0].teacher) + len(batch.seeds[0].reply)},
                                           template)
    assert not short.seeds and too_long["seed"] == 1  # the K soft tokens count against the length
    assert colab_train.seed_module().KINDS == sleep_job.KINDS  # loaded by path, as the job runs under python -I


def test_a_seeded_recipe_with_the_channel_is_one_job() -> None:
    block = channel_train.block(k=K, d=D, width=WIDTH, arm="state", exec_s=3600.0)
    sent = training_config(LoraCfg(objective="sft_dpo_seed", fold_rank=32), 4, warm=True, revision="a" * 40,
                           night=date(2026, 10, 24), channel=block)
    cfg = colab_train.load_config(json.dumps(sent))  # the channel and the one loss in one recipe, refused before
    assert cfg["channel"] == block and cfg["terms"] == {"fold_rank": 32, "dreams": False}
